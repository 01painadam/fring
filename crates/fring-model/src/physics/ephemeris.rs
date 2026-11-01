//! Building and interpolating the moon ephemeris (task C3).
//!
//! [`build_ephemeris`] integrates the moons with [`Ias15`] (f64) and samples
//! the dense output at equal intervals into a [`MoonEphemeris`], with states
//! and accelerations. [`interpolate_moons`] reads it back with the quintic
//! Hermite polynomial of [`hermite5`].

use super::hermite::hermite5;
use super::ias15::Ias15;
use super::moon_dynamics::MoonSystem;
use crate::{BodyState, Error, MoonEphemeris, MoonState, MoonTrack, Result, SaturnParams, Vec3};

/// Default table interval in seconds (ICR-002): quintic Hermite error below
/// 1e-3 km at mid-interval for Prometheus and Pandora.
pub const DEFAULT_INTERVAL_S: f64 = 1800.0;

/// Default IAS15 accuracy parameter.
pub const DEFAULT_EPSILON: f64 = 1.0e-9;

/// Name, GM and initial state of one moon (inertial, Saturn-centred, at t = 0).
#[derive(Clone, Debug, PartialEq)]
pub struct MoonInit {
    /// Name, for example "prometheus".
    pub name: String,
    /// GM in km³ s⁻².
    pub gm_km3_s2: f64,
    /// Position (km) and velocity (km/s) at t = 0.
    pub state: BodyState,
}

/// Integrates the moons from t = 0 and tabulates them every `interval_s`
/// seconds, with at least `duration_s` seconds covered (the last sample is at
/// or after `duration_s`). Uses [`DEFAULT_EPSILON`].
pub fn build_ephemeris(
    saturn: &SaturnParams,
    moons: &[MoonInit],
    duration_s: f64,
    interval_s: f64,
) -> Result<MoonEphemeris> {
    build_ephemeris_with(saturn, moons, duration_s, interval_s, DEFAULT_EPSILON)
}

/// As [`build_ephemeris`], with the IAS15 accuracy parameter `epsilon`.
pub fn build_ephemeris_with(
    saturn: &SaturnParams,
    moons: &[MoonInit],
    duration_s: f64,
    interval_s: f64,
    epsilon: f64,
) -> Result<MoonEphemeris> {
    saturn.validate()?;
    if moons.is_empty() {
        return Ok(MoonEphemeris::none());
    }
    if !(duration_s.is_finite() && duration_s > 0.0 && interval_s.is_finite() && interval_s > 0.0) {
        return Err(Error::Input(format!(
            "ephemeris: duration {duration_s} s and interval {interval_s} s must be positive"
        )));
    }
    let n_samples_f = (duration_s / interval_s).ceil() + 1.0;
    if n_samples_f > 1.0e8 {
        return Err(Error::Input(format!(
            "ephemeris: {n_samples_f} samples is too many (duration {duration_s} s, interval {interval_s} s)"
        )));
    }
    let n_samples = n_samples_f as usize;
    for m in moons {
        if !(m.gm_km3_s2.is_finite() && m.gm_km3_s2 > 0.0) {
            return Err(Error::Input(format!(
                "moon {:?}: GM {}",
                m.name, m.gm_km3_s2
            )));
        }
    }

    let n = moons.len();
    let system = MoonSystem::new(saturn, moons.iter().map(|m| m.gm_km3_s2).collect());
    let mut force = |x: &[f64], a: &mut [f64]| system.accelerations(x, a);

    let mut x0 = Vec::with_capacity(3 * n);
    let mut v0 = Vec::with_capacity(3 * n);
    for m in moons {
        let (p, v) = (m.state.position_km, m.state.velocity_km_s);
        x0.extend([p.x, p.y, p.z]);
        v0.extend([v.x, v.y, v.z]);
    }
    // First trial step: 1 % of the shortest dynamical time sqrt(r³/GM).
    let dt0 = moons
        .iter()
        .map(|m| {
            let r = m.state.position_km.norm();
            0.01 * (r * r * r / saturn.gm_km3_s2).sqrt()
        })
        .fold(f64::INFINITY, f64::min);
    let mut ias = Ias15::new(epsilon, 0.0, &x0, &v0, dt0)?;

    let mut tracks: Vec<MoonTrack> = moons
        .iter()
        .map(|m| MoonTrack {
            name: m.name.clone(),
            gm_km3_s2: m.gm_km3_s2,
            samples: Vec::with_capacity(n_samples),
            accelerations_km_s2: Vec::with_capacity(n_samples),
        })
        .collect();

    let mut x = vec![0.0; 3 * n];
    let mut v = vec![0.0; 3 * n];
    let mut a = vec![0.0; 3 * n];
    let push = |tracks: &mut Vec<MoonTrack>, x: &[f64], v: &[f64], a: &[f64]| {
        for (i, track) in tracks.iter_mut().enumerate() {
            let k = 3 * i;
            track.samples.push(BodyState {
                position_km: Vec3::new(x[k], x[k + 1], x[k + 2]),
                velocity_km_s: Vec3::new(v[k], v[k + 1], v[k + 2]),
            });
            track
                .accelerations_km_s2
                .push(Vec3::new(a[k], a[k + 1], a[k + 2]));
        }
    };

    // Sample 0 is the initial state.
    system.accelerations(&x0, &mut a);
    push(&mut tracks, &x0, &v0, &a);
    let mut next = 1;
    while next < n_samples {
        let dt = ias.step(&mut force)?;
        if !(dt.is_finite() && dt > 0.0) {
            return Err(Error::Backend(format!("ephemeris: bad step {dt} s")));
        }
        while next < n_samples && next as f64 * interval_s <= ias.time_s() {
            let t = next as f64 * interval_s;
            ias.dense_output(t, &mut x, &mut v);
            system.accelerations(&x, &mut a);
            push(&mut tracks, &x, &v, &a);
            next += 1;
        }
    }
    Ok(MoonEphemeris {
        start_s: 0.0,
        interval_s,
        tracks,
    })
}

/// Moon states at time `t_s` (seconds since the epoch), by quintic Hermite
/// interpolation of the table.
///
/// Returns [`Error::Input`] if `t_s` is not finite or outside
/// `[start_s, end_s()]`.
pub fn interpolate_moons(eph: &MoonEphemeris, t_s: f64) -> Result<MoonState> {
    let mut moons = vec![BodyState::default(); eph.tracks.len()];
    interpolate_moons_into(eph, t_s, &mut moons)?;
    Ok(MoonState { time_s: t_s, moons })
}

/// As [`interpolate_moons`], but writes into `out` (one entry per track) and
/// does not allocate. The ephemeris is not validated here; call
/// [`MoonEphemeris::validate`] once when it is built.
pub fn interpolate_moons_into(eph: &MoonEphemeris, t_s: f64, out: &mut [BodyState]) -> Result<()> {
    if out.len() != eph.tracks.len() {
        return Err(Error::Input(format!(
            "interpolate_moons: output has {} entries for {} moons",
            out.len(),
            eph.tracks.len()
        )));
    }
    if eph.tracks.is_empty() {
        return Ok(());
    }
    let n = eph.sample_count();
    if n < 2 {
        return Err(Error::Input(
            "interpolate_moons: fewer than 2 samples".into(),
        ));
    }
    let end = eph.end_s();
    if !(t_s >= eph.start_s && t_s <= end) {
        return Err(Error::Input(format!(
            "interpolate_moons: time {t_s} s is outside the ephemeris [{}, {end}] s",
            eph.start_s
        )));
    }
    let k = (((t_s - eph.start_s) / eph.interval_s).floor() as usize).min(n - 2);
    let s = (t_s - eph.sample_time_s(k)) / eph.interval_s;
    for (track, o) in eph.tracks.iter().zip(out.iter_mut()) {
        let (a, b) = (&track.samples[k], &track.samples[k + 1]);
        let (p, v) = hermite5(
            a.position_km,
            a.velocity_km_s,
            track.accelerations_km_s2[k],
            b.position_km,
            b.velocity_km_s,
            track.accelerations_km_s2[k + 1],
            eph.interval_s,
            s,
        );
        *o = BodyState {
            position_km: p,
            velocity_km_s: v,
        };
    }
    Ok(())
}
