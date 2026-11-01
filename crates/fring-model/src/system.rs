//! The fixed background of a run: Saturn and the pre-computed moon ephemeris.
//!
//! Moons are integrated before the run (f64, CPU) and given to every backend
//! as a table of states at equal time intervals (`docs/PLAN.md` section 6.3).
//! Backends interpolate the table (quintic Hermite, from positions, velocities
//! and accelerations; ICR-002).
//! So the CPU and GPU backends see the same moons. All states are inertial.

use crate::{Error, Result, Vec3};
use serde::{Deserialize, Serialize};

/// Saturn's gravity field: point mass plus J2, J4 and J6 (`docs/PLAN.md`
/// section 7.1; J6 since ICR-001). Values come from `params/saturn.toml`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SaturnParams {
    /// GM of Saturn in km³ s⁻².
    pub gm_km3_s2: f64,
    /// Reference radius R in km for which J2 and J4 are defined.
    pub reference_radius_km: f64,
    /// Zonal harmonic J2 (no unit).
    pub j2: f64,
    /// Zonal harmonic J4 (no unit).
    pub j4: f64,
    /// Zonal harmonic J6 (no unit).
    pub j6: f64,
}

impl SaturnParams {
    /// Checks that GM and R are positive and J2, J4, J6 are finite.
    pub fn validate(&self) -> Result<()> {
        let ok = self.gm_km3_s2.is_finite()
            && self.gm_km3_s2 > 0.0
            && self.reference_radius_km.is_finite()
            && self.reference_radius_km > 0.0
            && self.j2.is_finite()
            && self.j4.is_finite()
            && self.j6.is_finite();
        if ok {
            Ok(())
        } else {
            Err(Error::Input(format!(
                "Saturn parameters out of range: {self:?}"
            )))
        }
    }
}

/// Position and velocity of one body, inertial frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BodyState {
    /// Position in km.
    pub position_km: Vec3<f64>,
    /// Velocity in km/s.
    pub velocity_km_s: Vec3<f64>,
}

/// The ephemeris of one moon: states at `start_s + k × interval_s`.
#[derive(Clone, Debug, PartialEq)]
pub struct MoonTrack {
    /// Name, as in the config `[moons] include` list (for example "prometheus").
    pub name: String,
    /// GM of the moon in km³ s⁻².
    pub gm_km3_s2: f64,
    /// State at each sample time, in time order.
    pub samples: Vec<BodyState>,
    /// Inertial acceleration at each sample time in km s⁻² (same length as
    /// `samples`). With the states it gives the quintic Hermite
    /// interpolation (ICR-002).
    pub accelerations_km_s2: Vec<Vec3<f64>>,
}

/// Pre-computed states of all moons at equal time intervals.
///
/// Sample `k` of every track is at time `start_s + k × interval_s`. All tracks
/// have the same number of samples (at least 2). A run without moons has no
/// tracks ([`MoonEphemeris::none`]).
#[derive(Clone, Debug, PartialEq)]
pub struct MoonEphemeris {
    /// Time of sample 0, in seconds since the epoch.
    pub start_s: f64,
    /// Time between samples in seconds.
    pub interval_s: f64,
    /// One track per moon, in the order of the config `[moons] include` list.
    pub tracks: Vec<MoonTrack>,
}

impl MoonEphemeris {
    /// An ephemeris without moons.
    pub fn none() -> Self {
        Self {
            start_s: 0.0,
            interval_s: 1.0,
            tracks: Vec::new(),
        }
    }

    /// Number of samples per track (0 if there are no moons).
    pub fn sample_count(&self) -> usize {
        self.tracks.first().map_or(0, |t| t.samples.len())
    }

    /// Time of sample `k` in seconds since the epoch.
    pub fn sample_time_s(&self, k: usize) -> f64 {
        self.start_s + k as f64 * self.interval_s
    }

    /// Time of the last sample. A backend cannot interpolate after this time.
    pub fn end_s(&self) -> f64 {
        self.sample_time_s(self.sample_count().saturating_sub(1))
    }

    /// Checks the table: positive interval, at least 2 samples, the same
    /// number of samples in every track, one acceleration per sample,
    /// positive GM, finite states and accelerations.
    pub fn validate(&self) -> Result<()> {
        let fail = |msg: String| Err(Error::Input(format!("moon ephemeris: {msg}")));
        if !(self.start_s.is_finite() && self.interval_s.is_finite() && self.interval_s > 0.0) {
            return fail(format!(
                "start_s {} and interval_s {} must be finite, interval_s > 0",
                self.start_s, self.interval_s
            ));
        }
        let n = self.sample_count();
        for track in &self.tracks {
            if track.accelerations_km_s2.len() != track.samples.len() {
                return fail(format!(
                    "track {:?} has {} accelerations for {} samples",
                    track.name,
                    track.accelerations_km_s2.len(),
                    track.samples.len()
                ));
            }
            if track.samples.len() != n || n < 2 {
                return fail(format!(
                    "track {:?} has {} samples; every track needs the same number, at least 2",
                    track.name,
                    track.samples.len()
                ));
            }
            if !(track.gm_km3_s2.is_finite() && track.gm_km3_s2 > 0.0) {
                return fail(format!("track {:?} has GM {}", track.name, track.gm_km3_s2));
            }
            let finite = |v: Vec3<f64>| v.x.is_finite() && v.y.is_finite() && v.z.is_finite();
            if let Some(k) = track
                .samples
                .iter()
                .position(|s| !(finite(s.position_km) && finite(s.velocity_km_s)))
            {
                return fail(format!("track {:?} sample {k} is not finite", track.name));
            }
            if let Some(k) = track.accelerations_km_s2.iter().position(|a| !finite(*a)) {
                return fail(format!(
                    "track {:?} acceleration {k} is not finite",
                    track.name
                ));
            }
        }
        Ok(())
    }
}

/// Moon states at one time, as returned by [`Backend::moons`](crate::Backend::moons).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MoonState {
    /// Time in seconds since the epoch.
    pub time_s: f64,
    /// One state per moon, in the order of [`MoonEphemeris::tracks`].
    pub moons: Vec<BodyState>,
}

/// Everything that is fixed for a run: Saturn and the moon ephemeris.
#[derive(Clone, Debug, PartialEq)]
pub struct System {
    /// Saturn's gravity field.
    pub saturn: SaturnParams,
    /// Pre-computed moon states.
    pub moons: MoonEphemeris,
}

impl System {
    /// Checks Saturn's parameters and the moon ephemeris.
    pub fn validate(&self) -> Result<()> {
        self.saturn.validate()?;
        self.moons.validate()
    }
}
