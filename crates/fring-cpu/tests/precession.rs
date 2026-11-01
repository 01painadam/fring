//! Task C2: apsidal precession and nodal regression of an integrated orbit
//! against `docs/PLAN.md` section 7.2 (`physics::frequencies`).
//!
//! Method: start at radius r with e = i = 1e-3 (epicycle amplitudes), record
//! the time and longitude of each pericentre (r.v from negative to positive)
//! and each ascending node (z from negative to positive), interpolated
//! linearly inside the step, and fit the unwrapped longitude against time by
//! least squares. The slopes are the rates of the apse and the node.

use fring_model::physics::{SaturnField, dkd_step, frequencies, wh_step};
use fring_model::{SaturnParams, Vec3};
use serde::Deserialize;

const ECC: f64 = 1e-3;
const INC_RAD: f64 = 1e-3;
const N_ORBITS: u64 = 300;
const STEPS_PER_ORBIT: u64 = 150;
const RADII_KM: [f64; 3] = [100_000.0, 140_222.43, 180_000.0];

#[derive(Deserialize)]
struct SaturnFile {
    saturn: SaturnParams,
}

fn field() -> SaturnField<f64> {
    let text = include_str!("../../../params/saturn.toml");
    SaturnField::from_params(&toml::from_str::<SaturnFile>(text).unwrap().saturn)
}

/// Measured rates in rad/s.
struct Rates {
    apse: f64,
    node: f64,
}

/// Least-squares slope of the unwrapped longitudes against time.
fn rate(events: &[(f64, f64)]) -> f64 {
    let mut unwrapped = vec![events[0].1];
    for &(_, lon) in &events[1..] {
        let last = *unwrapped.last().unwrap();
        let d = (lon - last + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI)
            - std::f64::consts::PI;
        unwrapped.push(last + d);
    }
    let n = events.len() as f64;
    let tm = events.iter().map(|e| e.0).sum::<f64>() / n;
    let lm = unwrapped.iter().sum::<f64>() / n;
    let (mut sxy, mut sxx) = (0.0, 0.0);
    for (e, l) in events.iter().zip(&unwrapped) {
        sxy += (e.0 - tm) * (l - lm);
        sxx += (e.0 - tm) * (e.0 - tm);
    }
    sxy / sxx
}

/// Integrates with `step` (one step of `h` seconds) and measures the rates.
fn measure(
    field: &SaturnField<f64>,
    r_km: f64,
    h_s: f64,
    mut step: impl FnMut(&mut Vec3<f64>, &mut Vec3<f64>),
) -> Rates {
    let f = frequencies(field, r_km);
    let mut pos = Vec3::new(r_km, 0.0, 0.0);
    let mut vel = Vec3::new(
        ECC * r_km * f.kappa_rad_s,
        r_km * f.n_rad_s,
        INC_RAD * r_km * f.nu_rad_s,
    );
    let (mut peri, mut node) = (Vec::new(), Vec::new());
    let mut rdot_prev = pos.dot(vel);
    let mut t = 0.0;
    for _ in 0..N_ORBITS * STEPS_PER_ORBIT {
        let (p0, z0) = (pos, pos.z);
        step(&mut pos, &mut vel);
        t += h_s;
        let rdot = pos.dot(vel);
        let lon = |frac: f64| {
            let (a0, a1) = (p0.y.atan2(p0.x), pos.y.atan2(pos.x));
            let d = (a1 - a0 + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI)
                - std::f64::consts::PI;
            a0 + frac * d
        };
        if rdot_prev < 0.0 && rdot >= 0.0 {
            let frac = -rdot_prev / (rdot - rdot_prev);
            peri.push((t - h_s + frac * h_s, lon(frac)));
        }
        if z0 < 0.0 && pos.z >= 0.0 {
            let frac = -z0 / (pos.z - z0);
            node.push((t - h_s + frac * h_s, lon(frac)));
        }
        rdot_prev = rdot;
    }
    Rates {
        apse: rate(&peri),
        node: rate(&node),
    }
}

fn step_size(r_km: f64, f: &SaturnField<f64>) -> f64 {
    2.0 * std::f64::consts::PI / frequencies(f, r_km).n_rad_s / STEPS_PER_ORBIT as f64
}

#[test]
fn wisdom_holman_precession_within_0_1_percent() {
    let field = field();
    for r in RADII_KM {
        let f = frequencies(&field, r);
        let h = step_size(r, &field);
        let m = measure(&field, r, h, |p, v| {
            wh_step(&field, p, v, h).unwrap();
        });
        let apse_exp = f.n_rad_s - f.kappa_rad_s;
        let node_exp = -(f.nu_rad_s - f.n_rad_s);
        let (ea, en) = (m.apse / apse_exp - 1.0, m.node / node_exp - 1.0);
        println!(
            "WH 150/orbit, r = {r}: apse {:.6e} rad/s (expected {apse_exp:.6e}) error {ea:+.2e}; \
             node {:.6e} (expected {node_exp:.6e}) error {en:+.2e}",
            m.apse, m.node
        );
        assert!(ea.abs() < 1e-3, "apse error {ea:e} at r = {r}");
        assert!(en.abs() < 1e-3, "node error {en:e} at r = {r}");
    }
}

/// For the record (ADR 0003): the plain leapfrog is far outside 0.1 % at 150
/// steps per orbit. Printed, not asserted.
#[test]
fn leapfrog_bias_for_the_record() {
    let field = field();
    for r in RADII_KM {
        let f = frequencies(&field, r);
        let h = step_size(r, &field);
        let m = measure(&field, r, h, |p, v| dkd_step(&field, p, v, h));
        let ea = m.apse / (f.n_rad_s - f.kappa_rad_s) - 1.0;
        let en = m.node / (-(f.nu_rad_s - f.n_rad_s)) - 1.0;
        println!("leapfrog 150/orbit, r = {r}: apse error {ea:+.2e}, node error {en:+.2e}");
    }
}
