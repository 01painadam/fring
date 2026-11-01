//! Task C2: energy of the Wisdom-Holman step for a circular orbit at a0.
//! Same measures as the leapfrog test E2 (`energy.rs`, ADR 0002): the
//! largest |dE/E0| over all steps and the envelope drift.

use fring_model::physics::{SaturnField, circular_speed, wh_step};
use fring_model::{SaturnParams, Vec3};
use serde::Deserialize;

const A0_KM: f64 = 140_222.43;

#[derive(Deserialize)]
struct SaturnFile {
    saturn: SaturnParams,
}

fn field() -> SaturnField<f64> {
    let text = include_str!("../../../params/saturn.toml");
    SaturnField::from_params(&toml::from_str::<SaturnFile>(text).unwrap().saturn)
}

/// Least-squares slope of `y` against its index, times the index span.
fn slope_times_span(y: &[f64]) -> f64 {
    let n = y.len() as f64;
    let mean_k = (n - 1.0) / 2.0;
    let mean_y = y.iter().sum::<f64>() / n;
    let (mut sxy, mut sxx) = (0.0, 0.0);
    for (k, v) in y.iter().enumerate() {
        let dk = k as f64 - mean_k;
        sxy += dk * (v - mean_y);
        sxx += dk * dk;
    }
    sxy / sxx * (n - 1.0)
}

/// Runs `n_orbits` orbits of 150 steps and returns (max |dE/E0|, envelope drift).
fn run(vel_km_s: Vec3<f64>, n_orbits: u64) -> (f64, f64) {
    let f = field();
    let vc = circular_speed(&f, A0_KM);
    let steps = 150_u64;
    let h = 2.0 * std::f64::consts::PI * A0_KM / vc / steps as f64;
    let mut pos = Vec3::new(A0_KM, 0.0, 0.0);
    let mut vel = vel_km_s;
    let e0 = f.specific_energy(pos, vel);
    let mut max_err = 0.0_f64;
    let (mut hi, mut lo) = (Vec::new(), Vec::new());
    for _ in 0..n_orbits {
        let (mut mx, mut mn) = (f64::MIN, f64::MAX);
        for _ in 0..steps {
            wh_step(&f, &mut pos, &mut vel, h).unwrap();
            let err = (f.specific_energy(pos, vel) - e0) / e0.abs();
            mx = mx.max(err);
            mn = mn.min(err);
            max_err = max_err.max(err.abs());
        }
        hi.push(mx);
        lo.push(mn);
    }
    let drift = slope_times_span(&hi).abs().max(slope_times_span(&lo).abs());
    (max_err, drift)
}

/// Circular start at a0 (v = sqrt(r |a_r|)), 10^4 orbits.
#[test]
fn wisdom_holman_energy_circular_is_bounded_and_does_not_drift() {
    let vc = circular_speed(&field(), A0_KM);
    let (max_err, drift) = run(Vec3::new(0.0, vc, 0.0), 10_000);
    println!(
        "WH circular, 150 steps/orbit, 10^4 orbits: max |dE/E0| = {max_err:.3e}; drift = {drift:.3e}"
    );
    assert!(max_err < 1e-11, "max error {max_err:e}");
    assert!(drift < 1e-10, "drift {drift:e}");
}

/// Eccentric and inclined start (e = i = 0.01), 10^3 orbits. Here the kicks
/// matter, so the energy oscillates at the level of the split error.
#[test]
fn wisdom_holman_energy_eccentric_inclined_is_bounded_and_does_not_drift() {
    let f = field();
    let fr = fring_model::physics::frequencies(&f, A0_KM);
    let vel = Vec3::new(
        0.01 * A0_KM * fr.kappa_rad_s,
        A0_KM * fr.n_rad_s,
        0.01 * A0_KM * fr.nu_rad_s,
    );
    let (max_err, drift) = run(vel, 1_000);
    println!(
        "WH e = i = 0.01, 150 steps/orbit, 10^3 orbits: max |dE/E0| = {max_err:.3e}; drift = {drift:.3e}"
    );
    assert!(max_err < 5e-8, "max error {max_err:e}");
    assert!(drift < 1e-10, "drift {drift:e}");
}
