//! Task C1: energy conservation of the leapfrog for a circular orbit at a0.
//! Decision and numbers: `docs/adr/0002-c1-energy-test.md`.

use fring_model::physics::{SaturnField, circular_speed, dkd_step, leapfrog_circular_speed};
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

/// What a run measured. Errors are (E - E0) / |E0|, in units of the relative energy.
struct Run {
    /// Largest |error| over all steps.
    max_err: f64,
    /// Secular drift: the least-squares slope over the orbits of the per-orbit
    /// maximum (and of the per-orbit minimum) of the error, times the span;
    /// the larger of the two, as |.|.
    ///
    /// The envelope is used because the error at the orbit ends alone aliases
    /// the slow beat between the step phase and the epicycle phase (see the ADR).
    drift: f64,
}

/// Runs `n_orbits` orbits of `steps_per_orbit` steps of `h_s` seconds from
/// `(a0, 0, 0)` with tangential speed `v`.
fn run(field: &SaturnField<f64>, v: f64, steps_per_orbit: u64, n_orbits: u64, h_s: f64) -> Run {
    let mut pos = Vec3::new(A0_KM, 0.0, 0.0);
    let mut vel = Vec3::new(0.0, v, 0.0);
    let e0 = field.specific_energy(pos, vel);
    let mut max_err = 0.0_f64;
    let mut hi = Vec::with_capacity(n_orbits as usize);
    let mut lo = Vec::with_capacity(n_orbits as usize);
    for _ in 0..n_orbits {
        let (mut mx, mut mn) = (f64::MIN, f64::MAX);
        for _ in 0..steps_per_orbit {
            dkd_step(field, &mut pos, &mut vel, h_s);
            let err = (field.specific_energy(pos, vel) - e0) / e0.abs();
            mx = mx.max(err);
            mn = mn.min(err);
            max_err = max_err.max(err.abs());
        }
        hi.push(mx);
        lo.push(mn);
    }
    let drift = slope_times_span(&hi).abs().max(slope_times_span(&lo).abs());
    Run { max_err, drift }
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

/// Step for `steps_per_orbit` steps per orbit, with the orbit time of the
/// continuous circular orbit at a0.
fn step_size(field: &SaturnField<f64>, steps_per_orbit: u64) -> f64 {
    let vc = circular_speed(field, A0_KM);
    2.0 * std::f64::consts::PI * A0_KM / vc / steps_per_orbit as f64
}

/// E1: start on the leapfrog's own circular orbit. Every step is a pure
/// rotation, so only rounding is left.
#[test]
fn e1_leapfrog_circular_orbit_conserves_energy_to_1e_10() {
    let f = field();
    let h = step_size(&f, 150);
    let vc = circular_speed(&f, A0_KM);
    let vh = leapfrog_circular_speed(&f, A0_KM, h).unwrap();
    let r = run(&f, vh, 150, 10_000, h);
    println!(
        "E1: v_h - v_c = {:.6e} km/s ({:.3e} relative); max |dE/E| = {:.3e}; drift = {:.3e}",
        vh - vc,
        (vh - vc) / vc,
        r.max_err,
        r.drift
    );
    assert!(r.max_err < 1e-10, "max error {:e}", r.max_err);
    assert!(r.drift < 1e-10, "drift {:e}", r.drift);
}

/// E2: start on the exact continuous circular orbit (v = sqrt(r a_r)). The
/// energy oscillates with a bounded amplitude and does not drift.
#[test]
fn e2_continuous_circular_start_is_bounded_and_does_not_drift() {
    let f = field();
    let h = step_size(&f, 150);
    let vc = circular_speed(&f, A0_KM);
    let r = run(&f, vc, 150, 10_000, h);
    println!(
        "E2: 150 steps/orbit, 10^4 orbits: max |dE/E| = {:.3e}; drift = {:.3e}",
        r.max_err, r.drift
    );
    assert!(r.max_err < 5e-7, "max error {:e}", r.max_err);
    assert!(r.drift < 1e-10, "drift {:e}", r.drift);
}

/// Halving the step must shrink the oscillation. The start is off the
/// leapfrog circular orbit by delta ~ h^2, and the energy error is of second
/// order in delta, so the amplitude scales as h^4 (a factor 16), not h^2.
#[test]
fn e2_amplitude_scales_as_h_to_the_fourth() {
    let f = field();
    let vc = circular_speed(&f, A0_KM);
    let a150 = run(&f, vc, 150, 1000, step_size(&f, 150)).max_err;
    let a300 = run(&f, vc, 300, 1000, step_size(&f, 300)).max_err;
    let ratio = a150 / a300;
    println!(
        "E2 scaling: amplitude {a150:.3e} (150/orbit), {a300:.3e} (300/orbit), ratio {ratio:.2}"
    );
    assert!((12.8..=19.2).contains(&ratio), "ratio {ratio}");
}
