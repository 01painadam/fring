//! T1 of ICR-011: along-track error of the Wisdom-Holman split after one
//! Julian year at the production step, from the exact circular orbit, no
//! moons, f64. The drift about mu_eff (a0 = 140,220 km) must stay under
//! 0.5 km. The drift about GM must not: more than 10 km (the power arm).

use fring_model::physics::{SaturnField, circular_speed, frequencies, wh_steps};
use fring_model::{SaturnParams, Vec3};
use serde::Deserialize;

const A0_KM: f64 = 140_220.0;
const STEPS: u64 = 88_582;
const YEAR_S: f64 = 31_557_600.0;

#[derive(Deserialize)]
struct SaturnFile {
    saturn: SaturnParams,
}

fn params() -> SaturnParams {
    toml::from_str::<SaturnFile>(include_str!("../../../params/saturn.toml"))
        .unwrap()
        .saturn
}

/// Along-track error in km after `steps` steps of `h`: r (phi - n r t).
fn along_track_error_km(field: &SaturnField<f64>, r_km: f64, h: f64, steps: u64) -> f64 {
    let v = circular_speed(field, r_km);
    let n = frequencies(field, r_km).n_rad_s;
    let (mut p, mut vel) = (Vec3::new(r_km, 0.0, 0.0), Vec3::new(0.0, v, 0.0));
    wh_steps(field, &mut p, &mut vel, h, steps).unwrap();
    let t = h * steps as f64;
    let phi = p.y.atan2(p.x);
    let want = (n * t).rem_euclid(std::f64::consts::TAU);
    let mut d = phi - want;
    d -= std::f64::consts::TAU * (d / std::f64::consts::TAU).round();
    r_km * d
}

#[test]
fn drift_about_mu_eff_keeps_the_phase_and_gm_does_not() {
    let sp = params();
    let eff = SaturnField::<f64>::from_params(&sp).with_drift_radius(A0_KM);
    let old = SaturnField::<f64>::from_params(&sp);
    let h = YEAR_S / STEPS as f64;
    for r in [139_500.0, 140_220.0, 141_500.0] {
        let e = along_track_error_km(&eff, r, h, STEPS);
        let o = along_track_error_km(&old, r, h, STEPS);
        eprintln!("r {r}: mu_eff {e:+.4} km, GM {o:+.3} km");
        assert!(e.abs() < 0.5, "r {r}: mu_eff error {e} km");
        assert!(o.abs() > 10.0, "r {r}: GM error {o} km has no power");
    }
}
