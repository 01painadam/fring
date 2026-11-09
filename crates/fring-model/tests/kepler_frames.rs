//! Task C2: Kepler drift, Wisdom-Holman step and frame conversions.

use fring_model::physics::{
    LocalState, SaturnField, circular_speed, dkd_step, inertial_to_rotating, kepler_drift,
    local_to_rotating, reference_mean_motion, rotating_to_inertial, rotating_to_local, wh_step,
    wh_steps,
};
use fring_model::{SaturnParams, Vec3};
use serde::Deserialize;

const A0_KM: f64 = 140_222.43;

#[derive(Deserialize)]
struct SaturnFile {
    saturn: SaturnParams,
}

fn params() -> SaturnParams {
    let text = include_str!("../../../params/saturn.toml");
    toml::from_str::<SaturnFile>(text).unwrap().saturn
}

fn dist(a: Vec3<f64>, b: Vec3<f64>) -> f64 {
    Vec3::new(a.x - b.x, a.y - b.y, a.z - b.z).norm()
}

/// A tilted, eccentric bound orbit (e about 0.1) at about a0.
fn eccentric_state() -> (Vec3<f64>, Vec3<f64>, f64) {
    let gm = params().gm_km3_s2;
    let pos = Vec3::new(0.9 * A0_KM, 0.1 * A0_KM, 0.02 * A0_KM);
    let vc = (gm / pos.norm()).sqrt();
    let vel = Vec3::new(-0.15 * vc, 0.95 * vc, 0.05 * vc);
    let alpha = 2.0 / pos.norm() - vel.norm_squared() / gm;
    let a = 1.0 / alpha;
    let period = 2.0 * std::f64::consts::PI * (a * a * a / gm).sqrt();
    (pos, vel, period)
}

#[test]
fn kepler_drift_returns_after_one_period() {
    let gm = params().gm_km3_s2;
    let (p, v, period) = eccentric_state();
    let (p1, v1) = kepler_drift(gm, p, v, period).unwrap();
    let ep = dist(p1, p) / p.norm();
    let ev = dist(v1, v) / v.norm();
    println!("Kepler period return: position {ep:.2e}, velocity {ev:.2e}");
    assert!(ep < 1e-12 && ev < 1e-12);
    // Three periods and a back step.
    let (p3, _) = kepler_drift(gm, p, v, 3.0 * period).unwrap();
    assert!(dist(p3, p) / p.norm() < 1e-11);
    let (q, w) = kepler_drift(gm, p, v, 0.3 * period).unwrap();
    let (pb, vb) = kepler_drift(gm, q, w, -0.3 * period).unwrap();
    assert!(dist(pb, p) / p.norm() < 1e-12 && dist(vb, v) / v.norm() < 1e-12);
}

#[test]
fn kepler_drift_agrees_with_a_fine_leapfrog() {
    let sp = params();
    let point = SaturnField::<f64> {
        j2: 0.0,
        j4: 0.0,
        j6: 0.0,
        ..SaturnField::from_params(&sp)
    };
    let (p, v, period) = eccentric_state();
    let dt = period / 20.0;
    let (pk, vk) = kepler_drift(sp.gm_km3_s2, p, v, dt).unwrap();
    let (mut pl, mut vl) = (p, v);
    let n = 20_000;
    for _ in 0..n {
        dkd_step(&point, &mut pl, &mut vl, dt / n as f64);
    }
    let ep = dist(pk, pl) / p.norm();
    println!("Kepler drift versus 20000-step leapfrog: {ep:.2e}");
    assert!(ep < 1e-8);
    assert!(dist(vk, vl) / v.norm() < 1e-8);
}

#[test]
fn kepler_drift_f32_agrees_with_f64() {
    let gm = params().gm_km3_s2;
    let (p, v, period) = eccentric_state();
    let dt = period / 150.0;
    let (p64, v64) = kepler_drift(gm, p, v, dt).unwrap();
    let (p32, v32) = kepler_drift(gm as f32, p.cast::<f32>(), v.cast::<f32>(), dt as f32).unwrap();
    let ep = dist(p32.cast::<f64>(), p64) / p.norm();
    let ev = dist(v32.cast::<f64>(), v64) / v.norm();
    println!("Kepler drift f32 versus f64: position {ep:.2e}, velocity {ev:.2e}");
    assert!(ep < 5e-6 && ev < 5e-6);
}

#[test]
fn kepler_drift_unbound_returns_none() {
    let gm = params().gm_km3_s2;
    let p = Vec3::new(A0_KM, 0.0, 0.0);
    let v_esc = (2.0 * gm / A0_KM).sqrt();
    assert!(kepler_drift(gm, p, Vec3::new(0.0, v_esc, 0.0), 10.0).is_none());
    assert!(kepler_drift(gm, p, Vec3::new(0.0, 1.5 * v_esc, 0.0), 10.0).is_none());
    assert!(kepler_drift(gm, p, Vec3::new(0.0, f64::NAN, 0.0), 10.0).is_none());
}

#[test]
fn acceleration_splits_into_point_mass_and_perturbation() {
    let f = SaturnField::<f64>::from_params(&params());
    let p = Vec3::new(0.7 * A0_KM, -0.5 * A0_KM, 0.3 * A0_KM);
    let full = f.acceleration(p);
    let pert = f.acceleration_without_point_mass(p);
    let r3 = p.norm().powi(3);
    let pm = Vec3::new(
        -f.gm_km3_s2 * p.x / r3,
        -f.gm_km3_s2 * p.y / r3,
        -f.gm_km3_s2 * p.z / r3,
    );
    let sum = Vec3::new(pm.x + pert.x, pm.y + pert.y, pm.z + pert.z);
    assert!(dist(full, sum) / full.norm() < 1e-14);
    // The perturbation is about J2 (R/r)^2 of the full force.
    assert!(pert.norm() / full.norm() < 1e-2);
}

#[test]
fn wh_steps_equals_repeated_wh_step_and_f32_follows_f64() {
    let sp = params();
    let f64f = SaturnField::<f64>::from_params(&sp);
    let f32f = SaturnField::<f32>::from_params(&sp);
    let vc = circular_speed(&f64f, A0_KM);
    let h = 2.0 * std::f64::consts::PI * A0_KM / vc / 150.0;
    let start = (Vec3::new(A0_KM, 0.0, 10.0), Vec3::new(0.01, vc, 0.002));
    let (mut p1, mut v1) = start;
    for _ in 0..150 {
        wh_step(&f64f, &mut p1, &mut v1, h).unwrap();
    }
    let (mut p2, mut v2) = start;
    wh_steps(&f64f, &mut p2, &mut v2, h, 150).unwrap();
    assert!(dist(p1, p2) < 1e-6, "{} km", dist(p1, p2));
    assert!(dist(v1, v2) < 1e-11);
    let (mut p3, mut v3) = (start.0.cast::<f32>(), start.1.cast::<f32>());
    wh_steps(&f32f, &mut p3, &mut v3, h as f32, 150).unwrap();
    let d = dist(p3.cast::<f64>(), p2);
    println!("WH one orbit, f32 versus f64: {d:.3} km");
    assert!(d < 5.0, "{d} km");
}

#[test]
fn wh_step_unbound_leaves_state_unchanged() {
    let f = SaturnField::<f64>::from_params(&params());
    let (mut p, mut v) = (Vec3::new(A0_KM, 0.0, 0.0), Vec3::new(0.0, 40.0, 0.0));
    assert!(wh_step(&f, &mut p, &mut v, 360.0).is_none());
    assert_eq!(
        (p, v),
        (Vec3::new(A0_KM, 0.0, 0.0), Vec3::new(0.0, 40.0, 0.0))
    );
}

// ---- frames ----

fn n0() -> f64 {
    reference_mean_motion(&SaturnField::<f64>::from_params(&params()), A0_KM)
}

#[test]
fn frames_round_trip_f64() {
    let n0 = n0();
    let p = Vec3::new(-100_000.0, 85_000.0, 1_234.0);
    let v = Vec3::new(3.2, -8.1, 0.05);
    for t in [0.0, 1.0e4, 3.7e7, 9.9e8] {
        let (pr, vr) = inertial_to_rotating(p, v, n0, t);
        let (pi, vi) = rotating_to_inertial(pr, vr, n0, t);
        assert!(dist(pi, p) / p.norm() < 1e-12, "t = {t}");
        assert!(dist(vi, v) / v.norm() < 1e-12, "t = {t}");
    }
    let (pr, vr) = inertial_to_rotating(p, v, n0, 5.0e6);
    let l = rotating_to_local(pr, vr, A0_KM);
    let (pb, vb) = local_to_rotating(&l, A0_KM);
    assert!(dist(pb, pr) / pr.norm() < 1e-12);
    assert!(dist(vb, vr) / vr.norm() < 1e-12);
}

#[test]
fn frames_round_trip_f32() {
    let p = Vec3::new(-100_000.0_f32, 85_000.0, 1_234.0);
    let v = Vec3::new(3.2_f32, -8.1, 0.05);
    let n0 = n0() as f32;
    let t = 5.0e5_f32;
    let (pr, vr) = inertial_to_rotating(p, v, n0, t);
    let (pi, vi) = rotating_to_inertial(pr, vr, n0, t);
    let e = |a: Vec3<f32>, b: Vec3<f32>| {
        dist(a.cast::<f64>(), b.cast::<f64>()) / b.cast::<f64>().norm()
    };
    println!("frame round trip f32: {:.2e} {:.2e}", e(pi, p), e(vi, v));
    assert!(e(pi, p) < 1e-5 && e(vi, v) < 1e-5);
    let l = rotating_to_local(pr, vr, A0_KM as f32);
    let (pb, vb) = local_to_rotating(&l, A0_KM as f32);
    assert!(e(pb, pr) < 1e-5 && e(vb, vr) < 1e-5);
}

#[test]
fn identity_at_time_zero() {
    let p = Vec3::new(1.0e5, -2.0e4, 30.0);
    let v = Vec3::new(1.0, 2.0, 3.0);
    // With n0 t = 0 only the velocity term remains: v_rot = v - n0 z x r.
    let n0 = n0();
    let (pr, vr) = inertial_to_rotating(p, v, n0, 0.0);
    assert_eq!(pr, p);
    assert!((vr.x - (v.x + n0 * p.y)).abs() < 1e-15);
    assert!((vr.y - (v.y - n0 * p.x)).abs() < 1e-15);
    assert_eq!(vr.z, v.z);
}

#[test]
fn quarter_turn() {
    // After a quarter turn of the frame, the inertial x axis is the rotating -y axis.
    let n0 = 1.0e-4;
    let t = std::f64::consts::FRAC_PI_2 / n0;
    let (pr, _) = inertial_to_rotating(Vec3::new(1.0, 0.0, 0.0), Vec3::zero(), n0, t);
    assert!(pr.x.abs() < 1e-15 && (pr.y + 1.0).abs() < 1e-15);
}

#[test]
fn circular_orbit_at_a0_is_at_rest_in_the_rotating_frame() {
    let f = SaturnField::<f64>::from_params(&params());
    let n0 = reference_mean_motion(&f, A0_KM);
    let v = circular_speed(&f, A0_KM);
    assert!((v / A0_KM - n0).abs() / n0 < 1e-14);
    let t = 123_456.0;
    let th = n0 * t;
    let p = Vec3::new(A0_KM * th.cos(), A0_KM * th.sin(), 0.0);
    let vel = Vec3::new(-v * th.sin(), v * th.cos(), 0.0);
    let (pr, vr) = inertial_to_rotating(p, vel, n0, t);
    assert!((pr.x - A0_KM).abs() < 1e-8 && pr.y.abs() < 1e-8);
    assert!(vr.norm() < 1e-12, "{}", vr.norm());
    let l = rotating_to_local(pr, vr, A0_KM);
    assert!(l.x_km.abs() < 1e-8 && l.y_km.abs() < 1e-8);
    assert!(l.vx_km_s.abs() < 1e-12 && l.vy_km_s.abs() < 1e-12);
}

#[test]
fn particle_at_rest_inertially_moves_backwards_in_the_rotating_frame() {
    let n0 = n0();
    let p = Vec3::new(0.0, 140_000.0, 0.0);
    let (pr, vr) = inertial_to_rotating(p, Vec3::zero(), n0, 0.0);
    // v_rot = -n0 z x r = -n0 (-y, x, 0) = (n0 y, 0, 0) for r = (0, y, 0).
    assert_eq!(pr, p);
    assert!((vr.x - n0 * 140_000.0).abs() < 1e-12 && vr.y.abs() < 1e-12);
}

#[test]
fn local_coordinates_known_values_and_wrap() {
    // A point at 90 degrees, 10 km outside a0, moving outward at 1 m/s and
    // forward at 2 m/s (counter-clockwise).
    let rho = A0_KM + 10.0;
    let pos = Vec3::new(0.0, rho, 5.0);
    let vel = Vec3::new(-0.002, 0.001, 0.0003);
    let l = rotating_to_local(pos, vel, A0_KM);
    assert!((l.x_km - 10.0).abs() < 1e-9);
    assert!((l.y_km - A0_KM * std::f64::consts::FRAC_PI_2).abs() < 1e-8);
    assert_eq!(l.z_km, 5.0);
    assert!((l.vx_km_s - 0.001).abs() < 1e-15);
    // phi_dot = 0.002 / rho, so vy = a0 * 0.002 / rho
    assert!((l.vy_km_s - A0_KM * 0.002 / rho).abs() < 1e-15);
    assert_eq!(l.vz_km_s, 0.0003);
    // phi wraps to (-pi, pi]: the point on the negative x axis has phi = pi.
    let l = rotating_to_local(Vec3::new(-A0_KM, 0.0, 0.0), Vec3::zero(), A0_KM);
    assert!((l.y_km - A0_KM * std::f64::consts::PI).abs() < 1e-8);
    let l = rotating_to_local(Vec3::new(-A0_KM, -1e-6, 0.0), Vec3::zero(), A0_KM);
    assert!(l.y_km < 0.0 && l.y_km > -A0_KM * std::f64::consts::PI - 1e-8);
    let _ = LocalState::<f64>::default();
}
