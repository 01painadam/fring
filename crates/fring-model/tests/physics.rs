//! Task C1: the analytic acceleration of Saturn's field against a
//! finite-difference gradient of the potential, the same code in f32 and f64,
//! and the leapfrog step.

use fring_model::physics::{
    SaturnField, circular_speed, dkd_step, frequencies, leapfrog_circular_speed,
};
use fring_model::{SaturnParams, Vec3};
use serde::Deserialize;

#[derive(Deserialize)]
struct SaturnFile {
    saturn: SaturnParams,
}

fn params() -> SaturnParams {
    let text = include_str!("../../../params/saturn.toml");
    toml::from_str::<SaturnFile>(text).unwrap().saturn
}

/// Test points: radii in units of R, latitudes in degrees, several longitudes.
fn points() -> Vec<Vec3<f64>> {
    let r_ref = params().reference_radius_km;
    let mut v = Vec::new();
    for &r_over_r in &[1.05, 1.5, 140_222.43 / r_ref, 2.5, 8.0] {
        for &lat_deg in &[0.0_f64, 5.0, 30.0, 60.0, 89.0, -45.0] {
            for &lon_deg in &[0.0_f64, 37.0, 122.0, 251.0] {
                let r = r_over_r * r_ref;
                let (lat, lon) = (lat_deg.to_radians(), lon_deg.to_radians());
                v.push(Vec3::new(
                    r * lat.cos() * lon.cos(),
                    r * lat.cos() * lon.sin(),
                    r * lat.sin(),
                ));
            }
        }
    }
    v
}

/// Relative error |a - (-grad Phi)| / |a| with central differences, step 1e-5 r.
fn gradient_error(field: &SaturnField<f64>, p: Vec3<f64>) -> f64 {
    let h = 1e-5 * p.norm();
    let phi = |q: Vec3<f64>| field.potential(q);
    let d = |dx: f64, dy: f64, dz: f64| {
        let plus = phi(Vec3::new(p.x + dx, p.y + dy, p.z + dz));
        let minus = phi(Vec3::new(p.x - dx, p.y - dy, p.z - dz));
        -(plus - minus) / (2.0 * h)
    };
    let fd = Vec3::new(d(h, 0.0, 0.0), d(0.0, h, 0.0), d(0.0, 0.0, h));
    let a = field.acceleration(p);
    let diff = Vec3::new(a.x - fd.x, a.y - fd.y, a.z - fd.z);
    diff.norm() / a.norm()
}

#[test]
fn gradient_matches_finite_difference() {
    let full = SaturnField::<f64>::from_params(&params());
    let only = |j2: f64, j4: f64, j6: f64| SaturnField { j2, j4, j6, ..full };
    let cases = [
        ("all", full),
        ("point mass", only(0.0, 0.0, 0.0)),
        ("J2 only", only(full.j2, 0.0, 0.0)),
        ("J4 only", only(0.0, full.j4, 0.0)),
        ("J6 only", only(0.0, 0.0, full.j6)),
    ];
    for (name, field) in cases {
        let worst = points()
            .into_iter()
            .map(|p| gradient_error(&field, p))
            .fold(0.0, f64::max);
        println!("gradient, {name}: worst relative error {worst:.3e}");
        assert!(worst < 1e-8, "{name}: {worst:e}");
    }
}

#[test]
fn j6_changes_the_force_at_the_expected_size() {
    // Guards against a J term that is silently ignored: J6 alone must move
    // the force at the F ring by about 35/16 J6 q^6 (relative).
    let full = SaturnField::<f64>::from_params(&params());
    let no_j6 = SaturnField { j6: 0.0, ..full };
    let p = Vec3::new(140_222.43, 0.0, 0.0);
    let q = full.reference_radius_km / p.x;
    // |a| in units of the point-mass value GM/r^2.
    let k = full.gm_km3_s2 / (p.x * p.x);
    let expected = 35.0 / 16.0 * full.j6 * q.powi(6);
    let got = (full.acceleration(p).x - no_j6.acceleration(p).x).abs() / k;
    assert!((got - expected).abs() < 1e-8 * expected, "{got} {expected}");
}

#[test]
fn equatorial_force_agrees_with_section_7_2() {
    let field = SaturnField::<f64>::from_params(&params());
    for r in [80_000.0, 140_222.43, 300_000.0] {
        let v = circular_speed(&field, r);
        let f = frequencies(&field, r);
        let rel = (v / r - f.n_rad_s).abs() / f.n_rad_s;
        assert!(rel < 1e-14, "r = {r}: {rel:e}");
        // A circular orbit has kappa, nu close to n (within a few 1e-3).
        assert!((f.kappa_rad_s / f.n_rad_s - 1.0).abs() < 0.05);
        assert!((f.nu_rad_s / f.n_rad_s - 1.0).abs() < 0.05);
    }
}

#[test]
fn f32_agrees_with_f64() {
    let p = params();
    let f64_field = SaturnField::<f64>::from_params(&p);
    let f32_field = SaturnField::<f32>::from_params(&p);
    let mut worst = 0.0_f64;
    for pos in points() {
        let a64 = f64_field.acceleration(pos);
        let a32 = f32_field.acceleration(pos.cast::<f32>()).cast::<f64>();
        let diff = Vec3::new(a32.x - a64.x, a32.y - a64.y, a32.z - a64.z);
        worst = worst.max(diff.norm() / a64.norm());
    }
    println!("f32 acceleration versus f64: worst relative difference {worst:.3e}");
    assert!(worst < 1e-5, "{worst:e}");
}

#[test]
fn f32_leapfrog_follows_f64_for_one_orbit() {
    let p = params();
    let f64_field = SaturnField::<f64>::from_params(&p);
    let f32_field = SaturnField::<f32>::from_params(&p);
    let a0 = 140_222.43;
    let v = circular_speed(&f64_field, a0);
    let period = 2.0 * std::f64::consts::PI * a0 / v;
    let h = period / 150.0;
    let (mut p64, mut v64) = (Vec3::new(a0, 0.0, 0.0), Vec3::new(0.0, v, 0.0));
    let (mut p32, mut v32) = (p64.cast::<f32>(), v64.cast::<f32>());
    for _ in 0..150 {
        dkd_step(&f64_field, &mut p64, &mut v64, h);
        dkd_step(&f32_field, &mut p32, &mut v32, h as f32);
    }
    let q = p32.cast::<f64>();
    let d = Vec3::new(q.x - p64.x, q.y - p64.y, q.z - p64.z).norm();
    println!("f32 versus f64 leapfrog after one orbit: {d:.3} km");
    assert!(d < 5.0, "{d} km");
}

#[test]
fn leapfrog_circular_speed_is_a_pure_rotation() {
    let field = SaturnField::<f64>::from_params(&params());
    let a0 = 140_222.43;
    let vc = circular_speed(&field, a0);
    let h = 2.0 * std::f64::consts::PI * a0 / vc / 150.0;
    let vh = leapfrog_circular_speed(&field, a0, h).unwrap();
    println!(
        "v_h - v_circular = {:.6e} km/s (relative {:.3e})",
        vh - vc,
        (vh - vc) / vc
    );
    let (mut p, mut v) = (Vec3::new(a0, 0.0, 0.0), Vec3::new(0.0, vh, 0.0));
    dkd_step(&field, &mut p, &mut v, h);
    assert!(
        (p.norm() - a0).abs() < 1e-9,
        "radius changed by {}",
        p.norm() - a0
    );
    assert!(
        (v.norm() - vh).abs() < 1e-12,
        "speed changed by {}",
        v.norm() - vh
    );
    assert!(p.dot(v).abs() < 1e-9 * a0 * vh, "not tangential");
}
