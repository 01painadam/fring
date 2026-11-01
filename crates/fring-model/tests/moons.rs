//! Task C3: moon dynamics, IAS15 and the ephemeris table (ADR 0004, ICR-002).
//!
//! Fast tests run in a debug build. Long runs are `#[ignore]`:
//! `cargo test -p fring-model --release --test moons -- --ignored --nocapture`.

#![allow(clippy::needless_range_loop)]

use fring_model::params::{moons_from_toml_str, saturn_from_toml_str};
use fring_model::physics::{
    Ias15, MoonInit, MoonSystem, SaturnField, build_ephemeris, build_ephemeris_with, hermite3,
    interpolate_moons, interpolate_moons_into, kepler_drift,
};
use fring_model::{BodyState, Error, MoonEphemeris, MoonTrack, SaturnParams, Vec3};

const SATURN_TOML: &str = include_str!("../../../params/saturn.toml");
const MOONS_TOML: &str = include_str!("../../../params/moons.toml");
const YEAR_S: f64 = 365.25 * 86_400.0;

fn saturn() -> SaturnParams {
    saturn_from_toml_str(SATURN_TOML).unwrap()
}

fn moons() -> Vec<MoonInit> {
    moons_from_toml_str(MOONS_TOML).unwrap()
}

/// Integrates `moons` until `duration_s`, calling `on_step` after each step.
fn integrate<F: FnMut(&Ias15)>(
    saturn: &SaturnParams,
    moons: &[MoonInit],
    epsilon: f64,
    duration_s: f64,
    mut on_step: F,
) -> Ias15 {
    let system = MoonSystem::new(saturn, moons.iter().map(|m| m.gm_km3_s2).collect());
    let mut force = |x: &[f64], a: &mut [f64]| system.accelerations(x, a);
    let (mut x0, mut v0) = (Vec::new(), Vec::new());
    for m in moons {
        let (p, v) = (m.state.position_km, m.state.velocity_km_s);
        x0.extend([p.x, p.y, p.z]);
        v0.extend([v.x, v.y, v.z]);
    }
    let mut ias = Ias15::new(epsilon, 0.0, &x0, &v0, 60.0).unwrap();
    while ias.time_s() < duration_s {
        ias.step(&mut force).unwrap();
        on_step(&ias);
    }
    ias
}

fn vec(a: &[f64], i: usize) -> Vec3<f64> {
    Vec3::new(a[3 * i], a[3 * i + 1], a[3 * i + 2])
}

fn no_j(mut s: SaturnParams) -> SaturnParams {
    s.j2 = 0.0;
    s.j4 = 0.0;
    s.j6 = 0.0;
    s
}

#[test]
fn parameter_files_load() {
    let s = saturn();
    assert_eq!(s.gm_km3_s2, 37_931_206.234);
    let m = moons();
    assert_eq!(m.len(), 2);
    assert_eq!(m[0].name, "prometheus");
    assert_eq!(m[1].name, "pandora");
    assert_eq!(m[0].gm_km3_s2, 0.01071);
    assert_eq!(m[0].state.position_km.x, 3.294735820205955e+03);
    assert_eq!(m[1].state.velocity_km_s.z, 9.517031670933485e-03);
}

#[test]
fn one_moon_with_j_terms_conserves_energy_and_lz() {
    let saturn = saturn();
    let one = vec![moons().remove(0)];
    let gm_i = one[0].gm_km3_s2;
    let field = SaturnField::<f64>::from_params(&saturn);
    // Exactly conserved by the one-moon equations: the point-mass term has
    // GM + GM_i, the J terms have GM.
    let conserved = |p: Vec3<f64>, v: Vec3<f64>| {
        (
            0.5 * v.norm_squared() + field.potential(p) - gm_i / p.norm(),
            p.x * v.y - p.y * v.x,
        )
    };
    let (e0, lz0) = conserved(one[0].state.position_km, one[0].state.velocity_km_s);
    let (mut de, mut dl) = (0.0_f64, 0.0_f64);
    let ias = integrate(&saturn, &one, 1e-9, YEAR_S, |s| {
        let (e, lz) = conserved(vec(s.x(), 0), vec(s.v(), 0));
        de = de.max(((e - e0) / e0).abs());
        dl = dl.max(((lz - lz0) / lz0).abs());
    });
    println!(
        "one moon, 1 yr: max |dE/E| {de:e}, max |dLz/Lz| {dl:e}, {} steps, {} force calls",
        ias.steps(),
        ias.force_evaluations()
    );
    assert!(de < 1e-13 && dl < 1e-13, "{de:e} {dl:e}");
}

/// Barycentric energy/G and angular momentum/G of Saturn and the moons
/// without J terms (the masses are GM).
fn barycentric(saturn: &SaturnParams, gm: &[f64], x: &[f64], v: &[f64]) -> (f64, Vec3<f64>) {
    let n = gm.len();
    let total = saturn.gm_km3_s2 + gm.iter().sum::<f64>();
    let mut rs = Vec3::zero();
    let mut vs = Vec3::zero();
    for i in 0..n {
        rs -= vec(x, i) * gm[i];
        vs -= vec(v, i) * gm[i];
    }
    let (rs, vs) = (rs / total, vs / total);
    let mut e = 0.5 * saturn.gm_km3_s2 * vs.norm_squared();
    let mut l = rs.cross(vs) * saturn.gm_km3_s2;
    for i in 0..n {
        let (ri, vi) = (vec(x, i) + rs, vec(v, i) + vs);
        e += 0.5 * gm[i] * vi.norm_squared() - saturn.gm_km3_s2 * gm[i] / vec(x, i).norm();
        l += ri.cross(vi) * gm[i];
        for j in i + 1..n {
            e -= gm[i] * gm[j] / (vec(x, i) - vec(x, j)).norm();
        }
    }
    (e, l)
}

#[test]
fn two_moons_without_j_conserve_energy_and_angular_momentum() {
    let saturn = no_j(saturn());
    let two = moons();
    let gm: Vec<f64> = two.iter().map(|m| m.gm_km3_s2).collect();
    let (mut x0, mut v0) = (Vec::new(), Vec::new());
    for m in &two {
        let (p, v) = (m.state.position_km, m.state.velocity_km_s);
        x0.extend([p.x, p.y, p.z]);
        v0.extend([v.x, v.y, v.z]);
    }
    let (e0, l0) = barycentric(&saturn, &gm, &x0, &v0);
    let (mut de, mut dl) = (0.0_f64, 0.0_f64);
    integrate(&saturn, &two, 1e-9, 0.5 * YEAR_S, |s| {
        let (e, l) = barycentric(&saturn, &gm, s.x(), s.v());
        de = de.max(((e - e0) / e0).abs());
        dl = dl.max((l - l0).norm() / l0.norm());
    });
    println!("two moons, J = 0, half a year: max |dE/E| {de:e}, max |dL|/|L| {dl:e}");
    assert!(de < 1e-13 && dl < 1e-13, "{de:e} {dl:e}");
}

#[test]
fn ias15_matches_the_kepler_orbit() {
    // One moon, no J terms, GM + GM_i as the central mass: exact solution.
    let saturn = no_j(saturn());
    let one = vec![moons().remove(0)];
    let ias = integrate(&saturn, &one, 1e-9, 100.0 * 14.7 * 3600.0, |_| {});
    let t = ias.time_s();
    let (p, _) = kepler_drift(
        saturn.gm_km3_s2 + one[0].gm_km3_s2,
        one[0].state.position_km,
        one[0].state.velocity_km_s,
        t,
    )
    .unwrap();
    let err = (vec(ias.x(), 0) - p).norm();
    println!("Kepler check, 100 orbits: position error {err:e} km");
    assert!(err < 1e-6, "{err:e} km");
}

#[test]
fn epsilon_convergence_over_one_year() {
    let (saturn, two) = (saturn(), moons());
    let run = |eps: f64| integrate(&saturn, &two, eps, YEAR_S, |_| {});
    let (a, b, c) = (run(1e-9), run(1e-10), run(1e-11));
    // The runs stop at slightly different times: compare at the latest start
    // of the last steps (inside the last step of every run), with the dense
    // output of each run.
    let at = |s: &Ias15, t: f64| {
        let (mut x, mut v) = (vec![0.0; 6], vec![0.0; 6]);
        s.dense_output(t, &mut x, &mut v);
        x
    };
    let t = [&a, &b, &c]
        .iter()
        .map(|s| s.last_step_start_s().unwrap())
        .fold(0.0, f64::max);
    let (xa, xb, xc) = (at(&a, t), at(&b, t), at(&c, t));
    let d = |p: &[f64], q: &[f64]| {
        (0..2)
            .map(|i| (vec(p, i) - vec(q, i)).norm())
            .fold(0.0, f64::max)
    };
    println!(
        "1 yr: eps 1e-9 vs 1e-11: {:e} km, 1e-10 vs 1e-11: {:e} km; steps {} {} {}",
        d(&xa, &xc),
        d(&xb, &xc),
        a.steps(),
        b.steps(),
        c.steps()
    );
    assert!(d(&xa, &xc) < 1e-5 && d(&xb, &xc) < 1e-5);
}

/// A table of every `step`-th sample of a fine table.
fn subsample(fine: &MoonEphemeris, step: usize) -> MoonEphemeris {
    MoonEphemeris {
        start_s: fine.start_s,
        interval_s: fine.interval_s * step as f64,
        tracks: fine
            .tracks
            .iter()
            .map(|t| MoonTrack {
                name: t.name.clone(),
                gm_km3_s2: t.gm_km3_s2,
                samples: t.samples.iter().step_by(step).cloned().collect(),
                accelerations_km_s2: t
                    .accelerations_km_s2
                    .iter()
                    .step_by(step)
                    .cloned()
                    .collect(),
            })
            .collect(),
    }
}

/// Largest mid-interval position error (km) of the quintic table with `step`
/// fine samples per interval, and of the cubic one.
fn mid_interval_errors(fine: &MoonEphemeris, step: usize) -> (f64, f64) {
    assert!(step.is_multiple_of(2));
    let coarse = subsample(fine, step);
    let h = coarse.interval_s;
    let (mut quintic, mut cubic) = (0.0_f64, 0.0_f64);
    let mut out = vec![BodyState::default(); coarse.tracks.len()];
    for k in 0..coarse.sample_count() - 1 {
        let truth_index = k * step + step / 2;
        if truth_index >= fine.sample_count() {
            break;
        }
        let t = coarse.sample_time_s(k) + 0.5 * h;
        interpolate_moons_into(&coarse, t, &mut out).unwrap();
        for (m, track) in coarse.tracks.iter().enumerate() {
            let truth = fine.tracks[m].samples[truth_index].position_km;
            quintic = quintic.max((out[m].position_km - truth).norm());
            let (a, b) = (&track.samples[k], &track.samples[k + 1]);
            let (p, _) = hermite3(
                a.position_km,
                a.velocity_km_s,
                b.position_km,
                b.velocity_km_s,
                h,
                0.5,
            );
            cubic = cubic.max((p - truth).norm());
        }
    }
    (quintic, cubic)
}

#[test]
fn mid_interval_error_of_the_table() {
    // 60 days with a 150 s fine table; the truth at mid-interval is a fine
    // sample (dense output of IAS15, accurate to about 1e-8 km).
    let fine = build_ephemeris(&saturn(), &moons(), 60.0 * 86_400.0, 150.0).unwrap();
    let (q1800, c1800) = mid_interval_errors(&fine, 12);
    let (q300, c300) = mid_interval_errors(&fine, 2);
    let (q600, c600) = mid_interval_errors(&fine, 4);
    println!("mid-interval error (km), 60 days:");
    println!("  interval  300 s: quintic {q300:e}, cubic {c300:e}");
    println!("  interval  600 s: quintic {q600:e}, cubic {c600:e}");
    println!("  interval 1800 s: quintic {q1800:e}, cubic {c1800:e}");
    assert!(q1800 < 1e-3, "quintic 1800 s: {q1800:e} km");
    assert!(c300 < 1e-3, "cubic 300 s: {c300:e} km");
    assert!(c600 > 1e-3, "cubic must fail at 600 s: {c600:e} km");
}

#[test]
fn interpolation_reproduces_nodes_and_rejects_bad_times() {
    let eph = build_ephemeris(&saturn(), &moons(), 7200.0, 1800.0).unwrap();
    eph.validate().unwrap();
    assert_eq!(eph.sample_count(), 5);
    let m = interpolate_moons(&eph, 3600.0).unwrap();
    assert_eq!(m.time_s, 3600.0);
    assert_eq!(m.moons[1], eph.tracks[1].samples[2]);
    let end = interpolate_moons(&eph, eph.end_s()).unwrap();
    assert_eq!(end.moons[0], eph.tracks[0].samples[4]);
    for bad in [-1.0, eph.end_s() + 1.0, f64::NAN, f64::INFINITY] {
        assert!(
            matches!(interpolate_moons(&eph, bad), Err(Error::Input(_))),
            "{bad}"
        );
    }
    let mut wrong = [BodyState::default(); 1];
    assert!(interpolate_moons_into(&eph, 10.0, &mut wrong).is_err());
}

#[test]
fn table_acceleration_is_the_force() {
    let (s, two) = (saturn(), moons());
    let eph = build_ephemeris(&s, &two, 3600.0, 1800.0).unwrap();
    let sys = MoonSystem::new(&s, two.iter().map(|m| m.gm_km3_s2).collect());
    let mut x = Vec::new();
    for t in &eph.tracks {
        let p = t.samples[1].position_km;
        x.extend([p.x, p.y, p.z]);
    }
    let mut a = vec![0.0; 6];
    sys.accelerations(&x, &mut a);
    assert!((vec(&a, 0) - eph.tracks[0].accelerations_km_s2[1]).norm() < 1e-18);
}

#[test]
fn validate_checks_the_accelerations() {
    let mut eph = build_ephemeris(&saturn(), &moons(), 3600.0, 1800.0).unwrap();
    eph.tracks[1].accelerations_km_s2.pop();
    assert!(matches!(eph.validate(), Err(Error::Input(m)) if m.contains("accelerations")));
    let mut eph = build_ephemeris(&saturn(), &moons(), 3600.0, 1800.0).unwrap();
    eph.tracks[0].accelerations_km_s2[1].x = f64::NAN;
    assert!(eph.validate().is_err());
}

#[test]
fn bad_inputs_are_errors() {
    let (s, m) = (saturn(), moons());
    assert!(build_ephemeris(&s, &m, -1.0, 1800.0).is_err());
    assert!(build_ephemeris(&s, &m, 3600.0, 0.0).is_err());
    assert!(build_ephemeris(&s, &m, 3600.0, f64::NAN).is_err());
    assert!(
        build_ephemeris(&s, &[], 3600.0, 1800.0)
            .unwrap()
            .tracks
            .is_empty()
    );
}

#[test]
#[ignore = "30 years: about 1 s in release"]
fn thirty_year_table() {
    let t0 = std::time::Instant::now();
    let eph = build_ephemeris(&saturn(), &moons(), 30.0 * YEAR_S, 1800.0).unwrap();
    let secs = t0.elapsed().as_secs_f64();
    eph.validate().unwrap();
    let bytes = eph.tracks.len() * eph.sample_count() * (6 + 3) * 8;
    println!(
        "30 yr table: {} samples per moon, {:.1} MB (f64), built in {secs:.2} s",
        eph.sample_count(),
        bytes as f64 / 1e6
    );
    assert!(eph.end_s() >= 30.0 * YEAR_S);
}

#[test]
#[ignore = "20 years at three epsilons: seconds in release"]
fn epsilon_convergence_over_twenty_years() {
    let (s, m) = (saturn(), moons());
    let d = 20.0 * YEAR_S;
    let a = build_ephemeris_with(&s, &m, d, 1800.0, 1e-9).unwrap();
    let b = build_ephemeris_with(&s, &m, d, 1800.0, 1e-10).unwrap();
    let c = build_ephemeris_with(&s, &m, d, 1800.0, 1e-11).unwrap();
    let diff = |p: &MoonEphemeris, q: &MoonEphemeris| {
        let mut worst = 0.0_f64;
        for k in 0..p.tracks.len() {
            for i in 0..p.sample_count() {
                let (u, w) = (&p.tracks[k].samples[i], &q.tracks[k].samples[i]);
                worst = worst.max((u.position_km - w.position_km).norm());
            }
        }
        worst
    };
    println!(
        "20 yr max |dr|: eps 1e-9 vs 1e-11 {:e} km, 1e-10 vs 1e-11 {:e} km",
        diff(&a, &c),
        diff(&b, &c)
    );
}
