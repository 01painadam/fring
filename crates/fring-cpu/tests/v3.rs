//! Task C4: the CPU backend against the REBOUND reference of task V3.
//!
//! 100 non-chaotic test particles near the F ring core, Prometheus and
//! Pandora from `params/moons.toml`, 100 orbits (T0 = 53445.914 s each).
//! The initial states are read from `validation/reference/v3_ic.csv`
//! (17 digits, bit-exact), because 1 m in a is 1.8 km after 100 orbits.
//! The reference states every full orbit are in `v3_traj_orbits.csv`.
//!
//! The step size is T0/N, with N steps per orbit. The strict PLAN section 10
//! test (< 1 km) runs at N = 600; N = 150 (production) and N = 1500 are in
//! ADR 0005.
//!
//! Set `FRING_V3_RUST_CSV=<file>` to write the Rust trajectory in the long
//! CSV format that `validation/scripts/v3_compare.py --rust <file>` reads.

use fring_cpu::CpuBackend;
use fring_model::physics::{
    SaturnField, build_ephemeris, frequencies, interpolate_moons, wh_steps,
};
use fring_model::{Backend, InitialState, MoonEphemeris, Particles, RunConfig, System, Vec3};
use std::fmt::Write as _;
use std::sync::OnceLock;

const IC: &str = include_str!("../../../validation/reference/v3_ic.csv");
const TRAJ: &str = include_str!("../../../validation/reference/v3_traj_orbits.csv");
const MOONS: &str = include_str!("../../../validation/reference/v3_moons.csv");
const A0_KM: f64 = 140_222.43;
const N_ORBITS: usize = 100;
const N_PARTICLES: usize = 100;

/// Data rows of a reference CSV as numbers.
fn rows(text: &str) -> Vec<Vec<f64>> {
    text.lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| l.split(',').map(|x| x.parse().unwrap()).collect())
        .collect()
}

struct Setup {
    system: System,
    t0_s: f64,
    /// Reference states `[orbit][id]` = x, y, z, vx, vy, vz.
    reference: Vec<Vec<[f64; 6]>>,
}

fn setup() -> &'static Setup {
    static SETUP: OnceLock<Setup> = OnceLock::new();
    SETUP.get_or_init(|| {
        let saturn =
            fring_model::params::saturn_from_toml_str(include_str!("../../../params/saturn.toml"))
                .unwrap();
        let moons =
            fring_model::params::moons_from_toml_str(include_str!("../../../params/moons.toml"))
                .unwrap();
        let mut reference = vec![vec![[0.0; 6]; N_PARTICLES]; N_ORBITS + 1];
        let mut t0_s = 0.0;
        for r in rows(TRAJ) {
            let (k, id) = (r[0] as usize, r[2] as usize);
            if k == 10 {
                t0_s = r[1];
            }
            reference[k / 10][id].copy_from_slice(&r[3..9]);
        }
        let eph =
            build_ephemeris(&saturn, &moons, (N_ORBITS as f64) * t0_s + 7200.0, 1800.0).unwrap();
        Setup {
            system: System { saturn, moons: eph },
            t0_s,
            reference,
        }
    })
}

fn initial_state() -> InitialState {
    let mut particles = Particles::default();
    for r in rows(IC) {
        particles.push(
            r[0] as u32,
            Vec3::new(r[1], r[2], r[3]),
            Vec3::new(r[4], r[5], r[6]),
        );
    }
    assert_eq!(particles.len(), N_PARTICLES);
    InitialState {
        time_s: 0.0,
        particles,
    }
}

fn config(step_s: f64) -> RunConfig {
    let text = include_str!("../../fring-model/tests/data/example-run.toml");
    let text = text
        .replace("step_s = 360.0", &format!("step_s = {step_s:?}"))
        .replace("duration_years = 1.0", "duration_years = 0.1694");
    RunConfig::from_toml_str(&text).unwrap()
}

struct Result {
    max_pos_km: f64,
    max_vel_km_s: f64,
    worst_id: usize,
    worst_orbit: usize,
    /// Largest position difference at each full orbit.
    by_orbit_km: Vec<f64>,
}

/// Runs the backend for 100 orbits at `n` steps per orbit and compares with
/// the reference every full orbit.
fn run_v3(n: u32) -> Result {
    let s = setup();
    let h = s.t0_s / f64::from(n);
    let mut b = CpuBackend::new(&config(h), &s.system, &initial_state()).unwrap();
    let mut csv = String::from(
        "# Rust trajectory, same format as v3_traj_orbits.csv; columns: k,t_s,id,x_km,y_km,z_km,vx_km_s,vy_km_s,vz_km_s\n",
    );
    let out = std::env::var("FRING_V3_RUST_CSV").ok();
    let mut res = Result {
        max_pos_km: 0.0,
        max_vel_km_s: 0.0,
        worst_id: 0,
        worst_orbit: 0,
        by_orbit_km: Vec::new(),
    };
    for orbit in 0..=N_ORBITS {
        if orbit > 0 {
            b.step(u64::from(n)).unwrap();
        }
        let p = b.particles().unwrap().particles;
        let mut worst = 0.0_f64;
        for id in 0..N_PARTICLES {
            let r = &s.reference[orbit][id];
            let dp = (p.position_km(id) - Vec3::new(r[0], r[1], r[2])).norm();
            let dv = (p.velocity_km_s(id) - Vec3::new(r[3], r[4], r[5])).norm();
            worst = worst.max(dp);
            res.max_vel_km_s = res.max_vel_km_s.max(dv);
            if dp > res.max_pos_km {
                (res.max_pos_km, res.worst_id, res.worst_orbit) = (dp, id, orbit);
            }
            if out.is_some() {
                let (q, v) = (p.position_km(id), p.velocity_km_s(id));
                writeln!(
                    csv,
                    "{},{:.17e},{id},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e}",
                    10 * orbit,
                    orbit as f64 * s.t0_s,
                    q.x,
                    q.y,
                    q.z,
                    v.x,
                    v.y,
                    v.z
                )
                .unwrap();
            }
        }
        res.by_orbit_km.push(worst);
    }
    if let Some(path) = out {
        std::fs::write(format!("{path}.{n}"), csv).unwrap();
    }
    res
}

fn report(n: u32, r: &Result) {
    eprintln!(
        "V3 at {n} steps/orbit: max |dr| = {:.4e} km (particle {} at orbit {}), max |dv| = {:.3e} km/s; after 50 orbits {:.3e} km",
        r.max_pos_km, r.worst_id, r.worst_orbit, r.max_vel_km_s, r.by_orbit_km[50]
    );
}

#[test]
fn t0_is_two_pi_over_n_of_a0() {
    let s = setup();
    let field = SaturnField::<f64>::from_params(&s.system.saturn);
    let t0 = 2.0 * std::f64::consts::PI / frequencies(&field, A0_KM).n_rad_s;
    assert!((t0 / s.t0_s - 1.0).abs() < 1e-9, "{t0} {}", s.t0_s);
}

/// The moons of the Rust ephemeris against the moons of the REBOUND run.
#[test]
fn moons_agree_with_the_reference() {
    let s = setup();
    let mut worst = 0.0_f64;
    for r in rows(MOONS) {
        let m = interpolate_moons(&s.system.moons, r[1]).unwrap();
        for (j, moon) in m.moons.iter().enumerate() {
            let c = 2 + 6 * j;
            let q = Vec3::new(r[c], r[c + 1], r[c + 2]);
            worst = worst.max((moon.position_km - q).norm());
        }
    }
    eprintln!("moons against v3_moons.csv: max |dr| = {worst:.3e} km");
    assert!(worst < 1e-2, "{worst:e}");
}

/// PLAN section 10: position difference < 1 km for non-chaotic orbits.
#[test]
fn v3_600_steps_per_orbit_agree_within_1_km() {
    let r = run_v3(600);
    report(600, &r);
    assert!(r.max_pos_km < 1.0, "{:e}", r.max_pos_km);
}

/// The production step: the Wisdom-Holman error (h², along-track) is larger.
#[test]
fn v3_150_steps_per_orbit_production_step() {
    let r = run_v3(150);
    report(150, &r);
    assert!(r.max_pos_km < 12.0, "{:e}", r.max_pos_km);
}

/// Long: with a small step the difference is the reference's own level, so
/// any error in the moon terms would show.
#[test]
#[ignore = "long: 1.5e7 particle-steps; run with --ignored (release is fast)"]
fn v3_1500_steps_per_orbit_agree_within_0_1_km() {
    let r = run_v3(1500);
    report(1500, &r);
    assert!(r.max_pos_km < 0.1, "{:e}", r.max_pos_km);
}

/// Diagnostic: smallest b / (v_rel h) of a particle and a moon over the run,
/// with b the distance and v_rel the relative speed at a step boundary: the
/// number of steps in which the particle moves by its own distance to the
/// moon. Prints; asserts only that the encounters are resolved by 3 steps.
#[test]
#[ignore = "diagnostic: steps one at a time; run with --ignored --nocapture"]
fn encounter_metric() {
    let s = setup();
    for n in [150_u32, 600] {
        let h = s.t0_s / f64::from(n);
        let mut b = CpuBackend::new(&config(h), &s.system, &initial_state()).unwrap();
        let (mut metric, mut dist, mut at) = (f64::INFINITY, f64::INFINITY, (0, 0));
        for step in 0..(N_ORBITS as u64 * u64::from(n)) {
            b.step(1).unwrap();
            let m = b.moons();
            let p = b.particles().unwrap().particles;
            for id in 0..N_PARTICLES {
                for moon in &m.moons {
                    let d = (p.position_km(id) - moon.position_km).norm();
                    let v = (p.velocity_km_s(id) - moon.velocity_km_s).norm();
                    dist = dist.min(d);
                    if d / (v * h) < metric {
                        (metric, at) = (d / (v * h), (id, step));
                    }
                }
            }
        }
        eprintln!(
            "encounter at {n} steps/orbit: min b/(v_rel h) = {metric:.2} (particle {}, step {}); min distance {dist:.1} km",
            at.0, at.1
        );
        assert!(metric > 3.0);
    }
}

/// Serial Wisdom-Holman without moons (the model function) against the
/// backend with the moons switched off: a check of the backend glue.
#[test]
fn backend_without_moons_matches_the_model_loop() {
    let s = setup();
    let h = s.t0_s / 150.0;
    let sys = System {
        saturn: s.system.saturn,
        moons: MoonEphemeris::none(),
    };
    let ini = initial_state();
    let mut b = CpuBackend::new(&config(h), &sys, &ini).unwrap();
    b.step(300).unwrap();
    let field = SaturnField::<f64>::from_params(&sys.saturn)
        // ICR-011: the backend drifts about mu_eff at a0 (config a0 = 140,220 km).
        .with_drift_radius(140_220.0);
    let p = b.particles().unwrap().particles.clone();
    for i in 0..N_PARTICLES {
        let (mut pos, mut vel) = (ini.particles.position_km(i), ini.particles.velocity_km_s(i));
        wh_steps(&field, &mut pos, &mut vel, h, 300).unwrap();
        assert_eq!(p.position_km(i), pos);
        assert_eq!(p.velocity_km_s(i), vel);
    }
}
