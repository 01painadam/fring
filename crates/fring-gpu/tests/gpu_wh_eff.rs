//! T4 of ICR-011 (GPU T1): along-track error of the GPU Wisdom-Holman split
//! after one Julian year at the production step, from the exact circular
//! orbit, no moons. The drift is about mu_eff (a0 = 140,220 km). The same run
//! with the drift about GM is the power arm (the CPU T1 of `fring-cpu` gives
//! more than 10 km). Needs a GPU, so `#[ignore]`:
//! `cargo test -p fring-gpu --release --test gpu_wh_eff -- --ignored --nocapture`.

use fring_gpu::gpu::Gpu;
use fring_gpu::spike::{Consts, State, state_from_f64, state_to_f64};
use fring_gpu::wf32::model;
use fring_gpu::wh::{self, Mode};
use fring_gpu::wh_gpu::{DEFAULT_WORKGROUP, WhKernels};
use fring_model::Vec3;
use fring_model::physics::{SaturnField, circular_speed, frequencies};

const M4: u8 = model::M4;
const A0_KM: f64 = 140_220.0;
const STEPS: usize = 88_582;
const YEAR_S: f64 = 31_557_600.0;
const RADII_KM: [f64; 3] = [139_500.0, 140_220.0, 141_500.0];

/// Along-track errors in km at `RADII_KM` after one year on the GPU.
fn errors_km(mode: Mode, drift_radius_km: Option<f64>) -> Vec<f64> {
    errors_at_km(mode, drift_radius_km, &RADII_KM)
}

/// As [`errors_km`] for any start radii (km), in one `apply`.
fn errors_at_km(mode: Mode, drift_radius_km: Option<f64>, radii_km: &[f64]) -> Vec<f64> {
    let gpu = Gpu::new().expect("GPU");
    let kernels = WhKernels::new(&gpu, DEFAULT_WORKGROUP).expect("self-test");
    let sp = fring_model::params::saturn_from_toml_str(include_str!("../../../params/saturn.toml"))
        .unwrap();
    let field = SaturnField::<f64>::from_params(&sp);
    let h = YEAR_S / STEPS as f64;
    let c = Consts::<M4>::for_step_drift(&sp, &[], h, drift_radius_km);
    let prm = wh::params(&c, &[]);
    let table = vec![0f32; STEPS * wh::ROW];
    let mut states: Vec<State<M4>> = radii_km
        .iter()
        .map(|&r| {
            state_from_f64(
                Vec3::new(r, 0.0, 0.0),
                Vec3::new(0.0, circular_speed(&field, r), 0.0),
            )
        })
        .collect();
    let (status, _) = kernels.apply(mode.entry(), &prm, &table, STEPS, 0, &mut states);
    assert_eq!(status, fring_gpu::wh_gpu::NO_FAILURE);
    radii_km
        .iter()
        .zip(&states)
        .map(|(&r, s)| {
            let (p, _) = state_to_f64(s);
            let n = frequencies(&field, r).n_rad_s;
            let want = (n * h * STEPS as f64).rem_euclid(std::f64::consts::TAU);
            let mut d = p.y.atan2(p.x) - want;
            d -= std::f64::consts::TAU * (d / std::f64::consts::TAU).round();
            r * d
        })
        .collect()
}

/// Bounds in km from the measurement (M27), see the handoff.
fn bound_km(mode: Mode) -> Option<f64> {
    match mode {
        Mode::Dp => Some(BOUND_DP_KM),
        Mode::C => Some(BOUND_C_KM),
        Mode::D => None, // reported only (A0, 2026-11-09)
    }
}
/// Measured on the Apple M4: D+ 0.071, 0.0002, 0.103 km; C 0.073, 0.0013,
/// 0.110 km (the CPU f64 values are 0.0717, 0.0001, 0.1088). The O(h^2) split
/// error itself is 0.109 km at the band edge, so the bound is 0.25 km (margin
/// 2.3); the GM arm gives 12 to 13 km.
const BOUND_DP_KM: f64 = 0.25;
const BOUND_C_KM: f64 = 0.25;
/// The CPU f64 errors of `fring-cpu/tests/wh_eff.rs` (ICR-011 T1), km. The GPU
/// differs by at most 0.0054 km (D+) and 0.0013 km (C); bound 0.02 (margin 3.7).
const CPU_KM: [f64; 3] = [-0.0717, 0.0001, 0.1088];
const GPU_CPU_BOUND_KM: f64 = 0.02;

#[test]
#[ignore = "needs a GPU"]
fn gpu_one_year_phase_error_with_mu_eff() {
    if Gpu::new().is_none() {
        eprintln!("no GPU: skipped");
        return;
    }
    for mode in [Mode::D, Mode::Dp, Mode::C] {
        let e = errors_km(mode, Some(A0_KM));
        let o = errors_km(mode, None);
        eprintln!(
            "{}: mu_eff {:+.4} {:+.4} {:+.4} km; GM {:+.3} {:+.3} {:+.3} km",
            mode.name(),
            e[0],
            e[1],
            e[2],
            o[0],
            o[1],
            o[2]
        );
        if let Some(b) = bound_km(mode) {
            assert!(e.iter().all(|x| x.abs() < b), "{}: {e:?}", mode.name());
            for (g, c) in e.iter().zip(CPU_KM) {
                assert!((g - c).abs() < GPU_CPU_BOUND_KM, "{}: {e:?}", mode.name());
            }
            assert!(
                o.iter().all(|x| x.abs() > 10.0),
                "{}: no power {o:?}",
                mode.name()
            );
        }
    }
}

/// Error of the kick of one step against the f64 kick, in units of h GM / r^2
/// (the kick of the point mass), worst over positions in the band. With
/// `drift_radius_km = None` the truth is the J-only kick (before ICR-011).
fn kick_error(entry: &str, drift_radius_km: Option<f64>) -> f64 {
    let gpu = Gpu::new().expect("GPU");
    let kernels = WhKernels::new(&gpu, DEFAULT_WORKGROUP).expect("self-test");
    let sp = fring_model::params::saturn_from_toml_str(include_str!("../../../params/saturn.toml"))
        .unwrap();
    let h = 356.25;
    let c = Consts::<M4>::for_step_drift(&sp, &[], h, drift_radius_km);
    let prm = wh::params(&c, &[]);
    let table = vec![0f32; wh::ROW];
    let mut field = SaturnField::<f64>::from_params(&sp);
    if let Some(a0) = drift_radius_km {
        field = field.with_drift_radius(a0);
    }
    let pos: Vec<Vec3<f64>> = (0..200)
        .map(|k| {
            let r = 139_500.0 + 2_000.0 * (k % 10) as f64 / 9.0;
            let (th, z) = (0.37 * k as f64, 30.0 * ((k / 10) as f64 - 10.0));
            Vec3::new(r * th.cos(), r * th.sin(), z)
        })
        .collect();
    let mut states: Vec<State<M4>> = pos
        .iter()
        .map(|&p| state_from_f64(p, Vec3::new(0.0, 0.0, 0.0)))
        .collect();
    let (status, _) = kernels.apply(entry, &prm, &table, 1, 0, &mut states);
    assert_eq!(status, fring_gpu::wh_gpu::NO_FAILURE);
    let mut worst = 0.0f64;
    for (p, s) in pos.iter().zip(&states) {
        let (_, v) = state_to_f64(s);
        let a = if drift_radius_km.is_some() {
            field.kick_acceleration_saturn(*p)
        } else {
            field.acceleration_without_point_mass(*p)
        };
        let g = sp.gm_km3_s2 / p.norm_squared();
        // t_kick_acc_c returns the acceleration, the others the velocity change.
        let hh = if entry == "t_kick_acc_c" { 1.0 } else { h };
        worst = worst.max((v - a * hh).norm() / (g * hh));
    }
    worst
}

#[test]
#[ignore = "needs a GPU"]
fn kick_error_before_and_after_epsilon() {
    if Gpu::new().is_none() {
        eprintln!("no GPU: skipped");
        return;
    }
    for entry in ["t_kick_dir", "t_kick_dir_plus", "t_kick_acc_c"] {
        let (a, b) = (kick_error(entry, None), kick_error(entry, Some(A0_KM)));
        eprintln!("{entry:16} kick error / (h GM/r^2): GM drift {a:.2e}, mu_eff drift {b:.2e}");
    }
}

/// CPU f64 along-track error in km (the T1 of `fring-cpu`) at `r_km`, drift
/// about mu_eff at `drift_radius_km`.
fn cpu_error_km(drift_radius_km: f64, r_km: f64) -> f64 {
    let sp = fring_model::params::saturn_from_toml_str(include_str!("../../../params/saturn.toml"))
        .unwrap();
    let field = SaturnField::<f64>::from_params(&sp).with_drift_radius(drift_radius_km);
    let h = YEAR_S / STEPS as f64;
    let (mut p, mut v) = (
        Vec3::new(r_km, 0.0, 0.0),
        Vec3::new(0.0, circular_speed(&field, r_km), 0.0),
    );
    fring_model::physics::wh_steps(&field, &mut p, &mut v, h, STEPS as u64).unwrap();
    let want =
        (frequencies(&field, r_km).n_rad_s * h * STEPS as f64).rem_euclid(std::f64::consts::TAU);
    let mut d = p.y.atan2(p.x) - want;
    d -= std::f64::consts::TAU * (d / std::f64::consts::TAU).round();
    r_km * d
}

/// Bounds on |GPU - CPU| in km for start radii off 140,220 km (STEPPER-DRIFT,
/// A0 2026-11-10; M27). Measured 2026-11-10 on the Apple M4 over the 32-radius
/// scan of `e2_scan_start_radius`: D+ 0.0207 km, C 0.0022 km. D+ has an f32
/// bias that depends on the start radius (see the handoff); the bound is 2.4
/// times the maximum because 32 samples of a curve with +-0.02 km swing may
/// miss its peak. T1 above passes at 140,220 km only because that radius is
/// near a zero of the curve.
const SCAN_DP_BOUND_KM: f64 = 0.05;
const SCAN_C_BOUND_KM: f64 = 0.01;

/// STEPPER-DRIFT E1: does the start radius or the drift radius carry the
/// +0.009 km seen through `GpuStepper` at a0 = 140,222.43 km? The start radius
/// does. Measured D+ 0.0002, 0.0007, 0.0095, 0.0118 km; C 0.0013, -0.0013,
/// -0.0006, 0.0008 km.
#[test]
#[ignore = "needs a GPU"]
fn e1_start_and_drift_radius() {
    if Gpu::new().is_none() {
        eprintln!("no GPU: skipped");
        return;
    }
    for (drift, r) in [
        (140_220.0, 140_220.0),
        (140_222.43, 140_220.0),
        (140_222.43, 140_222.43),
        (140_220.0, 140_222.43),
    ] {
        let dp = errors_at_km(Mode::Dp, Some(drift), &[r])[0];
        let c = errors_at_km(Mode::C, Some(drift), &[r])[0];
        let cpu = cpu_error_km(drift, r);
        eprintln!("E1 drift {drift} r {r}: Dp {dp:+.4} C {c:+.4} CPU {cpu:+.4}");
        assert!((dp - cpu).abs() < SCAN_DP_BOUND_KM, "Dp {dp} CPU {cpu}");
        assert!((c - cpu).abs() < SCAN_C_BOUND_KM, "C {c} CPU {cpu}");
    }
}

/// STEPPER-DRIFT E2: along-track error against the start radius. Grid: 32
/// radii from 140,220 km in steps of 0.00390625 km (a quarter of the f32 ulp,
/// 0.015625 km), drift about mu_eff(140,220 km), one `apply` per mode.
#[test]
#[ignore = "needs a GPU"]
fn e2_scan_start_radius() {
    if Gpu::new().is_none() {
        eprintln!("no GPU: skipped");
        return;
    }
    let ulp = 0.015625;
    let radii: Vec<f64> = (0..32).map(|k| A0_KM + k as f64 * ulp / 4.0).collect();
    let dp = errors_at_km(Mode::Dp, Some(A0_KM), &radii);
    let c = errors_at_km(Mode::C, Some(A0_KM), &radii);
    let (mut mdp, mut mc) = (0.0f64, 0.0f64);
    for (k, r) in radii.iter().enumerate() {
        let cpu = cpu_error_km(A0_KM, *r);
        eprintln!(
            "E2 r {r:.5} (mod ulp {:.5}): Dp {:+.4} C {:+.4} CPU {cpu:+.4}",
            (r - A0_KM).rem_euclid(ulp),
            dp[k],
            c[k]
        );
        mdp = mdp.max((dp[k] - cpu).abs());
        mc = mc.max((c[k] - cpu).abs());
    }
    eprintln!("E2 max |GPU - CPU|: Dp {mdp:.4} km, C {mc:.4} km");
    assert!(mdp < SCAN_DP_BOUND_KM, "Dp {mdp}");
    assert!(mc < SCAN_C_BOUND_KM, "C {mc}");
}
