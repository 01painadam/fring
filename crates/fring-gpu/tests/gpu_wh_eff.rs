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
    let gpu = Gpu::new().expect("GPU");
    let kernels = WhKernels::new(&gpu, DEFAULT_WORKGROUP).expect("self-test");
    let sp = fring_model::params::saturn_from_toml_str(include_str!("../../../params/saturn.toml"))
        .unwrap();
    let field = SaturnField::<f64>::from_params(&sp);
    let h = YEAR_S / STEPS as f64;
    let c = Consts::<M4>::for_step_drift(&sp, &[], h, drift_radius_km);
    let prm = wh::params(&c, &[]);
    let table = vec![0f32; STEPS * wh::ROW];
    let mut states: Vec<State<M4>> = RADII_KM
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
    RADII_KM
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
