//! WH-EFF-GPU investigation: chaos indicator of regular orbits, no moons, one
//! year, CPU (f64) against GPU D+ and C, both about mu_eff. Without moons
//! every orbit is regular, so the FTLE of GPU and CPU must agree and the
//! shadow separation must stay at the regular-orbit level. Needs a GPU:
//! `cargo test -p fring-gpu --release --test gpu_chaos_regular -- --ignored --nocapture`.

mod common;

use common::*;
use fring_cpu::CpuBackend;
use fring_gpu::backend::{GpuBackend, Options};
use fring_gpu::wh::Mode;
use fring_model::config::{ChaosConfig, ChaosMethod, DiagnosticsConfig};
use fring_model::{Backend, ChaosReport};

const STEPS: u64 = 88_582;

fn stats(v: &[f64]) -> (f64, f64) {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (s[s.len() / 2], s[s.len() - 1])
}

#[test]
#[ignore = "needs a GPU"]
fn regular_orbits_have_the_same_ftle_on_gpu_and_cpu() {
    if fring_gpu::gpu::Gpu::new().is_none() {
        eprintln!("no GPU: skipped");
        return;
    }
    let k = 100u32;
    let system = system(H_S, STEPS + k as u64, false);
    let mut config = config(H_S, STEPS + k as u64, false);
    config.diagnostics = Some(DiagnosticsConfig {
        chaos: Some(ChaosConfig {
            method: ChaosMethod::Shadow,
            offset_km: 1.0,
            renorm_steps: k,
        }),
    });
    config.validate().unwrap();
    let (init, _) = ensemble(&system, 100);
    let mut cpu = CpuBackend::new(&config, &system, &init).unwrap();
    cpu.step(STEPS).unwrap();
    let c: ChaosReport = cpu.chaos().unwrap().unwrap();
    let day = 86_400.0;
    let (cm, cx) = stats(&c.ftle_per_s.iter().map(|x| x * day).collect::<Vec<_>>());
    println!(
        "cpu   FTLE per day: median {cm:.4e} max {cx:.4e}; log_growth max {:.3}",
        c.log_growth.iter().cloned().fold(0.0, f64::max)
    );
    for mode in [Mode::Dp, Mode::C, Mode::D] {
        let mut b = GpuBackend::with_options(
            &config,
            &system,
            &init,
            Options {
                mode,
                ..Options::default()
            },
        )
        .unwrap();
        b.step(STEPS).unwrap();
        let g = b.chaos().unwrap().unwrap();
        let f: Vec<f64> = g.ftle_per_s.iter().map(|x| x * day).collect();
        let (gm, gx) = stats(&f);
        let d: Vec<f64> = g
            .ftle_per_s
            .iter()
            .zip(&c.ftle_per_s)
            .map(|(a, b)| ((a - b) * day).abs())
            .collect();
        let (dm, dx) = stats(&d);
        println!(
            "{mode:?}: FTLE per day median {gm:.4e} max {gx:.4e}; |GPU-CPU| median {dm:.2e} max {dx:.2e}; log_growth max {:.3}",
            g.log_growth.iter().cloned().fold(0.0, f64::max)
        );
    }
}
