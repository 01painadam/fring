//! G3 throughput with and without the chaos indicator (ADR 0008): D+ and C,
//! N particles (default 100,000), 3,000 steps, both moons.
//! `cargo run -p fring-gpu --release --example g3_bench [N]`

use fring_gpu::backend::{GpuBackend, Options};
use fring_gpu::wh::Mode;
use fring_model::config::{ChaosConfig, ChaosMethod, DiagnosticsConfig, Distribution};
use fring_model::params::{moons_from_toml_str, saturn_from_toml_str};
use fring_model::physics::{SaturnField, build_ephemeris, generate_particles};
use fring_model::{Backend, InitialState, RunConfig, System};
use std::time::Instant;

const H: f64 = 356.2537;
const STEPS: u64 = 3000;

fn main() {
    let n: u32 = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(100_000);
    let text = include_str!("../../fring-model/tests/data/example-run.toml")
        .replace("step_s = 360.0", &format!("step_s = {H}"))
        .replace("duration_years = 1.0", "duration_years = 0.0335");
    let base = RunConfig::from_toml_str(&text).unwrap();
    let saturn = saturn_from_toml_str(include_str!("../../../params/saturn.toml")).unwrap();
    let all = moons_from_toml_str(include_str!("../../../params/moons.toml")).unwrap();
    let eph = build_ephemeris(&saturn, &all, STEPS as f64 * H + 7200.0, 1800.0).unwrap();
    let system = System { saturn, moons: eph };
    let field = SaturnField::<f64>::from_params(&system.saturn);
    let dist = Distribution::Band {
        min_semi_major_axis_km: 139_600.0,
        max_semi_major_axis_km: 140_600.0,
    };
    let init = InitialState {
        time_s: 0.0,
        particles: generate_particles(&field, &dist, n, 1).unwrap(),
    };
    println!("N = {n}, {STEPS} steps");
    for (name, mode) in [("D+", Mode::Dp), ("C", Mode::C)] {
        let mut rate = [0.0; 2];
        for (k, chaos) in [false, true].into_iter().enumerate() {
            let mut cfg = base.clone();
            if chaos {
                cfg.diagnostics = Some(DiagnosticsConfig {
                    chaos: Some(ChaosConfig {
                        method: ChaosMethod::Shadow,
                        offset_km: 1.0,
                        renorm_steps: 100,
                    }),
                });
            }
            let opts = Options {
                mode,
                ..Options::default()
            };
            let mut b = GpuBackend::with_options(&cfg, &system, &init, opts).unwrap();
            b.step(100).unwrap(); // warm-up (pipelines, caches)
            let t0 = Instant::now();
            b.step(STEPS - 100).unwrap();
            let _ = b.stats().unwrap();
            if chaos {
                let _ = b.chaos().unwrap();
            }
            let s = t0.elapsed().as_secs_f64();
            rate[k] = n as f64 * (STEPS - 100) as f64 / s;
        }
        println!(
            "{name:3} chaos off {:.3e} particle-steps/s, on {:.3e} (ratio {:.2})",
            rate[0],
            rate[1],
            rate[1] / rate[0]
        );
    }
}
