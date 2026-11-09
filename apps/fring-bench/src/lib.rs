//! Benchmarks: particle-steps per second for CPU and GPU (task B1).
//!
//! Owner: A5 (see `AGENTS.md`). Core particles with both moons run at the
//! production step (150 steps per orbit) through the `Backend` trait, as
//! `fring-run` uses it. Results go to a CSV file and to stdout as Markdown.

pub mod cli;
pub mod matrix;
pub mod measure;
pub mod report;

use cli::Args;
use fring_cpu::CpuBackend;
use fring_gpu::backend::{GpuBackend, Options};
use fring_gpu::wh::Mode;
use fring_model::config::{ChaosConfig, ChaosMethod, DiagnosticsConfig, Distribution};
use fring_model::params::{moons_from_toml_str, saturn_from_toml_str};
use fring_model::physics::{SaturnField, build_ephemeris, generate_particles};
use fring_model::{Backend, InitialState, RunConfig, System};
use matrix::{Point, Variant, build_matrix};
use report::{Outcome, RunEnv};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::time::Instant;

/// Production step: 150 steps per orbit at the core (seconds).
pub const STEP_S: f64 = 356.2537;
/// Seed of the initial particles.
pub const SEED: u64 = 1;
/// Time of one timed call (seconds), full and `--quick`.
pub const TARGET_S: f64 = 1.5;
/// Time of one timed call with `--quick`.
pub const QUICK_TARGET_S: f64 = 0.3;

/// Steps that the moon ephemeris must cover: warm-up, probes, 3 repeats and
/// read-back steps, each at most [`measure::MAX_STEPS`], with margin.
const EPHEMERIS_STEPS: u64 = 5 * measure::MAX_STEPS;

/// Everything that does not depend on N.
struct Setup {
    base: RunConfig,
    system: System,
    field: SaturnField<f64>,
}

fn setup() -> Result<Setup, String> {
    let text = include_str!("../../../crates/fring-model/tests/data/example-run.toml");
    let years = EPHEMERIS_STEPS as f64 * STEP_S / (365.25 * 86_400.0);
    let text = text
        .replace("step_s = 360.0", &format!("step_s = {STEP_S}"))
        .replace("duration_years = 1.0", &format!("duration_years = {years}"));
    let base = RunConfig::from_toml_str(&text).map_err(|e| e.to_string())?;
    let saturn = saturn_from_toml_str(include_str!("../../../params/saturn.toml"))
        .map_err(|e| e.to_string())?;
    let moons = moons_from_toml_str(include_str!("../../../params/moons.toml"))
        .map_err(|e| e.to_string())?;
    let eph = build_ephemeris(
        &saturn,
        &moons,
        EPHEMERIS_STEPS as f64 * STEP_S + 7200.0,
        1800.0,
    )
    .map_err(|e| e.to_string())?;
    let system = System { saturn, moons: eph };
    let field = SaturnField::<f64>::from_params(&system.saturn);
    Ok(Setup {
        base,
        system,
        field,
    })
}

/// The config of one point: chaos indicator as in ICR-003 (d0 = 1 km, K = 100).
fn point_config(base: &RunConfig, chaos: bool) -> RunConfig {
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
    cfg
}

fn core_particles(field: &SaturnField<f64>, n: usize) -> Result<InitialState, String> {
    let dist = Distribution::Core {
        semi_major_axis_km: 140_220.0,
        sigma_semi_major_axis_km: 5.0,
        eccentricity: 0.0024,
        apse_longitude_deg: 0.0,
        inclination_deg: 0.0,
    };
    let count = u32::try_from(n).map_err(|_| format!("{n} particles do not fit in u32"))?;
    let particles = generate_particles(field, &dist, count, SEED).map_err(|e| e.to_string())?;
    Ok(InitialState {
        time_s: 0.0,
        particles,
    })
}

/// Runs one point. Errors and panics (for example a GPU out of memory) become
/// a "not run" reason.
fn run_point(
    s: &Setup,
    init: &InitialState,
    p: Point,
    target_s: f64,
    env: &mut RunEnv,
) -> Result<measure::Timing, String> {
    let mut cfg = point_config(&s.base, p.chaos);
    cfg.particles.count = init.particles.len() as u32;
    let caught = catch_unwind(AssertUnwindSafe(|| -> Result<measure::Timing, String> {
        match p.variant {
            Variant::Cpu => {
                let mut b = CpuBackend::new(&cfg, &s.system, init).map_err(|e| e.to_string())?;
                measure::measure(&mut b, p.n, p.chaos, target_s, measure::MAX_STEPS)
            }
            v => {
                let mode = match v {
                    Variant::GpuD => Mode::D,
                    Variant::GpuC => Mode::C,
                    _ => Mode::Dp,
                };
                let opts = Options {
                    mode,
                    ..Options::default()
                };
                let mut b = GpuBackend::with_options(&cfg, &s.system, init, opts)
                    .map_err(|e| e.to_string())?;
                env.adapter = b.adapter_name();
                env.max_binding_bytes = b.limits().max_storage_buffer_binding_size;
                env.max_buffer_bytes = b.limits().max_buffer_size;
                measure::measure(&mut b, p.n, p.chaos, target_s, measure::MAX_STEPS)
            }
        }
    }));
    match caught {
        Ok(r) => r,
        Err(_) => Err("the backend panicked (probably out of memory)".to_string()),
    }
}

/// Runs the matrix and returns the outcomes. Progress goes to stderr.
pub fn run_matrix(args: &Args, env: &mut RunEnv) -> Result<Vec<Outcome>, String> {
    let s = setup()?;
    let target_s = if args.quick { QUICK_TARGET_S } else { TARGET_S };
    let points = build_matrix(&args.sizes, !args.no_mode_d);
    let mut outcomes = Vec::with_capacity(points.len());
    let mut current: Option<(usize, Result<InitialState, String>)> = None;
    for (i, p) in points.iter().enumerate() {
        if current.as_ref().is_none_or(|(n, _)| *n != p.n) {
            eprintln!("generating {} particles", p.n);
            current = Some((p.n, core_particles(&s.field, p.n)));
        }
        let t0 = Instant::now();
        let result = match &current {
            Some((_, Ok(init))) => run_point(&s, init, *p, target_s, env),
            Some((_, Err(why))) => Err(why.clone()),
            None => unreachable!(),
        };
        let line = match &result {
            Ok(t) => format!("{} particle-steps/s", report::sci(t.rate_best)),
            Err(why) => format!("not run: {why}"),
        };
        eprintln!(
            "[{}/{}] {} N={} chaos={}: {line} ({:.1} s)",
            i + 1,
            points.len(),
            p.variant.name(),
            p.n,
            if p.chaos { "on" } else { "off" },
            t0.elapsed().as_secs_f64()
        );
        outcomes.push(Outcome { point: *p, result });
    }
    Ok(outcomes)
}

/// Runs the benchmark: matrix, CSV file, Markdown on stdout.
pub fn run(args: &Args) -> Result<(), String> {
    let t0 = Instant::now();
    let facts = fring_io::sysinfo::machine_facts();
    let mut env = RunEnv {
        wgpu_version: report::wgpu_version(include_str!("../../../Cargo.lock")),
        git: fring_io::sysinfo::git_hash(
            &std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        ),
        ..RunEnv::default()
    };
    let outcomes = run_matrix(args, &mut env)?;
    env.wall_s = t0.elapsed().as_secs_f64();

    let path = args
        .out
        .clone()
        .unwrap_or_else(|| PathBuf::from(format!("runs/bench/bench-{}.csv", facts.date)));
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    std::fs::write(&path, report::csv(&outcomes))
        .map_err(|e| format!("{}: {e}", path.display()))?;
    eprintln!("wrote {}", path.display());

    println!("{}", report::markdown_machine(&facts, &env));
    println!("{}", report::markdown_table(&outcomes));
    if env.max_binding_bytes > 0 {
        println!(
            "Largest N whose state buffer fits the GPU limits: {} ({} bytes per particle).",
            report::max_particles(env.max_binding_bytes, env.max_buffer_bytes),
            report::GPU_STATE_BYTES
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny CPU-only matrix needs no GPU, so it runs in CI.
    #[test]
    fn tiny_cpu_point_runs() {
        let s = setup().unwrap();
        let init = core_particles(&s.field, 50).unwrap();
        let mut env = RunEnv::default();
        for chaos in [false, true] {
            let p = Point {
                variant: Variant::Cpu,
                n: 50,
                chaos,
            };
            let t = run_point(&s, &init, p, 0.02, &mut env).unwrap();
            assert!(t.rate_best >= t.rate_min && t.rate_min > 0.0);
            assert!(t.steps_per_call >= measure::MIN_STEPS);
        }
    }

    /// Needs a GPU: `cargo test -p fring-bench --release -- --ignored`.
    #[test]
    #[ignore = "needs a GPU"]
    fn tiny_gpu_matrix_runs() {
        let args = Args {
            sizes: vec![1000],
            quick: true,
            no_mode_d: false,
            out: None,
        };
        let mut env = RunEnv::default();
        let out = run_matrix(&args, &mut env).unwrap();
        // CPU 2 + three GPU modes x 2.
        assert_eq!(out.len(), 8);
        for o in &out {
            let t = o
                .result
                .as_ref()
                .unwrap_or_else(|e| panic!("{:?}: {e}", o.point));
            assert!(t.rate_best > 0.0);
        }
        assert!(env.adapter.contains("Metal") || !env.adapter.is_empty());
        assert!(env.max_binding_bytes > 0);
    }
}
