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
use fring_model::config::{
    ChaosConfig, ChaosMethod, Collisions, DiagnosticsConfig, Distribution, LossConfig,
    MoonRadiusRule, OnUnbound, Restitution, SubstepConfig,
};
use fring_model::params::{moon_radii_from_toml_str, moons_from_toml_str, saturn_from_toml_str};
use fring_model::physics::{SaturnField, build_ephemeris, generate_particles};
use fring_model::{Backend, InitialState, RunConfig, System};
use matrix::{B2Spec, Feature, Point, State, Variant, build_matrix, build_matrix_b2};
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

/// Seed of the band particles: the A3-ANALYSIS seed.
pub const BAND_SEED: u64 = 20_261_102;
/// Band edges (km) and reference orbit (km) of the A3-ANALYSIS runs, from
/// `docs/results/configs/chaos-map-full30.toml`.
pub const BAND_MIN_KM: f64 = 139_700.0;
/// See [`BAND_MIN_KM`].
pub const BAND_MAX_KM: f64 = 141_100.0;
/// See [`BAND_MIN_KM`].
pub const BAND_REFERENCE_KM: f64 = 140_222.43;

/// Collision parameters of the B2 rows (`crates/fring-collide/SPEC.md`
/// section 10): radius s = 1 cm, cells L_x = 0.05 km, L_y = 10 km,
/// L_z = 0.04 km (H / 2 for c = 1 cm/s), constant restitution 0.5 (a cost
/// figure, not a physical choice).
pub const PARTICLE_RADIUS_KM: f64 = 1.0e-5;
/// Radial cell size (km); the sensitivity point uses [`WIDE_CELL_KM`].
pub const CELL_RADIAL_KM: f64 = 0.05;
/// Radial cell size of the sensitivity point (km).
pub const WIDE_CELL_KM: f64 = 0.2;
/// Azimuthal cell size (km).
pub const CELL_AZIMUTHAL_KM: f64 = 10.0;
/// Vertical cell size (km).
pub const CELL_VERTICAL_KM: f64 = 0.04;
/// Weight W = 8e8 for 10^6 superparticles (tau = 0.3, SPEC section 10).
pub const WEIGHT_AT_1E6: f64 = 8.0e8;

/// Weight of one superparticle for `n` superparticles: W = 8e8 x (10^6 / n).
pub fn weight_for(n: usize) -> f64 {
    WEIGHT_AT_1E6 * 1.0e6 / n as f64
}

/// Everything that does not depend on N.
struct Setup {
    base: RunConfig,
    system: System,
    field: SaturnField<f64>,
    /// Moon radii in the track order of the ephemeris (km).
    radii_km: Vec<f64>,
}

fn setup(repeats: usize) -> Result<Setup, String> {
    // Steps that the moon ephemeris must cover: warm-up, probes, the repeats and
    // read-back steps, each at most MAX_STEPS, with margin (5 x for 3 repeats).
    let ephemeris_steps = (repeats as u64 + 2).max(5) * measure::MAX_STEPS;
    let text = include_str!("../../../crates/fring-model/tests/data/example-run.toml");
    let years = ephemeris_steps as f64 * STEP_S / (365.25 * 86_400.0);
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
        ephemeris_steps as f64 * STEP_S + 7200.0,
        1800.0,
    )
    .map_err(|e| e.to_string())?;
    let system = System { saturn, moons: eph };
    let field = SaturnField::<f64>::from_params(&system.saturn);
    let named = moon_radii_from_toml_str(include_str!("../../../params/moons.toml"))
        .map_err(|e| e.to_string())?;
    let radii_km = system
        .moons
        .tracks
        .iter()
        .map(|t| {
            named
                .iter()
                .find(|(name, _)| *name == t.name)
                .map(|(_, r)| *r)
                .ok_or_else(|| format!("no radius_km for moon {:?}", t.name))
        })
        .collect::<Result<Vec<f64>, String>>()?;
    Ok(Setup {
        base,
        system,
        field,
        radii_km,
    })
}

/// The config of one point: chaos indicator as in ICR-003 (d0 = 1 km, K = 100),
/// and the B2 feature tables. A `Base` point on the core is the B1 config.
fn point_config(base: &RunConfig, p: &Point) -> RunConfig {
    let mut cfg = base.clone();
    let f = p.feature;
    if f.freeze() {
        cfg.loss = Some(LossConfig {
            on_unbound: OnUnbound::Freeze,
            moon_radius: if f.radius_rule() {
                MoonRadiusRule::Params
            } else {
                MoonRadiusRule::Off
            },
        });
    }
    if f == Feature::Substep {
        cfg.substep = Some(SubstepConfig {
            accuracy: 0.1,
            max_level: 4,
        });
    }
    if f.collisions() {
        cfg.collisions = Collisions::Stochastic {
            particle_radius_km: PARTICLE_RADIUS_KM,
            weight: weight_for(p.n),
            cell_radial_km: if p.wide_cells {
                WIDE_CELL_KM
            } else {
                CELL_RADIAL_KM
            },
            cell_azimuthal_km: CELL_AZIMUTHAL_KM,
            cell_vertical_km: CELL_VERTICAL_KM,
            every_steps: p.every_steps,
            restitution: Restitution::Constant { epsilon: 0.5 },
        };
    }
    if p.state == State::Band {
        cfg.frame.reference_semi_major_axis_km = BAND_REFERENCE_KM;
    }
    if p.chaos {
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

/// The initial particles of a state: the B1 cold core (seed 1) or the
/// A3-ANALYSIS band (seed 20261102), generated directly.
fn make_particles(
    field: &SaturnField<f64>,
    n: usize,
    state: State,
) -> Result<InitialState, String> {
    let (dist, seed) = match state {
        State::Core => (
            Distribution::Core {
                semi_major_axis_km: 140_220.0,
                sigma_semi_major_axis_km: 5.0,
                eccentricity: 0.0024,
                apse_longitude_deg: 0.0,
                inclination_deg: 0.0,
            },
            SEED,
        ),
        State::Band => (
            Distribution::Band {
                min_semi_major_axis_km: BAND_MIN_KM,
                max_semi_major_axis_km: BAND_MAX_KM,
            },
            BAND_SEED,
        ),
    };
    let count = u32::try_from(n).map_err(|_| format!("{n} particles do not fit in u32"))?;
    let particles = generate_particles(field, &dist, count, seed).map_err(|e| e.to_string())?;
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
    repeats: usize,
    env: &mut RunEnv,
) -> Result<measure::Timing, String> {
    let mut cfg = point_config(&s.base, &p);
    cfg.particles.count = init.particles.len() as u32;
    let opt = measure::Options {
        repeats,
        collisions_every: p.feature.collisions().then_some(p.every_steps),
    };
    let caught = catch_unwind(AssertUnwindSafe(|| -> Result<measure::Timing, String> {
        match p.variant {
            Variant::Cpu => {
                let mut b = CpuBackend::new(&cfg, &s.system, init).map_err(|e| e.to_string())?;
                if p.feature.radius_rule() {
                    b.set_moon_radii_km(&s.radii_km)
                        .map_err(|e| e.to_string())?;
                }
                measure::measure_with(&mut b, p.n, p.chaos, target_s, measure::MAX_STEPS, opt)
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
                if p.feature.radius_rule() {
                    b.set_moon_radii_km(&s.radii_km)
                        .map_err(|e| e.to_string())?;
                }
                measure::measure_with(&mut b, p.n, p.chaos, target_s, measure::MAX_STEPS, opt)
            }
        }
    }));
    match caught {
        Ok(r) => r,
        Err(_) => Err("the backend panicked (probably out of memory)".to_string()),
    }
}

/// Short label of a point for the progress lines.
fn point_label(p: &Point) -> String {
    format!(
        "{} {} {} N={} every={}{} chaos={}",
        p.variant.name(),
        p.feature.name(),
        p.state.name(),
        p.n,
        p.every_steps,
        if p.wide_cells { " wide" } else { "" },
        if p.chaos { "on" } else { "off" }
    )
}

/// Runs the matrix and returns the outcomes. Progress goes to stderr.
pub fn run_matrix(args: &Args, env: &mut RunEnv) -> Result<Vec<Outcome>, String> {
    let repeats = args.repeats();
    let s = setup(repeats)?;
    let target_s = if args.quick { QUICK_TARGET_S } else { TARGET_S };
    let points = if args.is_b2() {
        build_matrix_b2(
            &args.sizes,
            &B2Spec {
                features: args.features.clone(),
                states: args.states.clone(),
                every: args.every.clone(),
                sensitivity: args.sensitivity,
            },
        )
    } else {
        build_matrix(&args.sizes, !args.no_mode_d)
    };
    let mut outcomes = Vec::with_capacity(points.len());
    let mut current: Option<((usize, State), Result<InitialState, String>)> = None;
    for (i, p) in points.iter().enumerate() {
        if current.as_ref().is_none_or(|(k, _)| *k != (p.n, p.state)) {
            eprintln!("generating {} particles ({})", p.n, p.state.name());
            current = Some(((p.n, p.state), make_particles(&s.field, p.n, p.state)));
        }
        let t0 = Instant::now();
        let result = match &current {
            Some((_, Ok(init))) => run_point(&s, init, *p, target_s, repeats, env),
            Some((_, Err(why))) => Err(why.clone()),
            None => unreachable!(),
        };
        let line = match &result {
            Ok(t) => format!("{} particle-steps/s", report::sci(t.rate_best)),
            Err(why) => format!("not run: {why}"),
        };
        eprintln!(
            "[{}/{}] {}: {line} ({:.1} s)",
            i + 1,
            points.len(),
            point_label(p),
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
    let csv_text = if args.is_b2() {
        report::csv_b2(&outcomes)
    } else {
        report::csv(&outcomes)
    };
    std::fs::write(&path, csv_text).map_err(|e| format!("{}: {e}", path.display()))?;
    eprintln!("wrote {}", path.display());

    println!("{}", report::markdown_machine(&facts, &env));
    if args.is_b2() {
        println!("{}", report::markdown_table_b2(&outcomes));
    } else {
        println!("{}", report::markdown_table(&outcomes));
    }
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
        let s = setup(3).unwrap();
        let init = make_particles(&s.field, 50, State::Core).unwrap();
        let mut env = RunEnv::default();
        for chaos in [false, true] {
            let p = Point::b1(Variant::Cpu, 50, chaos);
            let t = run_point(&s, &init, p, 0.02, 3, &mut env).unwrap();
            assert!(t.rate_best >= t.rate_min && t.rate_min > 0.0);
            assert!(t.steps_per_call >= measure::MIN_STEPS);
        }
    }

    fn feature_point(f: Feature, state: State, n: usize) -> Point {
        Point {
            variant: Variant::Cpu,
            n,
            chaos: false,
            feature: f,
            state,
            every_steps: 10,
            wide_cells: false,
        }
    }

    /// Every feature config passes `RunConfig::validate`, and the weight
    /// follows W = 8e8 x (10^6 / N).
    #[test]
    fn feature_configs_validate() {
        let s = setup(3).unwrap();
        for f in [
            Feature::Base,
            Feature::Freeze,
            Feature::Loss,
            Feature::Coll,
            Feature::CollLoss,
            Feature::Substep,
        ] {
            for state in [State::Core, State::Band] {
                let p = feature_point(f, state, 1_000_000);
                let mut cfg = point_config(&s.base, &p);
                cfg.particles.count = 1_000_000;
                cfg.validate().unwrap_or_else(|e| panic!("{f:?}: {e}"));
                assert_eq!(cfg.collisions != Collisions::None, f.collisions());
                assert_eq!(cfg.freeze_unbound(), f.freeze());
                assert_eq!(cfg.moon_radius_rule(), f.radius_rule());
                assert_eq!(cfg.substep.is_some(), f == Feature::Substep);
            }
        }
        assert_eq!(weight_for(1_000_000), 8.0e8);
        assert_eq!(weight_for(10_000_000), 8.0e7);
        assert_eq!(s.radii_km.len(), 2);
    }

    /// Tiny CPU points with each feature run and give sensible statistics.
    #[test]
    fn tiny_cpu_feature_points_run() {
        let s = setup(2).unwrap();
        let mut env = RunEnv::default();
        for state in [State::Core, State::Band] {
            let init = make_particles(&s.field, 400, state).unwrap();
            for f in [
                Feature::Freeze,
                Feature::Loss,
                Feature::Coll,
                Feature::CollLoss,
                Feature::Substep,
            ] {
                let p = feature_point(f, state, 400);
                let t = run_point(&s, &init, p, 0.02, 2, &mut env)
                    .unwrap_or_else(|e| panic!("{f:?} {state:?}: {e}"));
                assert_eq!(t.raw_rates.len(), 2);
                assert!(t.rate_best >= t.rate_min && t.rate_min > 0.0);
                assert_eq!(t.steps_per_call % 10, 0);
                assert_eq!(t.coll.is_some(), f.collisions());
                if let Some(c) = t.coll {
                    assert_eq!(c.coll_steps, 2 * t.steps_per_call / 10);
                }
            }
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
            features: Vec::new(),
            states: vec![State::Core],
            every: vec![1, 10],
            repeats: None,
            sensitivity: false,
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
