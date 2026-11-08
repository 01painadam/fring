//! E1-GPU, test G6 (ICR-004 amendment 2): throughput of `GpuBackend` with the
//! `[loss]` keys absent, with `on_unbound = "freeze"`, and with freeze plus
//! `moon_radius = "params"` (radii set). The population is the core of the
//! `fring-bench` matrix (a = 140,220 km), so no particle is lost: the numbers
//! are the cost of the checks alone. One call is timed three times for each
//! variant, and the variants take turns (off, freeze, rule, off, ...) so that
//! a drift of the GPU clock does not favour one.
//!
//! `cargo run -p fring-gpu --release --example e1_bench [N ...]`
//! (default N = 10000 100000 1000000). Limit: the best rate of the rule variant is
//! within 5 % of the best rate with the keys absent.

use fring_gpu::backend::GpuBackend;
use fring_model::config::Distribution;
use fring_model::params::{moons_from_toml_str, saturn_from_toml_str};
use fring_model::physics::{SaturnField, build_ephemeris, generate_particles};
use fring_model::{Backend, InitialState, RunConfig, System};
use std::time::Instant;

const H: f64 = 356.2537;
const REPEATS: usize = 3;
const TARGET_S: f64 = 1.5;
/// Radii of Prometheus and Pandora, km (`params/moons.toml`).
const RADII_KM: [f64; 2] = [43.09, 40.63];

fn make(
    extra: &str,
    radii: bool,
    system: &System,
    init: &InitialState,
    n: usize,
    years: f64,
) -> GpuBackend {
    let text = include_str!("../../fring-model/tests/data/example-run.toml")
        .replace("step_s = 360.0", &format!("step_s = {H}"))
        .replace("duration_years = 1.0", &format!("duration_years = {years}"))
        .replace("count = 10000", &format!("count = {n}"))
        + "\n"
        + extra;
    let cfg = RunConfig::from_toml_str(&text).unwrap();
    let mut b = GpuBackend::new(&cfg, system, init).unwrap();
    if radii {
        b.set_moon_radii_km(&RADII_KM).unwrap();
    }
    b
}

fn main() {
    let sizes: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let sizes = if sizes.is_empty() {
        vec![10_000, 100_000, 1_000_000]
    } else {
        sizes
    };
    let max_steps = 30_000u64;
    let saturn = saturn_from_toml_str(include_str!("../../../params/saturn.toml")).unwrap();
    let all = moons_from_toml_str(include_str!("../../../params/moons.toml")).unwrap();
    let years = 5.0 * max_steps as f64 * H / (365.25 * 86_400.0);
    let eph = build_ephemeris(&saturn, &all, 5.0 * max_steps as f64 * H + 7200.0, 1800.0).unwrap();
    let system = System { saturn, moons: eph };
    let field = SaturnField::<f64>::from_params(&system.saturn);
    let variants: [(&str, &str, bool); 3] = [
        ("keys absent", "", false),
        ("freeze", "[loss]\non_unbound = \"freeze\"\n", false),
        (
            "freeze + radius rule",
            "[loss]\non_unbound = \"freeze\"\nmoon_radius = \"params\"\n",
            true,
        ),
    ];
    println!(
        "| N | variant | best rate (particle-steps/s) | slowest | ratio to keys absent (best) |"
    );
    println!("|---|---|---|---|---|");
    for n in sizes {
        let dist = Distribution::Core {
            semi_major_axis_km: 140_220.0,
            sigma_semi_major_axis_km: 5.0,
            eccentricity: 0.0024,
            apse_longitude_deg: 0.0,
        };
        let init = InitialState {
            time_s: 0.0,
            particles: generate_particles(&field, &dist, n as u32, 1).unwrap(),
        };
        let mut b: Vec<GpuBackend> = variants
            .iter()
            .map(|(_, extra, radii)| make(extra, *radii, &system, &init, n, years))
            .collect();
        // Warm-up and calibration on the first variant: about TARGET_S per call.
        let t = Instant::now();
        for x in &mut b {
            x.step(10).unwrap();
        }
        let _ = t;
        let t = Instant::now();
        b[0].step(100).unwrap();
        let per_step = t.elapsed().as_secs_f64() / 100.0;
        let steps = ((TARGET_S / per_step) as u64 / 10 * 10).clamp(10, max_steps);
        let mut rates = vec![Vec::new(); b.len()];
        for _ in 0..REPEATS {
            for (v, x) in b.iter_mut().enumerate() {
                let t = Instant::now();
                x.step(steps).unwrap();
                rates[v].push(n as f64 * steps as f64 / t.elapsed().as_secs_f64());
            }
        }
        let best = |r: &Vec<f64>| r.iter().cloned().fold(0.0, f64::max);
        let slow = |r: &Vec<f64>| r.iter().cloned().fold(f64::INFINITY, f64::min);
        let base = best(&rates[0]);
        for (v, (name, _, _)) in variants.iter().enumerate() {
            println!(
                "| {n} | {name} | {:.4e} | {:.4e} | {:.3} |",
                best(&rates[v]),
                slow(&rates[v]),
                best(&rates[v]) / base
            );
        }
        eprintln!("N = {n}: {steps} steps per call");
    }
}
