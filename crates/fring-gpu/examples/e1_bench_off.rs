//! E1-GPU, test G6: the "keys absent" rate only, using only the API that
//! `main` already has, so the same file runs on the tree before E1-GPU and on
//! the E1-GPU tree (same population, steps and method as `e1_bench`).
//! `cargo run -p fring-gpu --release --example e1_bench_off [N ...]`

use fring_gpu::backend::GpuBackend;
use fring_model::config::Distribution;
use fring_model::params::{moons_from_toml_str, saturn_from_toml_str};
use fring_model::physics::{SaturnField, build_ephemeris, generate_particles};
use fring_model::{Backend, InitialState, RunConfig, System};
use std::time::Instant;

const H: f64 = 356.2537;
const REPEATS: usize = 5;
const TARGET_S: f64 = 1.5;

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
    let years = 8.0 * max_steps as f64 * H / (365.25 * 86_400.0);
    let eph = build_ephemeris(&saturn, &all, 8.0 * max_steps as f64 * H + 7200.0, 1800.0).unwrap();
    let system = System { saturn, moons: eph };
    let field = SaturnField::<f64>::from_params(&system.saturn);
    println!("| N | best rate (particle-steps/s) | slowest |");
    println!("|---|---|---|");
    for n in sizes {
        let dist = Distribution::Core {
            semi_major_axis_km: 140_220.0,
            sigma_semi_major_axis_km: 5.0,
            eccentricity: 0.0024,
            apse_longitude_deg: 0.0,
            inclination_deg: 0.0,
        };
        let init = InitialState {
            time_s: 0.0,
            particles: generate_particles(&field, &dist, n as u32, 1).unwrap(),
        };
        let text = include_str!("../../fring-model/tests/data/example-run.toml")
            .replace("step_s = 360.0", &format!("step_s = {H}"))
            .replace("duration_years = 1.0", &format!("duration_years = {years}"))
            .replace("count = 10000", &format!("count = {n}"));
        let cfg = RunConfig::from_toml_str(&text).unwrap();
        let mut b = GpuBackend::new(&cfg, &system, &init).unwrap();
        b.step(10).unwrap();
        let t = Instant::now();
        b.step(100).unwrap();
        let per_step = t.elapsed().as_secs_f64() / 100.0;
        let steps = ((TARGET_S / per_step) as u64 / 10 * 10).clamp(10, max_steps);
        let mut rates = Vec::new();
        for _ in 0..REPEATS {
            let t = Instant::now();
            b.step(steps).unwrap();
            rates.push(n as f64 * steps as f64 / t.elapsed().as_secs_f64());
        }
        let best = rates.iter().cloned().fold(0.0, f64::max);
        let slow = rates.iter().cloned().fold(f64::INFINITY, f64::min);
        println!("| {n} | {best:.4e} | {slow:.4e} |");
    }
}
