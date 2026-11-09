//! K5-WIRE benchmark: the cost of one collision step in `GpuBackend`
//! (bin pass + collide pass + the tally read), for N particles (default 10^5
//! and 10^6). Cells are scaled to a mean occupancy of about 5 in a 1000 km wide
//! band of eccentric orbits (e = 1e-4, no moons).
//!
//! Two differences, so that the block structure is the same on both sides:
//! - `every_steps = 1`: ten `step(1)` calls with collisions against ten
//!   `step(1)` calls without (each call waits for the GPU once);
//! - `every_steps = 10`: one `step(10)` with collisions against one without.
//!
//! `cargo run -p fring-gpu --release --example k5_wire_bench [-- --n 100000,1000000 --reps 7]`

use fring_gpu::backend::{GpuBackend, Options};
use fring_gpu::wh::Mode;
use fring_model::config::Distribution;
use fring_model::physics::{SaturnField, generate_particles};
use fring_model::{Backend, InitialState, MoonEphemeris, RunConfig, System};
use std::time::Instant;

fn arg(name: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter()
        .position(|x| x == name)
        .and_then(|i| a.get(i + 1))
        .cloned()
}

fn config(collisions: &str) -> RunConfig {
    let text = include_str!("../../fring-model/tests/data/example-run.toml")
        .replace("duration_years = 1.0", "duration_years = 0.1")
        .replace(r#"include = ["prometheus", "pandora"]"#, "include = []")
        .replace("[collisions]\nmodel = \"none\"", collisions);
    RunConfig::from_toml_str(&text).unwrap()
}

fn stochastic(every: u32, n: usize) -> String {
    // Occupancy 5: N lx ly / (1000 km x 2 pi 140,000 km) = 5, lx = 2 km.
    let lx = 2.0;
    let ly = 5.0 * 1000.0 * 879_646.0 / (n as f64 * lx);
    format!(
        "[collisions]\nmodel = \"stochastic\"\nparticle_radius_km = 1.0e-5\nweight = 1.0e12\n\
         cell_radial_km = {lx}\ncell_azimuthal_km = {ly}\ncell_vertical_km = 1.0\n\
         every_steps = {every}\n[collisions.restitution]\nlaw = \"constant\"\nepsilon = 0.5\n"
    )
}

/// Median wall time in ms of `reps` runs of `calls` calls of `step(per_call)`.
fn time(
    cfg: &RunConfig,
    sys: &System,
    init: &InitialState,
    calls: u64,
    per_call: u64,
    reps: usize,
) -> (f64, f64) {
    let opts = Options {
        mode: Mode::Dp,
        ..Default::default()
    };
    let mut b = GpuBackend::with_options(cfg, sys, init, opts).unwrap();
    b.step(per_call).unwrap(); // warm-up (compiles nothing new, touches the buffers)
    let mut v: Vec<f64> = (0..reps)
        .map(|_| {
            let t0 = Instant::now();
            for _ in 0..calls {
                b.step(per_call).unwrap();
            }
            t0.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    let s = b.stats().unwrap();
    assert!(cfg.collisions_are_none() || s.collision.occupancy_sum > 0);
    v.sort_by(f64::total_cmp);
    (v[0], v[reps / 2])
}

trait NoneCheck {
    fn collisions_are_none(&self) -> bool;
}
impl NoneCheck for RunConfig {
    fn collisions_are_none(&self) -> bool {
        matches!(self.collisions, fring_model::config::Collisions::None)
    }
}

fn main() {
    let sizes: Vec<usize> = arg("--n")
        .map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect())
        .unwrap_or_else(|| vec![100_000, 1_000_000]);
    let reps: usize = arg("--reps").and_then(|s| s.parse().ok()).unwrap_or(7);
    let saturn =
        fring_model::params::saturn_from_toml_str(include_str!("../../../params/saturn.toml"))
            .unwrap();
    let sys = System {
        saturn,
        moons: MoonEphemeris::none(),
    };
    let field = SaturnField::<f64>::from_params(&sys.saturn);
    let none = config("[collisions]\nmodel = \"none\"");
    println!(
        "| N | every | baseline min / median ms | with collisions min / median ms | collision steps | cost per collision step: difference of minima / difference of medians, ms |"
    );
    println!("|---|---|---|---|---|---|");
    for n in sizes {
        let dist = Distribution::EccentricBand {
            min_semi_major_axis_km: 139_600.0,
            max_semi_major_axis_km: 140_600.0,
            eccentricity: 1.0e-4,
            apse_longitude_deg: 0.0,
        };
        let init = InitialState {
            time_s: 0.0,
            particles: generate_particles(&field, &dist, n as u32, 1).unwrap(),
        };
        for every in [1u32, 10] {
            let (calls, per_call) = if every == 1 { (10, 1) } else { (1, 10) };
            let base = time(&none, &sys, &init, calls, per_call, reps);
            let with = time(
                &config(&stochastic(every, n)),
                &sys,
                &init,
                calls,
                per_call,
                reps,
            );
            let k = (10 / every) as f64;
            println!(
                "| {n} | {every} | {:.2} / {:.2} | {:.2} / {:.2} | {k} | {:.2} / {:.2} |",
                base.0,
                base.1,
                with.0,
                with.1,
                (with.0 - base.0) / k,
                (with.1 - base.1) / k
            );
        }
    }
}
