//! G1 throughput benchmark (ADR 0006): particle-steps per second of every
//! option on the GPU, and of the f64 CPU code on the same machine.
//!
//! ```sh
//! cargo run -p fring-gpu --release --example g1_bench > docs/adr/0006/bench_m4.csv
//! ```
//!
//! Setup: 10⁶ particles in the core (a = 140,222 km ± 50 km, e = 0.0024,
//! i = 1.1e-4), both moons, h = 356.25 s, one dispatch = 150 steps (one
//! orbit, all steps in registers). Each option runs once to warm up, then
//! three times; the best GPU time counts (from submit to completion, without
//! upload and read-back). The CPU runs 10⁵ particles for 150 steps with
//! rayon on all cores (the code path of `fring-cpu`).

use fring_gpu::gpu::Gpu;
use fring_gpu::spike::{Consts, Opt, State, Tables, state_from_f64};
use fring_gpu::spike_gpu::GpuSpike;
use fring_gpu::wf32::model;
use fring_model::Vec3;
use fring_model::params::{moons_from_toml_str, saturn_from_toml_str};
use fring_model::physics::{
    EpicycleElements, KickTable, SaturnField, build_ephemeris, epicycle_state, wh_steps_moons,
};
use rayon::prelude::*;
use std::time::Instant;

const M4: u8 = model::M4;
const H_S: f64 = 356.25;
const STEPS: usize = 150;

fn main() {
    let n_gpu: usize = std::env::args()
        .nth(1)
        .map_or(1_000_000, |s| s.parse().unwrap());
    let sp = saturn_from_toml_str(include_str!("../../../params/saturn.toml")).unwrap();
    let moons = moons_from_toml_str(include_str!("../../../params/moons.toml")).unwrap();
    let eph = build_ephemeris(&sp, &moons, STEPS as f64 * H_S + 7200.0, 1800.0).unwrap();
    let field = SaturnField::<f64>::from_params(&sp);
    let c = Consts::<M4>::new(&sp, &eph, H_S);
    let mut tab = Tables::<M4>::new(&eph);
    tab.fill(&eph, 0.0, H_S, 0, STEPS).unwrap();
    let init = |i: usize| {
        let el = EpicycleElements {
            a_km: 140_222.43 + 100.0 * ((i as f64 * 0.618_034) % 1.0 - 0.5),
            e: 0.0024,
            i_rad: 1.1e-4,
            mean_anomaly_rad: 2.399_963 * i as f64,
            z_phase_rad: 1.3 * i as f64,
            mean_longitude_rad: std::f64::consts::TAU * (i as f64 * 0.754_877_7 % 1.0),
        };
        epicycle_state(&field, &el)
    };

    println!("backend,option,trig,n_particles,steps,seconds,particle_steps_per_s,ratio_to_cpu");

    // CPU: the f64 code path of fring-cpu, rayon over particles.
    let n_cpu = 100_000;
    let mut t64 = KickTable::<f64>::for_ephemeris(&eph);
    t64.fill(&eph, 0.0, H_S, 0, STEPS).unwrap();
    let mut cpu: Vec<(Vec3<f64>, Vec3<f64>)> = (0..n_cpu).map(init).collect();
    let mut best_cpu = f64::INFINITY;
    for _ in 0..3 {
        let t0 = Instant::now();
        cpu.par_iter_mut().for_each(|(p, v)| {
            wh_steps_moons(&field, &t64, p, v, H_S).unwrap();
        });
        best_cpu = best_cpu.min(t0.elapsed().as_secs_f64());
    }
    let cpu_rate = (n_cpu * STEPS) as f64 / best_cpu;
    println!("cpu_f64_rayon,-,libm,{n_cpu},{STEPS},{best_cpu:.4},{cpu_rate:.3e},1.00");

    let Some(gpu) = Gpu::new() else {
        eprintln!("no GPU");
        return;
    };
    eprintln!("GPU: {} ({:?})", gpu.info.name, gpu.info.backend);
    let states: Vec<State<M4>> = (0..n_gpu)
        .map(|i| {
            let (p, v) = init(i);
            state_from_f64(p, v)
        })
        .collect();
    let builtin = GpuSpike::builtin_trig_source();
    for (trig, spike) in [
        ("own", GpuSpike::new(&gpu).expect("self-test")),
        (
            "builtin",
            GpuSpike::with_source(&gpu, &builtin).expect("self-test"),
        ),
    ] {
        for opt in Opt::ALL {
            let mut st = states.clone();
            spike.advance(opt, &c, &tab, &mut st); // warm-up
            let mut best = f64::INFINITY;
            for _ in 0..3 {
                let mut st = states.clone();
                best = best.min(spike.advance(opt, &c, &tab, &mut st).as_secs_f64());
            }
            let rate = (n_gpu * STEPS) as f64 / best;
            println!(
                "gpu_m4,{},{trig},{n_gpu},{STEPS},{best:.4},{rate:.3e},{:.2}",
                opt.name(),
                rate / cpu_rate
            );
        }
    }
}
