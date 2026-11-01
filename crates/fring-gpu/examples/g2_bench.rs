//! Task G2 benchmark: particle-steps per second of `GpuBackend`.
//!
//! 1. Kernel throughput by mode (G1 method: 10^6 core particles, both moons,
//!    150 steps per call, best of 3) next to `CpuBackend` (10^5 particles).
//! 2. Workgroup sweep for each mode.
//! 3. Throughput through the `Backend` interface with read-back at a given
//!    output cadence (a frame every 150 and every 1500 steps).
//!
//! `cargo run -p fring-gpu --release --example g2_bench [-- --n 1000000]`

use fring_cpu::CpuBackend;
use fring_gpu::backend::{GpuBackend, Options};
use fring_gpu::wh::Mode;
use fring_model::params::{moons_from_toml_str, saturn_from_toml_str};
use fring_model::physics::{EpicycleElements, SaturnField, build_ephemeris, epicycle_state};
use fring_model::{Backend, InitialState, Particles, RunConfig, System};
use std::time::Instant;

const H_S: f64 = 356.2537;

fn setup(n: usize, steps: u64) -> (RunConfig, System, InitialState) {
    let text = include_str!("../../fring-model/tests/data/example-run.toml");
    let years = steps as f64 * H_S / (365.25 * 86_400.0);
    let t = text
        .replace("step_s = 360.0", &format!("step_s = {H_S}"))
        .replace("duration_years = 1.0", &format!("duration_years = {years}"));
    let config = RunConfig::from_toml_str(&t).unwrap();
    let saturn = saturn_from_toml_str(include_str!("../../../params/saturn.toml")).unwrap();
    let moons = moons_from_toml_str(include_str!("../../../params/moons.toml")).unwrap();
    let eph = build_ephemeris(&saturn, &moons, steps as f64 * H_S + 7200.0, 1800.0).unwrap();
    let system = System { saturn, moons: eph };
    let field = SaturnField::<f64>::from_params(&system.saturn);
    let mut p = Particles::with_capacity(n);
    for i in 0..n {
        let k = i as f64;
        let el = EpicycleElements {
            a_km: 140_222.43 + 20.0 * ((k * 0.618_033_988_749_894_9) % 1.0 - 0.5),
            e: 0.0024,
            i_rad: 1.1e-4,
            mean_anomaly_rad: std::f64::consts::TAU * ((k * 0.754_877_666_246_693) % 1.0),
            z_phase_rad: std::f64::consts::TAU * ((k * 0.569_840_290_998_053_2) % 1.0),
            mean_longitude_rad: std::f64::consts::TAU * ((k * 0.381_966_011_250_105_1) % 1.0),
        };
        let (pos, vel) = epicycle_state(&field, &el);
        p.push(i as u32, pos, vel);
    }
    (
        config,
        system,
        InitialState {
            time_s: 0.0,
            particles: p,
        },
    )
}

fn best_of_3(b: &mut dyn Backend, n: usize, steps: u64) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..3 {
        let t = Instant::now();
        b.step(steps).unwrap();
        best = best.min(t.elapsed().as_secs_f64());
    }
    n as f64 * steps as f64 / best
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args
        .iter()
        .position(|a| a == "--n")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(1_000_000);
    let (config, system, init) = setup(n, 20_000);
    let modes = [Mode::D, Mode::Dp, Mode::C];

    println!(
        "section,mode,workgroup,n_particles,steps_per_call,output_every,particle_steps_per_s,note"
    );
    // CPU, 10^5 particles.
    {
        let (c, s, i) = setup(100_000, 20_000);
        let mut cpu = CpuBackend::new(&c, &s, &i).unwrap();
        let r = best_of_3(&mut cpu, 100_000, 150);
        println!("kernel,cpu_f64,-,100000,150,-,{r:.3e},CpuBackend");
    }
    for mode in modes {
        let mut g = GpuBackend::with_options(
            &config,
            &system,
            &init,
            Options {
                mode,
                ..Default::default()
            },
        )
        .unwrap();
        let r = best_of_3(&mut g, n, 150);
        println!(
            "kernel,{},64,{n},150,-,{r:.3e},{} block {}",
            mode.name(),
            g.adapter_name(),
            g.block_steps()
        );
    }
    for mode in modes {
        for wg in [32u32, 64, 128, 256] {
            let mut g = GpuBackend::with_options(
                &config,
                &system,
                &init,
                Options {
                    mode,
                    workgroup: wg,
                    ..Default::default()
                },
            )
            .unwrap();
            let r = best_of_3(&mut g, n, 150);
            println!("sweep,{},{wg},{n},150,-,{r:.3e},", mode.name());
        }
    }
    for mode in modes {
        for every in [150u64, 1500] {
            let mut g = GpuBackend::with_options(
                &config,
                &system,
                &init,
                Options {
                    mode,
                    ..Default::default()
                },
            )
            .unwrap();
            let total = 3000u64;
            let t = Instant::now();
            let mut done = 0;
            while done < total {
                g.step(every).unwrap();
                done += every;
                let v = g.particles().unwrap();
                std::hint::black_box(v.particles.x_km[0]);
            }
            let secs = t.elapsed().as_secs_f64();
            println!(
                "interface,{},64,{n},{every},{every},{:.3e},frame read-back every {every} steps",
                mode.name(),
                n as f64 * total as f64 / secs
            );
        }
    }
}
