//! Task G2, done when: the GPU matches the CPU (PLAN section 10) for 10^4 test
//! particles over 10^3 orbits, with both moons, through the `Backend` trait.
//!
//! Non-chaotic orbits are chosen as in G1 (ADR 0006): an f64 shadow particle
//! starts 1 mm further out. A particle is regular if, at every checkpoint (10
//! orbits) up to 10^3 orbits, the shadow's |Δa| is below 10 times its first
//! value and its phase difference below 10 times the linear Kepler shear.
//! For the regular particles: Δa < 0.01 km and Δλ < 1e-5 rad.
//!
//! Needs the GPU and takes about 2 minutes: `#[ignore]`. Run with
//! `cargo test -p fring-gpu --release --test done_when -- --ignored --nocapture`.

mod common;

use common::*;
use fring_cpu::CpuBackend;
use fring_gpu::backend::{GpuBackend, Options};
use fring_gpu::spike::{phase_difference, semi_major_axis};
use fring_gpu::wh::Mode;
use fring_model::{Backend, InitialState, Particles, Vec3};
use std::time::Instant;

const N: usize = 10_000;
const STEPS_PER_ORBIT: u64 = 150;
const ORBITS: u64 = 1000;
const STEPS: u64 = 150_030; // about 1000 orbits at H_S
const CHECK: u64 = 1500; // checkpoint every 10 orbits

fn state(p: &Particles, i: usize) -> (Vec3<f64>, Vec3<f64>) {
    (p.position_km(i), p.velocity_km_s(i))
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * q).round() as usize]
}

#[test]
#[ignore = "needs a GPU, about 2 minutes"]
fn gpu_matches_cpu_for_1e4_particles_over_1e3_orbits() {
    let _ = ORBITS;
    let system = system(H_S, STEPS, true);
    let config = config(H_S, STEPS, true);
    let gm = system.saturn.gm_km3_s2;
    let (init, kinds) = ensemble(&system, N);

    // Shadows: 1 mm further out.
    let mut shadow_init = InitialState {
        time_s: 0.0,
        particles: init.particles.clone(),
    };
    for i in 0..N {
        let p = init.particles.position_km(i);
        let q = p * (1.0 + 1e-6 / p.norm());
        shadow_init.particles.x_km[i] = q.x;
        shadow_init.particles.y_km[i] = q.y;
        shadow_init.particles.z_km[i] = q.z;
    }

    let t0 = Instant::now();
    let mut cpu = CpuBackend::new(&config, &system, &init).unwrap();
    let mut shadow = CpuBackend::new(&config, &system, &shadow_init).unwrap();
    let a_init: Vec<f64> = (0..N)
        .map(|i| {
            let (p, v) = state(&init.particles, i);
            semi_major_axis(gm, p, v)
        })
        .collect();
    // amp_a: max |Δa_sh| / |Δa_sh(first checkpoint)|; amp_l: max Δλ_sh / Δλ_lin.
    let mut da0 = vec![0.0f64; N];
    let mut amp_a = vec![0.0f64; N];
    let mut amp_l = vec![0.0f64; N];
    let mut done = 0u64;
    let mut cp = 0u64;
    while done < STEPS {
        let n = CHECK.min(STEPS - done);
        cpu.step(n).unwrap();
        shadow.step(n).unwrap();
        done += n;
        cp += 1;
        let orbits = done as f64 / STEPS_PER_ORBIT as f64;
        let (pc, ps) = (
            cpu.particles().unwrap().particles.clone(),
            shadow.particles().unwrap().particles.clone(),
        );
        for i in 0..N {
            let (p, v) = state(&pc, i);
            let (q, w) = state(&ps, i);
            let da = (semi_major_axis(gm, q, w) - semi_major_axis(gm, p, v)).abs();
            let dl = phase_difference(p, q).abs();
            if cp == 1 {
                da0[i] = da.max(1e-12);
            }
            let lin = 1.5 * da0[i] / A0_KM * std::f64::consts::TAU * orbits;
            amp_a[i] = amp_a[i].max(da / da0[i]);
            amp_l[i] = amp_l[i].max(dl / lin);
        }
    }
    let regular: Vec<usize> = (0..N)
        .filter(|&i| amp_a[i] < 10.0 && amp_l[i] < 10.0)
        .collect();
    println!(
        "CPU and shadow: {:.0} s. Regular at {} orbits: {} of {} (circular {}, core {}, stress {})",
        t0.elapsed().as_secs_f64(),
        done / STEPS_PER_ORBIT,
        regular.len(),
        N,
        regular.iter().filter(|&&i| kinds[i] == 0).count(),
        regular.iter().filter(|&&i| kinds[i] == 1).count(),
        regular.iter().filter(|&&i| kinds[i] == 2).count(),
    );
    let pc = cpu.particles().unwrap().particles.clone();

    let mut failures = Vec::new();
    for mode in [Mode::D, Mode::C] {
        let t1 = Instant::now();
        let mut gpu = GpuBackend::with_options(
            &config,
            &system,
            &init,
            Options {
                mode,
                ..Default::default()
            },
        )
        .unwrap();
        gpu.step(STEPS).unwrap();
        let t_gpu = t1.elapsed().as_secs_f64();
        let pg = gpu.particles().unwrap().particles.clone();
        let (mut da, mut dl): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
        for &i in &regular {
            let (p, v) = state(&pc, i);
            let (q, w) = state(&pg, i);
            da.push((semi_major_axis(gm, q, w) - semi_major_axis(gm, p, v)).abs());
            dl.push(phase_difference(p, q).abs());
        }
        let _ = &a_init;
        da.sort_by(|a, b| a.total_cmp(b));
        dl.sort_by(|a, b| a.total_cmp(b));
        let (ma, md) = (*da.last().unwrap(), *dl.last().unwrap());
        println!(
            "GPU {} ({:.1} s wall, {}): {} regular particles. da km: max {:.2e}, p99 {:.2e}, median {:.2e}. dphase rad: max {:.2e}, p99 {:.2e}, median {:.2e}",
            mode.name(),
            t_gpu,
            gpu.adapter_name(),
            regular.len(),
            ma,
            quantile(&da, 0.99),
            quantile(&da, 0.5),
            md,
            quantile(&dl, 0.99),
            quantile(&dl, 0.5),
        );
        if ma >= 0.01 || md >= 1e-5 {
            failures.push(format!(
                "mode {}: max da {ma:.2e} km, max dphase {md:.2e} rad",
                mode.name()
            ));
        }
    }
    assert!(failures.is_empty(), "PLAN section 10 not met: {failures:?}");
}
