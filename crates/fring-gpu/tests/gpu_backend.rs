//! `GpuBackend` through the `Backend` trait: against `CpuBackend` over one
//! orbit, the failure flag, energy statistics, and the bit-identity of
//! `wh.wgsl` with the frozen G1 kernels in `spike.wgsl`. Needs a GPU, so
//! `#[ignore]`: `cargo test -p fring-gpu --release --test gpu_backend -- --ignored --nocapture`.

mod common;

use common::*;
use fring_cpu::CpuBackend;
use fring_gpu::backend::{GpuBackend, Options};
use fring_gpu::gpu::Gpu;
use fring_gpu::spike::{Consts, Opt, Tables, state_from_f64};
use fring_gpu::spike_gpu::GpuSpike;
use fring_gpu::wf32::model;
use fring_gpu::wh::{self, Mode};
use fring_gpu::wh_gpu::{DEFAULT_WORKGROUP, NO_FAILURE, WhKernels};
use fring_model::{Backend, Error, Vec3};

fn max_rel(a: &fring_model::Particles, b: &fring_model::Particles) -> (f64, f64) {
    let (mut wp, mut wv) = (0.0f64, 0.0f64);
    for i in 0..a.len() {
        let (pa, pb) = (a.position_km(i), b.position_km(i));
        let (va, vb) = (a.velocity_km_s(i), b.velocity_km_s(i));
        wp = wp.max((pa - pb).norm() / pb.norm());
        wv = wv.max((va - vb).norm() / vb.norm());
    }
    (wp, wv)
}

#[test]
#[ignore = "needs a GPU"]
fn matches_the_cpu_backend_over_one_orbit_with_and_without_moons() {
    if Gpu::new().is_none() {
        eprintln!("no GPU: skipped");
        return;
    }
    for moons in [true, false] {
        for mode in [Mode::D, Mode::Dp, Mode::C] {
            let steps = 150;
            let system = system(H_S, steps, moons);
            let config = config(H_S, steps, moons);
            let (init, _) = ensemble(&system, 600);
            let mut cpu = CpuBackend::with_point_mass_drift(&config, &system, &init).unwrap();
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
            // Several calls and a block size that does not divide them.
            for n in [1, 60, 89] {
                cpu.step(n).unwrap();
                gpu.step(n).unwrap();
            }
            assert_eq!(gpu.time_s(), cpu.time_s());
            let pc = cpu.particles().unwrap().particles.clone();
            let pg = gpu.particles().unwrap().particles.clone();
            let (wp, wv) = max_rel(&pg, &pc);
            println!(
                "moons {moons}, mode {}: after 150 steps GPU against CPU: position {wp:.2e}, velocity {wv:.2e} (relative)",
                mode.name()
            );
            let bound = match mode {
                Mode::C => 1e-9,
                Mode::Dp => 1e-8,
                Mode::D => 1e-7,
            };
            assert!(wp < bound && wv < bound);
            let (sg, sc) = (gpu.stats().unwrap(), cpu.stats().unwrap());
            assert_eq!(sg.steps_total, sc.steps_total);
            assert_eq!(
                sg.max_rel_energy_error.is_some(),
                sc.max_rel_energy_error.is_some()
            );
            if let (Some(a), Some(b)) = (sg.max_rel_energy_error, sc.max_rel_energy_error) {
                println!("  energy error: GPU {a:.2e}, CPU {b:.2e}");
                assert!(a < 1e-9);
            }
        }
    }
}

#[test]
#[ignore = "needs a GPU"]
fn small_blocks_and_one_big_block_agree() {
    if Gpu::new().is_none() {
        return;
    }
    let steps = 400;
    let system = system(H_S, steps, true);
    let config = config(H_S, steps, true);
    let (init, _) = ensemble(&system, 300);
    let run = |block: usize| {
        let mut g = GpuBackend::with_options(
            &config,
            &system,
            &init,
            Options {
                block_steps: Some(block),
                ..Default::default()
            },
        )
        .unwrap();
        g.step(steps).unwrap();
        g.particles().unwrap().particles.clone()
    };
    // Different block sizes split the half drifts differently, but the merged
    // drift-kick-drift sequence is the same map: equal up to DS rounding.
    let (a, b) = (run(400), run(37));
    let (wp, wv) = max_rel(&a, &b);
    println!("block 400 against block 37: position {wp:.2e}, velocity {wv:.2e}");
    assert!(wp < 1e-8 && wv < 1e-8);
}

#[test]
#[ignore = "needs a GPU"]
fn an_unbound_particle_is_reported_with_its_id() {
    if Gpu::new().is_none() {
        return;
    }
    let steps = 20;
    let system = system(H_S, steps, true);
    let config = config(H_S, steps, true);
    let (mut init, _) = ensemble(&system, 200);
    // Particle with index 77 (id 1077) gets 3 times the escape speed.
    let bad = 77;
    init.particles.id[bad] = 1077;
    init.particles.vy_km_s[bad] *= 3.0;
    let mut gpu = GpuBackend::new(&config, &system, &init).unwrap();
    match gpu.step(steps) {
        Err(Error::Backend(m)) => {
            println!("{m}");
            assert!(m.contains("particle 1077") && m.contains("index 77"), "{m}");
        }
        other => panic!("expected Error::Backend, got {other:?}"),
    }
}

#[test]
#[ignore = "needs a GPU"]
fn a_huge_particle_count_gives_a_clear_error() {
    let Some(gpu) = Gpu::new() else {
        return;
    };
    println!(
        "{}: max_storage_buffer_binding_size {} bytes, max_buffer_size {} bytes, max workgroup size x {}",
        gpu.info.name,
        gpu.limits.max_storage_buffer_binding_size,
        gpu.limits.max_buffer_size,
        gpu.limits.max_compute_workgroup_size_x
    );
}

/// `wh.wgsl` is a copy of the G1 kernels with a new layout: without moons
/// (identical tables) the results must be identical bits, with moons only the
/// rounding of the indirect term differs.
#[test]
#[ignore = "needs a GPU"]
fn wh_wgsl_gives_the_bits_of_spike_wgsl() {
    let Some(gpu) = Gpu::new() else {
        return;
    };
    const M4: u8 = model::M4;
    let h = 356.25; // exact in f32: the G1 kernels need it
    let spike = GpuSpike::new(&gpu).unwrap();
    let k = WhKernels::new(&gpu, DEFAULT_WORKGROUP).unwrap();
    for moons in [false, true] {
        let steps = 150;
        let system = system(h, steps, moons);
        let (init, _) = ensemble(&system, 128);
        let gms: Vec<f64> = system.moons.tracks.iter().map(|t| t.gm_km3_s2).collect();
        let states: Vec<_> = (0..128)
            .map(|i| {
                state_from_f64::<M4>(
                    init.particles.position_km(i),
                    init.particles.velocity_km_s(i),
                )
            })
            .collect();
        let c = Consts::<M4>::new(&system.saturn, &system.moons, h);
        let mut tab = Tables::<M4>::new(&system.moons);
        tab.fill(&system.moons, 0.0, h, 0, steps as usize).unwrap();
        let mut kt = fring_model::physics::KickTable::<f64>::for_ephemeris(&system.moons);
        kt.fill(&system.moons, 0.0, h, 0, steps as usize).unwrap();
        let packed = wh::pack_table(&kt, 1.0);
        let c2 = Consts::<M4>::for_step(&system.saturn, &gms, h);
        let prm = wh::params(&c2, &gms);
        for (opt, mode) in [(Opt::D, Mode::D), (Opt::C, Mode::C)] {
            let mut a = states.clone();
            spike.advance(opt, &c, &tab, &mut a);
            let mut b = states.clone();
            let (status, _) = k.apply(
                mode.entry(),
                &prm,
                &packed,
                steps as usize,
                gms.len(),
                &mut b,
            );
            assert_eq!(status, NO_FAILURE);
            let same = a.iter().zip(&b).filter(|(x, y)| x == y).count();
            let mut worst = 0.0f64;
            for (x, y) in a.iter().zip(&b) {
                let (px, py) = (
                    fring_gpu::spike::state_to_f64(x),
                    fring_gpu::spike::state_to_f64(y),
                );
                worst = worst
                    .max((px.0 - py.0).norm() / px.0.norm())
                    .max((px.1 - py.1).norm() / px.1.norm());
            }
            println!(
                "moons {moons}, {}: identical states {same}/128, worst relative difference {worst:.2e}",
                mode.name()
            );
            if !moons {
                assert_eq!(same, 128, "bit-identity without moons");
            } else {
                assert!(worst < 1e-8);
            }
        }
    }
    let _ = Vec3::new(0.0, 0.0, 0.0);
}
