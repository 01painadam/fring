//! ICR-005 item 5: `GpuStepper` against `GpuBackend::step`, bit for bit.
//! Needs a GPU, so `#[ignore]`:
//! `cargo test -p fring-gpu --release --test gpu_stepper -- --ignored --nocapture`.

mod common;

use common::*;
use fring_gpu::backend::{GpuBackend, Options};
use fring_gpu::gpu::Gpu;
use fring_gpu::stepper::GpuStepper;
use fring_gpu::wh::Mode;
use fring_gpu::wh_gpu::{DEFAULT_WORKGROUP, WhKernels};
use fring_model::Backend;
use fring_model::config::{ChaosConfig, ChaosMethod, DiagnosticsConfig};

/// Copies a buffer to the host (blocking; for tests).
fn read(gpu: &Gpu, buf: &wgpu::Buffer) -> Vec<u8> {
    let d = gpu.device();
    let stage = d.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: buf.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = d.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(buf, 0, &stage, 0, buf.size());
    gpu.queue().submit([enc.finish()]);
    stage.map_async(wgpu::MapMode::Read, .., |r| r.expect("map failed"));
    d.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let v = stage.get_mapped_range(..).expect("range").to_vec();
    stage.unmap();
    v
}

#[test]
#[ignore = "needs a GPU"]
fn stepper_gives_the_same_bits_as_the_backend() {
    let Some(gpu) = Gpu::new() else {
        eprintln!("no GPU: skipped");
        return;
    };
    let kernels = WhKernels::new(&gpu, DEFAULT_WORKGROUP).expect("self-test");
    let chaos = ChaosConfig {
        method: ChaosMethod::Shadow,
        offset_km: 1.0,
        renorm_steps: 5,
    };
    // (moons, chaos, mode, block steps). 7 divides neither the calls nor 5.
    let cases = [
        (true, None, Mode::Dp, Some(7)),
        (false, None, Mode::Dp, Some(7)),
        (true, Some(chaos), Mode::Dp, Some(7)),
        (false, Some(chaos), Mode::Dp, Some(7)),
        (true, None, Mode::Dp, None),
        (true, Some(chaos), Mode::D, Some(7)),
        (true, None, Mode::C, Some(7)),
    ];
    for (moons, chaos, mode, block_steps) in cases {
        let label = format!("moons {moons}, chaos {chaos:?}, {mode:?}, block {block_steps:?}");
        let calls = [1u64, 60, 89];
        let steps: u64 = calls.iter().sum();
        let system = system(H_S, steps, moons);
        let mut cfg = config(H_S, steps, moons);
        cfg.diagnostics = chaos.map(|c| DiagnosticsConfig { chaos: Some(c) });
        let (init, _) = ensemble(&system, 100);
        let opts = Options {
            mode,
            block_steps,
            ..Default::default()
        };
        let mut backend = GpuBackend::with_options(&cfg, &system, &init, opts).unwrap();
        let mut stepper = GpuStepper::new(
            &gpu,
            &kernels,
            &system,
            &init,
            H_S,
            backend.block_steps(),
            mode,
            chaos,
        )
        .unwrap();
        assert_eq!(stepper.len(), 100);
        assert_eq!(stepper.capacity(), backend.stepper().capacity());
        let bgpu = backend.gpu().clone();
        for n in calls {
            backend.step(n).unwrap();
            let mut enc = gpu.device().create_command_encoder(&Default::default());
            stepper.encode_steps(&mut enc, n).unwrap();
            gpu.queue().submit([enc.finish()]);
            assert_eq!(
                stepper.time_s().to_bits(),
                backend.time_s().to_bits(),
                "time, {label}"
            );
            let (bs, ss) = (backend.stepper(), &stepper);
            assert!(
                read(&bgpu, bs.state_buffer()) == read(&gpu, ss.state_buffer()),
                "state bytes after {n} steps, {label}"
            );
            assert!(
                read(&bgpu, bs.status_buffer()) == read(&gpu, ss.status_buffer()),
                "status bytes after {n} steps, {label}"
            );
            if let (Some(a), Some(b)) = (bs.chaos_buffer(), ss.chaos_buffer()) {
                assert!(
                    read(&bgpu, a) == read(&gpu, b),
                    "chaos accumulators after {n} steps, {label}"
                );
            }
        }
        eprintln!("same bits: {label}");
    }
}
