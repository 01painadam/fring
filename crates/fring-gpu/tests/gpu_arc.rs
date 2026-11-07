//! Task D1: the arc wrap on the GPU. Needs a GPU, so `#[ignore]`:
//! `cargo test -p fring-gpu --release --test gpu_arc -- --ignored --nocapture`.

mod common;

use common::*;
use fring_gpu::backend::{GpuBackend, Options};
use fring_gpu::gpu::Gpu;
use fring_gpu::wh::Mode;
use fring_model::Backend;

/// FNV-1a of the f64 bits of the particle positions and velocities after a run
/// with the full ring (arc off). Taken on the commit before D1 changed any
/// shader (`db282d5`): the arc code must not change a full-ring run by one bit.
const GOLDEN_FULL_RING: u64 = 0xa999_c5ad_4c72_2112;

fn full_ring_hash(mode: Mode) -> u64 {
    let steps = 150;
    let system = system(H_S, steps, true);
    let config = config(H_S, steps, true);
    let (init, _) = ensemble(&system, 300);
    let mut gpu = GpuBackend::with_options(
        &config,
        &system,
        &init,
        Options {
            mode,
            block_steps: Some(64),
            ..Default::default()
        },
    )
    .unwrap();
    gpu.step(150).unwrap();
    let p = gpu.particles().unwrap().particles.clone();
    let mut bytes = Vec::new();
    for i in 0..p.len() {
        let (a, b) = (p.position_km(i), p.velocity_km_s(i));
        for x in [a.x, a.y, a.z, b.x, b.y, b.z] {
            bytes.extend_from_slice(&x.to_bits().to_le_bytes());
        }
    }
    fnv1a(&bytes)
}

#[test]
#[ignore = "needs a GPU"]
fn full_ring_run_is_bit_identical_to_before_d1() {
    if Gpu::new().is_none() {
        eprintln!("no GPU: skipped");
        return;
    }
    let h = full_ring_hash(Mode::Dp);
    println!("full ring hash {h:#018x}");
    assert_eq!(h, GOLDEN_FULL_RING);
}
