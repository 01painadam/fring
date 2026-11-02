//! ICR-006 (W2-DROP-GPU): dropped particles in `GpuStepper`. Needs a GPU, so
//! `#[ignore]`:
//! `cargo test -p fring-gpu --release --test gpu_drop -- --ignored --nocapture`.

mod common;

use common::*;
use fring_gpu::gpu::Gpu;
use fring_gpu::stepper::GpuStepper;
use fring_gpu::wh::Mode;
use fring_gpu::wh_gpu::{DEFAULT_WORKGROUP, WhKernels};

/// Hashes of the state buffer of the rule-off stepper before ICR-006 existed
/// (commit 938344d): 100 particles of the G1 ensemble with the real moons,
/// 150 steps in blocks of 7, modes D and Dp.
///
/// GPU-specific: measured on the Apple M4 (Metal) of AJ's Mac mini. On another
/// GPU the test may fail without any fault. Regenerate these numbers only
/// with a decision of A0: they are the proof that ICR-006 leaves the results
/// of live particles bit for bit unchanged.
const GOLDEN_D: u64 = 0x7d03_1159_242d_8d73;
const GOLDEN_DP: u64 = 0xb2fc_1bda_9983_cfef;

fn run_hash(mode: Mode) -> u64 {
    let gpu = Gpu::new().expect("GPU");
    let kernels = WhKernels::new(&gpu, DEFAULT_WORKGROUP).expect("self-test");
    let steps = 150;
    let system = system(H_S, steps, true);
    let (init, _) = ensemble(&system, 100);
    let mut st = GpuStepper::new(&gpu, &kernels, &system, &init, H_S, 7, mode, None).unwrap();
    let mut enc = gpu.device().create_command_encoder(&Default::default());
    st.encode_steps(&mut enc, steps).unwrap();
    gpu.queue().submit([enc.finish()]);
    fnv1a(&read_buffer(&gpu, st.state_buffer()))
}

#[test]
#[ignore = "needs a GPU"]
fn rule_off_state_bits_are_the_golden_bits() {
    if Gpu::new().is_none() {
        eprintln!("no GPU: skipped");
        return;
    }
    let (d, dp) = (run_hash(Mode::D), run_hash(Mode::Dp));
    eprintln!("hash D {d:#018x} Dp {dp:#018x}");
    assert_eq!(d, GOLDEN_D, "mode D");
    assert_eq!(dp, GOLDEN_DP, "mode Dp");
}

#[test]
#[ignore = "needs a GPU"]
fn status_buffer_has_8_plus_cap_words_with_word_0_no_failure() {
    let Some(gpu) = Gpu::new() else {
        eprintln!("no GPU: skipped");
        return;
    };
    let kernels = WhKernels::new(&gpu, DEFAULT_WORKGROUP).expect("self-test");
    let system = system(H_S, 10, true);
    let (init, _) = ensemble(&system, 100);
    let st = GpuStepper::new(&gpu, &kernels, &system, &init, H_S, 7, Mode::Dp, None).unwrap();
    let b = read_buffer(&gpu, st.status_buffer());
    assert_eq!(b.len(), 4 * (8 + st.capacity()));
    assert_eq!(b[..4], u32::MAX.to_le_bytes());
    assert!(b[4..].iter().all(|x| *x == 0));
    assert!(
        st.status_buffer()
            .usage()
            .contains(wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC)
    );
}
