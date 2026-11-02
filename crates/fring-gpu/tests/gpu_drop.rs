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

/// The 6 planes of particle `i` (hi and lo words, 8 bytes each) from a state
/// buffer read back.
fn particle_bytes(state: &[u8], cap: usize, i: usize) -> Vec<u8> {
    (0..6)
        .flat_map(|c| state[8 * (c * cap + i)..8 * (c * cap + i) + 8].to_vec())
        .collect()
}

fn word(b: &[u8], w: usize) -> u32 {
    u32::from_le_bytes(b[4 * w..4 * w + 4].try_into().unwrap())
}

/// Runs `steps` steps in `calls` and returns (state bytes, status bytes).
fn run(
    gpu: &Gpu,
    kernels: &WhKernels,
    system: &fring_model::System,
    init: &fring_model::InitialState,
    mode: Mode,
    block: usize,
    calls: &[u64],
) -> (Vec<u8>, Vec<u8>, usize) {
    let mut st = GpuStepper::new(gpu, kernels, system, init, H_S, block, mode, None).unwrap();
    for n in calls {
        let mut enc = gpu.device().create_command_encoder(&Default::default());
        st.encode_steps(&mut enc, *n).unwrap();
        gpu.queue().submit([enc.finish()]);
    }
    (
        read_buffer(gpu, st.state_buffer()),
        read_buffer(gpu, st.status_buffer()),
        st.capacity(),
    )
}

#[test]
#[ignore = "needs a GPU"]
fn an_unbound_particle_gets_reason_1_and_is_frozen_and_others_are_unchanged() {
    let Some(gpu) = Gpu::new() else {
        eprintln!("no GPU: skipped");
        return;
    };
    let kernels = WhKernels::new(&gpu, DEFAULT_WORKGROUP).expect("self-test");
    let system = system(H_S, 40, true);
    let (init, _) = ensemble(&system, 100);
    let mut bad_init = init.clone();
    let (a, b) = (77usize, 12usize);
    bad_init.particles.vy_km_s[a] *= 3.0; // above the escape speed
    bad_init.particles.vy_km_s[b] *= 3.0;
    for mode in [Mode::D, Mode::Dp] {
        // Three blocks of 7 steps and the end of a fourth.
        let calls = [20u64, 1];
        let (s_ok, st_ok, cap) = run(&gpu, &kernels, &system, &init, mode, 7, &calls);
        let (s_bad, st_bad, _) = run(&gpu, &kernels, &system, &bad_init, mode, 7, &calls);
        assert_eq!(
            word(&st_ok, 0),
            u32::MAX,
            "{mode:?}: no drop in the clean run"
        );
        assert_eq!(
            word(&st_bad, 0),
            b as u32,
            "{mode:?}: word 0 is the smallest index"
        );
        assert_eq!(word(&st_bad, 1), 2, "{mode:?}: two unbound");
        assert!((2..8).all(|w| word(&st_bad, w) == 0));
        for i in 0..100 {
            let r = word(&st_bad, 8 + i);
            assert_eq!(r, u32::from(i == a || i == b), "{mode:?}: reason {i}");
            if i == a || i == b {
                // Frozen at the state it was loaded with, in every block.
                let (s0, _, _) = run(&gpu, &kernels, &system, &bad_init, mode, 7, &[]);
                assert_eq!(
                    particle_bytes(&s_bad, cap, i),
                    particle_bytes(&s0, cap, i),
                    "{mode:?}: frozen state {i}"
                );
            } else {
                assert_eq!(
                    particle_bytes(&s_bad, cap, i),
                    particle_bytes(&s_ok, cap, i),
                    "{mode:?}: live particle {i} changed"
                );
            }
        }
    }
}
