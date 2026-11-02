//! W2-COMPARE on a native GPU: the reference copy of `web.rs` ("Set to now"
//! and the copy at t = 0) and the projection of the reference. `web.rs` is
//! wasm32 only, so these tests repeat its GPU calls exactly: the copy is
//! `copy_buffer_to_buffer(state, 0, reference, 0, ref_bytes(cap))` in the
//! same encoder as the steps. Needs a GPU, so `#[ignore]`:
//! `cargo test -p fring-web --release --test compare_gpu -- --ignored --nocapture`.

use fring_gpu::gpu::Gpu;
use fring_gpu::stepper::GpuStepper;
use fring_gpu::wh::Mode;
use fring_gpu::wh_gpu::{DEFAULT_WORKGROUP, WhKernels};
use fring_web::live::{BLOCK_STEPS, Band, STEP_S, Setup, ref_bytes};

fn read_words(gpu: &Gpu, buf: &wgpu::Buffer, bytes: u64) -> Vec<u32> {
    let d = gpu.device();
    let stage = d.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = d.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(buf, 0, &stage, 0, bytes);
    gpu.queue().submit([enc.finish()]);
    stage.map_async(wgpu::MapMode::Read, .., |r| r.expect("map"));
    d.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let v = stage.get_mapped_range(..).expect("range");
    v.as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

fn reference_buffer(gpu: &Gpu, bytes: u64) -> wgpu::Buffer {
    gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

#[test]
#[ignore = "needs a GPU"]
fn reference_copy_is_the_state_at_the_time_of_the_copy() {
    let Some(gpu) = Gpu::new() else {
        eprintln!("no GPU: skipped");
        return;
    };
    let setup = Setup::builtin().unwrap();
    let full = setup.build_ephemeris(5.0 * 86_400.0).unwrap();
    let kernels = WhKernels::new(&gpu, DEFAULT_WORKGROUP).unwrap();
    let system = setup.system(&full, true);
    let n = 5000u32;
    let init = setup.initial(Band::Core, n).unwrap();
    let mut st = GpuStepper::new(
        &gpu,
        &kernels,
        &system,
        &init,
        STEP_S,
        BLOCK_STEPS,
        Mode::Dp,
        None,
    )
    .unwrap();
    let cap = st.capacity() as u64;
    let bytes = ref_bytes(cap);
    assert_eq!(bytes % 4, 0);
    let reference = reference_buffer(&gpu, bytes);

    // 1. Copy at t = 0 (what make_stepper does), then step 40 times.
    let mut enc = gpu.device().create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(st.state_buffer(), 0, &reference, 0, bytes);
    st.encode_steps(&mut enc, 40).unwrap();
    gpu.queue().submit([enc.finish()]);
    let state0 = read_words(&gpu, st.state_buffer(), bytes);
    let ref0 = read_words(&gpu, &reference, bytes);
    assert_eq!(ref0.len() as u64, bytes / 4);
    // The reference is the t = 0 state: it differs from the stepped state, and
    // it holds non-zero low words (the tail of each double-single value).
    let differing = ref0.iter().zip(&state0).filter(|(a, b)| a != b).count();
    assert!(differing > ref0.len() / 2, "differing words: {differing}");
    assert!(ref0.iter().skip(1).step_by(2).any(|w| *w != 0));

    // 2. "Set to now": the copy goes before the steps of the same encoder, so
    // the reference is the state of the previous frame, not of this one.
    let before = read_words(&gpu, st.state_buffer(), bytes);
    let mut enc = gpu.device().create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(st.state_buffer(), 0, &reference, 0, bytes);
    st.encode_steps(&mut enc, 40).unwrap();
    gpu.queue().submit([enc.finish()]);
    let ref1 = read_words(&gpu, &reference, bytes);
    let after = read_words(&gpu, st.state_buffer(), bytes);
    assert_eq!(ref1, before, "reference = state before this frame's steps");
    assert_ne!(ref1, after);
    assert_eq!(st.time_s(), 80.0 * STEP_S);
}
