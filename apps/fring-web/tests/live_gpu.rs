//! The live set-up on a native GPU: `Setup` builds a `System` (with and without
//! Pandora) and the initial particles, `GpuStepper` runs them, and the particles
//! stay in the ring. Needs a GPU, so `#[ignore]`:
//! `cargo test -p fring-web --release --test live_gpu -- --ignored --nocapture`.

use fring_gpu::gpu::Gpu;
use fring_gpu::stepper::GpuStepper;
use fring_gpu::wh::Mode;
use fring_gpu::wh_gpu::{DEFAULT_WORKGROUP, NO_FAILURE, WhKernels};
use fring_web::live::{BLOCK_STEPS, Band, STEP_S, Setup, count_fits, ds_word};

fn read_words(gpu: &Gpu, buf: &wgpu::Buffer, bytes: u64) -> Vec<f32> {
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
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

#[test]
#[ignore = "needs a GPU"]
fn live_setup_runs_on_the_stepper_with_and_without_pandora() {
    let Some(gpu) = Gpu::new() else {
        eprintln!("no GPU: skipped");
        return;
    };
    let setup = Setup::builtin().unwrap();
    let full = setup.build_ephemeris(5.0 * 86_400.0).unwrap();
    let kernels = WhKernels::new(&gpu, DEFAULT_WORKGROUP).unwrap();
    let n = 2000u32;
    for (pandora, band) in [
        (true, Band::Core),
        (false, setup.preset_band(2)),
        (true, setup.preset_band(1)),
    ] {
        let system = setup.system(&full, pandora);
        let init = setup.initial(band, n).unwrap();
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
        // Two calls of 150 steps (15 hours): the second one reuses the slots.
        for _ in 0..2 {
            let mut enc = gpu.device().create_command_encoder(&Default::default());
            st.encode_steps(&mut enc, 150).unwrap();
            gpu.queue().submit([enc.finish()]);
        }
        assert_eq!(st.time_s(), 300.0 * STEP_S);
        let status = read_words(&gpu, st.status_buffer(), 4);
        assert_eq!(status[0].to_bits(), NO_FAILURE);
        let cap = st.capacity();
        let w = read_words(&gpu, st.state_buffer(), (6 * cap * 8) as u64);
        for i in 0..n as usize {
            let (hx, lx) = (w[ds_word(0, i, cap)], w[ds_word(0, i, cap) + 1]);
            let (hy, ly) = (w[ds_word(1, i, cap)], w[ds_word(1, i, cap) + 1]);
            let r = (f64::from(hx) + f64::from(lx)).hypot(f64::from(hy) + f64::from(ly));
            // Radius changes by at most a few hundred km in 15 hours.
            assert!(
                (138_500.0..143_000.0).contains(&r),
                "pandora {pandora}, i {i}: r = {r}"
            );
        }
        eprintln!(
            "pandora {pandora}: ok, {} moons in the system",
            system.moons.tracks.len()
        );
    }
}

/// The largest count of the page (10^6) fits the device limits, runs steps and
/// stays in the ring.
#[test]
#[ignore = "needs a GPU"]
fn one_million_particles_fit_and_run() {
    let Some(gpu) = Gpu::new() else {
        eprintln!("no GPU: skipped");
        return;
    };
    let l = gpu.limits.clone();
    count_fits(
        1_000_000,
        DEFAULT_WORKGROUP,
        l.max_storage_buffer_binding_size,
        l.max_buffer_size,
        l.max_compute_workgroups_per_dimension,
    )
    .unwrap();
    let setup = Setup::builtin().unwrap();
    let full = setup.build_ephemeris(5.0 * 86_400.0).unwrap();
    let kernels = WhKernels::new(&gpu, DEFAULT_WORKGROUP).unwrap();
    let system = setup.system(&full, true);
    let init = setup.initial(Band::Core, 1_000_000).unwrap();
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
    let mut enc = gpu.device().create_command_encoder(&Default::default());
    st.encode_steps(&mut enc, 20).unwrap();
    gpu.queue().submit([enc.finish()]);
    assert_eq!(st.time_s(), 20.0 * STEP_S);
    let status = read_words(&gpu, st.status_buffer(), 4);
    assert_eq!(status[0].to_bits(), NO_FAILURE);
    let cap = st.capacity();
    assert_eq!(cap, 1_000_000);
    // Spot check: the last particle is still near the core.
    let i = 999_999;
    let bytes = 16 * cap as u64;
    let w = read_words(&gpu, st.state_buffer(), bytes);
    let r = (f64::from(w[ds_word(0, i, cap)]) + f64::from(w[ds_word(0, i, cap) + 1]))
        .hypot(f64::from(w[ds_word(1, i, cap)]) + f64::from(w[ds_word(1, i, cap) + 1]));
    assert!((138_500.0..143_000.0).contains(&r), "r = {r}");
}
