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

/// Along-track error (km) of one circular orbit at a0 after `steps` steps of
/// `STEP_S`, no moons, against the angle n(r) t of the exact circular orbit.
fn along_track_error_km(gpu: &Gpu, kernels: &WhKernels, setup: &Setup, drift: Option<f64>) -> f64 {
    use fring_gpu::stepper::StepperOptions;
    use fring_model::physics::{SaturnField, frequencies};
    let a0 = setup.a0_km();
    let init = setup
        .initial(
            Band::Circular {
                min_km: a0 - 1.0e-3,
                max_km: a0 + 1.0e-3,
            },
            1,
        )
        .unwrap();
    let (x0, y0) = (init.particles.x_km[0], init.particles.y_km[0]);
    let (r0, th0) = (x0.hypot(y0), y0.atan2(x0));
    let n = frequencies(&SaturnField::<f64>::from_params(&setup.saturn), r0).n_rad_s;
    let mut system = setup.system(&setup.build_ephemeris(5.0 * 86_400.0).unwrap(), false);
    system.moons.tracks.clear();
    let mut st = GpuStepper::new_with_drift(
        gpu,
        kernels,
        &system,
        &init,
        StepperOptions {
            step_s: STEP_S,
            block_steps: BLOCK_STEPS,
            mode: Mode::Dp,
            chaos: None,
            drift_radius_km: drift,
        },
    )
    .unwrap();
    let steps = 87_660u64; // one Julian year at 360 s
    for _ in 0..(steps / 1000) {
        let mut enc = gpu.device().create_command_encoder(&Default::default());
        st.encode_steps(&mut enc, 1000).unwrap();
        gpu.queue().submit([enc.finish()]);
    }
    let mut enc = gpu.device().create_command_encoder(&Default::default());
    st.encode_steps(&mut enc, steps % 1000).unwrap();
    gpu.queue().submit([enc.finish()]);
    let status = read_words(gpu, st.status_buffer(), 4);
    assert_eq!(status[0].to_bits(), NO_FAILURE);
    let cap = st.capacity();
    let w = read_words(gpu, st.state_buffer(), (6 * cap * 8) as u64);
    let x = f64::from(w[ds_word(0, 0, cap)]) + f64::from(w[ds_word(0, 0, cap) + 1]);
    let y = f64::from(w[ds_word(1, 0, cap)]) + f64::from(w[ds_word(1, 0, cap) + 1]);
    let want = th0 + n * st.time_s();
    let d = (y.atan2(x) - want + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI)
        - std::f64::consts::PI;
    r0 * d
}

/// ICR-011 on the web path: one circular orbit at a0, no moons, one Julian year
/// at the web step (87,660 steps of 360 s). The drift about GM slips along the
/// track by about 12.8 km (ICR-011 T1, 356 s); about mu_eff at a0 it does not.
/// Measured (M4, Metal, release, the SEED start angle): mu_eff +0.0101 km, GM
/// -13.0565 km. Bounds (M27): mu_eff below 0.1 km (10 times the measurement;
/// the ICR-011 T1 band-edge error is 0.07 to 0.11 km, and at a0 it is 0.0001
/// km); GM above 10 km (T1 limit; measured 13.06, so the test has power).
#[test]
#[ignore = "needs a GPU"]
fn web_stepper_with_mu_eff_keeps_the_circular_orbit_at_a0() {
    let Some(gpu) = Gpu::new() else {
        eprintln!("no GPU: skipped");
        return;
    };
    let setup = Setup::builtin().unwrap();
    let kernels = WhKernels::new(&gpu, DEFAULT_WORKGROUP).unwrap();
    let eff = along_track_error_km(&gpu, &kernels, &setup, Some(setup.a0_km()));
    let gm = along_track_error_km(&gpu, &kernels, &setup, None);
    eprintln!("along-track error after 1 year: mu_eff {eff:+.4} km, GM {gm:+.4} km");
    assert!(eff.abs() < 0.1, "mu_eff arm: {eff} km");
    assert!(gm.abs() > 10.0, "GM arm has no power: {gm} km");
}
