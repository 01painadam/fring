//! W2-DROP on a native GPU: a particle dropped on purpose is counted in the
//! status buffer (ICR-006) and hidden by `shaders/project.wgsl`, live and in a
//! reference copy of the status buffer. The state is a real `GpuStepper` state,
//! so its double-single low words are not zero (M24). `web.rs` is wasm32 only,
//! so this test repeats its GPU calls: `set_moon_radii_km` with the radii of
//! `moon_radii_for`, a copy of the whole status buffer for the reference, and
//! the projection with binding 3 = the status buffer. Needs a GPU:
//! `cargo test -p fring-web --release --test drop_gpu -- --ignored --nocapture`.

use fring_gpu::gpu::Gpu;
use fring_gpu::stepper::GpuStepper;
use fring_gpu::wh::Mode;
use fring_gpu::wh_gpu::{DEFAULT_WORKGROUP, WhKernels};
use fring_web::live::{
    BLOCK_STEPS, Band, ProjectionUniform, STEP_S, Setup, builtin_moon_radii_for, ref_bytes,
};
use fring_web::view::ViewTransform;

const SHADER: &str = include_str!("../shaders/project.wgsl");

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

fn entry(binding: u32, b: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: b.as_entire_binding(),
    }
}

fn buffer(gpu: &Gpu, size: u64) -> wgpu::Buffer {
    gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

/// Projects `n` particles of `state` with `status` as the reason words and
/// returns the disc diameters (z of each instance).
fn sizes(
    gpu: &Gpu,
    state: &wgpu::Buffer,
    status: &wgpu::Buffer,
    n: u32,
    cap: u32,
    size_px: f32,
) -> Vec<f32> {
    let (d, q) = (gpu.device(), gpu.queue());
    let mut u = ProjectionUniform::new(&ViewTransform::IDENTITY, [139_000.0, 0.0], n, cap, 0);
    u.size_px = size_px;
    let inst = buffer(gpu, u64::from(n) * 16);
    let uni = d.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 48,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    q.write_buffer(&uni, 0, &u.to_bytes());
    let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: Some("project"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind = d.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            entry(0, state),
            entry(1, &inst),
            entry(2, &uni),
            entry(3, status),
        ],
    });
    let mut enc = d.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(n.div_ceil(64), 1, 1);
    }
    q.submit([enc.finish()]);
    let w = read_words(gpu, &inst, u64::from(n) * 16);
    (0..n as usize)
        .map(|i| f32::from_bits(w[4 * i + 2]))
        .collect()
}

#[test]
#[ignore = "needs a GPU"]
fn a_dropped_particle_is_counted_and_hidden_live_and_in_the_reference() {
    let Some(gpu) = Gpu::new() else {
        eprintln!("no GPU: skipped");
        return;
    };
    let (d, q) = (gpu.device(), gpu.queue());
    let setup = Setup::builtin().unwrap();
    let full = setup.build_ephemeris(5.0 * 86_400.0).unwrap();
    let kernels = WhKernels::new(&gpu, DEFAULT_WORKGROUP).unwrap();
    let system = setup.system(&full, true);
    let n = 2000u32;
    let mut init = setup.initial(Band::Core, n).unwrap();
    let (a, b) = (5usize, 1234usize);
    init.particles.vy_km_s[a] *= 3.0; // above the escape speed: unbound
    init.particles.vy_km_s[b] *= 3.0;
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
    st.set_moon_radii_km(&builtin_moon_radii_for(&system).unwrap())
        .unwrap();
    let cap = st.capacity();
    let status_bytes = ((8 + cap) * 4) as u64;
    let ref_state = buffer(&gpu, ref_bytes(cap as u64));
    let ref_status = buffer(&gpu, status_bytes);

    // The reference at t = 0: no particle is dropped yet.
    let mut enc = d.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(st.state_buffer(), 0, &ref_state, 0, ref_bytes(cap as u64));
    enc.copy_buffer_to_buffer(st.status_buffer(), 0, &ref_status, 0, status_bytes);
    q.submit([enc.finish()]);
    let at_zero = sizes(&gpu, &ref_state, &ref_status, n, cap as u32, 2.5);
    assert!(at_zero.iter().all(|s| *s == 2.5), "t = 0: all visible");

    // Steps (three blocks), then the counts.
    let mut enc = d.create_command_encoder(&Default::default());
    st.encode_steps(&mut enc, 3 * BLOCK_STEPS as u64).unwrap();
    q.submit([enc.finish()]);
    let status = read_words(&gpu, st.status_buffer(), status_bytes);
    assert_eq!(status[0], a as u32, "smallest dropped index");
    assert_eq!(status[1], 2, "two unbound");
    assert!(status[2..8].iter().all(|w| *w == 0), "none inside a moon");
    for i in 0..n as usize {
        let want = u32::from(i == a || i == b);
        assert_eq!(status[8 + i], want, "reason of {i}");
    }

    // M24: the state is a real stepper state, so some low words are not zero.
    let words = read_words(&gpu, st.state_buffer(), st.state_buffer().size());
    let low_nonzero = (0..n as usize)
        .filter(|i| f32::from_bits(words[2 * i + 1]) != 0.0)
        .count();
    assert!(low_nonzero > 0, "no low word is non-zero");

    // Live: the two dropped particles have diameter 0, the others 2.5.
    let live = sizes(
        &gpu,
        st.state_buffer(),
        st.status_buffer(),
        n,
        cap as u32,
        2.5,
    );
    for (i, s) in live.iter().enumerate() {
        let want = if i == a || i == b { 0.0 } else { 2.5 };
        assert_eq!(*s, want, "live size of {i}");
    }

    // Reference taken now: the same particles are hidden in the copy, and the
    // copy of the t = 0 reference (taken before) still shows all.
    let mut enc = d.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(st.state_buffer(), 0, &ref_state, 0, ref_bytes(cap as u64));
    enc.copy_buffer_to_buffer(st.status_buffer(), 0, &ref_status, 0, status_bytes);
    q.submit([enc.finish()]);
    let r = sizes(&gpu, &ref_state, &ref_status, n, cap as u32, 2.5);
    assert_eq!(r, live, "reference copy hides the same particles");
}
