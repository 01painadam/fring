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
use fring_web::live::{
    BLOCK_STEPS, Band, ProjectionUniform, STEP_S, Setup, instance_bytes, project_twin,
    ref_base_slot, ref_bytes,
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

fn entry(binding: u32, buf: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buf.as_entire_binding(),
    }
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

/// The reference copy, projected with its own uniform at its own time, into
/// the second half of the instance buffer, agrees with the twin and with the
/// exact f64 value. The state has non-zero low words (M24).
#[test]
#[ignore = "needs a GPU"]
fn reference_projection_matches_the_twin_in_the_second_half_of_the_buffer() {
    let Some(gpu) = Gpu::new() else {
        eprintln!("no GPU: skipped");
        return;
    };
    let (d, q) = (gpu.device(), gpu.queue());
    let setup = Setup::builtin().unwrap();
    let full = setup.build_ephemeris(5.0 * 86_400.0).unwrap();
    let kernels = WhKernels::new(&gpu, DEFAULT_WORKGROUP).unwrap();
    let system = setup.system(&full, true);
    let n = 20_000u32;
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
    let cap = st.capacity();
    let bytes = ref_bytes(cap as u64);
    let reference = reference_buffer(&gpu, bytes);
    let mut enc = d.create_command_encoder(&Default::default());
    st.encode_steps(&mut enc, 100).unwrap();
    enc.copy_buffer_to_buffer(st.state_buffer(), 0, &reference, 0, bytes);
    q.submit([enc.finish()]);
    let t_ref_s = st.time_s();
    let words = read_words(&gpu, &reference, bytes);
    let f = |w: u32| f64::from(f32::from_bits(w));
    let pos = |i: usize| {
        let (kx, ky) = (2 * i, 2 * (cap + i));
        [
            f(words[kx]) + f(words[kx + 1]),
            f(words[ky]) + f(words[ky + 1]),
        ]
    };

    // The view at the reference time: rotating frame, -n0 t.
    let view = ViewTransform::new([0.0, 0.0], -setup.n0_rad_s() * t_ref_s);
    let near = view.apply(pos(0));
    let eye = [near[0] + 3.0, near[1] - 2.0];
    let base = ref_base_slot(n);
    let head = 3u32;
    let u = ProjectionUniform::new(&view, eye, n, cap as u32, base + head);
    let inst_bytes = instance_bytes(n);
    let mk = |size: u64, usage| {
        d.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage,
            mapped_at_creation: false,
        })
    };
    let inst = mk(
        inst_bytes,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let uni = mk(
        48,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    );
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
        entries: &[entry(0, &reference), entry(1, &inst), entry(2, &uni)],
    });
    let mut enc = d.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(n.div_ceil(64), 1, 1);
    }
    q.submit([enc.finish()]);
    let out = read_words(&gpu, &inst, inst_bytes);

    // The live half (before the reference base) was not touched.
    assert!(out[..4 * base as usize].iter().all(|w| *w == 0));
    let mut checked = 0;
    let mut worst = 0f64;
    for i in 0..n as usize {
        let k = 4 * (base + head) as usize + 4 * i;
        let got = [f(out[k]), f(out[k + 1])];
        assert_eq!(f(out[k + 2]), f64::from(u.size_px));
        let v = view.apply(pos(i));
        let exact = [v[0] - eye[0], v[1] - eye[1]];
        let dist = exact[0].hypot(exact[1]);
        if dist > 500.0 {
            continue;
        }
        checked += 1;
        let tol = 1e-6 + 6e-7 * dist;
        let xs = |c: usize| {
            let (kx, ky) = (2 * i, 2 * (cap + i));
            [
                f32::from_bits(words[if c == 0 { kx } else { ky }]),
                f32::from_bits(words[if c == 0 { kx + 1 } else { ky + 1 }]),
            ]
        };
        let twin = project_twin(xs(0), xs(1), &u);
        let e_exact = (got[0] - exact[0]).hypot(got[1] - exact[1]);
        let e_twin = (got[0] - f64::from(twin[0])).hypot(got[1] - f64::from(twin[1]));
        assert!(e_exact < tol, "i {i}: {e_exact} km at {dist} km");
        assert!(e_twin < tol, "i {i}: {e_twin} km at {dist} km");
        worst = worst.max(e_exact / tol);
    }
    eprintln!(
        "{checked} particles within 500 km of the eye; worst error {worst:.3} of the tolerance"
    );
    assert!(checked > 10, "only {checked} particles near the eye");
}
