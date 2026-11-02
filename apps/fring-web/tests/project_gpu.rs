//! M16 check for `shaders/project.wgsl`: Metal compiles WGSL with fast math, so
//! the GPU result must be compared with the Rust twin (`project_twin`) and with
//! the exact f64 value. Needs a GPU, so `#[ignore]`:
//! `cargo test -p fring-web --release --test project_gpu -- --ignored --nocapture`.
//!
//! Tolerance for a particle at distance d from the eye: 1e-6 km + 6e-7 d. The
//! twin error against f64 is at most about 3e-7 d + 1e-9 km (see `live.rs`),
//! and the GPU may differ from the twin by one more rounding of the same size
//! (fused multiply-add), so 6e-7 d. For d = 20 km that is 1.3e-5 km, which is
//! 1.3e-3 px at 100 px/km (a window 10 km wide). For d = 4000 km (a window of
//! 4000 px at 1 px/km) it is 2.4e-3 km, 2.4e-3 px. A shader that lost the low
//! parts (a re-associated sum) would be wrong by up to 0.008 km, which is 0.8 px
//! at 100 px/km and 600 times the tolerance at d = 20 km.

use fring_gpu::gpu::Gpu;
use fring_web::live::{ProjectionUniform, ds_split, ds_word, project_twin};
use fring_web::view::ViewTransform;

const SHADER: &str = include_str!("../shaders/project.wgsl");

/// Tolerance for a particle `dist_km` from the eye: 1e-6 km + 6e-7 |d|.
fn tolerance_km(dist_km: f64) -> f64 {
    1e-6 + 6e-7 * dist_km
}

/// Positions near 1.4e5 km with non-zero low parts; the eye is a few km away
/// (case 0) or 2000 km away (case 1).
fn positions(n: usize) -> Vec<[f64; 2]> {
    let mut x = 0x2545_f491_4f6c_dd1du64;
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut pts: Vec<[f64; 2]> = (0..n)
        .map(|_| {
            let (r, a) = (139_000.0 + 3_000.0 * next(), std::f64::consts::TAU * next());
            [r * a.cos() + next() * 0.01, r * a.sin() + next() * 0.01]
        })
        .collect();
    // A cluster within 10 km of particle 0, so that case 0 has many particles.
    for i in 1..=2_000 {
        pts[i] = [
            pts[0][0] + 20.0 * next() - 10.0,
            pts[0][1] + 20.0 * next() - 10.0,
        ];
    }
    pts
}

fn run(gpu: &Gpu, pts: &[[f64; 2]], cap: usize, u: &ProjectionUniform) -> Vec<[f32; 2]> {
    let (d, q) = (gpu.device(), gpu.queue());
    let n = pts.len();
    let mut words = vec![0f32; 2 * 6 * cap];
    for (i, p) in pts.iter().enumerate() {
        for (plane, v) in [p[0], p[1]].into_iter().enumerate() {
            let w = ds_word(plane, i, cap);
            let [hi, lo] = ds_split(v);
            words[w] = hi;
            words[w + 1] = lo;
        }
    }
    let bytes = |f: &[f32]| f.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let mk = |size: u64, usage| {
        d.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage,
            mapped_at_creation: false,
        })
    };
    let state = mk(
        (words.len() * 4) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    q.write_buffer(&state, 0, &bytes(&words));
    let inst_size = ((u.first as usize + n) * 16) as u64;
    let inst = mk(
        inst_size,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let uni = mk(
        48,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    );
    q.write_buffer(&uni, 0, &u.to_bytes());
    let stage = mk(
        inst_size,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    );
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
            wgpu::BindGroupEntry {
                binding: 0,
                resource: state.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: inst.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: uni.as_entire_binding(),
            },
        ],
    });
    let mut enc = d.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(n.div_ceil(64) as u32, 1, 1);
    }
    enc.copy_buffer_to_buffer(&inst, 0, &stage, 0, inst_size);
    q.submit([enc.finish()]);
    stage.map_async(wgpu::MapMode::Read, .., |r| r.expect("map"));
    d.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let view = stage.get_mapped_range(..).expect("range");
    let f: Vec<f32> = view
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    (0..n)
        .map(|i| {
            let k = 4 * (u.first as usize + i);
            assert_eq!(f[k + 2], u.size_px);
            [f[k], f[k + 1]]
        })
        .collect()
}

#[test]
#[ignore = "needs a GPU"]
fn gpu_projection_matches_the_twin_and_the_exact_value() {
    let Some(gpu) = Gpu::new() else {
        eprintln!("no GPU: skipped");
        return;
    };
    let n: usize = 20_000;
    let cap = n.next_multiple_of(64);
    let pts = positions(n);
    let views = [
        ViewTransform::IDENTITY,
        ViewTransform::new([0.0, 0.0], -0.7),
        ViewTransform::new([139_380.0, 1_200.0], 1.4),
    ];
    for (vi, view) in views.iter().enumerate() {
        for (ci, eye_offset) in [[3.0, -2.0], [2000.0, 1500.0]].into_iter().enumerate() {
            let near = view.apply(pts[0]);
            let eye = [near[0] + eye_offset[0], near[1] + eye_offset[1]];
            let u = ProjectionUniform::new(view, eye, n as u32, cap as u32, 3);
            let got = run(&gpu, &pts, cap, &u);
            let mut checked = 0;
            let (mut worst_exact, mut worst_twin) = (0f64, 0f64);
            for (p, g) in pts.iter().zip(&got) {
                let v = view.apply(*p);
                let exact = [v[0] - eye[0], v[1] - eye[1]];
                let dist = exact[0].hypot(exact[1]);
                // Case 0: particles within 20 km of the eye. Case 1: within 4000 km.
                if dist > if ci == 0 { 20.0 } else { 4000.0 } {
                    continue;
                }
                checked += 1;
                let tol = tolerance_km(dist);
                let twin = project_twin(ds_split(p[0]), ds_split(p[1]), &u);
                let e_exact = (f64::from(g[0]) - exact[0]).hypot(f64::from(g[1]) - exact[1]);
                let e_twin = f64::from(g[0] - twin[0]).hypot(f64::from(g[1] - twin[1]));
                assert!(
                    e_exact < tol,
                    "view {vi} case {ci}: {e_exact} km at {dist} km"
                );
                assert!(
                    e_twin < tol,
                    "view {vi} case {ci}: {e_twin} km at {dist} km"
                );
                worst_exact = worst_exact.max(e_exact / tol);
                worst_twin = worst_twin.max(e_twin / tol);
            }
            eprintln!(
                "view {vi} case {ci}: {checked} particles; worst error as a fraction of the tolerance: vs exact {worst_exact:.3}, vs twin {worst_twin:.3}"
            );
            assert!(checked > 0, "view {vi} case {ci}: no particle near the eye");
        }
    }
}
