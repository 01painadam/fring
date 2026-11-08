//! Task G3: the WGSL kernel `chaos_renorm` against its Rust twin
//! (`fring_gpu::chaos::renorm`). Needs a GPU, so `#[ignore]`:
//! `cargo test -p fring-gpu --release --test gpu_chaos -- --ignored --nocapture`.

use fring_gpu::chaos::{ACC_PLANES, Acc, renorm};
use fring_gpu::gpu::Gpu;
use fring_gpu::spike::state_from_f64;
use fring_gpu::wf32::model;
use fring_gpu::wh;
use fring_gpu::wh_gpu::{DEFAULT_WORKGROUP, WhKernels, grid, pack_states, unpack_states};
use fring_model::{Real, Vec3};

const M4: u8 = model::M4;

fn bytes(w: &[f32]) -> Vec<u8> {
    w.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[test]
#[ignore = "needs a GPU"]
fn chaos_renorm_matches_its_rust_twin() {
    let Some(gpu) = Gpu::new() else {
        eprintln!("no GPU: skipped");
        return;
    };
    let k = WhKernels::new(&gpu, DEFAULT_WORKGROUP).expect("self-test");
    let (d, q) = (gpu.device(), gpu.queue());
    let n = 192usize;
    let cap = n;
    let d0 = 1.0e-3f32;
    let mut main = Vec::new();
    let mut shadow = Vec::new();
    for i in 0..n {
        let a = 139_000.0 + 3000.0 * (i % 13) as f64 / 13.0;
        let ang = 0.1 * i as f64;
        let p = Vec3::new(a * ang.cos(), a * ang.sin(), 0.5 * (i % 5) as f64);
        let vc = (3.79e7f64 / a).sqrt();
        let v = Vec3::new(-vc * ang.sin(), vc * ang.cos(), 1e-4);
        let ang2 = 0.77 * i as f64 + 0.3;
        let dir = Vec3::new(ang2.cos(), ang2.sin() * 0.6, 0.8 * (1.7 * ang2).sin());
        let dir = dir * (1.0 / dir.norm());
        // Separations from 0.1 d0 to 1000 d0; a velocity offset for some.
        let size = 1e-3 * 10f64.powf(-1.0 + 4.0 * (i % 7) as f64 / 6.0);
        let dv = dir * (if i % 2 == 0 { 2e-7 } else { 0.0 });
        let sep = if i == 3 {
            Vec3::new(0.0, 0.0, 0.0)
        } else {
            dir * size
        };
        main.push(state_from_f64::<M4>(p, v));
        shadow.push(state_from_f64::<M4>(p + sep, v + dv));
    }
    // GPU.
    let rw =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let uni = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
    let mk = |size: usize, usage| {
        d.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size.max(16) as u64,
            usage,
            mapped_at_creation: false,
        })
    };
    let st = mk(6 * cap * 8, rw);
    q.write_buffer(&st, 0, &bytes(&pack_states(&main, cap)));
    let sh = mk(6 * cap * 8, rw);
    q.write_buffer(&sh, 0, &bytes(&pack_states(&shadow, cap)));
    let acc = mk(ACC_PLANES * cap * 8, rw);
    let prm = mk(wh::PARAM_WORDS * 4, uni);
    let sp = fring_model::params::saturn_from_toml_str(include_str!("../../../params/saturn.toml"))
        .unwrap();
    let c = fring_gpu::spike::Consts::<M4>::for_step(&sp, &[], 356.2537);
    q.write_buffer(&prm, 0, &bytes(&wh::params(&c, &[])));
    let ctl = mk(32, uni);
    let cw: Vec<u8> = wh::control(1, n, cap, 0)
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    q.write_buffer(&ctl, 0, &cw);
    let status = mk(4 * fring_gpu::wh_gpu::status_words(cap), rw);
    q.write_buffer(&status, 0, &fring_gpu::wh_gpu::status_init_bytes(cap));
    let cprm = mk(16, uni);
    q.write_buffer(&cprm, 0, &bytes(&[d0, 0.0, 0.0, 0.0]));
    let bind = k.chaos_bind_group(&st, &prm, &ctl, &status, &sh, &acc, &cprm);
    let pipe = k.chaos_pipeline();
    let rounds = 3;
    for _ in 0..rounds {
        let mut enc = d.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipe);
            pass.set_bind_group(0, &bind, &[]);
            let (gx, gy) = grid(cap, k.workgroup);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        q.submit([enc.finish()]);
    }
    let stage = mk(
        (6 + ACC_PLANES) * cap * 8,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    );
    let mut enc = d.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(&sh, 0, &stage, 0, (6 * cap * 8) as u64);
    enc.copy_buffer_to_buffer(
        &acc,
        0,
        &stage,
        (6 * cap * 8) as u64,
        (ACC_PLANES * cap * 8) as u64,
    );
    q.submit([enc.finish()]);
    stage.map_async(wgpu::MapMode::Read, .., |r| r.expect("map"));
    d.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let words: Vec<f32> = stage
        .get_mapped_range(..)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    let mut gpu_shadow = shadow.clone();
    unpack_states(&words[..12 * cap], cap, &mut gpu_shadow);
    let aw = &words[12 * cap..];

    // Twin.
    let mut tw = shadow.clone();
    let mut accs = vec![Acc::<M4>::default(); n];
    for _ in 0..rounds {
        for i in 0..n {
            renorm(&main[i], &mut tw[i], &mut accs[i], d0);
        }
    }
    let (mut wpos, mut wvel, mut wl, mut ws, mut wy) = (0f64, 0f64, 0f64, 0f64, 0f64);
    let ds = |plane: usize, i: usize| {
        aw[2 * (plane * cap + i)] as f64 + aw[2 * (plane * cap + i) + 1] as f64
    };
    for i in 0..n {
        for c in 0..3 {
            wpos = wpos.max((gpu_shadow[i][c].to_f64() - tw[i][c].to_f64()).abs());
            wvel = wvel.max((gpu_shadow[i][c + 3].to_f64() - tw[i][c + 3].to_f64()).abs());
        }
        wl = wl.max((ds(0, i) - (accs[i].l.hi as f64 + accs[i].l.lo as f64)).abs());
        let (s_t, y_t) = (
            accs[i].s.hi as f64 + accs[i].s.lo as f64,
            accs[i].y.hi as f64 + accs[i].y.lo as f64,
        );
        ws = ws.max((ds(1, i) - s_t).abs() / s_t.abs().max(1.0));
        wy = wy.max((ds(2, i) - y_t).abs() / y_t.abs().max(1.0));
        assert_eq!(aw[2 * (3 * cap + i)], accs[i].j);
        assert_eq!(aw[2 * (3 * cap + i) + 1], accs[i].flags, "flag of {i}");
    }
    println!(
        "chaos_renorm vs twin: shadow pos {wpos:.2e} km, vel {wvel:.2e} km/s, L {wl:.2e}, S {ws:.2e}, Y {wy:.2e}"
    );
    assert!(wpos < 1e-9 && wvel < 1e-12, "shadow state");
    assert!(wl < 5e-6 && ws < 5e-6 && wy < 5e-6, "sums");
}
