//! The production kernels (`shaders/wh.wgsl`) against their Rust twins
//! (PLAN section 10: WGSL function against Rust twin, relative < 1e-6), and
//! the start-up checks. Needs a GPU, so `#[ignore]`: run with
//! `cargo test -p fring-gpu --release --test gpu_wh -- --ignored --nocapture`.

use fring_gpu::ds::Ds;
use fring_gpu::gpu::Gpu;
use fring_gpu::spike::{Consts, State, state_from_f64, state_to_f64};
use fring_gpu::wf32::model;
use fring_gpu::wh::{self, HostTable};
use fring_gpu::wh_gpu::{DEFAULT_WORKGROUP, NO_FAILURE, WhKernels};
use fring_model::params::{moons_from_toml_str, saturn_from_toml_str};
use fring_model::physics::{
    EpicycleElements, KickTable, SaturnField, build_ephemeris, epicycle_state,
};
use fring_model::{Real, SaturnParams, Vec3};

const M4: u8 = model::M4;
/// A step that is not exact in f32 (ADR 0007).
const H: f64 = 356.2537;

struct Setup {
    sp: SaturnParams,
    gms: Vec<f64>,
    c: Consts<M4>,
    prm: Vec<f32>,
    table: KickTable<f64>,
    packed: Vec<f32>,
    host: HostTable<M4>,
    states: Vec<State<M4>>,
}

fn setup(n_steps: usize, n_particles: usize) -> Setup {
    let sp = saturn_from_toml_str(include_str!("../../../params/saturn.toml")).unwrap();
    let moons = moons_from_toml_str(include_str!("../../../params/moons.toml")).unwrap();
    let eph = build_ephemeris(&sp, &moons, 2.0 * n_steps as f64 * H + 7200.0, 1800.0).unwrap();
    let gms: Vec<f64> = eph.tracks.iter().map(|t| t.gm_km3_s2).collect();
    let c = Consts::<M4>::for_step(&sp, &gms, H);
    let prm = wh::params(&c, &gms);
    let mut table = KickTable::<f64>::for_ephemeris(&eph);
    table.fill(&eph, 0.0, H, 0, n_steps).unwrap();
    let scale = fring_gpu::spike::kick_scale(H);
    let packed = wh::pack_table(&table, scale);
    let host = HostTable::<M4>::from_kick_table(&table, scale);
    let field = SaturnField::<f64>::from_params(&sp);
    let states = (0..n_particles)
        .map(|k| {
            let el = EpicycleElements {
                a_km: 139_000.0 + 3_000.0 * ((k * 7) % 11) as f64 / 10.0,
                e: [0.0, 0.0024, 0.01][k % 3],
                i_rad: [0.0, 1.1e-4, 1e-3][k % 3],
                mean_anomaly_rad: 0.37 * k as f64,
                z_phase_rad: 0.11 * k as f64,
                mean_longitude_rad: 0.0981 * k as f64,
            };
            let (p, v) = epicycle_state(&field, &el);
            state_from_f64::<M4>(p, v)
        })
        .collect();
    Setup {
        sp,
        gms,
        c,
        prm,
        table,
        packed,
        host,
        states,
    }
}

fn gpu_and_kernels() -> Option<(Gpu, WhKernels)> {
    let Some(gpu) = Gpu::new() else {
        eprintln!("no GPU: skipped");
        return None;
    };
    let k = WhKernels::new(&gpu, DEFAULT_WORKGROUP).expect("self-test");
    Some((gpu, k))
}

fn rel_vec(a: Vec3<f64>, b: Vec3<f64>) -> f64 {
    (a - b).norm() / b.norm().max(f64::MIN_POSITIVE)
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(f64::MIN_POSITIVE)
}

/// Runs `name` on the states, compares each output state with `twin`
/// (position and velocity vectors, relative to their norms), prints the
/// worst error and asserts the PLAN bound.
fn check_vectors(
    name: &str,
    s: &Setup,
    k: &WhKernels,
    twin: impl Fn(&Setup, &State<M4>) -> State<M4>,
) {
    let mut g = s.states.clone();
    let (status, _) = k.apply(name, &s.prm, &s.packed, 1, 2, &mut g);
    assert_eq!(status, NO_FAILURE, "{name}: unexpected failure flag");
    let (mut wp, mut wv) = (0.0f64, 0.0f64);
    for (st, gs) in s.states.iter().zip(&g) {
        let t = twin(s, st);
        let (tp, tv) = state_to_f64(&t);
        let (gp, gv) = state_to_f64(gs);
        wp = wp.max(rel_vec(gp, tp));
        wv = wv.max(rel_vec(gv, tv));
    }
    println!("{name:14} worst relative error: first triple {wp:.2e}, second triple {wv:.2e}");
    assert!(wp < 1e-6 && wv < 1e-6, "{name}");
}

fn vecs(st: &State<M4>) -> (Vec3<Ds<M4>>, Vec3<Ds<M4>>) {
    (
        Vec3::new(st[0], st[1], st[2]),
        Vec3::new(st[3], st[4], st[5]),
    )
}

fn join(p: Vec3<Ds<M4>>, v: Vec3<Ds<M4>>) -> State<M4> {
    [p.x, p.y, p.z, v.x, v.y, v.z]
}

#[test]
#[ignore = "needs a GPU"]
fn wgsl_functions_match_their_rust_twins() {
    let Some((_gpu, k)) = gpu_and_kernels() else {
        return;
    };
    let s = setup(1, 192);
    let half = Ds::<M4>::split(0.5 * H);

    check_vectors("t_drift_d", &s, &k, |s, st| {
        let (p, v) = vecs(st);
        let (p, v) = wh::drift(&s.c, p, v, half).unwrap();
        join(p, v)
    });
    check_vectors("t_kepler_c", &s, &k, |s, st| {
        let (p, v) = vecs(st);
        let (p, v) = wh::kepler(&s.c, p, v, half).unwrap();
        join(p, v)
    });
    check_vectors("t_accel_j_c", &s, &k, |s, st| {
        let (p, _) = vecs(st);
        let a = s.c.field_d.acceleration_without_point_mass(p);
        join(p, a)
    });
    check_vectors("t_kick_acc_c", &s, &k, |s, st| {
        let (p, _) = vecs(st);
        let a = wh::kick_acc_c(&s.c, &s.host, 0, p);
        join(p, a)
    });
    check_vectors("t_kick_dir_plus", &s, &k, |s, st| {
        let (p, v) = vecs(st);
        join(p, wh::kick_dir_plus(&s.c, &s.host, 0, p, v))
    });
    check_vectors("t_kick_dir", &s, &k, |s, st| {
        let (p, v) = vecs(st);
        join(p, wh::kick_dir(&s.c, &s.host, 0, p, v))
    });

    // The six DS scalars of the Kepler drift, one by one.
    let mut g = s.states.clone();
    let (status, _) = k.apply("t_ds_scalars", &s.prm, &s.packed, 1, 2, &mut g);
    assert_eq!(status, NO_FAILURE);
    let mut worst = [0.0f64; 6];
    for (st, gs) in s.states.iter().zip(&g) {
        let (p, v) = vecs(st);
        let t = wh::scalars(&s.c, p, v, half).unwrap();
        for c in 0..6 {
            // e cos E0 and e sin E0 vanish on circular orbits: compare them
            // with the scale of 1/a-like quantities (absolute, relative to 1).
            let tv = t[c].to_f64();
            let gv = gs[c].to_f64();
            let e = if c == 3 || c == 4 {
                (gv - tv).abs()
            } else {
                rel(gv, tv)
            };
            worst[c] = worst[c].max(e);
        }
    }
    println!(
        "t_ds_scalars   worst error: r0 {:.1e}, a {:.1e}, n {:.1e}, ecos {:.1e} (abs), esin {:.1e} (abs), m {:.1e}",
        worst[0], worst[1], worst[2], worst[3], worst[4], worst[5]
    );
    assert!(worst.iter().all(|&e| e < 1e-6));

    // newton_f32 on (e cos E0, e sin E0, m) of three orbit types.
    let mut inp: Vec<State<M4>> = Vec::new();
    for k3 in 0..96 {
        let e = [0.0, 0.0024, 0.05][k3 % 3];
        let ph = 0.21 * k3 as f64;
        let m = 0.0418 * (1.0 + 0.003 * k3 as f64);
        let z = Ds::<M4>::ZERO;
        inp.push([
            Ds::split(e * ph.cos()),
            Ds::split(e * ph.sin()),
            Ds::split(m),
            z,
            z,
            z,
        ]);
    }
    let mut g = inp.clone();
    let (status, _) = k.apply("t_newton_f32", &s.prm, &s.packed, 1, 2, &mut g);
    assert_eq!(status, NO_FAILURE);
    let mut w = 0.0f64;
    for (i, o) in inp.iter().zip(&g) {
        let t = wh::newton::<M4>(i[0].hi, i[1].hi, i[2].hi);
        w = w.max(rel(o[3].hi as f64, t as f64));
    }
    println!("t_newton_f32   worst relative error {w:.2e}");
    assert!(w < 1e-6);
    let _ = (&s.sp, &s.gms, &s.table);
}
