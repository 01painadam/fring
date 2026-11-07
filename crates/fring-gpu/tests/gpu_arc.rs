//! Task D1: the arc wrap on the GPU. Needs a GPU, so `#[ignore]`:
//! `cargo test -p fring-gpu --release --test gpu_arc -- --ignored --nocapture`.

mod common;

use common::*;
use fring_gpu::backend::{GpuBackend, Options};
use fring_gpu::gpu::Gpu;
use fring_gpu::wh::Mode;
use fring_model::Backend;

/// FNV-1a of the f64 bits of the particle positions and velocities after a run
/// with the full ring (arc off). Taken on the commit before D1 changed any
/// shader (`db282d5`): the arc code must not change a full-ring run by one bit.
const GOLDEN_FULL_RING: u64 = 0xa999_c5ad_4c72_2112;

fn full_ring_hash(mode: Mode) -> u64 {
    let steps = 150;
    let system = system(H_S, steps, true);
    let config = config(H_S, steps, true);
    let (init, _) = ensemble(&system, 300);
    let mut gpu = GpuBackend::with_options(
        &config,
        &system,
        &init,
        Options {
            mode,
            block_steps: Some(64),
            ..Default::default()
        },
    )
    .unwrap();
    gpu.step(150).unwrap();
    let p = gpu.particles().unwrap().particles.clone();
    let mut bytes = Vec::new();
    for i in 0..p.len() {
        let (a, b) = (p.position_km(i), p.velocity_km_s(i));
        for x in [a.x, a.y, a.z, b.x, b.y, b.z] {
            bytes.extend_from_slice(&x.to_bits().to_le_bytes());
        }
    }
    fnv1a(&bytes)
}

#[test]
#[ignore = "needs a GPU"]
fn full_ring_run_is_bit_identical_to_before_d1() {
    if Gpu::new().is_none() {
        eprintln!("no GPU: skipped");
        return;
    }
    let h = full_ring_hash(Mode::Dp);
    println!("full ring hash {h:#018x}");
    assert_eq!(h, GOLDEN_FULL_RING);
}

// ---------------------------------------------------------------------------

use fring_gpu::arc::{ArcConsts, extra_rows, set_kick_angles, wrap_ds};
use fring_gpu::spike::{Consts, State, state_from_f64, state_to_f64};
use fring_gpu::wf32::model;
use fring_gpu::wh;
use fring_gpu::wh_gpu::{DEFAULT_WORKGROUP, WhKernels};
use fring_model::Vec3;
use fring_model::config::{ChaosConfig, ChaosMethod, DiagnosticsConfig, Domain};
use fring_model::physics::{ArcGeometry, SaturnField, reference_mean_motion};

const M4: u8 = model::M4;

/// The reference radius of the example config (the arc turns at n(a0)).
const A0_CONFIG_KM: f64 = 140_220.0;

fn n0_of(system: &fring_model::System) -> f64 {
    let field = SaturnField::<f64>::from_params(&system.saturn);
    reference_mean_motion(&field, A0_CONFIG_KM)
}

/// `n` particles with semi-major axes within +-`spread_km` of the reference
/// radius (they drift against the arc at 1.5 n da/a), the three orbit types of
/// G1 and longitudes all round the ring.
fn spread_ensemble(
    system: &fring_model::System,
    n: usize,
    spread_km: f64,
) -> fring_model::InitialState {
    let field = SaturnField::<f64>::from_params(&system.saturn);
    let mut p = fring_model::Particles::with_capacity(n);
    let mut rng = Rng(0x00DD_BA11_C0FF_EE01);
    for i in 0..n {
        let (_, e, inc) = ORBIT_TYPES[i % 3];
        let el = fring_model::physics::EpicycleElements {
            a_km: A0_CONFIG_KM + spread_km * 2.0 * (rng.next() - 0.5),
            e,
            i_rad: inc,
            mean_anomaly_rad: std::f64::consts::TAU * rng.next(),
            z_phase_rad: std::f64::consts::TAU * rng.next(),
            mean_longitude_rad: std::f64::consts::TAU * rng.next(),
        };
        let (pos, vel) = fring_model::physics::epicycle_state(&field, &el);
        p.push(i as u32, pos, vel);
    }
    fring_model::InitialState {
        time_s: 0.0,
        particles: p,
    }
}

fn dang(a: f64, b: f64) -> f64 {
    let tau = std::f64::consts::TAU;
    let d = (a - b).rem_euclid(tau);
    if d > std::f64::consts::PI { d - tau } else { d }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// The initial state with every particle folded into the arc.
fn fold_init(g: &ArcGeometry, init: &fring_model::InitialState) -> fring_model::InitialState {
    let mut p = fring_model::Particles::with_capacity(init.particles.len());
    for i in 0..init.particles.len() {
        let w = g
            .fold(
                init.particles.position_km(i),
                init.particles.velocity_km_s(i),
                init.time_s,
            )
            .unwrap();
        p.push(init.particles.id[i], w.pos_km, w.vel_km_s);
    }
    fring_model::InitialState {
        time_s: init.time_s,
        particles: p,
    }
}

fn lz(p: Vec3<f64>, v: Vec3<f64>) -> f64 {
    p.x * v.y - p.y * v.x
}

/// `t_arc_wrap` against its Rust twin and against the f64 wrap: the DS error of
/// one wrap, and E and Lz before and after.
#[test]
#[ignore = "needs a GPU"]
fn wrap_kernel_matches_its_twin_and_the_f64_wrap() {
    let Some(gpu) = Gpu::new() else {
        eprintln!("no GPU: skipped");
        return;
    };
    let k = WhKernels::new(&gpu, DEFAULT_WORKGROUP).expect("self-test");
    let system = system(H_S, 10, false);
    let field = SaturnField::<f64>::from_params(&system.saturn);
    let n0 = n0_of(&system);
    let (start_s, w_deg) = (9.0e8_f64, 7.0_f64);
    let geom = ArcGeometry::new(w_deg.to_radians(), (-35f64).to_radians(), n0).unwrap();
    let w = geom.width_rad();
    let t_kick = start_s + 0.5 * H_S;
    let psi = geom.centre_at_rad(t_kick);

    // The table of a block of one step: kick row 0, extra rows 1 and 2.
    let mut table = vec![0f32; 3 * wh::ROW];
    set_kick_angles(&mut table[..wh::ROW], &geom, start_s, H_S, 0, 1);
    table[wh::ROW..].copy_from_slice(&extra_rows(&geom, start_s, H_S, 0, 1));
    let psi32 = (
        table[fring_gpu::arc::PSI_COS],
        table[fring_gpu::arc::PSI_SIN],
    );
    let consts = ArcConsts::<M4>::new(&geom);

    let c = Consts::<M4>::for_step(&system.saturn, &[], H_S);
    let prm = wh::params(&c, &[]);
    let mut rng = Rng(0x1234_5678_9ABC_DEF1);
    let mut states: Vec<State<M4>> = Vec::new();
    let mut kinds = Vec::new(); // expected |k| by construction
    for i in 0..4096usize {
        // Offsets in units of the width from the centre; 3.2 widths cannot be wrapped.
        let off = if i < 16 {
            [3.2, -3.2][i % 2] * w
        } else {
            (rng.next() - 0.5) * 2.0 * 2.4 * w
        };
        let r = 139_000.0 + 3_000.0 * rng.next();
        let lon = psi + off;
        let vc = (37_931_206.0_f64 / r).sqrt();
        let vr = 0.3 * (rng.next() - 0.5);
        let vt = vc * (1.0 + 1e-3 * (rng.next() - 0.5));
        let (cl, sl) = (lon.cos(), lon.sin());
        let p = Vec3::new(r * cl, r * sl, 10.0 * (rng.next() - 0.5));
        let v = Vec3::new(
            vr * cl - vt * sl,
            vr * sl + vt * cl,
            0.01 * (rng.next() - 0.5),
        );
        states.push(state_from_f64::<M4>(p, v));
        kinds.push(off);
    }
    let input = states.clone();
    let (_, _) = k.apply("t_arc_wrap", &prm, &table, 1, 0, &mut states);

    let (mut w_twin, mut w_p, mut w_v, mut w_e, mut w_l) = (0f64, 0f64, 0f64, 0f64, 0f64);
    let (mut n_k, mut n_fail, mut n_disagree) = ([0usize; 5], 0usize, 0usize);
    for (i, (s_in, s_out)) in input.iter().zip(&states).enumerate() {
        let tw = wrap_ds::<M4>(
            [s_in[0], s_in[1], s_in[2]],
            [s_in[3], s_in[4], s_in[5]],
            psi32,
            &consts,
        );
        let (p_in, v_in) = state_to_f64(s_in);
        let reference = geom.wrap(p_in, v_in, t_kick);
        let failed = s_out[0].hi.is_nan();
        assert_eq!(
            failed, !tw.ok,
            "twin and GPU disagree on failure, particle {i}"
        );
        if failed {
            n_fail += 1;
            assert!(
                reference.is_none(),
                "f64 wrap should fail too, particle {i}"
            );
            continue;
        }
        let (p_out, v_out) = state_to_f64(s_out);
        // GPU against the twin.
        let tws: State<M4> = [tw.p[0], tw.p[1], tw.p[2], tw.v[0], tw.v[1], tw.v[2]];
        let (p_tw, v_tw) = state_to_f64(&tws);
        w_twin = w_twin
            .max((p_out - p_tw).norm() / p_tw.norm())
            .max((v_out - v_tw).norm() / v_tw.norm());
        // GPU against f64 (the f32 centre angle can flip a particle that sits
        // within 1e-6 rad of an edge; none are expected).
        let rf = reference.expect("f64 wrap");
        let kf = rf.k;
        if (tw.dirs[0] + tw.dirs[1]) != kf {
            n_disagree += 1;
            continue;
        }
        n_k[kf.unsigned_abs() as usize] += 1;
        w_p = w_p.max((p_out - rf.pos_km).norm() / rf.pos_km.norm());
        w_v = w_v.max((v_out - rf.vel_km_s).norm() / rf.vel_km_s.norm());
        let (e0, e1) = (
            field.specific_energy(p_in, v_in),
            field.specific_energy(p_out, v_out),
        );
        w_e = w_e.max(((e1 - e0) / e0).abs());
        w_l = w_l.max(((lz(p_out, v_out) - lz(p_in, v_in)) / lz(p_in, v_in)).abs());
    }
    println!(
        "one wrap (7 deg arc), {} particles: k=0 {}, |k|=1 {}, |k|=2 {}, failures {n_fail}, \
         decision disagreements {n_disagree}",
        input.len(),
        n_k[0],
        n_k[1],
        n_k[2]
    );
    println!(
        "GPU vs twin: {w_twin:.2e} (relative). GPU vs f64 wrap: position {w_p:.2e}, \
         velocity {w_v:.2e} (relative). One wrap on the GPU: |dE/E| {w_e:.2e}, |dLz/Lz| {w_l:.2e}"
    );
    assert_eq!(n_fail, 16, "only the 16 particles at 3.2 widths fail");
    assert!(n_k[2] > 100 && n_k[1] > 100 && n_k[0] > 100);
    assert_eq!(n_disagree, 0);
    assert!(w_twin < 1e-6, "GPU against twin");
    assert!(w_p < 1e-12 && w_v < 1e-12, "DS error of the wrap");
    assert!(
        w_e < 1e-12 && w_l < 1e-12,
        "energy and angular momentum of one wrap"
    );
    let _ = kinds;
}

fn arc_config(
    moons: bool,
    steps: u64,
    width_deg: f64,
    centre_deg: f64,
    chaos: Option<ChaosConfig>,
) -> fring_model::RunConfig {
    let mut cfg = config(H_S, steps, moons);
    cfg.domain = Domain::Arc {
        width_deg,
        centre_longitude_deg: centre_deg,
    };
    cfg.diagnostics = chaos.map(|c| DiagnosticsConfig { chaos: Some(c) });
    cfg.validate().unwrap();
    cfg
}

fn run(
    cfg: &fring_model::RunConfig,
    system: &fring_model::System,
    init: &fring_model::InitialState,
    steps: &[u64],
    mode: Mode,
    block: usize,
) -> GpuBackend {
    let mut b = GpuBackend::with_options(
        cfg,
        system,
        init,
        Options {
            mode,
            block_steps: Some(block),
            ..Default::default()
        },
    )
    .unwrap();
    for &n in steps {
        b.step(n).unwrap();
    }
    b
}

/// With no moons the wrap is an exact symmetry: every particle of the arc run
/// is a rotation, by a whole number of widths, of the same particle of the
/// full-ring run.
#[test]
#[ignore = "needs a GPU"]
fn arc_run_is_the_full_ring_run_rotated_by_whole_widths() {
    let Some(_) = Gpu::new() else {
        eprintln!("no GPU: skipped");
        return;
    };
    let steps = 1500; // about 10 orbits
    let system = system(H_S, steps, false);
    let init = spread_ensemble(&system, 600, 400.0);
    let field = SaturnField::<f64>::from_params(&system.saturn);
    let n0 = n0_of(&system);
    for (w_deg, c_deg, mode) in [
        (10.0_f64, 20.0_f64, Mode::Dp),
        (5.0, -170.0, Mode::D),
        (5.0, -170.0, Mode::Dp),
        (30.0, 175.0, Mode::Dp),
    ] {
        let geom = ArcGeometry::new(w_deg.to_radians(), c_deg.to_radians(), n0).unwrap();
        let init = fold_init(&geom, &init);
        let cfg_arc = arc_config(false, steps, w_deg, c_deg, None);
        let cfg_ring = config(H_S, steps, false);
        let calls = [1u64, 700, steps - 701]; // blocks of 64 do not divide the calls
        let mut ring = run(&cfg_ring, &system, &init, &calls, mode, 64);
        let mut arc = run(&cfg_arc, &system, &init, &calls, mode, 64);
        // The noise floor: the same full-ring run from the initial state
        // turned by 0.7 rad, turned back (the f32 rounding of the kick differs,
        // the physics does not).
        let mut turned = fring_model::Particles::with_capacity(init.particles.len());
        let (c7, s7) = (0.7_f64.cos(), 0.7_f64.sin());
        let rot =
            |a: Vec3<f64>, c: f64, s: f64| Vec3::new(c * a.x - s * a.y, s * a.x + c * a.y, a.z);
        for i in 0..init.particles.len() {
            turned.push(
                init.particles.id[i],
                rot(init.particles.position_km(i), c7, s7),
                rot(init.particles.velocity_km_s(i), c7, s7),
            );
        }
        let init_turned = fring_model::InitialState {
            time_s: 0.0,
            particles: turned,
        };
        let mut ring_turned = run(&cfg_ring, &system, &init_turned, &calls, mode, 64);
        let t = arc.time_s();
        let w = geom.width_rad();
        let psi = geom.centre_at_rad(t);
        let pr = ring.particles().unwrap().particles.clone();
        let pa = arc.particles().unwrap().particles.clone();
        let pt = ring_turned.particles().unwrap().particles.clone();
        let (mut floor_p, mut floor_v) = (0f64, 0f64);
        for i in 0..pr.len() {
            let back = |a: Vec3<f64>| rot(a, c7, -s7);
            floor_p = floor_p.max((pr.position_km(i) - back(pt.position_km(i))).norm());
            floor_v = floor_v.max((pr.velocity_km_s(i) - back(pt.velocity_km_s(i))).norm());
        }
        let (mut max_off, mut max_dp, mut max_dv, mut max_de, mut max_dl) =
            (0f64, 0f64, 0f64, 0f64, 0f64);
        let mut by_m = [0f64; 4];
        let (mut ring_de, mut ring_dl) = (0f64, 0f64);
        let (mut moved, mut max_m) = (0usize, 0i64);
        for i in 0..pr.len() {
            let (p0, v0) = (pr.position_km(i), pr.velocity_km_s(i));
            let (p1, v1) = (pa.position_km(i), pa.velocity_km_s(i));
            max_off = max_off.max(dang(p1.y.atan2(p1.x), psi).abs() - 0.5 * w);
            let m = (dang(p1.y.atan2(p1.x), p0.y.atan2(p0.x)) / w).round();
            if m != 0.0 {
                moved += 1;
            }
            max_m = max_m.max(m.abs() as i64);
            let (c, s) = ((m * w).cos(), (m * w).sin());
            let rp = Vec3::new(c * p0.x - s * p0.y, s * p0.x + c * p0.y, p0.z);
            let rv = Vec3::new(c * v0.x - s * v0.y, s * v0.x + c * v0.y, v0.z);
            max_dp = max_dp.max((p1 - rp).norm());
            by_m[(m.abs() as usize).min(3)] = by_m[(m.abs() as usize).min(3)].max((p1 - rp).norm());
            max_dv = max_dv.max((v1 - rv).norm());
            // Drift of E and Lz from the initial state: the arc run against the ring run.
            let (pi, vi) = (
                init.particles.position_km(i),
                init.particles.velocity_km_s(i),
            );
            let ei = field.specific_energy(pi, vi);
            let li = lz(pi, vi);
            max_de = max_de.max(((field.specific_energy(p1, v1) - ei) / ei).abs());
            max_dl = max_dl.max(((lz(p1, v1) - li) / li).abs());
            ring_de = ring_de.max(((field.specific_energy(p0, v0) - ei) / ei).abs());
            ring_dl = ring_dl.max(((lz(p0, v0) - li) / li).abs());
        }
        println!(
            "{w_deg} deg arc, mode {}, {steps} steps: {moved}/{} particles rotated (max |m| {max_m}); \
             outside the arc by at most {:.1e} rad; against the rotated full-ring run: \
             position {max_dp:.2e} km, velocity {max_dv:.2e} km/s (rounding floor, full ring from a state turned by 0.7 rad, turned back: {floor_p:.2e} km, {floor_v:.2e} km/s), drift from the initial state, arc run: |dE/E| {max_de:.2e}, |dLz/Lz| {max_dl:.2e}; full-ring run: {ring_de:.2e}, {ring_dl:.2e}",
            mode.name(),
            pr.len(),
            max_off.max(0.0)
        );
        println!("  position difference by |m| = 0, 1, 2, 3: {:?}", by_m);
        // 2e-6 rad: the edge slack EDGE_TOL = 1e-6 rad plus the f32 rounding of the test.
        assert!(max_off < 2e-6, "all particles are inside the arc");
        assert!(moved > pr.len() / 10, "enough wraps to test");
        // Tolerance: 4 times the rounding floor (+ 1e-6 km, 1e-9 km/s). The arc run
        // and the ring run use different f32 rounding of the kick (the state is
        // rotated), so they differ by the same amount as a ring run from a rotated
        // state. The floor is that difference measured; D1 adds only the DS error of
        // a wrap (1e-14 relative, 1e-9 km). Mode D has about 100 times the noise of
        // mode D+ (f32 J2 term), which is why the floor is not a fixed number. The
        // measured ratio arc/floor is 1.0 to 2.5 (see the handoff D1).
        assert!(
            max_dp < 4.0 * floor_p + 1e-6 && max_dv < 4.0 * floor_v + 1e-9,
            "same orbit up to rotation"
        );
        // Tolerance: the drift of E and Lz of the arc run may be 2 times that of the
        // ring run plus 3e-13 (about 3 wraps per particle times 1.3e-14 each,
        // with margin).
        assert!(
            max_de < 2.0 * ring_de + 3e-13 && max_dl < 2.0 * ring_dl + 3e-13,
            "the wraps add no energy or Lz drift"
        );
    }
}

/// An arc of 360 degrees never wraps: the same orbits as the full ring (the
/// compiler may fuse the arithmetic of the two kernels differently, so the
/// bits can differ in the last place).
#[test]
#[ignore = "needs a GPU"]
fn arc_of_360_degrees_equals_the_full_ring() {
    let Some(_) = Gpu::new() else {
        eprintln!("no GPU: skipped");
        return;
    };
    let steps = 150;
    let system = system(H_S, steps, true);
    let (init, _) = ensemble(&system, 300);
    let mut a = run(
        &arc_config(true, steps, 360.0, 0.0, None),
        &system,
        &init,
        &[150],
        Mode::Dp,
        64,
    );
    let mut r = run(
        &config(H_S, steps, true),
        &system,
        &init,
        &[150],
        Mode::Dp,
        64,
    );
    let (pa, pr) = (
        a.particles().unwrap().particles.clone(),
        r.particles().unwrap().particles.clone(),
    );
    let (mut dp, mut dv, mut same) = (0f64, 0f64, 0usize);
    for i in 0..pa.len() {
        dp = dp.max((pa.position_km(i) - pr.position_km(i)).norm());
        dv = dv.max((pa.velocity_km_s(i) - pr.velocity_km_s(i)).norm());
        same += usize::from(pa.position_km(i) == pr.position_km(i));
    }
    println!(
        "360 deg arc against the full ring after 150 steps: {same}/{} particles bit-identical, \
         largest difference {dp:.2e} km, {dv:.2e} km/s",
        pa.len()
    );
    // Tolerance: 1e-4 km and 1e-8 km/s. The ring and the arc kernels do the same
    // operations, but the Metal compiler fuses the arithmetic differently in the
    // two, so the f32 rounding of the kick differs. After 150 steps the measured
    // difference is 9e-6 km and 1.4e-9 km/s; the limit is about 10 times that, and
    // 1e-4 km is 7e-10 of the orbit radius, far below any physical effect.
    assert!(dp < 1e-4 && dv < 1e-8);
}

/// With chaos, the shadow of a particle takes the same rotations as the
/// particle: the separation stays near d0 (a shadow that wrapped on its own
/// would be one arc width, 24000 km, away).
#[test]
#[ignore = "needs a GPU"]
fn chaos_shadows_take_the_same_rotations_as_their_particles() {
    let Some(_) = Gpu::new() else {
        eprintln!("no GPU: skipped");
        return;
    };
    let steps = 1000;
    let system = system(H_S, steps, true);
    let init = spread_ensemble(&system, 300, 400.0);
    let chaos = ChaosConfig {
        method: ChaosMethod::Shadow,
        offset_km: 1.0,
        renorm_steps: 5,
    };
    for mode in [Mode::D, Mode::Dp] {
        let geom =
            ArcGeometry::new(10f64.to_radians(), 20f64.to_radians(), n0_of(&system)).unwrap();
        let init = fold_init(&geom, &init);
        let cfg = arc_config(true, steps, 10.0, 20.0, Some(chaos));
        let mut with = run(&cfg, &system, &init, &[1, 600, 399], mode, 64);
        let ring = run(
            &config(H_S, steps, true),
            &system,
            &init,
            &[1, 600, 399],
            mode,
            64,
        );
        let pw = with.particles().unwrap().particles.clone();
        let mut ring = ring;
        let pr = ring.particles().unwrap().particles.clone();
        let wraps = (0..pw.len())
            .filter(|&i| (pr.position_km(i) - pw.position_km(i)).norm() > 1.0)
            .count();
        // Parent and shadow in the state buffers.
        let gpu = with.gpu().clone();
        let st = with.stepper();
        let cap = st.capacity();
        let a = read_buffer(&gpu, st.state_buffer());
        let b = read_buffer(&gpu, st.shadow_buffer().unwrap());
        let word =
            |buf: &[u8], w: usize| f32::from_le_bytes(buf[4 * w..4 * w + 4].try_into().unwrap());
        let val = |buf: &[u8], c: usize, i: usize| {
            let w = 2 * (c * cap + i);
            f64::from(word(buf, w)) + f64::from(word(buf, w + 1))
        };
        let mut max_sep = 0f64;
        for i in 0..pw.len() {
            let d: f64 = (0..3)
                .map(|c| (val(&a, c, i) - val(&b, c, i)).powi(2))
                .sum::<f64>()
                .sqrt();
            max_sep = max_sep.max(d);
        }
        println!(
            "mode {}: {wraps}/{} particles differ from the full ring (wrapped); largest particle-shadow \
             separation {max_sep:.3} km (d0 = 1 km, arc 10 deg = {:.0} km)",
            mode.name(),
            pw.len(),
            A0_KM * 10f64.to_radians()
        );
        assert!(wraps > 100, "many particles wrapped");
        assert!(max_sep < 50.0, "shadow wrapped like its particle");
    }
}

/// The arc is refused in mode C (no arc kernel there).
#[test]
#[ignore = "needs a GPU"]
fn arc_is_refused_in_mode_c() {
    let Some(_) = Gpu::new() else {
        eprintln!("no GPU: skipped");
        return;
    };
    let system = system(H_S, 10, false);
    let (init, _) = ensemble(&system, 10);
    let r = GpuBackend::with_options(
        &arc_config(false, 10, 10.0, 0.0, None),
        &system,
        &init,
        Options {
            mode: Mode::C,
            ..Default::default()
        },
    );
    let msg = match r {
        Err(e) => e.to_string(),
        Ok(_) => panic!("mode C with an arc must fail"),
    };
    assert!(msg.contains("mode C"), "{msg}");
}

/// Cost of the wrap: 10^5 particles (all round the ring, so a 10 deg arc folds
/// them), 300 steps, no moons (the cheapest kick, so the ratio is the largest), mode D+.
#[test]
#[ignore = "needs a GPU; prints times"]
fn arc_cost() {
    let Some(_) = Gpu::new() else {
        eprintln!("no GPU: skipped");
        return;
    };
    let steps = 1500;
    let system = system(H_S, steps, false);
    let n = 100_000;
    let init = spread_ensemble(&system, n, 400.0);
    let geom = ArcGeometry::new(10f64.to_radians(), 0.0, n0_of(&system)).unwrap();
    let init_arc = fold_init(&geom, &init);
    let mut times = Vec::new();
    for arc in [false, true] {
        let cfg = if arc {
            arc_config(false, steps, 10.0, 0.0, None)
        } else {
            config(H_S, steps, false)
        };
        let i = if arc { &init_arc } else { &init };
        let mut b = run(&cfg, &system, i, &[1], Mode::Dp, 64); // warm up
        let t0 = std::time::Instant::now();
        b.step(steps - 1)
            .unwrap_or_else(|e| panic!("arc {arc}: {e}"));
        let _ = b.particles().unwrap();
        times.push(t0.elapsed().as_secs_f64());
    }
    println!(
        "{n} particles, {} steps: full ring {:.3} s, arc {:.3} s (x{:.3})",
        steps - 1,
        times[0],
        times[1],
        times[1] / times[0]
    );
}
