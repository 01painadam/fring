//! ICR-006 (W2-DROP-GPU): dropped particles in `GpuStepper`. Needs a GPU, so
//! `#[ignore]`:
//! `cargo test -p fring-gpu --release --test gpu_drop -- --ignored --nocapture`.

mod common;

use common::*;
use fring_gpu::gpu::Gpu;
use fring_gpu::spike::{Consts, State, kick_scale, state_from_f64};
use fring_gpu::stepper::GpuStepper;
use fring_gpu::wf32::model;
use fring_gpu::wh::{self, HostTable, Mode};
use fring_gpu::wh_gpu::{DEFAULT_WORKGROUP, WhKernels};
use fring_model::physics::{KickTable, interpolate_moons};
use fring_model::{InitialState, Particles, System, Vec3};

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
    run_rule(gpu, kernels, system, init, mode, block, calls, None)
}

#[allow(clippy::too_many_arguments)]
fn run_rule(
    gpu: &Gpu,
    kernels: &WhKernels,
    system: &fring_model::System,
    init: &fring_model::InitialState,
    mode: Mode,
    block: usize,
    calls: &[u64],
    radii: Option<&[f64]>,
) -> (Vec<u8>, Vec<u8>, usize) {
    let mut st = GpuStepper::new(gpu, kernels, system, init, H_S, block, mode, None).unwrap();
    if let Some(r) = radii {
        st.set_moon_radii_km(r).unwrap();
    }
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

const M4: u8 = model::M4;
/// Mean radii of Prometheus and Pandora, km (`params/moons.toml`).
const RADII_KM: [f64; 2] = [43.09, 40.63];

/// A particle that passes moon `k` at the distance `b_km` (about) at the
/// time `t_cross_s`: at t = 0 it is `u t_cross` behind the moon in the radial
/// direction with the relative velocity `u` (km/s) outwards, and `b` to the
/// side. The relative path is nearly straight (tidal bending only).
fn crafted(system: &System, k: usize, u: f64, b: f64, t_cross_s: f64) -> (Vec3<f64>, Vec3<f64>) {
    let m = &interpolate_moons(&system.moons, 0.0).unwrap().moons[k];
    let (pos, vel) = (m.position_km, m.velocity_km_s);
    let rh = pos / pos.norm();
    let th = Vec3::new(0.0, 0.0, 1.0).cross(rh);
    (pos - rh * (u * t_cross_s) + th * b, vel + rh * u)
}

fn init_of(list: &[(Vec3<f64>, Vec3<f64>)]) -> InitialState {
    let mut p = Particles::with_capacity(list.len());
    for (i, (x, v)) in list.iter().enumerate() {
        p.push(i as u32, *x, *v);
    }
    InitialState {
        time_s: 0.0,
        particles: p,
    }
}

/// What the Rust twin says about each particle over the whole run.
struct Twin {
    reason: Vec<u32>,
    /// Block of the drop.
    block: Vec<Option<usize>>,
    /// Smallest distance to each moon over the checked drifts, km.
    min_km: Vec<[f32; 2]>,
    /// Distances of the ends of the drift that hit, km.
    ends_km: Vec<[f32; 2]>,
}

fn twin(
    system: &System,
    init: &InitialState,
    mode: Mode,
    block: usize,
    steps: usize,
    radii_km: &[f64],
) -> Twin {
    let gms: Vec<f64> = system.moons.tracks.iter().map(|t| t.gm_km3_s2).collect();
    let c = Consts::<M4>::for_step(&system.saturn, &gms, H_S);
    let p = &init.particles;
    let mut st: Vec<State<M4>> = (0..p.len())
        .map(|i| state_from_f64(p.position_km(i), p.velocity_km_s(i)))
        .collect();
    let r2: Vec<f32> = radii_km.iter().map(|r| (r * r) as f32).collect();
    let mut t = Twin {
        reason: vec![0; st.len()],
        block: vec![None; st.len()],
        min_km: vec![[f32::INFINITY; 2]; st.len()],
        ends_km: vec![[0.0; 2]; st.len()],
    };
    let mut table = KickTable::<f64>::for_ephemeris(&system.moons);
    let (mut first, mut b) = (0usize, 0usize);
    while first < steps {
        let n = block.min(steps - first);
        table
            .fill(&system.moons, init.time_s, H_S, first as u64, n)
            .unwrap();
        let tab = HostTable::<M4>::from_kick_table(&table, kick_scale(H_S));
        let rows = wh::sync_rows(&system.moons, init.time_s, H_S, first as u64, n);
        let sync = wh::sync_moons::<M4>(&rows, 2);
        for (i, state) in st.iter_mut().enumerate() {
            if t.reason[i] != 0 {
                continue;
            }
            let o = wh::advance_dropping(mode, &c, &tab, &sync, &r2, state);
            for j in 0..2 {
                t.min_km[i][j] = t.min_km[i][j].min(o.min_d2_km2[j].sqrt());
            }
            if o.reason != 0 {
                t.reason[i] = o.reason;
                t.block[i] = Some(b);
                t.ends_km[i] = o.ends_d2_km2.map(f32::sqrt);
            }
        }
        first += n;
        b += 1;
    }
    t
}

/// Finds the crafted particle of moon `k` whose closest approach (twin, rule
/// off) is nearest to 25 km. Returns its state and that distance.
fn pick_crossing(system: &System, k: usize, mode: Mode) -> ((Vec3<f64>, Vec3<f64>), f64) {
    let t_cross = 5.0 * H_S;
    let list: Vec<_> = (-30..=30)
        .map(|j| crafted(system, k, 0.4, 6.0 * f64::from(j), t_cross))
        .collect();
    let tw = twin(system, &init_of(&list), mode, 3, 12, &[0.0, 0.0]);
    let (best, d) = tw
        .min_km
        .iter()
        .enumerate()
        .map(|(i, m)| (i, f64::from(m[k])))
        .min_by(|a, b| (a.1 - 25.0).abs().total_cmp(&(b.1 - 25.0).abs()))
        .unwrap();
    (list[best], d)
}

#[test]
#[ignore = "needs a GPU"]
fn a_crossing_between_two_kicks_is_dropped_in_the_right_block_and_r_plus_1_km_is_not() {
    let Some(gpu) = Gpu::new() else {
        eprintln!("no GPU: skipped");
        return;
    };
    let kernels = WhKernels::new(&gpu, DEFAULT_WORKGROUP).expect("self-test");
    let steps = 12;
    let system = system(H_S, steps as u64, true);
    assert_eq!(system.moons.tracks[0].name, "prometheus");
    assert_eq!(system.moons.tracks[1].name, "pandora");
    for mode in [Mode::D, Mode::Dp] {
        for k in 0..2usize {
            let (cross, dmin) = pick_crossing(&system, k, mode);
            println!("{mode:?} moon {k}: closest approach {dmin:.2} km");
            assert!((10.0..40.0).contains(&dmin), "closest approach {dmin}");
            // Particle 5 is the crossing; the others are the ensemble.
            let (base, _) = ensemble(&system, 40);
            let mut list: Vec<_> = (0..40)
                .map(|i| {
                    (
                        base.particles.position_km(i),
                        base.particles.velocity_km_s(i),
                    )
                })
                .collect();
            list[5] = cross;
            let init = init_of(&list);
            let mut radii = [0.0f64; 2];
            // Radius dmin + 1 km: dropped. Radius dmin - 1 km: not dropped.
            radii[k] = dmin + 1.0;
            let tw = twin(&system, &init, mode, 3, steps, &radii);
            assert_eq!(tw.reason[5], 2 + k as u32, "twin reason");
            let blk = tw.block[5].unwrap();
            assert!(
                tw.ends_km[5].iter().all(|e| f64::from(*e) > radii[k]),
                "both ends outside the radius: {:?}",
                tw.ends_km[5]
            );
            assert!(blk >= 1, "crossing in a later block, got {blk}");
            let calls = [steps as u64];
            let (s_hit, st_hit, cap) = run_rule(
                &gpu,
                &kernels,
                &system,
                &init,
                mode,
                3,
                &calls,
                Some(&radii),
            );
            // Reasons and counts.
            for i in 0..40 {
                let want = if i == 5 { 2 + k as u32 } else { 0 };
                assert_eq!(word(&st_hit, 8 + i), want, "{mode:?} moon {k}: reason {i}");
                assert_eq!(tw.reason[i], want, "twin reason {i}");
            }
            assert_eq!(word(&st_hit, 0), 5);
            assert_eq!(word(&st_hit, 1), 0);
            for w in 2..8 {
                assert_eq!(word(&st_hit, w), u32::from(w == 2 + k), "count word {w}");
            }
            // Dropped in block `blk`: the state is the one at the start of that
            // block (3 steps per block), also after more blocks.
            let (s_at, _, _) = run_rule(
                &gpu,
                &kernels,
                &system,
                &init,
                mode,
                3,
                &[3 * blk as u64],
                Some(&radii),
            );
            assert_eq!(
                particle_bytes(&s_hit, cap, 5),
                particle_bytes(&s_at, cap, 5),
                "{mode:?} moon {k}: state frozen at the start of block {blk}"
            );
            // Radius dmin - 1 km: nobody is dropped.
            radii[k] = dmin - 1.0;
            let (s_miss, st_miss, _) = run_rule(
                &gpu,
                &kernels,
                &system,
                &init,
                mode,
                3,
                &calls,
                Some(&radii),
            );
            assert_eq!(
                word(&st_miss, 0),
                u32::MAX,
                "{mode:?} moon {k}: not dropped"
            );
            assert!((1..8).all(|w| word(&st_miss, w) == 0));
            // The other particles are bit for bit those of the run without a drop.
            for i in (0..40).filter(|i| *i != 5) {
                assert_eq!(
                    particle_bytes(&s_hit, cap, i),
                    particle_bytes(&s_miss, cap, i),
                    "{mode:?} moon {k}: particle {i}"
                );
            }
        }
    }
}

#[test]
#[ignore = "needs a GPU"]
fn rule_with_tiny_radii_gives_the_bits_of_the_rule_off_stepper() {
    let Some(gpu) = Gpu::new() else {
        eprintln!("no GPU: skipped");
        return;
    };
    let kernels = WhKernels::new(&gpu, DEFAULT_WORKGROUP).expect("self-test");
    let system = system(H_S, 150, true);
    let (init, _) = ensemble(&system, 100);
    for mode in [Mode::D, Mode::Dp] {
        let (off, st_off, _) = run(&gpu, &kernels, &system, &init, mode, 7, &[150]);
        let (on, st_on, _) = run_rule(
            &gpu,
            &kernels,
            &system,
            &init,
            mode,
            7,
            &[150],
            Some(&[1e-3, 1e-3]),
        );
        assert!(off == on, "{mode:?}: state bits");
        assert!(st_off == st_on, "{mode:?}: status bits");
    }
}

#[test]
#[ignore = "needs a GPU"]
fn gpu_and_twin_agree_on_who_is_dropped_near_the_moons() {
    let Some(gpu) = Gpu::new() else {
        eprintln!("no GPU: skipped");
        return;
    };
    let kernels = WhKernels::new(&gpu, DEFAULT_WORKGROUP).expect("self-test");
    let steps = 12;
    let system = system(H_S, steps as u64, true);
    let mut list = Vec::new();
    for k in 0..2usize {
        for u in [0.25, 0.4, -0.5] {
            for j in -30..=30 {
                list.push(crafted(&system, k, u, 3.0 * f64::from(j), 5.0 * H_S));
            }
        }
    }
    let init = init_of(&list);
    for mode in [Mode::D, Mode::Dp] {
        let tw = twin(&system, &init, mode, 3, steps, &RADII_KM);
        let (_, st, _) = run_rule(
            &gpu,
            &kernels,
            &system,
            &init,
            mode,
            3,
            &[steps as u64],
            Some(&RADII_KM),
        );
        let (mut dropped, mut border) = (0, 0);
        let mut counts = [0u32; 6];
        for i in 0..list.len() {
            let g = word(&st, 8 + i);
            if g != 0 {
                dropped += 1;
                counts[g as usize - 1] += 1;
            }
            if g != tw.reason[i] {
                // Allowed only on the edge: the closest approach is within 0.05 km of a radius.
                let near = (0..2).any(|j| (f64::from(tw.min_km[i][j]) - RADII_KM[j]).abs() < 0.05);
                assert!(
                    near,
                    "{mode:?}: particle {i}: GPU {g}, twin {}",
                    tw.reason[i]
                );
                border += 1;
            }
        }
        println!(
            "{mode:?}: {dropped} of {} dropped, {border} on the edge",
            list.len()
        );
        assert!(dropped > 20 && dropped < list.len() - 20);
        assert_eq!(word(&st, 1), counts[0]);
        assert_eq!(word(&st, 2), counts[1]);
        assert_eq!(word(&st, 3), counts[2]);
        assert!(border <= 2);
    }
}

#[test]
#[ignore = "needs a GPU"]
fn set_moon_radii_checks_its_arguments() {
    let Some(gpu) = Gpu::new() else {
        eprintln!("no GPU: skipped");
        return;
    };
    let kernels = WhKernels::new(&gpu, DEFAULT_WORKGROUP).expect("self-test");
    let system = system(H_S, 20, true);
    let (init, _) = ensemble(&system, 10);
    let make =
        |mode, chaos| GpuStepper::new(&gpu, &kernels, &system, &init, H_S, 7, mode, chaos).unwrap();
    let is_config = |r: fring_model::Result<()>| matches!(r, Err(fring_model::Error::Config(_)));
    assert!(make(Mode::Dp, None).set_moon_radii_km(&RADII_KM).is_ok());
    assert!(is_config(make(Mode::Dp, None).set_moon_radii_km(&[43.0])));
    assert!(is_config(
        make(Mode::Dp, None).set_moon_radii_km(&[-1.0, 1.0])
    ));
    assert!(is_config(
        make(Mode::Dp, None).set_moon_radii_km(&[f64::NAN, 1.0])
    ));
    assert!(is_config(make(Mode::C, None).set_moon_radii_km(&RADII_KM)));
    let chaos = fring_model::config::ChaosConfig {
        method: fring_model::config::ChaosMethod::Shadow,
        offset_km: 1.0,
        renorm_steps: 5,
    };
    assert!(is_config(
        make(Mode::Dp, Some(chaos)).set_moon_radii_km(&RADII_KM)
    ));
    let mut st = make(Mode::D, None);
    let mut enc = gpu.device().create_command_encoder(&Default::default());
    st.encode_steps(&mut enc, 1).unwrap();
    gpu.queue().submit([enc.finish()]);
    assert!(is_config(st.set_moon_radii_km(&RADII_KM)));
}
