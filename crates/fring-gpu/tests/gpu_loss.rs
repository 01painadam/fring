//! E1-GPU, tests G2 and G3 (ICR-004, amendment 2): flag and freeze, the radius
//! rule and `GpuBackend::dropped` on the GPU. Needs a GPU, so `#[ignore]`:
//! `cargo test -p fring-gpu --release --test gpu_loss -- --ignored --nocapture`.
//!
//! The `drop_step` convention is the GPU one (see
//! `GpuStepper::status_buffer`): the drift after kick `k` belongs to step
//! `first + k`, the first half drift of a block to `first`, and the end half
//! drift to `first + n - 1`.

mod common;

use common::*;
use fring_gpu::backend::{GpuBackend, Options};
use fring_gpu::gpu::Gpu;
use fring_gpu::spike::{Consts, State, kick_scale, state_from_f64};
use fring_gpu::wf32::model::M4;
use fring_gpu::wh::{self, HostTable, Mode};
use fring_model::physics::{KickTable, interpolate_moons};
use fring_model::{
    Backend, DropReport, Error, InitialState, Particles, RunConfig, System, Vec3, drop_inside_moon,
};

/// Mean radii of Prometheus and Pandora, km (`params/moons.toml`).
const RADII_KM: [f64; 2] = [43.09, 40.63];
const LOSS_FREEZE: &str = "[loss]\non_unbound = \"freeze\"\n";
const LOSS_RULE: &str = "[loss]\non_unbound = \"freeze\"\nmoon_radius = \"params\"\n";
const CHAOS: &str = "[diagnostics.chaos]\nmethod = \"shadow\"\noffset_km = 1.0\nrenorm_steps = 5\n";

fn have_gpu() -> bool {
    if Gpu::new().is_none() {
        eprintln!("no GPU: skipped");
        return false;
    }
    true
}

fn cfg(steps: u64, extra: &str, arc_deg: Option<f64>) -> RunConfig {
    RunConfig::from_toml_str(&config_text(H_S, steps, true, extra, arc_deg)).unwrap()
}

fn backend(
    config: &RunConfig,
    system: &System,
    init: &InitialState,
    block: usize,
    mode: Mode,
) -> fring_model::Result<GpuBackend> {
    let opts = Options {
        mode,
        block_steps: Some(block),
        ..Options::default()
    };
    GpuBackend::with_options(config, system, init, opts)
}

fn with_radii(mut b: GpuBackend) -> GpuBackend {
    b.set_moon_radii_km(&RADII_KM).unwrap();
    b
}

/// The six planes of every particle as bytes (position and velocity as f64).
fn snapshot(b: &mut GpuBackend) -> Vec<[u64; 6]> {
    // `step(0)` makes the snapshot come from the GPU (double-single words
    // summed in f64), also before the first step.
    b.step(0).unwrap();
    let v = b.particles().unwrap();
    let p = v.particles;
    (0..p.len())
        .map(|i| {
            [
                p.x_km[i].to_bits(),
                p.y_km[i].to_bits(),
                p.z_km[i].to_bits(),
                p.vx_km_s[i].to_bits(),
                p.vy_km_s[i].to_bits(),
                p.vz_km_s[i].to_bits(),
            ]
        })
        .collect()
}

fn init_with(list: &[(Vec3<f64>, Vec3<f64>)], base: &InitialState) -> InitialState {
    let mut p = Particles::with_capacity(base.particles.len() + list.len());
    let n = base.particles.len();
    for i in 0..n {
        p.push(
            i as u32,
            base.particles.position_km(i),
            base.particles.velocity_km_s(i),
        );
    }
    for (j, (x, v)) in list.iter().enumerate() {
        p.push((n + j) as u32, *x, *v);
    }
    InitialState {
        time_s: 0.0,
        particles: p,
    }
}

/// A particle that passes moon `k` at about `b_km` at the time `t_cross_s`
/// (as in `gpu_drop.rs`).
fn crafted(system: &System, k: usize, u: f64, b: f64, t_cross_s: f64) -> (Vec3<f64>, Vec3<f64>) {
    let m = &interpolate_moons(&system.moons, 0.0).unwrap().moons[k];
    let (pos, vel) = (m.position_km, m.velocity_km_s);
    let rh = pos / pos.norm();
    let th = Vec3::new(0.0, 0.0, 1.0).cross(rh);
    (pos - rh * (u * t_cross_s) + th * b, vel + rh * u)
}

/// What the Rust twin says about each particle over `steps` steps in blocks of
/// `block`: the reason, the `drop_step` of the GPU convention (for reason 1
/// only meaningful with `block = 1`) and the smallest distance to each moon.
struct Twin {
    reason: Vec<u32>,
    step: Vec<u64>,
    min_km: Vec<[f32; 2]>,
}

fn twin(
    system: &System,
    list: &[(Vec3<f64>, Vec3<f64>)],
    block: usize,
    steps: usize,
    radii_km: &[f64],
) -> Twin {
    let gms: Vec<f64> = system.moons.tracks.iter().map(|t| t.gm_km3_s2).collect();
    let c = Consts::<M4>::for_step(&system.saturn, &gms, H_S);
    let mut st: Vec<State<M4>> = list.iter().map(|(x, v)| state_from_f64(*x, *v)).collect();
    let r2: Vec<f32> = radii_km.iter().map(|r| (r * r) as f32).collect();
    let mut t = Twin {
        reason: vec![0; st.len()],
        step: vec![u64::MAX; st.len()],
        min_km: vec![[f32::INFINITY; 2]; st.len()],
    };
    let mut table = KickTable::<f64>::for_ephemeris(&system.moons);
    let mut first = 0usize;
    while first < steps {
        let n = block.min(steps - first);
        table
            .fill(&system.moons, 0.0, H_S, first as u64, n)
            .unwrap();
        let tab = HostTable::<M4>::from_kick_table(&table, kick_scale(H_S));
        let rows = wh::sync_rows(&system.moons, 0.0, H_S, first as u64, n);
        let sync = wh::sync_moons::<M4>(&rows, 2);
        for (i, state) in st.iter_mut().enumerate() {
            if t.reason[i] != 0 {
                continue;
            }
            let o = wh::advance_dropping(Mode::Dp, &c, &tab, &sync, &r2, state);
            for j in 0..2 {
                t.min_km[i][j] = t.min_km[i][j].min(o.min_d2_km2[j].sqrt());
            }
            if o.reason != 0 {
                t.reason[i] = o.reason;
                let seg = o.segment.unwrap_or(0);
                t.step[i] = (first + seg.max(1) - 1) as u64;
            }
        }
        first += n;
    }
    t
}

/// The crafted passes of moon `k` whose closest distance (rule off) is nearest
/// to `target_km`. Returns the state and that distance.
fn pick_pass(system: &System, k: usize, target_km: f64) -> ((Vec3<f64>, Vec3<f64>), f64) {
    let list: Vec<_> = (-120..=120)
        .map(|j| crafted(system, k, 0.4, 1.5 * f64::from(j), 5.0 * H_S))
        .collect();
    let tw = twin(system, &list, 3, 12, &[0.0, 0.0]);
    let (best, d) = tw
        .min_km
        .iter()
        .enumerate()
        .map(|(i, m)| (i, f64::from(m[k])))
        .min_by(|a, b| (a.1 - target_km).abs().total_cmp(&(b.1 - target_km).abs()))
        .unwrap();
    (list[best], d)
}

/// A pass of moon `k` so close (< 0.75 km at a kick) that the kick makes the
/// orbit unbound. Searched with the twin, over the impact parameter and the
/// time of the crossing, with blocks of 1 step, so that the block index of the
/// loss is its step.
fn pick_kicked_unbound(system: &System, k: usize) -> ((Vec3<f64>, Vec3<f64>), u64) {
    let mut list = Vec::new();
    for tj in 0..90 {
        let t_cross = 4.5 * H_S + 4.0 * f64::from(tj);
        for bj in -300..=300 {
            list.push(crafted(system, k, 0.05, 0.2 * f64::from(bj), t_cross));
        }
    }
    let tw = twin(system, &list, 1, 12, &[0.0, 0.0]);
    let i = (0..list.len())
        .filter(|i| tw.reason[*i] == 1 && tw.step[*i] >= 3)
        .min_by_key(|i| tw.step[*i])
        .expect("a pass that the kick of the moon makes unbound");
    (list[i], tw.step[i])
}

fn dropped(b: &mut GpuBackend) -> DropReport {
    b.dropped().unwrap().expect("a report with freeze")
}

fn unbound_at_start(system: &System, init: &InitialState, i: usize) -> (Vec3<f64>, Vec3<f64>) {
    let _ = system;
    (
        init.particles.position_km(i),
        init.particles.velocity_km_s(i) * 3.0,
    )
}

/// C2 on the GPU: unbound particles with freeze, for block lengths 8, 37 and
/// 500: reason 1, exact `drop_step` (0 for the two that start unbound, the
/// step of the twin for a pass that the kick of Prometheus makes unbound), the
/// other particles bit for bit as in a run without the bad ones, the frozen
/// state equal to the state at the start of the block, `dropped()` counts and
/// ids; and the `"error"` default still returns the old error.
#[test]
#[ignore = "needs a GPU"]
fn g3_unbound_particles_are_frozen_with_exact_drop_steps() {
    if !have_gpu() {
        return;
    }
    let steps = 40u64;
    let system = system(H_S, steps, true);
    let (base, _) = ensemble(&system, 90);
    let kicked = pick_kicked_unbound(&system, 0);
    println!("pass made unbound by the kick: step {}", kicked.1);
    let bad = [
        unbound_at_start(&system, &base, 5),
        unbound_at_start(&system, &base, 6),
        kicked.0,
    ];
    let n0 = base.particles.len();
    let init_bad = init_with(&bad, &base);
    // The same run with benign particles in the three places.
    let benign = [
        (
            base.particles.position_km(5),
            base.particles.velocity_km_s(5),
        ),
        (
            base.particles.position_km(6),
            base.particles.velocity_km_s(6),
        ),
        (
            base.particles.position_km(7),
            base.particles.velocity_km_s(7),
        ),
    ];
    let init_ok = init_with(&benign, &base);
    for block in [8usize, 37, 500] {
        let config = cfg(steps, LOSS_FREEZE, None);
        let mut a = backend(&config, &system, &init_bad, block, Mode::Dp).unwrap();
        let start = snapshot(&mut a);
        a.step(steps).unwrap();
        let rep = dropped(&mut a);
        let mut o = backend(&config, &system, &init_ok, block, Mode::Dp).unwrap();
        o.step(steps).unwrap();
        assert!(dropped(&mut o).reason.iter().all(|r| *r == 0));
        let (sa, so) = (snapshot(&mut a), snapshot(&mut o));
        let want = [(n0, 0u64), (n0 + 1, 0), (n0 + 2, kicked.1)];
        for i in 0..n0 + 3 {
            let w = want.iter().find(|(j, _)| *j == i);
            match w {
                Some((_, s)) => {
                    assert_eq!(rep.reason[i], 1, "block {block}: reason {i}");
                    assert_eq!(rep.drop_step[i], *s, "block {block}: drop_step {i}");
                }
                None => {
                    assert_eq!(rep.reason[i], 0, "block {block}: live {i}");
                    assert_eq!(rep.drop_step[i], u64::MAX);
                    assert_eq!(sa[i], so[i], "block {block}: live particle {i} changed");
                }
            }
        }
        assert_eq!(rep.n_unbound, 3);
        assert_eq!(rep.n_inside_moon, vec![0, 0]);
        // Frozen state: the state at the start of the block of the loss. The
        // two that start unbound keep the initial state.
        for i in [n0, n0 + 1] {
            assert_eq!(sa[i], start[i], "block {block}: frozen {i}");
        }
        let b0 = (kicked.1 as usize / block) * block;
        let mut r = backend(&config, &system, &init_bad, block, Mode::Dp).unwrap();
        if b0 > 0 {
            r.step(b0 as u64).unwrap();
        }
        assert_eq!(
            snapshot(&mut r)[n0 + 2],
            sa[n0 + 2],
            "block {block}: the kicked particle is frozen at the start of its block ({b0})"
        );
    }
    // The default (no freeze) still returns the old error with the id.
    let mut init_err = init_ok.clone();
    init_err.particles.id[77] = 1077;
    init_err.particles.vy_km_s[77] *= 3.0;
    for extra in ["", "[loss]\n"] {
        let config = cfg(steps, extra, None);
        let mut e = backend(&config, &system, &init_err, 8, Mode::Dp).unwrap();
        match e.step(steps) {
            Err(Error::Backend(m)) => {
                assert!(m.contains("particle 1077") && m.contains("index 77"), "{m}")
            }
            other => panic!("expected the old error, got {other:?}"),
        }
        assert!(e.dropped().unwrap().is_none());
    }
}

/// C3 on the GPU: a crossing between two kicks is dropped with reason 2 + k and
/// the exact `drop_step` of the twin, for both moons and block lengths 8, 37
/// and 500; a pass at radius + 1 km is not dropped; one at radius - 1 km is;
/// the frozen state is the state at the start of the block.
#[test]
#[ignore = "needs a GPU"]
fn g3_a_pass_inside_the_radius_is_dropped_with_the_exact_step() {
    if !have_gpu() {
        return;
    }
    let steps = 40u64;
    let system = system(H_S, steps, true);
    let (base, _) = ensemble(&system, 60);
    let mut list = Vec::new();
    for (k, r) in RADII_KM.iter().enumerate() {
        for dr in [-1.0, 1.0, -10.0, 10.0] {
            let (s, d) = pick_pass(&system, k, r + dr);
            println!(
                "moon {k}: wanted {:.1} km, crafted pass at {d:.2} km",
                r + dr
            );
            list.push((k, dr, d, s));
        }
    }
    let states: Vec<_> = list.iter().map(|l| l.3).collect();
    let n0 = base.particles.len();
    let init = init_with(&states, &base);
    for block in [8usize, 37, 500] {
        let tw = twin(&system, &states, block, steps as usize, &RADII_KM);
        let config = cfg(steps, LOSS_RULE, None);
        let mut b = with_radii(backend(&config, &system, &init, block, Mode::Dp).unwrap());
        b.step(steps).unwrap();
        let rep = dropped(&mut b);
        let after = snapshot(&mut b);
        for (j, (k, dr, d, _)) in list.iter().enumerate() {
            let i = n0 + j;
            let hit = *d < RADII_KM[*k];
            if hit {
                let want_reason = drop_inside_moon(*k);
                assert_eq!(
                    rep.reason[i], want_reason,
                    "block {block}: moon {k} dr {dr}"
                );
                assert_eq!(u32::from(rep.reason[i]), tw.reason[j]);
                assert_eq!(
                    rep.drop_step[i], tw.step[j],
                    "block {block}: moon {k} dr {dr}"
                );
                // Frozen state = state at the start of the block of the loss.
                let b0 = (rep.drop_step[i] as usize / block) * block;
                let mut r = with_radii(backend(&config, &system, &init, block, Mode::Dp).unwrap());
                if b0 > 0 {
                    r.step(b0 as u64).unwrap();
                }
                assert_eq!(
                    snapshot(&mut r)[i],
                    after[i],
                    "block {block}: frozen {k} {dr}"
                );
            } else {
                assert_eq!(
                    rep.reason[i], 0,
                    "block {block}: moon {k} dr {dr} is not dropped"
                );
            }
        }
        // Counts per moon agree with the reasons.
        for k in 0..2 {
            let n = rep
                .reason
                .iter()
                .filter(|r| **r == drop_inside_moon(k))
                .count();
            assert_eq!(rep.n_inside_moon[k], n as u64, "block {block}");
        }
        assert!(rep.n_inside_moon.iter().sum::<u64>() >= 4, "{rep:?}");
    }
}

/// The radius rule with the radii not set is a `Config` error at the first
/// step; the radii cannot be set after a step; `[substep]`, mode C with
/// `[loss]` and the rule with the arc are `Config` errors.
#[test]
#[ignore = "needs a GPU"]
fn g3_config_errors() {
    if !have_gpu() {
        return;
    }
    let steps = 12u64;
    let system = system(H_S, steps, true);
    let (init, _) = ensemble(&system, 20);
    let is_cfg = |r: fring_model::Result<GpuBackend>| matches!(r, Err(Error::Config(_)));
    let substep = "[loss]\non_unbound = \"freeze\"\n[substep]\naccuracy = 0.1\nmax_level = 4\n";
    assert!(is_cfg(backend(
        &cfg(steps, substep, None),
        &system,
        &init,
        8,
        Mode::Dp
    )));
    assert!(is_cfg(backend(
        &cfg(steps, LOSS_FREEZE, None),
        &system,
        &init,
        8,
        Mode::C
    )));
    assert!(is_cfg(backend(
        &cfg(steps, LOSS_RULE, Some(10.0)),
        &system,
        &init,
        8,
        Mode::Dp
    )));
    // Defaults only: nothing changes, also in mode C.
    assert!(backend(&cfg(steps, "[loss]\n", None), &system, &init, 8, Mode::C).is_ok());
    // Radii.
    let mut b = backend(&cfg(steps, LOSS_RULE, None), &system, &init, 8, Mode::Dp).unwrap();
    assert!(matches!(b.step(4), Err(Error::Config(_))));
    assert!(matches!(
        b.set_moon_radii_km(&[43.0]),
        Err(Error::Config(_))
    ));
    assert!(matches!(
        b.set_moon_radii_km(&[43.0, 0.0]),
        Err(Error::Config(_))
    ));
    assert!(matches!(
        b.set_moon_radii_km(&[f64::NAN, 40.0]),
        Err(Error::Config(_))
    ));
    b.set_moon_radii_km(&RADII_KM).unwrap();
    b.step(4).unwrap();
    assert!(matches!(
        b.set_moon_radii_km(&RADII_KM),
        Err(Error::Config(_))
    ));
}

/// G2 on the backend: with `[loss]` absent, with only the defaults, with
/// freeze but no loss, and with a radius rule of tiny radii the state bits are
/// the same; `dropped()` is `None` unless `freeze`.
#[test]
#[ignore = "needs a GPU"]
fn g2_off_means_off_on_the_backend() {
    if !have_gpu() {
        return;
    }
    let steps = 40u64;
    let system = system(H_S, steps, true);
    let (init, _) = ensemble(&system, 100);
    for mode in [Mode::D, Mode::Dp] {
        let mut off = backend(&cfg(steps, "", None), &system, &init, 7, mode).unwrap();
        off.step(steps).unwrap();
        assert!(off.dropped().unwrap().is_none());
        let reference = snapshot(&mut off);
        let mut dflt = backend(&cfg(steps, "[loss]\n", None), &system, &init, 7, mode).unwrap();
        dflt.step(steps).unwrap();
        assert!(dflt.dropped().unwrap().is_none());
        assert_eq!(snapshot(&mut dflt), reference, "{mode:?}: [loss] defaults");
        let mut fr = backend(&cfg(steps, LOSS_FREEZE, None), &system, &init, 7, mode).unwrap();
        fr.step(steps).unwrap();
        assert!(dropped(&mut fr).reason.iter().all(|r| *r == 0));
        assert_eq!(snapshot(&mut fr), reference, "{mode:?}: freeze, no loss");
        let mut tiny = backend(&cfg(steps, LOSS_RULE, None), &system, &init, 7, mode).unwrap();
        tiny.set_moon_radii_km(&[1.0e-3, 1.0e-3]).unwrap();
        tiny.step(steps).unwrap();
        assert!(dropped(&mut tiny).reason.iter().all(|r| *r == 0));
        assert_eq!(snapshot(&mut tiny), reference, "{mode:?}: tiny radii");
    }
}

/// C7 on the GPU: with chaos, a lost particle (unbound, or inside a moon) is
/// frozen with its shadow, its chaos `flags` get bit 1 (value 2), its
/// accumulators stop (equal at two times after the loss), and the other
/// particles have the accumulators of a run in which no particle is lost.
#[test]
#[ignore = "needs a GPU"]
fn g3_chaos_pair_is_frozen_and_flagged() {
    if !have_gpu() {
        return;
    }
    let steps = 60u64;
    let system = system(H_S, steps, true);
    let (base, _) = ensemble(&system, 60);
    let (hit, _) = pick_pass(&system, 0, 20.0);
    let bad = [unbound_at_start(&system, &base, 5), hit];
    let n0 = base.particles.len();
    let init_bad = init_with(&bad, &base);
    let benign = [
        (
            base.particles.position_km(5),
            base.particles.velocity_km_s(5),
        ),
        (
            base.particles.position_km(6),
            base.particles.velocity_km_s(6),
        ),
    ];
    let init_ok = init_with(&benign, &base);
    let config = cfg(steps, &format!("{LOSS_RULE}{CHAOS}"), None);
    let mut a = with_radii(backend(&config, &system, &init_bad, 10, Mode::Dp).unwrap());
    let mut o = with_radii(backend(&config, &system, &init_ok, 10, Mode::Dp).unwrap());
    a.step(30).unwrap();
    o.step(30).unwrap();
    let (c30, o30) = (a.chaos().unwrap().unwrap(), o.chaos().unwrap().unwrap());
    a.step(30).unwrap();
    o.step(30).unwrap();
    let (c60, o60) = (a.chaos().unwrap().unwrap(), o.chaos().unwrap().unwrap());
    let rep = dropped(&mut a);
    assert_eq!(rep.reason[n0], 1);
    assert_eq!(rep.reason[n0 + 1], drop_inside_moon(0));
    for i in 0..n0 {
        assert_eq!(c60.flags[i], 0);
        assert_eq!(
            c60.log_growth[i].to_bits(),
            o60.log_growth[i].to_bits(),
            "live {i}"
        );
        assert_eq!(
            c60.megno_mean[i].to_bits(),
            o60.megno_mean[i].to_bits(),
            "live {i}"
        );
    }
    for i in [n0, n0 + 1] {
        assert_eq!(c60.flags[i] & 2, 2, "flags bit 1 of lost particle {i}");
        assert_eq!(c30.flags[i] & 2, 2, "already flagged at step 30");
        assert_eq!(
            c60.log_growth[i].to_bits(),
            c30.log_growth[i].to_bits(),
            "L {i}"
        );
        assert_eq!(
            c60.megno_mean[i].to_bits(),
            c30.megno_mean[i].to_bits(),
            "Y {i}"
        );
    }
    assert!(
        o60.log_growth[5].to_bits() != o30.log_growth[5].to_bits(),
        "live ones go on"
    );
}

/// Freeze with the arc: a frozen particle is not wrapped (it keeps its state
/// bit for bit), the others wrap (some are rotated by the arc width and
/// differ from a full-ring run), and they are as in a run without the loss.
#[test]
#[ignore = "needs a GPU"]
fn g3_freeze_with_the_arc_skips_the_frozen_particle() {
    if !have_gpu() {
        return;
    }
    let steps = 300u64;
    let system = system(H_S, steps, true);
    let (base, _) = ensemble(&system, 240);
    let bad = [unbound_at_start(&system, &base, 5)];
    let n0 = base.particles.len();
    let init_bad = init_with(&bad, &base);
    let benign = [(
        base.particles.position_km(5),
        base.particles.velocity_km_s(5),
    )];
    let init_ok = init_with(&benign, &base);
    let width = 2.0;
    let arc_cfg = cfg(steps, LOSS_FREEZE, Some(width));
    let mut a = backend(&arc_cfg, &system, &init_bad, 50, Mode::Dp).unwrap();
    let start = snapshot(&mut a);
    a.step(steps).unwrap();
    let rep = dropped(&mut a);
    assert_eq!((rep.reason[n0], rep.drop_step[n0]), (1, 0));
    let sa = snapshot(&mut a);
    assert_eq!(
        sa[n0], start[n0],
        "the frozen particle is not wrapped or moved"
    );
    let mut o = backend(&arc_cfg, &system, &init_ok, 50, Mode::Dp).unwrap();
    o.step(steps).unwrap();
    let so = snapshot(&mut o);
    for i in 0..n0 {
        assert_eq!(
            sa[i], so[i],
            "particle {i} is as in the run without the loss"
        );
    }
    // The same (already folded) particles on the full ring: the ones that the
    // arc rotated during the run differ, the others have the same bits.
    let so0 = snapshot(&mut backend(&arc_cfg, &system, &init_ok, 50, Mode::Dp).unwrap());
    let mut folded = init_ok.clone();
    #[allow(clippy::needless_range_loop)] // six parallel arrays
    for i in 0..n0 {
        folded.particles.x_km[i] = f64::from_bits(so0[i][0]);
        folded.particles.y_km[i] = f64::from_bits(so0[i][1]);
        folded.particles.z_km[i] = f64::from_bits(so0[i][2]);
        folded.particles.vx_km_s[i] = f64::from_bits(so0[i][3]);
        folded.particles.vy_km_s[i] = f64::from_bits(so0[i][4]);
        folded.particles.vz_km_s[i] = f64::from_bits(so0[i][5]);
    }
    let mut full = backend(&cfg(steps, "", None), &system, &folded, 50, Mode::Dp).unwrap();
    full.step(steps).unwrap();
    let sf = snapshot(&mut full);
    let rotated: Vec<f64> = (0..n0)
        .filter(|i| sa[*i] != sf[*i])
        .map(|i| {
            let (a, f) = (f64::from_bits(sa[i][0]), f64::from_bits(sf[i][0]));
            (a - f).abs()
        })
        .collect();
    println!(
        "{} of {n0} live particles were wrapped by the arc",
        rotated.len()
    );
    assert!(!rotated.is_empty() && rotated.len() < n0);
}
