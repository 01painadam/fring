//! E1-GPU, test G1 (ICR-004 section 4, amendment 2): the f32 twins of the
//! radius rule against the generic functions of `fring-model`
//! (`moon_radius::segment_min_d2`, `drift_hits_moon`) with `T = f32` and
//! `T = f64`. No GPU is needed, so these are normal tests.
//! `cargo test -p fring-gpu --test twins_loss`.

mod common;

use common::*;
use fring_gpu::spike::{Consts, kick_scale, state_from_f64};
use fring_gpu::wf32::model::M4;
use fring_gpu::wh::{self, HostTable, Mode};
use fring_model::physics::moon_radius::{drift_hits_moon, segment_min_d2};
use fring_model::physics::{KickTable, interpolate_moons};
use fring_model::{System, Vec3};

/// Deterministic pseudo-random numbers in [0, 1) (SplitMix64), so that the test
/// needs no dependency and no seed argument.
struct Sm(u64);
impl Sm {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (z ^ (z >> 31)) as f64 / 18_446_744_073_709_551_616.0
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next()
    }
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(f64::MIN_POSITIVE)
}

/// Two measures, because `d2 = |a + s (b - a)|²` cancels when the segment
/// passes close to the moon centre: the f32 result then has an absolute error of
/// about `eps (|a|² + |b|²)` and a relative error that grows as `d2` falls.
/// (a) The plain relative error must be below 1e-6 whenever there is no heavy
/// cancellation (`d2 >= 0.1 max(|a|², |b|²)`; this includes every pass near the
/// radii of the moons). (b) The error scaled by `|a|² + |b|²` must be below
/// 1e-6 in every case. Even the generic function in f32 against the same in f64
/// differs by 2.5e-6 relative at `d2 = 4 km²` and `|a|² = 8,000 km²`.
#[test]
fn segment_min_d2_twin_against_the_generic_function_in_f32_and_f64() {
    let mut rng = Sm(7);
    let (mut rel32, mut rel64, mut sc32, mut sc64, mut n_rel) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0);
    let cases = 20_000;
    for _ in 0..cases {
        // Relative positions of a drift near a moon: tens to a few hundred km.
        let a = [
            rng.range(-200.0, 200.0),
            rng.range(-200.0, 200.0),
            rng.range(-20.0, 20.0),
        ];
        let b = [
            a[0] + rng.range(-120.0, 120.0),
            a[1] + rng.range(-120.0, 120.0),
            a[2] + rng.range(-20.0, 20.0),
        ];
        let (a32, b32) = (a.map(|x| x as f32), b.map(|x| x as f32));
        let twin = f64::from(wh::segment_min_d2::<M4>(a32, b32));
        let g32 = f64::from(segment_min_d2::<f32>(a32, b32));
        // The generic function in f64 on the same (f32 rounded) inputs.
        let g64 = segment_min_d2::<f64>(a32.map(f64::from), b32.map(f64::from));
        let sq = |x: [f32; 3]| x.iter().map(|c| f64::from(*c).powi(2)).sum::<f64>();
        let (sa, sb) = (sq(a32), sq(b32));
        let scale = sa + sb;
        sc32 = sc32.max((twin - g32).abs() / scale);
        sc64 = sc64.max((twin - g64).abs() / scale);
        if g64 >= 0.1 * sa.max(sb) {
            rel32 = rel32.max(rel(twin, g32));
            rel64 = rel64.max(rel(twin, g64));
            n_rel += 1;
        }
    }
    eprintln!(
        "segment_min_d2, {cases} cases ({n_rel} without heavy cancellation): relative error to f32 {rel32:.2e}, to f64 {rel64:.2e}; scaled error to f32 {sc32:.2e}, to f64 {sc64:.2e}"
    );
    assert!(n_rel > 1000);
    assert!(rel32 < 1e-6 && rel64 < 1e-6, "{rel32} {rel64}");
    assert!(sc32 < 1e-6 && sc64 < 1e-6, "{sc32} {sc64}");
}

#[test]
fn segment_min_d2_twin_hand_cases_equal_the_generic_function() {
    let cases: [([f32; 3], [f32; 3], f32); 4] = [
        ([-70.0, 20.0, 0.0], [70.0, 20.0, 0.0], 400.0),
        ([-70.0, 20.0, 0.0], [-30.0, 20.0, 0.0], 1300.0),
        ([3.0, 4.0, 0.0], [3.0, 4.0, 0.0], 25.0),
        ([0.0, 0.0, 0.0], [10.0, 0.0, 0.0], 0.0),
    ];
    for (a, b, want) in cases {
        let twin = wh::segment_min_d2::<M4>(a, b);
        let g32 = segment_min_d2::<f32>(a, b);
        let g64 = segment_min_d2::<f64>(a.map(f64::from), b.map(f64::from));
        assert!(
            (f64::from(twin) - g64).abs() <= 1e-6 * g64.max(1.0),
            "{a:?} {b:?}"
        );
        assert!(
            (f64::from(g32) - f64::from(want)).abs() <= 1e-3,
            "{a:?} {b:?}"
        );
    }
}

#[test]
fn drift_hits_moon_passes_at_radius_plus_and_minus_1_km_for_both_moons() {
    for (name, r) in [("prometheus", 43.09f32), ("pandora", 40.63f32)] {
        for (dy, hit) in [(1.0f32, false), (-1.0, true)] {
            let a = [-60.0, r + dy, 0.0];
            let b = [60.0, r + dy, 0.0];
            let g32 = drift_hits_moon::<f32>(a, b, r);
            let g64 = drift_hits_moon::<f64>(a.map(f64::from), b.map(f64::from), f64::from(r));
            // The twin has no separate boolean: the rule is d2 < r2.
            let twin = wh::segment_min_d2::<M4>(a, b) < r * r;
            assert_eq!((g32, g64, twin), (hit, hit, hit), "{name} dy {dy}");
        }
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

#[test]
fn advance_dropping_agrees_with_the_generic_rule_on_what_it_found() {
    let n = 12usize;
    let system = system(H_S, n as u64, true);
    let gms: Vec<f64> = system.moons.tracks.iter().map(|t| t.gm_km3_s2).collect();
    let c = Consts::<M4>::for_step(&system.saturn, &gms, H_S);
    let mut table = KickTable::<f64>::for_ephemeris(&system.moons);
    table.fill(&system.moons, 0.0, H_S, 0, n).unwrap();
    let tab = HostTable::<M4>::from_kick_table(&table, kick_scale(H_S));
    let rows = wh::sync_rows(&system.moons, 0.0, H_S, 0, n);
    let sync = wh::sync_moons::<M4>(&rows, 2);
    let radii = [43.09f64, 40.63];
    let r2: Vec<f32> = radii.iter().map(|r| (r * r) as f32).collect();
    let (mut hits, mut clear, mut skipped) = (0, 0, 0);
    for mode in [Mode::D, Mode::Dp] {
        for k in 0..2usize {
            for j in -30..=30 {
                let (x, v) = crafted(&system, k, 0.4, 3.0 * f64::from(j), 5.0 * H_S);
                let mut s = state_from_f64::<M4>(x, v);
                let o = wh::advance_dropping(mode, &c, &tab, &sync, &r2, &mut s);
                if o.reason >= 2 {
                    let kk = (o.reason - 2) as usize;
                    let [a, b] = o.hit_rel_km;
                    let rk = radii[kk];
                    // The distance of the ends agrees with the segment rule, and
                    // the generic function in f32 and f64 says "hit" too, unless
                    // the pass is within 1e-5 of the radius (rounding).
                    let d2 = f64::from(segment_min_d2::<f32>(a, b));
                    if rel(d2, rk * rk) < 1e-5 {
                        skipped += 1;
                        continue;
                    }
                    assert!(
                        drift_hits_moon::<f32>(a, b, rk as f32),
                        "f32 {mode:?} {k} {j}"
                    );
                    assert!(
                        drift_hits_moon::<f64>(a.map(f64::from), b.map(f64::from), rk),
                        "f64 {mode:?} {k} {j}"
                    );
                    hits += 1;
                } else {
                    // Not dropped: no checked drift came within the radius.
                    for (m, r) in radii.iter().enumerate() {
                        let d = f64::from(o.min_d2_km2[m]);
                        assert!(d >= r * r * (1.0 - 1e-5), "{mode:?} {k} {j} moon {m}: {d}");
                    }
                    clear += 1;
                }
            }
        }
    }
    eprintln!(
        "advance_dropping: {hits} hits confirmed, {clear} clear, {skipped} within 1e-5 of the radius"
    );
    assert!(
        hits >= 20 && clear >= 20,
        "the crafted set must exercise both outcomes"
    );
}
