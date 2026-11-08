//! ICR-004 tests C4 to C7: sub-stepping.

use fring_cpu::BLOCK_STEPS;
use fring_cpu::CpuBackend;
use fring_cpu::extended::{ExtParams, LEVEL_UNKNOWN, LevelHook, advance_ext_hooked};
use fring_cpu::integrate::State;
use fring_model::config::Distribution;
use fring_model::physics::{KickTable, SaturnField, build_ephemeris, generate_particles};
use fring_model::{Backend, InitialState, Particles, RunConfig, System};

const H: f64 = 360.0;
const RADII: [f64; 2] = [43.09, 40.63];
const SUB: &str = "[substep]\naccuracy = 0.1\nmax_level = 4\n";

fn system() -> System {
    let saturn =
        fring_model::params::saturn_from_toml_str(include_str!("../../../params/saturn.toml"))
            .unwrap();
    let all = fring_model::params::moons_from_toml_str(include_str!("../../../params/moons.toml"))
        .unwrap();
    let moons = build_ephemeris(&saturn, &all, 1.0e7, 1800.0).unwrap();
    System { saturn, moons }
}

fn config(extra: &str, chaos: bool) -> RunConfig {
    let text = include_str!("../../fring-model/tests/data/example-run.toml");
    let mut t = text.replace("duration_years = 1.0", "duration_years = 0.3");
    if chaos {
        t += "\n[diagnostics.chaos]\nmethod = \"shadow\"\noffset_km = 1e-3\nrenorm_steps = 100\n";
    }
    RunConfig::from_toml_str(&format!("{t}\n{extra}")).unwrap()
}

fn band(sys: &System, count: u32, lo: f64, hi: f64) -> InitialState {
    let field = SaturnField::<f64>::from_params(&sys.saturn);
    let dist = Distribution::Band {
        min_semi_major_axis_km: lo,
        max_semi_major_axis_km: hi,
    };
    InitialState {
        time_s: 0.0,
        particles: generate_particles(&field, &dist, count, 20_261_108).unwrap(),
    }
}

fn bits(p: &Particles) -> Vec<u64> {
    [
        &p.x_km, &p.y_km, &p.z_km, &p.vx_km_s, &p.vy_km_s, &p.vz_km_s,
    ]
    .iter()
    .flat_map(|v| v.iter().map(|x| x.to_bits()))
    .collect()
}

/// Particles on circular orbits near Prometheus's orbit, ahead of the moon on
/// the outside and behind it on the inside, so that the moon overtakes them
/// within about 100 days at radial offsets of 20 to 150 km.
fn encounter_set(sys: &System, count: usize) -> InitialState {
    let field = SaturnField::<f64>::from_params(&sys.saturn);
    let m = &sys.moons.tracks[0].samples[0];
    let (mp, r0) = (m.position_km, m.position_km.norm());
    let mut x = 0x2545_f491_4f6c_dd1du64;
    let mut u = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut p = Particles::with_capacity(count);
    for id in 0..count {
        let dr = (20.0 + 130.0 * u()) * if u() < 0.5 { 1.0 } else { -1.0 };
        let phi = (0.04 + 0.26 * u()) * dr.signum();
        let r = r0 + dr;
        let (c, sn) = (phi.cos(), phi.sin());
        let rad = fring_model::Vec3::new(
            (mp.x * c - mp.y * sn) / r0,
            (mp.x * sn + mp.y * c) / r0,
            0.0,
        );
        let v = fring_model::physics::circular_speed(&field, r);
        let tan = fring_model::Vec3::new(-rad.y, rad.x, 0.0);
        p.push(
            id as u32,
            rad * r + fring_model::Vec3::new(0.0, 0.0, 0.3 * (u() - 0.5)),
            tan * v,
        );
    }
    InitialState {
        time_s: 0.0,
        particles: p,
    }
}

/// Calls of `block` steps (the last call is shorter) that add up to `total`.
fn calls(total: u64, block: u64) -> Vec<u64> {
    let mut v = vec![block; (total / block) as usize];
    if !total.is_multiple_of(block) {
        v.push(total % block);
    }
    v
}

fn run_backend(
    cfg: &RunConfig,
    sys: &System,
    ini: &InitialState,
    calls: &[u64],
    threads: usize,
) -> (
    Particles,
    Option<fring_model::ChaosReport>,
    Option<fring_model::DropReport>,
) {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    pool.install(|| {
        let mut b = CpuBackend::new(cfg, sys, ini).unwrap();
        b.set_moon_radii_km(&RADII).unwrap();
        for &n in calls {
            b.step(n).unwrap();
        }
        let p = b.particles().unwrap().particles.clone();
        (p, b.chaos().unwrap(), b.dropped().unwrap())
    })
}

/// Result of the serial loop over `advance_ext_hooked`.
struct Serial {
    state: Vec<State>,
    shadow: Vec<State>,
    reason: Vec<u8>,
    drop_step: Vec<u64>,
    trace: Vec<Vec<u8>>,
}

/// The serial loop: the blocks of the backend (calls of at most
/// `BLOCK_STEPS`), one particle at a time, the levels recorded.
#[allow(clippy::needless_range_loop)]
fn run_serial(sys: &System, ini: &InitialState, calls: &[u64], offset: Option<f64>) -> Serial {
    let field = SaturnField::<f64>::from_params(&sys.saturn);
    let p = &ini.particles;
    let n = p.len();
    let state: Vec<State> = (0..n)
        .map(|i| {
            [
                p.x_km[i],
                p.y_km[i],
                p.z_km[i],
                p.vx_km_s[i],
                p.vy_km_s[i],
                p.vz_km_s[i],
            ]
        })
        .collect();
    let shadow: Vec<State> = state
        .iter()
        .map(|s| {
            let d = offset.unwrap_or(0.0);
            [s[0] + d, s[1], s[2], s[3], s[4], s[5]]
        })
        .collect();
    let mut out = Serial {
        state,
        shadow,
        reason: vec![0; n],
        drop_step: vec![u64::MAX; n],
        trace: vec![Vec::new(); n],
    };
    let mut level = vec![LEVEL_UNKNOWN; n];
    let mut table = KickTable::<f64>::for_ephemeris(&sys.moons);
    let mut done = 0u64;
    for &c in calls {
        let mut left = c;
        while left > 0 {
            let mut block = left.min(BLOCK_STEPS);
            if offset.is_some() {
                // Chaos: blocks end at the renormalisation steps (K = 100).
                block = block.min(100 - done % 100);
            }
            table
                .fill_ext(&sys.moons, ini.time_s, H, done, block as usize)
                .unwrap();
            let prm = ExtParams {
                field: &field,
                table: &table,
                h_s: H,
                first_step: done,
                radii_km: &RADII,
                radius_rule: false,
                substep: Some((0.1, 4)),
            };
            for i in 0..n {
                let sh = offset.map(|_| &mut out.shadow[i]);
                advance_ext_hooked(
                    &prm,
                    &mut out.state[i],
                    sh,
                    &mut out.reason[i],
                    &mut out.drop_step[i],
                    &mut level[i],
                    LevelHook::Record(&mut out.trace[i]),
                );
            }
            done += block;
            left -= block;
        }
    }
    out
}

/// C4: with `[substep]` on and particles far from the moons, the bits are
/// those of a run with it off, for block lengths 4096, 1000 and 37.
#[test]
fn c4_level_zero_is_today() {
    let sys = system();
    let ini = band(&sys, 60, 137_000.0, 138_000.0);
    for block in [4096u64, 1000, 37] {
        let c = calls(4200, block);
        let (off, _, _) = {
            let mut b = CpuBackend::new(&config("", false), &sys, &ini).unwrap();
            for &n in &c {
                b.step(n).unwrap();
            }
            (b.particles().unwrap().particles.clone(), (), ())
        };
        let (on, _, _) = run_backend(&config(SUB, false), &sys, &ini, &c, 4);
        assert_eq!(bits(&off), bits(&on), "block length {block}");
        let s = run_serial(&sys, &ini, &c, None);
        assert!(
            s.trace.iter().all(|t| t.iter().all(|&l| l == 0)),
            "levels are 0"
        );
        assert!(s.reason.iter().all(|&r| r == 0));
    }
}

/// The runs of C5: 60 particles with encounters, 100 days, three block lengths.
fn c5_runs(sys: &System) -> (usize, Vec<Serial>) {
    let ini = encounter_set(sys, 60);
    let total = 24_000u64; // 100 days
    let runs = [4096u64, 1000, 37]
        .iter()
        .map(|&b| run_serial(sys, &ini, &calls(total, b), None))
        .collect();
    (ini.particles.len(), runs)
}

/// Level agreement, and the position differences of the live particles
/// (sorted), against the run with blocks of 4096.
fn c5_numbers(n: usize, runs: &[Serial]) -> (f64, usize, usize, Vec<f64>) {
    let (mut same, mut all, mut live, mut n_enc) = (0usize, 0usize, 0usize, 0usize);
    let mut diffs = Vec::new();
    for i in 0..n {
        if runs.iter().any(|r| r.reason[i] != 0) {
            continue;
        }
        live += 1;
        n_enc += usize::from(runs[0].trace[i].iter().any(|&l| l > 0));
        for r in &runs[1..] {
            assert_eq!(r.trace[i].len(), runs[0].trace[i].len());
            all += r.trace[i].len();
            same += r.trace[i]
                .iter()
                .zip(&runs[0].trace[i])
                .filter(|(a, b)| a == b)
                .count();
            diffs.push(
                (0..3)
                    .map(|c| (r.state[i][c] - runs[0].state[i][c]).powi(2))
                    .sum::<f64>()
                    .sqrt(),
            );
        }
    }
    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (same as f64 / all as f64, live, n_enc, diffs)
}

/// C5 (ICR-004 amendment 2): with encounters, other block lengths give the
/// same levels in at least 99.9 % of the steps, and the median position
/// difference after 100 days is at most 1e-3 km. The 90th percentile and the
/// maximum are printed, not limits: particles with close encounters are
/// chaotic (a change of 1e-12 km in one start coordinate gives 4e-5 km after
/// 100 days), so the maximum measures the chaos and not the code. A mistake at
/// the block ends would move every particle, so the median finds it.
#[test]
#[ignore = "release: 100 days, 60 particles, three block lengths"]
fn c5_block_length() {
    let sys = system();
    let (n, runs) = c5_runs(&sys);
    let (frac, live, n_enc, diffs) = c5_numbers(n, &runs);
    let q = |f: f64| diffs[((diffs.len() - 1) as f64 * f) as usize];
    eprintln!(
        "C5: {live} live of {n}, {n_enc} with L>0; levels agree in {frac:.6} of the steps; \
         position difference (km) median {:e}, p90 {:e}, max {:e}",
        q(0.5),
        q(0.9),
        q(1.0)
    );
    assert!(live >= 30 && n_enc >= 5, "the test has no power");
    assert!(frac >= 0.999, "levels agree in {frac}");
    assert!(q(0.5) <= 1e-3, "median position difference {} km", q(0.5));
}

/// C6: bit for bit the same for 1, 4 and 8 threads and call splits, and equal
/// to the serial loop, with `[substep]` on (and the chaos shadow).
#[test]
fn c6_threads() {
    let sys = system();
    let ini = encounter_set(&sys, 40);
    let c = [1500u64, 1000, 1500];
    let cfg = config(SUB, true);
    let a = run_backend(&cfg, &sys, &ini, &c, 1);
    for t in [4, 8] {
        let b = run_backend(&cfg, &sys, &ini, &c, t);
        assert_eq!(bits(&a.0), bits(&b.0), "{t} threads");
        assert_eq!(
            a.1.as_ref().unwrap().log_growth,
            b.1.as_ref().unwrap().log_growth
        );
    }
    // The serial loop with the same blocks.
    let s = run_serial(&sys, &ini, &c, Some(1e-3));
    for i in 0..ini.particles.len() {
        assert_eq!(a.0.x_km[i].to_bits(), s.state[i][0].to_bits());
        assert_eq!(a.0.vz_km_s[i].to_bits(), s.state[i][5].to_bits());
    }
    // And some particles really were sub-stepped.
    assert!(s.trace.iter().any(|t| t.iter().any(|&l| l > 0)));
}

/// Particles 45 to 80 km radially from Prometheus at the start, within 300 km
/// along the orbit: a close encounter in the first steps.
fn close_set(sys: &System, count: usize) -> Vec<State> {
    let field = SaturnField::<f64>::from_params(&sys.saturn);
    let m = &sys.moons.tracks[0].samples[0];
    let r0 = m.position_km.norm();
    let rad = m.position_km / r0;
    let tan = fring_model::Vec3::new(-rad.y, rad.x, 0.0);
    (0..count)
        .map(|i| {
            let f = i as f64 / count as f64;
            let dr = (45.0 + 35.0 * f) * if i % 2 == 0 { 1.0 } else { -1.0 };
            let ds = (-300.0 + 600.0 * ((i * 7 % count) as f64 / count as f64)) / 1.0;
            let r = r0 + dr;
            let v = fring_model::physics::circular_speed(&field, r);
            let p = rad * r + tan * ds;
            let w = tan * v;
            [p.x, p.y, p.z, w.x, w.y, w.z]
        })
        .collect()
}

/// C7: the shadow uses the levels of its main particle. The main particle
/// does not depend on the shadow. A shadow with its own levels would give
/// other bits.
#[test]
fn c7_shadow_follows_the_level_of_its_main_particle() {
    let sys = system();
    let field = SaturnField::<f64>::from_params(&sys.saturn);
    let n = 600usize;
    let mut table = KickTable::<f64>::for_ephemeris(&sys.moons);
    table.fill_ext(&sys.moons, 0.0, H, 0, n).unwrap();
    let prm = ExtParams {
        field: &field,
        table: &table,
        h_s: H,
        first_step: 0,
        radii_km: &RADII,
        radius_rule: false,
        substep: Some((0.1, 4)),
    };
    let (mut with_levels, mut own_differs, mut checked) = (0, 0, 0);
    for s0 in close_set(&sys, 16) {
        let off = {
            let r = fring_model::Vec3::new(s0[0], s0[1], s0[2]);
            r * (30.0 / r.norm())
        };
        let sh0 = [
            s0[0] + off.x,
            s0[1] + off.y,
            s0[2] + off.z,
            s0[3],
            s0[4],
            s0[5],
        ];
        let (mut s, mut sh) = (s0, sh0);
        let (mut re, mut dr, mut lv) = (0u8, u64::MAX, LEVEL_UNKNOWN);
        let mut trace = Vec::new();
        advance_ext_hooked(
            &prm,
            &mut s,
            Some(&mut sh),
            &mut re,
            &mut dr,
            &mut lv,
            LevelHook::Record(&mut trace),
        );
        if re != 0 {
            continue;
        }
        checked += 1;
        // The main particle alone.
        let mut s2 = s0;
        let (mut re2, mut dr2, mut lv2) = (0u8, u64::MAX, LEVEL_UNKNOWN);
        let mut t2 = Vec::new();
        advance_ext_hooked(
            &prm,
            &mut s2,
            None,
            &mut re2,
            &mut dr2,
            &mut lv2,
            LevelHook::Record(&mut t2),
        );
        assert_eq!(s.map(f64::to_bits), s2.map(f64::to_bits));
        assert_eq!(trace, t2);
        // The shadow alone, with the levels of the main particle.
        let mut s3 = sh0;
        let (mut re3, mut dr3, mut lv3) = (0u8, u64::MAX, LEVEL_UNKNOWN);
        advance_ext_hooked(
            &prm,
            &mut s3,
            None,
            &mut re3,
            &mut dr3,
            &mut lv3,
            LevelHook::Replay(&trace),
        );
        assert_eq!(sh.map(f64::to_bits), s3.map(f64::to_bits));
        // The shadow alone, with its own levels.
        let mut s4 = sh0;
        let (mut re4, mut dr4, mut lv4) = (0u8, u64::MAX, LEVEL_UNKNOWN);
        let mut t4 = Vec::new();
        advance_ext_hooked(
            &prm,
            &mut s4,
            None,
            &mut re4,
            &mut dr4,
            &mut lv4,
            LevelHook::Record(&mut t4),
        );
        with_levels += usize::from(trace.iter().any(|&l| l > 0));
        if t4 != trace {
            own_differs += 1;
            assert_ne!(sh.map(f64::to_bits), s4.map(f64::to_bits));
        }
    }
    eprintln!(
        "C7: {checked} checked, {with_levels} sub-stepped, {own_differs} where own levels differ"
    );
    assert!(checked >= 8 && with_levels >= 5, "the test has no power");
    assert!(
        own_differs >= 1,
        "the test has no power: shadow levels never differ"
    );
}
