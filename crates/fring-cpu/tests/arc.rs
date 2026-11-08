//! D1-CPU: the co-moving arc on the CPU backend (`domain.kind = "arc"`).
//!
//! Primary test: the wrap changes neither E nor Lz (PLAN section 9, D1). It
//! compares both just before and just after the wrap call, at the same time,
//! so that no integrator error is in the numbers.
#![allow(clippy::needless_range_loop)]

use fring_cpu::CpuBackend;
use fring_cpu::collide::CollisionState;
use fring_cpu::integrate::State;
use fring_cpu::wrap_states;
use fring_model::config::{Distribution, Domain};
use fring_model::physics::{
    ArcGeometry, SaturnField, build_ephemeris, generate_particles, reference_mean_motion,
};
use fring_model::{
    Backend, Error, InitialState, MoonEphemeris, Particles, RunConfig, System, Vec3,
};

const A0: f64 = 140_220.0;

fn saturn() -> fring_model::SaturnParams {
    fring_model::params::saturn_from_toml_str(include_str!("../../../params/saturn.toml")).unwrap()
}

fn system() -> System {
    System {
        saturn: saturn(),
        moons: MoonEphemeris::none(),
    }
}

fn field() -> SaturnField<f64> {
    SaturnField::<f64>::from_params(&saturn())
}

fn config(width_deg: f64, extra: &str) -> RunConfig {
    let text = include_str!("../../fring-model/tests/data/example-run.toml");
    let t = text
        .replace("duration_years = 1.0", "duration_years = 0.1")
        .replace(r#"include = ["prometheus", "pandora"]"#, "include = []")
        .replace(
            r#"kind = "full_ring"              # or"#,
            &format!("kind = \"arc\"\nwidth_deg = {width_deg}\ncentre_longitude_deg = 0.0\n# or"),
        );
    RunConfig::from_toml_str(&format!("{t}\n{extra}")).unwrap()
}

fn ring_config(extra: &str) -> RunConfig {
    let mut c = config(10.0, extra);
    c.domain = Domain::FullRing;
    c
}

fn band(count: u32, lo: f64, hi: f64) -> InitialState {
    let dist = Distribution::Band {
        min_semi_major_axis_km: lo,
        max_semi_major_axis_km: hi,
    };
    InitialState {
        time_s: 0.0,
        particles: generate_particles(&field(), &dist, count, 20_261_108).unwrap(),
    }
}

fn geom(width_deg: f64) -> ArcGeometry {
    ArcGeometry::from_domain_in_field(
        &Domain::Arc {
            width_deg,
            centre_longitude_deg: 0.0,
        },
        &field(),
        A0,
    )
    .unwrap()
    .unwrap()
}

fn bits(p: &Particles) -> Vec<u64> {
    [
        &p.x_km, &p.y_km, &p.z_km, &p.vx_km_s, &p.vy_km_s, &p.vz_km_s,
    ]
    .iter()
    .flat_map(|v| v.iter().map(|x| x.to_bits()))
    .collect()
}

fn lz(s: &State) -> f64 {
    s[0] * s[4] - s[1] * s[3]
}

fn energy(f: &SaturnField<f64>, s: &State) -> f64 {
    f.specific_energy(Vec3::new(s[0], s[1], s[2]), Vec3::new(s[3], s[4], s[5]))
}

/// PLAN D1: E and Lz are the same just before and just after the wrap.
#[test]
fn wrap_keeps_e_and_lz() {
    let f = field();
    let g = geom(0.5);
    let init = band(300, 139_800.0, 140_600.0);
    let h = 360.0;
    let ids = init.particles.id.clone();
    // Start inside the arc (fold), then integrate step by step in Saturn's
    // field and wrap after each step, as the backend does.
    let mut p = Particles::with_capacity(300);
    for i in 0..300 {
        let w = g
            .fold(
                init.particles.position_km(i),
                init.particles.velocity_km_s(i),
                0.0,
            )
            .unwrap();
        p.push(ids[i], w.pos_km, w.vel_km_s);
    }
    let (mut worst_e, mut worst_l, mut events) = (0.0f64, 0.0f64, 0u64);
    for step in 1..=3000u64 {
        fring_cpu::integrate::step_particles(&f, &mut p, h, 1).unwrap();
        let mut st: Vec<State> = (0..p.len())
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
        let before = st.clone();
        wrap_states(&g, step as f64 * h, step, &ids, &mut st, None, None).unwrap();
        for i in 0..st.len() {
            if st[i] == before[i] {
                continue;
            }
            events += 1;
            worst_e = worst_e.max(
                ((energy(&f, &st[i]) - energy(&f, &before[i])) / energy(&f, &before[i])).abs(),
            );
            worst_l = worst_l.max(((lz(&st[i]) - lz(&before[i])) / lz(&before[i])).abs());
        }
        for (i, s) in st.iter().enumerate() {
            p.x_km[i] = s[0];
            p.y_km[i] = s[1];
            p.z_km[i] = s[2];
            p.vx_km_s[i] = s[3];
            p.vy_km_s[i] = s[4];
            p.vz_km_s[i] = s[5];
        }
    }
    println!("wrap events {events}: max |dE/E| {worst_e:.3e}, max |dLz/Lz| {worst_l:.3e}");
    assert!(events > 500, "too few wraps: {events}");
    assert!(worst_e < 1e-13 && worst_l < 1e-13);
}

/// The arc run against the full ring from the same folded start, turned by
/// whole widths. Reported, not the PLAN bound: the two runs round differently
/// in the integrator.
#[test]
fn arc_against_rotated_ring() {
    let f = field();
    let g = geom(0.5);
    let init = band(300, 139_800.0, 140_600.0);
    let mut folded = Particles::with_capacity(300);
    for i in 0..300 {
        let w = g
            .fold(
                init.particles.position_km(i),
                init.particles.velocity_km_s(i),
                0.0,
            )
            .unwrap();
        folded.push(init.particles.id[i], w.pos_km, w.vel_km_s);
    }
    let ring_init = InitialState {
        time_s: 0.0,
        particles: folded,
    };
    let mut arc = CpuBackend::new(&config(0.5, ""), &system(), &init).unwrap();
    let mut ring = CpuBackend::new(&ring_config(""), &system(), &ring_init).unwrap();
    arc.step(3000).unwrap();
    ring.step(3000).unwrap();
    let a = arc.particles().unwrap().particles.clone();
    let r = ring.particles().unwrap().particles.clone();
    let w = g.width_rad();
    let (mut dpos, mut de, mut dl, mut rotated) = (0.0f64, 0.0f64, 0.0f64, 0);
    for i in 0..a.len() {
        let sa = [
            a.x_km[i],
            a.y_km[i],
            a.z_km[i],
            a.vx_km_s[i],
            a.vy_km_s[i],
            a.vz_km_s[i],
        ];
        let sr = [
            r.x_km[i],
            r.y_km[i],
            r.z_km[i],
            r.vx_km_s[i],
            r.vy_km_s[i],
            r.vz_km_s[i],
        ];
        let m = ((sa[1].atan2(sa[0]) - sr[1].atan2(sr[0])) / w).round();
        rotated += usize::from(m != 0.0);
        let (c, s) = ((m * w).cos(), (m * w).sin());
        let (x, y) = (c * sr[0] - s * sr[1], s * sr[0] + c * sr[1]);
        dpos =
            dpos.max(((x - sa[0]).powi(2) + (y - sa[1]).powi(2) + (sr[2] - sa[2]).powi(2)).sqrt());
        de = de.max(((energy(&f, &sa) - energy(&f, &sr)) / energy(&f, &sr)).abs());
        dl = dl.max(((lz(&sa) - lz(&sr)) / lz(&sr)).abs());
    }
    println!(
        "arc vs rotated ring: {rotated}/300 rotated, max position {dpos:.3e} km, |dE/E| {de:.3e}, |dLz/Lz| {dl:.3e}"
    );
    assert!(rotated > 100);
    assert!(dpos < 1e-6 && de < 1e-12 && dl < 1e-12);
}

fn run(cfg: &RunConfig, init: &InitialState, calls: &[u64]) -> Vec<u64> {
    let mut b = CpuBackend::new(cfg, &system(), init).unwrap();
    for &n in calls {
        b.step(n).unwrap();
    }
    bits(b.particles().unwrap().particles)
}

#[test]
fn threads_and_call_splits_are_bit_identical() {
    let init = band(500, 139_800.0, 140_600.0);
    let cfg = config(0.5, "");
    let whole = run(&cfg, &init, &[1500]);
    for calls in [vec![700, 800], vec![1, 1499], vec![0, 3, 1497]] {
        assert_eq!(whole, run(&cfg, &init, &calls), "{calls:?}");
    }
    for n in [1, 4, 8] {
        let r = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .unwrap()
            .install(|| run(&cfg, &init, &[900, 600]));
        assert_eq!(whole, r, "{n} threads");
    }
}

/// A 360 degree arc never wraps and is the full ring (blocks of one step
/// only change the rounding).
#[test]
fn arc_360_stays_the_ring() {
    let init = band(100, 139_800.0, 140_600.0);
    let a = run(&config(360.0, ""), &init, &[400]);
    let r = run(&ring_config(""), &init, &[400]);
    let (fa, fr) = (
        a.iter().map(|b| f64::from_bits(*b)).collect::<Vec<_>>(),
        r.iter().map(|b| f64::from_bits(*b)).collect::<Vec<_>>(),
    );
    let worst = fa
        .iter()
        .zip(&fr)
        .map(|(x, y)| (x - y).abs() / y.abs().max(1e-30))
        .fold(0.0f64, f64::max);
    println!("360 deg arc against ring: max relative difference {worst:.3e}");
    assert!(worst < 1e-9);
}

/// The shadow gets the rotation of its main particle: the chaos indicator of
/// the arc run equals that of the ring run (a shadow wrapped alone would be
/// a whole width away).
#[test]
fn chaos_shadow_follows_the_wrap() {
    let chaos = "[diagnostics.chaos]\nmethod = \"shadow\"\noffset_km = 1e-3\nrenorm_steps = 100\n";
    let init = band(200, 139_800.0, 140_600.0);
    let mut arc = CpuBackend::new(&config(0.5, chaos), &system(), &init).unwrap();
    let mut ring = CpuBackend::new(&ring_config(chaos), &system(), &init).unwrap();
    arc.step(2000).unwrap();
    ring.step(2000).unwrap();
    let a = arc.chaos().unwrap().unwrap();
    let r = ring.chaos().unwrap().unwrap();
    let worst = a
        .log_growth
        .iter()
        .zip(&r.log_growth)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max);
    println!("chaos log growth, arc against ring: max difference {worst:.3e}");
    assert!(a.log_growth.iter().all(|l| l.is_finite() && l.abs() < 20.0));
    assert!(worst < 1e-3);
}

/// A step longer than `MAX_WRAPS` widths: error, or reason 1 with freeze.
#[test]
fn wrap_failure_errors_or_freezes() {
    let init = band(50, 139_000.0, 141_000.0);
    let err = CpuBackend::new(&config(1e-4, ""), &system(), &init)
        .and_then(|mut b| b.step(50).map(|_| b))
        .err()
        .expect("must fail");
    match err {
        Error::Backend(m) => assert!(m.contains("unbound or invalid orbit"), "{m}"),
        e => panic!("{e:?}"),
    }
    let cfg = config(1e-4, "[loss]\non_unbound = \"freeze\"\n");
    let mut b = CpuBackend::new(&cfg, &system(), &init).unwrap();
    b.step(50).unwrap();
    let d = b.dropped().unwrap().unwrap();
    println!("frozen by a failed wrap: {} of 50", d.n_unbound);
    assert!(d.n_unbound > 0);
    assert!(d.reason.iter().all(|&r| r <= 1));
    assert!(
        d.drop_step
            .iter()
            .zip(&d.reason)
            .all(|(&s, &r)| (r == 0) == (s == u64::MAX))
    );
    b.step(10).unwrap();
    assert_eq!(b.dropped().unwrap().unwrap().n_unbound, d.n_unbound);
}

#[test]
fn moons_and_arc_run() {
    let saturn = saturn();
    let m = fring_model::params::moons_from_toml_str(include_str!("../../../params/moons.toml"))
        .unwrap();
    let sys = System {
        moons: build_ephemeris(&saturn, &m, 40.0 * 86_400.0, 1800.0).unwrap(),
        saturn,
    };
    let init = band(200, 139_000.0, 141_500.0);
    let text = include_str!("../../fring-model/tests/data/example-run.toml");
    let cfg = RunConfig::from_toml_str(
        &text
            .replace("duration_years = 1.0", "duration_years = 0.1")
            .replace(
                r#"kind = "full_ring"              # or"#,
                "kind = \"arc\"\nwidth_deg = 3.0\ncentre_longitude_deg = 0.0\n# or",
            ),
    )
    .unwrap();
    let mut b = CpuBackend::new(&cfg, &sys, &init).unwrap();
    b.step(1000).unwrap();
    let g = geom(3.0);
    let t = b.time_s();
    let p = b.particles().unwrap().particles.clone();
    for i in 0..p.len() {
        let w = g
            .wrap(p.position_km(i), p.velocity_km_s(i), t)
            .expect("wraps");
        assert_eq!(w.k, 0, "particle {i} is outside the arc");
    }
}

fn stoch(cells: &str) -> String {
    format!(
        "[collisions]\nmodel = \"stochastic\"\nparticle_radius_km = 1.0e-5\nweight = 1.0e18\n\
         {cells}every_steps = 1\n[collisions.restitution]\nlaw = \"constant\"\nepsilon = 0.5\n"
    )
}

#[test]
fn stochastic_collisions_in_an_arc() {
    let text = include_str!("../../fring-model/tests/data/example-run.toml");
    let cfg = RunConfig::from_toml_str(
        &text
            .replace("duration_years = 1.0", "duration_years = 0.1")
            .replace(r#"include = ["prometheus", "pandora"]"#, "include = []")
            .replace(
                r#"kind = "full_ring"              # or"#,
                "kind = \"arc\"\nwidth_deg = 3.0\ncentre_longitude_deg = 0.0\n# or",
            )
            .replace(
                "[collisions]\nmodel = \"none\"",
                &stoch("cell_radial_km = 100.0\ncell_azimuthal_km = 1000.0\ncell_vertical_km = 100.0\n"),
            ),
    )
    .unwrap();
    let init = band(2000, 140_000.0, 140_400.0);
    let a = run(&cfg, &init, &[300]);
    assert_eq!(a, run(&cfg, &init, &[100, 200]));
    let mut b = CpuBackend::new(&cfg, &system(), &init).unwrap();
    b.step(300).unwrap();
    let s = b.stats().unwrap();
    println!("arc, stochastic: {} collisions", s.collisions_since_last);
    assert!(s.collisions_since_last > 0);
}

/// Hard spheres: a pair across the seam gives the same collision as the same
/// pair in the middle of the arc.
#[test]
fn hard_sphere_pair_across_the_seam() {
    let text = include_str!("../../fring-model/tests/data/example-run.toml");
    let cfg = RunConfig::from_toml_str(
        &text
            .replace("duration_years = 1.0", "duration_years = 0.1")
            .replace(r#"include = ["prometheus", "pandora"]"#, "include = []")
            .replace(
                r#"kind = "full_ring"              # or"#,
                "kind = \"arc\"\nwidth_deg = 10.0\ncentre_longitude_deg = 0.0\n# or",
            )
            .replace(
                "[collisions]\nmodel = \"none\"",
                "[collisions]\nmodel = \"hard_sphere\"\nparticle_radius_km = 0.05\n\
                 [collisions.restitution]\nlaw = \"constant\"\nepsilon = 0.5\n",
            ),
    )
    .unwrap();
    let f = field();
    let n0 = reference_mean_motion(&f, A0);
    let g = geom(10.0);
    let w = g.width_rad();
    let vc = (f.gm_km3_s2 / A0).sqrt();
    let at = |phi: f64, dv: f64| -> State {
        let v = vc + dv;
        [
            A0 * phi.cos(),
            A0 * phi.sin(),
            0.0,
            -v * phi.sin(),
            v * phi.cos(),
            0.0,
        ]
    };
    let rot = |s: &State, a: f64| -> State {
        let (c, sn) = (a.cos(), a.sin());
        [
            c * s[0] - sn * s[1],
            sn * s[0] + c * s[1],
            s[2],
            c * s[3] - sn * s[4],
            sn * s[3] + c * s[4],
            s[5],
        ]
    };
    // Gap 0.08 km < contact 0.1 km, approaching at 1 m/s.
    let gap = 0.08 / A0;
    let a_img = at(0.5 * w - 0.4 * gap, 1.0e-3);
    let b_img = at(0.5 * w + 0.6 * gap, 0.0);
    let collide = |sa: State, sb: State| -> (State, State, u64) {
        let mut st = vec![sa, sb];
        let mut cs = CollisionState::from_config(&cfg, &f).unwrap().unwrap();
        let t = 0.0;
        cs.step(&f, t, 1, cfg.time.step_s, &[0, 1], None, &mut st)
            .unwrap();
        (st[0], st[1], cs.tally.collisions)
    };
    // Seam: B is the image of b_img turned back by one width (inside the arc).
    let (sa, sb, n_seam) = collide(a_img, rot(&b_img, -w));
    // Middle: both turned by -w/2.
    let (ma, mb, n_mid) = collide(rot(&a_img, -0.5 * w), rot(&b_img, -0.5 * w));
    assert_eq!((n_seam, n_mid), (1, 1));
    // Compare the velocity changes in the frame of the middle pair.
    let dv = |after: &State, before: &State| [after[3] - before[3], after[4] - before[4]];
    let seam_a = dv(&rot(&sa, -0.5 * w), &rot(&a_img, -0.5 * w));
    let mid_a = dv(&ma, &rot(&a_img, -0.5 * w));
    let seam_b = dv(&rot(&rot(&sb, w), -0.5 * w), &rot(&b_img, -0.5 * w));
    let mid_b = dv(&mb, &rot(&b_img, -0.5 * w));
    let rel = |x: [f64; 2], y: [f64; 2]| ((x[0] - y[0]).hypot(x[1] - y[1])) / y[0].hypot(y[1]);
    let (ra, rb) = (rel(seam_a, mid_a), rel(seam_b, mid_b));
    println!(
        "seam pair against mid-arc pair: relative dv difference A {ra:.3e}, B {rb:.3e}; n0 {n0:.3e}"
    );
    assert!(ra < 2e-8 && rb < 2e-8);
}

/// Cost per step at 10^5 particles, arc (blocks of one step) against the full
/// ring (blocks of up to 4096), no moons. Run with
/// `cargo test -p fring-cpu --release --test arc -- --ignored --nocapture`.
#[test]
#[ignore]
fn arc_cost() {
    let init = band(100_000, 139_800.0, 140_600.0);
    for (name, cfg) in [("ring", ring_config("")), ("arc 5 deg", config(5.0, ""))] {
        let mut b = CpuBackend::new(&cfg, &system(), &init).unwrap();
        b.step(20).unwrap();
        let t = std::time::Instant::now();
        b.step(500).unwrap();
        let dt = t.elapsed().as_secs_f64();
        println!(
            "{name}: {:.3} ms per step ({:.1} ns per particle-step)",
            dt / 500.0 * 1e3,
            dt / 500.0 / 1e5 * 1e9
        );
    }
}
