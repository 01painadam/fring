//! K3-CPU: the hard-sphere collision step in `CpuBackend`. K3 uses no random
//! numbers, so the comparisons are exact (bit equality), except the energy
//! balance of one collision step (1e-9 relative).
//!
//! The particle radius is 50 m, much larger than real ring particles, so that
//! a small clump (600 particles, Gaussian width 6 km) has many collisions in
//! a few hundred steps.

use fring_cpu::CpuBackend;
use fring_cpu::collide::CollisionState;
use fring_model::config::Distribution;
use fring_model::physics::{SaturnField, generate_particles};
use fring_model::{
    Backend, Error, InitialState, MoonEphemeris, Particles, RunConfig, StepStats, System, Vec3,
};

const RADIUS_KM: f64 = 0.05;

fn system() -> System {
    let saturn =
        fring_model::params::saturn_from_toml_str(include_str!("../../../params/saturn.toml"))
            .unwrap();
    System {
        saturn,
        moons: MoonEphemeris::none(),
    }
}

fn config(collisions: &str, extra: &str) -> RunConfig {
    let text = include_str!("../../fring-model/tests/data/example-run.toml");
    let t = text
        .replace("duration_years = 1.0", "duration_years = 0.04")
        .replace(r#"include = ["prometheus", "pandora"]"#, "include = []")
        .replace("[collisions]\nmodel = \"none\"", collisions);
    RunConfig::from_toml_str(&format!("{t}\n{extra}")).unwrap()
}

fn hard_sphere(radius_km: f64) -> String {
    format!(
        "[collisions]\nmodel = \"hard_sphere\"\nparticle_radius_km = {radius_km}\n\
         [collisions.restitution]\nlaw = \"constant\"\nepsilon = 0.5\n"
    )
}

/// A clump of `count` particles around the core radius (the reference orbit
/// of the example config is 140 200 km).
fn clump(count: u32, sigma_km: f64) -> InitialState {
    let field = SaturnField::<f64>::from_params(&system().saturn);
    let dist = Distribution::Clump {
        semi_major_axis_km: 140_200.0,
        longitude_deg: 10.0,
        sigma_km,
    };
    InitialState {
        time_s: 0.0,
        particles: generate_particles(&field, &dist, count, 20_261_108).unwrap(),
    }
}

/// Smallest distance between two particles of `p` (brute force), in km.
fn min_distance_km(p: &Particles) -> f64 {
    let mut m = f64::INFINITY;
    for i in 0..p.len() {
        for j in 0..i {
            let d = ((p.x_km[i] - p.x_km[j]).powi(2)
                + (p.y_km[i] - p.y_km[j]).powi(2)
                + (p.z_km[i] - p.z_km[j]).powi(2))
            .sqrt();
            m = m.min(d);
        }
    }
    m
}

fn bits(p: &Particles) -> Vec<u64> {
    [
        &p.x_km, &p.y_km, &p.z_km, &p.vx_km_s, &p.vy_km_s, &p.vz_km_s,
    ]
    .iter()
    .flat_map(|v| v.iter().map(|x| x.to_bits()))
    .collect()
}

fn stats_bits(s: &StepStats) -> Vec<u64> {
    let c = &s.collision;
    vec![
        s.steps_total,
        s.collisions_since_last,
        c.energy_loss_km2_s2.to_bits(),
        c.candidates,
        c.occupied_cells,
        c.deep_collisions,
    ]
}

fn run(cfg: &RunConfig, init: &InitialState, calls: &[u64]) -> (Vec<u64>, Vec<u64>) {
    let mut b = CpuBackend::new(cfg, &system(), init).unwrap();
    for &n in calls {
        b.step(n).unwrap();
    }
    let s = b.stats().unwrap();
    (bits(b.particles().unwrap().particles), stats_bits(&s))
}

fn state_of(p: &Particles) -> Vec<[f64; 6]> {
    (0..p.len())
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
        .collect()
}

#[test]
fn hs_nonzero_stats() {
    let init = clump(600, 6.0);
    assert!(
        min_distance_km(&init.particles) > 2.0 * RADIUS_KM,
        "overlap"
    );
    let cfg = config(&hard_sphere(RADIUS_KM), "");
    let mut b = CpuBackend::new(&cfg, &system(), &init).unwrap();
    b.step(250).unwrap();
    let s = b.stats().unwrap();
    println!(
        "hard spheres: {} collisions, loss {:.3e} km2/s2, deep {}",
        s.collisions_since_last, s.collision.energy_loss_km2_s2, s.collision.deep_collisions
    );
    assert_eq!(s.steps_total, 250);
    assert!(s.collisions_since_last > 0, "no collisions");
    assert!(s.collision.energy_loss_km2_s2 > 0.0);
    assert!(s.max_rel_energy_error.is_none());
    // Not a stochastic-model field.
    assert_eq!(s.collision.candidates, 0);
    // The stats are since the previous call.
    let s2 = b.stats().unwrap();
    assert_eq!(s2.collisions_since_last, 0);
    assert_eq!(s2.collision.energy_loss_km2_s2, 0.0);
    // step(0) does nothing.
    let before = bits(b.particles().unwrap().particles);
    b.step(0).unwrap();
    assert_eq!(bits(b.particles().unwrap().particles), before);
}

/// The tally's energy loss equals the loss of kinetic energy across the
/// collision step alone (velocities just before and after, one step), summed
/// over snapshots of a run. The orbit's own energy change is not in it.
#[test]
fn hs_energy_balance_of_the_collision_step() {
    let init = clump(600, 6.0);
    let sys = system();
    let field = SaturnField::<f64>::from_params(&sys.saturn);
    let cfg = config(&hard_sphere(RADIUS_KM), "");
    // The orbits come from a run without collisions, so the pairs that overlap
    // at the end of a step are still there for the collision step.
    let free = config("[collisions]\nmodel = \"none\"", "");
    let mut b = CpuBackend::new(&free, &sys, &init).unwrap();
    let mut c = CollisionState::from_config(&cfg, &field).unwrap().unwrap();
    let (mut loss, mut dk, mut n) = (0.0, 0.0, 0u64);
    for step in 1..=250u64 {
        b.step(1).unwrap();
        let p = b.particles().unwrap().particles.clone();
        let before = state_of(&p);
        let mut after = before.clone();
        let t_s = step as f64 * 360.0;
        c.step(&field, t_s, step, 360.0, &p.id, None, &mut after)
            .unwrap();
        // Positions do not change; the kinetic energy loss per unit mass.
        for (a, b) in after.iter().zip(&before) {
            assert_eq!(a[..3], b[..3]);
            for k in 3..6 {
                dk += 0.5 * (b[k] - a[k]) * (b[k] + a[k]);
            }
        }
        let t = std::mem::take(&mut c.tally);
        loss += t.stats.energy_loss_km2_s2;
        n += t.collisions;
    }
    println!("{n} collisions, tally loss {loss:.6e}, kinetic energy loss {dk:.6e}");
    assert!(n > 0);
    assert!(((loss - dk) / loss).abs() <= 1e-9, "{loss} {dk}");
}

#[test]
fn hs_same_input_bit_identical() {
    let init = clump(600, 6.0);
    let cfg = config(&hard_sphere(RADIUS_KM), "");
    let a = run(&cfg, &init, &[200]);
    assert!(a.1[1] > 0);
    assert_eq!(a, run(&cfg, &init, &[200]));
    // The seed is not used by K3.
    let mut other = config(&hard_sphere(RADIUS_KM), "");
    other.run.seed = 7;
    assert_eq!(a, run(&other, &init, &[200]));
}

#[test]
fn hs_threads_1_4_8() {
    let init = clump(600, 6.0);
    let cfg = config(&hard_sphere(RADIUS_KM), "");
    let on = |n: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .unwrap()
            .install(|| run(&cfg, &init, &[120, 80]))
    };
    let one = on(1);
    assert!(one.1[1] > 0);
    assert_eq!(one, on(4));
    assert_eq!(one, on(8));
}

/// Every step is a collision step and a block of one step, so any split of
/// the calls gives the same bits.
#[test]
fn hs_call_splits() {
    let init = clump(600, 6.0);
    let cfg = config(&hard_sphere(RADIUS_KM), "");
    let whole = run(&cfg, &init, &[200]);
    assert!(whole.1[1] > 0);
    for calls in [vec![100, 100], vec![7, 193], vec![0, 6, 194], vec![1; 200]] {
        assert_eq!(whole, run(&cfg, &init, &calls), "{calls:?}");
    }
}

/// Frozen particles (unbound, `on_unbound = "freeze"`) stay out. Five frozen
/// particles are put where free particles will be at the end of step 1, in
/// front of their relative motion, so that they would be hit if they took part.
#[test]
fn hs_frozen_do_not_collide() {
    let sys = system();
    let field = SaturnField::<f64>::from_params(&sys.saturn);
    let base = clump(600, 6.0);
    let n_free = base.particles.len();
    let freeze = "[loss]\non_unbound = \"freeze\"\n";
    let free_cfg = config("[collisions]\nmodel = \"none\"", freeze);
    let hs_cfg = config(&hard_sphere(RADIUS_KM), freeze);

    // Free positions and velocities after step 1.
    let mut b = CpuBackend::new(&free_cfg, &sys, &base).unwrap();
    b.step(1).unwrap();
    let p1 = b.particles().unwrap().particles.clone();
    let mut init = base.clone();
    let frozen_v = Vec3::new(0.0, 40.0, 0.0);
    for k in 0..5usize {
        let (r, v) = (p1.position_km(k), p1.velocity_km_s(k));
        let g = v - frozen_v;
        let ahead = r + g * (0.05 / g.norm());
        init.particles.push(10_000 + k as u32, ahead, frozen_v);
    }
    assert!(
        min_distance_km(&init.particles) > 2.0 * RADIUS_KM,
        "overlap"
    );

    // Control: with all particles taking part, the step finds collisions with
    // the frozen ones.
    let mut b = CpuBackend::new(&free_cfg, &sys, &init).unwrap();
    b.step(1).unwrap();
    let after1 = state_of(b.particles().unwrap().particles);
    let ids = b.particles().unwrap().particles.id.clone();
    let reason = b.dropped().unwrap().unwrap().reason;
    assert_eq!(reason.iter().filter(|&&r| r != 0).count(), 5);
    let count = |reason: Option<&[u8]>| {
        let mut c = CollisionState::from_config(&hs_cfg, &field)
            .unwrap()
            .unwrap();
        let mut st = after1.clone();
        c.step(&field, 360.0, 1, 360.0, &ids, reason, &mut st)
            .unwrap();
        c.tally.collisions
    };
    let (all, masked) = (count(None), count(Some(&reason)));
    println!("collisions at step 1: all take part {all}, frozen left out {masked}");
    assert!(all > masked);

    // The run: the 600 free particles are bit-identical to a run without the
    // frozen ones, and the frozen ones do not change.
    let with = |init: &InitialState| {
        let mut b = CpuBackend::new(&hs_cfg, &sys, init).unwrap();
        b.step(60).unwrap();
        let s = b.stats().unwrap();
        (b.particles().unwrap().particles.clone(), s)
    };
    let (pw, sw) = with(&init);
    let (po, so) = with(&base);
    assert!(sw.collisions_since_last > 0);
    assert_eq!(stats_bits(&sw), stats_bits(&so));
    let (bw, bo) = (bits(&pw), bits(&po));
    for f in 0..6 {
        let (a, b) = (f * pw.len(), f * po.len());
        assert_eq!(bw[a..a + n_free], bo[b..b + n_free], "field {f}");
    }
    for i in n_free..pw.len() {
        assert_eq!(pw.vy_km_s[i].to_bits(), 40.0f64.to_bits());
        assert_eq!(pw.vx_km_s[i].to_bits(), 0.0f64.to_bits());
        assert_eq!(pw.x_km[i].to_bits(), init.particles.x_km[i].to_bits());
    }
}

#[test]
fn hs_refusals() {
    let init = clump(20, 6.0);
    let sys = system();
    let err = |cfg: RunConfig| match CpuBackend::new(&cfg, &sys, &init) {
        Err(Error::Config(m)) => m,
        Err(e) => panic!("{e:?}"),
        Ok(_) => panic!("accepted"),
    };
    assert!(CpuBackend::new(&config(&hard_sphere(RADIUS_KM), ""), &sys, &init).is_ok());
    let chaos = "[diagnostics.chaos]\nmethod = \"shadow\"\noffset_km = 1e-3\nrenorm_steps = 100\n";
    assert!(err(config(&hard_sphere(RADIUS_KM), chaos)).contains("hard_sphere"));
    let mut arc = config(&hard_sphere(RADIUS_KM), "");
    arc.domain = fring_model::config::Domain::Arc {
        width_deg: 10.0,
        centre_longitude_deg: 0.0,
    };
    assert!(err(arc).contains("D1-CPU"));
    // K3 counts the cells along the full ring in a u32: 1 cm particles do not fit.
    assert!(err(config(&hard_sphere(1.0e-5), "")).contains("too small"));
}

/// Cost of one hard-sphere step by stage, and of the integrator at blocks of
/// 1 and 4096 steps (for B2 and K4). Run with
/// `cargo test -p fring-cpu --release --test collisions_hs -- --ignored --nocapture`.
#[test]
#[ignore]
fn hs_step_timing() {
    use std::time::Instant;
    let sys = system();
    let field = SaturnField::<f64>::from_params(&sys.saturn);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .unwrap();
    let steps_per_year = 365.25 * 86_400.0 / 360.0;
    for n in [100_000u32, 1_000_000] {
        // Dense clump (many pairs) and a wide band (almost none).
        let cases: Vec<(&str, f64, InitialState)> = vec![
            ("clump", 0.01, clump(n, 40.0)),
            ("band", 0.01, {
                let dist = Distribution::Band {
                    min_semi_major_axis_km: 140_000.0,
                    max_semi_major_axis_km: 140_400.0,
                };
                InitialState {
                    time_s: 0.0,
                    particles: generate_particles(&field, &dist, n, 1).unwrap(),
                }
            }),
        ];
        for (name, s, init) in cases {
            let cfg = config(&hard_sphere(s), "");
            let p = &init.particles;
            let mut state = state_of(p);
            let mut c = CollisionState::from_config(&cfg, &field).unwrap().unwrap();
            pool.install(|| {
                let mut tot = 0.0;
                for step in 1..=5u64 {
                    c.step(&field, step as f64 * 360.0, step, 360.0, &p.id, None, &mut state)
                        .unwrap();
                    let t = c.last_times;
                    let total = t.mask_s + t.find_s + t.resolve_s + t.write_s;
                    tot = total;
                    println!(
                        "N {n} {name} step {step}: mask {:.2} ms, find {:.2} ms, resolve {:.2} ms, copy {:.2} ms, total {:.2} ms",
                        t.mask_s * 1e3, t.find_s * 1e3, t.resolve_s * 1e3, t.write_s * 1e3, total * 1e3
                    );
                }
                println!(
                    "N {n} {name}: collisions {}, last step {:.1} ms, {:.0} s per model year",
                    c.tally.collisions, tot * 1e3, tot * steps_per_year
                );
            });
        }
        // Integrator alone: blocks of 1 step against one block.
        let cfg = config("[collisions]\nmodel = \"none\"", "");
        let init = clump(n, 40.0);
        let mut b = CpuBackend::new(&cfg, &sys, &init).unwrap();
        pool.install(|| {
            b.step(10).unwrap();
            let t = Instant::now();
            for _ in 0..200 {
                b.step(1).unwrap();
            }
            let one = t.elapsed().as_secs_f64() / 200.0;
            let t = Instant::now();
            b.step(200).unwrap();
            let many = t.elapsed().as_secs_f64() / 200.0;
            println!(
                "N {n} integrator per step: block 1 {:.3} ms, block 4096 {:.3} ms",
                one * 1e3,
                many * 1e3
            );
        });
    }
}
