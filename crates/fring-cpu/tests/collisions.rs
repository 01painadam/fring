//! K2-CPU: the stochastic collision step in `CpuBackend`. All comparisons are
//! exact (bit equality); there is no tolerance.

use fring_cpu::CpuBackend;
use fring_model::config::Distribution;
use fring_model::physics::{SaturnField, generate_particles};
use fring_model::{
    Backend, Error, InitialState, MoonEphemeris, Particles, RunConfig, StepStats, System, Vec3,
};

fn system() -> System {
    let saturn =
        fring_model::params::saturn_from_toml_str(include_str!("../../../params/saturn.toml"))
            .unwrap();
    System {
        saturn,
        moons: MoonEphemeris::none(),
    }
}

/// Example config without moons; `collisions` replaces the `[collisions]`
/// section and `extra` is appended.
fn config(collisions: &str, extra: &str) -> RunConfig {
    let text = include_str!("../../fring-model/tests/data/example-run.toml");
    let t = text
        .replace("duration_years = 1.0", "duration_years = 0.04")
        .replace(r#"include = ["prometheus", "pandora"]"#, "include = []")
        .replace("[collisions]\nmodel = \"none\"", collisions);
    RunConfig::from_toml_str(&format!("{t}\n{extra}")).unwrap()
}

/// Dense enough for many collisions: 3000 particles in 400 km, 4 x 44 cells.
fn stochastic(every: u32) -> String {
    format!(
        "[collisions]\nmodel = \"stochastic\"\nparticle_radius_km = 1.0e-5\nweight = 1.0e18\n\
         cell_radial_km = 100.0\ncell_azimuthal_km = 20000.0\ncell_vertical_km = 100.0\n\
         every_steps = {every}\n[collisions.restitution]\nlaw = \"constant\"\nepsilon = 0.5\n"
    )
}

fn band(count: u32) -> InitialState {
    let field = SaturnField::<f64>::from_params(&system().saturn);
    let dist = Distribution::Band {
        min_semi_major_axis_km: 140_000.0,
        max_semi_major_axis_km: 140_400.0,
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

/// Every field of the stats as bits, summed over the `stats()` calls.
fn stats_bits(s: &StepStats) -> Vec<u64> {
    let c = &s.collision;
    vec![
        s.steps_total,
        s.collisions_since_last,
        c.energy_loss_km2_s2.to_bits(),
        c.candidates,
        c.occupied_cells,
        c.occupancy_sum,
        c.occupancy_sq_sum,
        c.lonely_particles,
        c.overfull_cells,
        c.gmax_exceeded,
        c.deep_collisions,
    ]
}

/// Runs the calls, reads the stats once at the end.
fn run(cfg: &RunConfig, init: &InitialState, calls: &[u64]) -> (Vec<u64>, Vec<u64>) {
    let mut b = CpuBackend::new(cfg, &system(), init).unwrap();
    for &n in calls {
        b.step(n).unwrap();
    }
    let s = b.stats().unwrap();
    (bits(b.particles().unwrap().particles), stats_bits(&s))
}

#[test]
fn collisions_nonzero_stats() {
    let init = band(3000);
    let mut b = CpuBackend::new(&config(&stochastic(1), ""), &system(), &init).unwrap();
    b.step(200).unwrap();
    let s = b.stats().unwrap();
    let c = &s.collision;
    assert_eq!(s.steps_total, 200);
    assert!(s.collisions_since_last > 0, "no collisions");
    assert!(c.candidates >= s.collisions_since_last);
    assert!(c.occupied_cells > 0 && c.occupancy_sum > 0);
    assert_eq!(c.occupancy_sum, 3000 * 200, "k = 1: every step collides");
    assert!(c.occupancy_sq_sum >= c.occupancy_sum);
    assert!(c.energy_loss_km2_s2 > 0.0);
    assert!(s.max_rel_energy_error.is_none());
    println!(
        "collisions {} candidates {} occupied {} loss {:.3e} overfull {} gmax_exceeded {}",
        s.collisions_since_last,
        c.candidates,
        c.occupied_cells,
        c.energy_loss_km2_s2,
        c.overfull_cells,
        c.gmax_exceeded
    );
    // The stats are since the previous call.
    let s2 = b.stats().unwrap();
    assert_eq!(s2.collisions_since_last, 0);
    assert_eq!(s2.collision.candidates, 0);
    // A step() of 0 steps does nothing.
    let before = bits(b.particles().unwrap().particles);
    b.step(0).unwrap();
    assert_eq!(bits(b.particles().unwrap().particles), before);
    assert_eq!(b.stats().unwrap().collision.occupancy_sum, 0);
}

#[test]
fn same_seed_bit_identical_other_seed_differs() {
    let init = band(3000);
    let cfg = config(&stochastic(1), "");
    let a = run(&cfg, &init, &[150]);
    assert_eq!(a, run(&cfg, &init, &[150]));
    let mut other = config(&stochastic(1), "");
    other.run.seed = 7;
    let b = run(&other, &init, &[150]);
    assert_ne!(a.0, b.0);
}

#[test]
fn threads_1_4_8() {
    let init = band(3000);
    let cfg = config(&stochastic(2), "");
    let on = |n: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .unwrap()
            .install(|| run(&cfg, &init, &[80, 40]))
    };
    let one = on(1);
    assert!(one.1[1] > 0);
    assert_eq!(one, on(4));
    assert_eq!(one, on(8));
}

/// Largest |difference| of two states given as bit patterns.
fn max_diff(a: &[u64], b: &[u64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (f64::from_bits(*x) - f64::from_bits(*y)).abs())
        .fold(0.0, f64::max)
}

/// Splits at multiples of `every_steps` give identical bits (the blocks are
/// the same). Splits at other steps add a synchronisation, which changes the
/// rounding of the integrator itself (about 1e-12 km, with or without
/// collisions). The collision steps do not change: same counts, same
/// accepted pairs, states equal to that rounding.
#[test]
fn call_splits() {
    let init = band(3000);
    let cfg = config(&stochastic(3), "");
    let whole = run(&cfg, &init, &[60]);
    assert!(whole.1[1] > 0);
    for calls in [vec![30, 30], vec![3, 57], vec![0, 6, 54], vec![3; 20]] {
        assert_eq!(whole, run(&cfg, &init, &calls), "{calls:?}");
    }
    let none = config("[collisions]\nmodel = \"none\"", "");
    let integrator = max_diff(&run(&none, &init, &[60]).0, &run(&none, &init, &[7, 53]).0);
    println!("integrator split rounding {integrator:.3e}");
    for calls in [vec![7, 53], vec![2, 0, 58], vec![5, 1, 4, 50], vec![1; 60]] {
        let r = run(&cfg, &init, &calls);
        let d = max_diff(&whole.0, &r.0);
        println!("{calls:?}: max state difference {d:.3e}");
        let (mut w, mut s) = (whole.1.clone(), r.1.clone());
        // Energy loss: equal to rounding, not bit for bit.
        let (ew, es) = (f64::from_bits(w[2]), f64::from_bits(s[2]));
        assert!((ew - es).abs() <= 1e-5 * ew, "{calls:?}: {ew} {es}");
        (w[2], s[2]) = (0, 0);
        assert_eq!(w, s, "{calls:?}");
        assert!(d <= 1e-8, "{calls:?}: {d}");
    }
}

#[test]
fn every_steps_cadence() {
    let init = band(3000);
    let cfg = config(&stochastic(4), "");
    let mut b = CpuBackend::new(&cfg, &system(), &init).unwrap();
    for n in [3u64, 1, 3, 1, 2] {
        b.step(n).unwrap();
        // Total steps 3, 4, 7, 8, 10: collision steps at 4 and 8 only.
    }
    let s = b.stats().unwrap();
    assert_eq!(s.steps_total, 10);
    assert_eq!(s.collision.occupancy_sum, 3000 * 2);
}

#[test]
fn frozen_do_not_collide() {
    let mut init = band(3000);
    // 20 unbound particles inside the dense band, frozen at step 0.
    for k in 0..20 {
        init.particles.push(
            10_000 + k,
            Vec3::new(140_100.0 + 10.0 * f64::from(k), 0.0, 0.0),
            Vec3::new(0.0, 40.0, 0.0),
        );
    }
    let cfg = config(&stochastic(1), "[loss]\non_unbound = \"freeze\"\n");
    let mut b = CpuBackend::new(&cfg, &system(), &init).unwrap();
    b.step(100).unwrap();
    let s = b.stats().unwrap();
    assert_eq!(b.dropped().unwrap().unwrap().n_unbound, 20);
    assert_eq!(s.collision.occupancy_sum, 3000 * 100);
    assert!(s.collisions_since_last > 0);
    let p = &b.particles().unwrap().particles;
    for i in 3000..3020 {
        assert_eq!(p.vy_km_s[i].to_bits(), 40.0f64.to_bits());
        assert_eq!(p.vx_km_s[i].to_bits(), 0.0f64.to_bits());
        assert_eq!(p.x_km[i].to_bits(), init.particles.x_km[i].to_bits());
    }
}

#[test]
fn refusals() {
    let init = band(10);
    let sys = system();
    let err = |cfg: RunConfig| match CpuBackend::new(&cfg, &sys, &init) {
        Err(Error::Config(m)) => m,
        Err(e) => panic!("{e:?}"),
        Ok(_) => panic!("accepted"),
    };
    let chaos = "[diagnostics.chaos]\nmethod = \"shadow\"\noffset_km = 1e-3\nrenorm_steps = 100\n";
    assert!(err(config(&stochastic(1), chaos)).contains("shadow"));
    let hs = "[collisions]\nmodel = \"hard_sphere\"\nparticle_radius_km = 1e-5\n\
              [collisions.restitution]\nlaw = \"constant\"\nepsilon = 0.5\n";
    assert!(err(config(hs, "")).contains("K3-CPU"));
    let mut arc = config(&stochastic(1), "");
    arc.domain = fring_model::config::Domain::Arc {
        width_deg: 10.0,
        centre_longitude_deg: 0.0,
    };
    assert!(err(arc).contains("D1-CPU"));
}

/// Time of one collision step by stage (for B2). Run with
/// `cargo test -p fring-cpu --release --test collisions -- --ignored --nocapture`.
#[test]
#[ignore]
fn collision_step_timing() {
    use fring_cpu::collide::CollisionState;
    let sys = system();
    let field = SaturnField::<f64>::from_params(&sys.saturn);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .unwrap();
    for n in [100_000u32, 1_000_000] {
        // W such that about 0.05 candidates per particle per step.
        let w = 1.0e15 * 1.0e6 / f64::from(n);
        let cfg = config(
            &stochastic(1)
                .replace("cell_azimuthal_km = 20000.0", "cell_azimuthal_km = 2000.0")
                .replace("weight = 1.0e18", &format!("weight = {w:e}")),
            "",
        );
        let p = band(n).particles;
        let mut state: Vec<[f64; 6]> = (0..p.len())
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
        // Circular orbits have u = 0 and no candidates: add 1 cm/s of noise.
        let mut x = 0x9e37_79b9_7f4a_7c15u64;
        for s in state.iter_mut() {
            for v in &mut s[3..6] {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                *v += 1e-5 * ((x >> 11) as f64 / (1u64 << 53) as f64 - 0.5);
            }
        }
        let mut c = CollisionState::from_config(&cfg, &field).unwrap().unwrap();
        pool.install(|| {
            for step in 1..=4u64 {
                c.step(&field, step as f64 * 360.0, step, 360.0, &p.id, None, &mut state)
                    .unwrap();
                let t = c.last_times;
                let total = t.keys_s + t.group_s + t.cells_s + t.apply_s;
                println!(
                    "N {n} step {step}: keys {:.1} ms, group {:.1} ms, cells {:.1} ms, apply {:.1} ms, total {:.1} ms, group share {:.0} %",
                    t.keys_s * 1e3, t.group_s * 1e3, t.cells_s * 1e3, t.apply_s * 1e3,
                    total * 1e3, 100.0 * t.group_s / total
                );
            }
        });
        println!(
            "N {n}: collisions so far {} stats {:?}",
            c.tally.collisions, c.tally.stats
        );
    }
}
