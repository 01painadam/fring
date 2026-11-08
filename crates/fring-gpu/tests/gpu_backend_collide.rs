//! K5-WIRE: stochastic collisions in `GpuBackend` (bin, collide, tally every
//! `every_steps`). Tests marked `#[ignore]` need a GPU:
//! `cargo test -p fring-gpu --release --test gpu_backend_collide -- --ignored --nocapture`
//!
//! The comparison with the CPU backend is statistical: the difference of the
//! means of several seeds is at most 4 combined standard errors
//! (`crates/fring-collide/SPEC.md`, ICR-010). Trajectories are never compared.

use fring_cpu::CpuBackend;
use fring_gpu::backend::{GpuBackend, Options};
use fring_gpu::wh::Mode;
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

fn config(collisions: &str, extra: &str) -> RunConfig {
    let text = include_str!("../../fring-model/tests/data/example-run.toml");
    let t = text
        .replace("duration_years = 1.0", "duration_years = 0.04")
        .replace(r#"include = ["prometheus", "pandora"]"#, "include = []")
        .replace("[collisions]\nmodel = \"none\"", collisions);
    RunConfig::from_toml_str(&format!("{t}\n{extra}")).unwrap()
}

/// Dense enough for many collisions: 4 x 44 cells of 100 km x 20,000 km x 100 km.
fn stochastic(every: u32) -> String {
    format!(
        "[collisions]\nmodel = \"stochastic\"\nparticle_radius_km = 1.0e-5\nweight = 1.0e18\n\
         cell_radial_km = 100.0\ncell_azimuthal_km = 20000.0\ncell_vertical_km = 100.0\n\
         every_steps = {every}\n[collisions.restitution]\nlaw = \"constant\"\nepsilon = 0.5\n"
    )
}

/// Eccentric band of epicyclic amplitude a e = 14 km: the peculiar speed is
/// about a e n = 3e-3 km/s, 2000 times the stepper noise of the circular
/// orbits (1.6e-6 km/s, K5 handoff), so the collision rate does not measure the
/// noise.
fn band(count: u32) -> InitialState {
    let field = SaturnField::<f64>::from_params(&system().saturn);
    let dist = Distribution::EccentricBand {
        min_semi_major_axis_km: 140_000.0,
        max_semi_major_axis_km: 140_400.0,
        eccentricity: 1.0e-4,
        apse_longitude_deg: 0.0,
    };
    InitialState {
        time_s: 0.0,
        particles: generate_particles(&field, &dist, count, 20_261_108).unwrap(),
    }
}

fn gpu(cfg: &RunConfig, init: &InitialState) -> GpuBackend {
    let opts = Options {
        mode: Mode::Dp,
        ..Default::default()
    };
    GpuBackend::with_options(cfg, &system(), init, opts).unwrap()
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
        c.occupancy_sum,
        c.occupancy_sq_sum,
        c.lonely_particles,
        c.overfull_cells,
        c.gmax_exceeded,
    ]
}

fn run(cfg: &RunConfig, init: &InitialState, calls: &[u64]) -> (Vec<u64>, Vec<u64>) {
    let mut b = gpu(cfg, init);
    for &n in calls {
        b.step(n).unwrap();
    }
    let s = b.stats().unwrap();
    (bits(b.particles().unwrap().particles), stats_bits(&s))
}

#[test]
fn refusals_need_no_gpu() {
    let init = band(10);
    let sys = system();
    let err = |cfg: RunConfig, mode: Mode| {
        let opts = Options {
            mode,
            ..Default::default()
        };
        match GpuBackend::with_options(&cfg, &sys, &init, opts) {
            Err(Error::Config(m)) => m,
            Err(e) => panic!("{e:?}"),
            Ok(_) => panic!("accepted"),
        }
    };
    let chaos = "[diagnostics.chaos]\nmethod = \"shadow\"\noffset_km = 1e-3\nrenorm_steps = 100\n";
    assert!(err(config(&stochastic(1), chaos), Mode::Dp).contains("shadow"));
    let hs = "[collisions]\nmodel = \"hard_sphere\"\nparticle_radius_km = 1e-5\n\
              [collisions.restitution]\nlaw = \"constant\"\nepsilon = 0.5\n";
    assert!(err(config(hs, ""), Mode::Dp).contains("K3"));
    let mut arc = config(&stochastic(1), "");
    arc.domain = fring_model::config::Domain::Arc {
        width_deg: 10.0,
        centre_longitude_deg: 0.0,
    };
    assert!(err(arc, Mode::Dp).contains("arc"));
    assert!(err(config(&stochastic(1), ""), Mode::C).contains("mode C"));
}

#[test]
#[ignore = "needs a GPU"]
fn collisions_nonzero_stats() {
    let init = band(3000);
    let mut b = gpu(&config(&stochastic(1), ""), &init);
    b.step(200).unwrap();
    let s = b.stats().unwrap();
    let c = &s.collision;
    assert_eq!(s.steps_total, 200);
    assert!(s.collisions_since_last > 0, "no collisions");
    assert!(c.candidates >= s.collisions_since_last);
    assert_eq!(c.occupancy_sum, 3000 * 200, "k = 1: every step collides");
    assert!(c.occupied_cells > 0 && c.occupancy_sq_sum >= c.occupancy_sum);
    assert!(c.energy_loss_km2_s2 > 0.0);
    assert_eq!(c.overfull_cells, 0);
    assert!(s.max_rel_energy_error.is_none());
    println!(
        "collisions {} candidates {} occupied {} loss {:.3e}",
        s.collisions_since_last, c.candidates, c.occupied_cells, c.energy_loss_km2_s2
    );
    // The stats are since the previous call.
    let s2 = b.stats().unwrap();
    assert_eq!(s2.collisions_since_last, 0);
    assert_eq!(s2.collision.occupancy_sum, 0);
    // A step of 0 steps does nothing.
    let before = bits(b.particles().unwrap().particles);
    b.step(0).unwrap();
    assert_eq!(bits(b.particles().unwrap().particles), before);
    assert_eq!(b.stats().unwrap().collision.occupancy_sum, 0);
}

#[test]
#[ignore = "needs a GPU"]
fn model_none_has_no_collision_stats() {
    let init = band(3000);
    let mut b = gpu(&config("[collisions]\nmodel = \"none\"", ""), &init);
    b.step(50).unwrap();
    let s = b.stats().unwrap();
    assert_eq!(s.collisions_since_last, 0);
    assert_eq!(s.collision.occupancy_sum, 0);
    assert!(s.max_rel_energy_error.is_some());
}

#[test]
#[ignore = "needs a GPU"]
fn same_seed_bit_identical_other_seed_differs() {
    let init = band(3000);
    let cfg = config(&stochastic(1), "");
    let a = run(&cfg, &init, &[150]);
    assert!(a.1[1] > 0);
    assert_eq!(a, run(&cfg, &init, &[150]));
    let mut other = cfg.clone();
    other.run.seed += 1;
    let b = run(&other, &init, &[150]);
    assert_ne!(a.0, b.0);
}

#[test]
#[ignore = "needs a GPU"]
fn call_splits_and_cadence() {
    let init = band(3000);
    let cfg = config(&stochastic(4), "");
    let whole = run(&cfg, &init, &[150]);
    assert!(whole.1[1] > 0);
    // Splits at multiples of every_steps give the same blocks, so the same bits.
    // A split inside a block adds a synchronisation (a rounding change, as on
    // the CPU), so the collision draws can differ: only the counts of the
    // cadence are exact then.
    let aligned: [&[u64]; 2] = [&[72, 78], &[4, 8, 138]];
    for calls in aligned {
        assert_eq!(whole, run(&cfg, &init, calls), "{calls:?}");
    }
    let inside: [&[u64]; 3] = [&[70, 80], &[3, 147], &[1, 1, 148]];
    for calls in inside {
        let r = run(&cfg, &init, calls);
        assert_eq!((r.1[0], r.1[5]), (whole.1[0], whole.1[5]), "{calls:?}");
        assert!(r.1[1] > 0);
    }
    // Total steps 3, 4, 7, 8, 10: collision steps at 4 and 8 only.
    let mut b = gpu(&cfg, &init);
    for n in [3u64, 1, 3, 1, 2] {
        b.step(n).unwrap();
    }
    let s = b.stats().unwrap();
    assert_eq!(s.steps_total, 10);
    assert_eq!(s.collision.occupancy_sum, 3000 * 2);
    // A block that does not end on a multiple of 4 leaves no tally.
    let mut b = gpu(&cfg, &init);
    b.step(3).unwrap();
    let s = b.stats().unwrap();
    assert_eq!(s.collisions_since_last, 0);
    assert_eq!(s.collision.occupancy_sum, 0);
    assert_eq!(s.collision.candidates, 0);
    assert_eq!(s.collision.energy_loss_km2_s2, 0.0);
    b.step(2).unwrap();
    b.step(1).unwrap();
    assert_eq!(b.stats().unwrap().collision.occupancy_sum, 3000 * 1);
    // The step that follows (6 to 7) is not a collision step.
    b.step(1).unwrap();
    assert_eq!(b.stats().unwrap().collision.occupancy_sum, 0);
}

#[test]
#[ignore = "needs a GPU"]
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
    let mut b = gpu(&cfg, &init);
    b.step(100).unwrap();
    let s = b.stats().unwrap();
    assert_eq!(b.dropped().unwrap().unwrap().n_unbound, 20);
    assert_eq!(s.collision.occupancy_sum, 3000 * 100);
    assert!(s.collisions_since_last > 0);
    let p = &b.particles().unwrap().particles;
    for i in 3000..3020 {
        assert_eq!(p.vy_km_s[i].to_bits(), init.particles.vy_km_s[i].to_bits());
        assert_eq!(p.vx_km_s[i].to_bits(), init.particles.vx_km_s[i].to_bits());
        assert_eq!(p.x_km[i].to_bits(), init.particles.x_km[i].to_bits());
    }
}

/// Mean and standard error of the mean.
fn mean_se(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let m = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0);
    (m, (var / n).sqrt())
}

/// GPU against CPU: collisions, candidates and energy loss over 120 steps of
/// 3000 particles, 12 seeds on each backend (the same initial state).
#[test]
#[ignore = "needs a GPU"]
fn statistics_agree_with_the_cpu_backend() {
    let init = band(3000);
    let (steps, seeds) = (120u64, 12u64);
    let mut samples_c = [Vec::new(), Vec::new(), Vec::new()]; // occupancy_sq / sum, occupied
    let mut samples_g = [Vec::new(), Vec::new(), Vec::new()];
    let mut gpu_vals = [Vec::new(), Vec::new(), Vec::new()];
    let mut cpu_vals = [Vec::new(), Vec::new(), Vec::new()];
    for seed in 0..seeds {
        let mut cfg = config(&stochastic(1), "");
        cfg.run.seed = 1000 + seed;
        let mut g = gpu(&cfg, &init);
        g.step(steps).unwrap();
        let sg = g.stats().unwrap();
        let mut c = CpuBackend::new(&cfg, &system(), &init).unwrap();
        c.step(steps).unwrap();
        let sc = c.stats().unwrap();
        for (s, vals, extra) in [
            (&sg, &mut gpu_vals, &mut samples_g),
            (&sc, &mut cpu_vals, &mut samples_c),
        ] {
            vals[0].push(s.collisions_since_last as f64);
            vals[1].push(s.collision.candidates as f64);
            vals[2].push(s.collision.energy_loss_km2_s2);
            extra[0].push(s.collision.occupancy_sq_sum as f64 / s.collision.occupancy_sum as f64);
            extra[1].push(s.collision.occupied_cells as f64);
        }
    }
    let names = ["collisions", "candidates", "energy loss km2/s2"];
    for k in 0..3 {
        let (mg, sg) = mean_se(&gpu_vals[k]);
        let (mc, sc) = mean_se(&cpu_vals[k]);
        let comb = (sg * sg + sc * sc).sqrt();
        println!(
            "{:20} GPU {mg:.6e} (SE/mean {:.2e})  CPU {mc:.6e} (SE/mean {:.2e})  diff {:.2} SE",
            names[k],
            sg / mg,
            sc / mc,
            (mg - mc).abs() / comb
        );
        assert!(
            (mg - mc).abs() <= 4.0 * comb,
            "{}: GPU {mg} CPU {mc}, 4 combined SE {}",
            names[k],
            4.0 * comb
        );
    }
    // The cell statistics are deterministic functions of the (nearly equal)
    // positions: reported, and they must agree to 1e-3.
    for (name, a, b) in [
        ("<N^2>/<N>", &samples_g[0], &samples_c[0]),
        ("occupied cells", &samples_g[1], &samples_c[1]),
    ] {
        let (ma, _) = mean_se(a);
        let (mb, _) = mean_se(b);
        println!("{name:20} GPU {ma:.6} CPU {mb:.6}");
        assert!((ma - mb).abs() <= 1e-3 * mb, "{name}: {ma} {mb}");
    }
}
