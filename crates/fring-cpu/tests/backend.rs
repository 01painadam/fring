//! `CpuBackend` through the `Backend` trait: it steps, reports energy, and
//! refuses an ephemeris that does not cover the run. Moons: `tests/v3.rs`.

use fring_cpu::CpuBackend;
use fring_cpu::integrate::step_particles;
use fring_model::physics::{SaturnField, circular_speed};
use fring_model::{
    Backend, BodyState, Error, InitialState, MoonEphemeris, MoonTrack, Particles, RunConfig,
    SaturnParams, System, Vec3,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct SaturnFile {
    saturn: SaturnParams,
}

fn saturn() -> SaturnParams {
    let text = include_str!("../../../params/saturn.toml");
    toml::from_str::<SaturnFile>(text).unwrap().saturn
}

fn config() -> RunConfig {
    // The example config with an empty moon list: Saturn only.
    let text = include_str!("../../fring-model/tests/data/example-run.toml");
    let from = r#"include = ["prometheus", "pandora"]"#;
    assert!(text.contains(from), "example-run.toml changed");
    RunConfig::from_toml_str(&text.replace(from, "include = []")).unwrap()
}

fn init(n: u32) -> InitialState {
    let field = SaturnField::<f64>::from_params(&saturn());
    let mut particles = Particles::with_capacity(n as usize);
    for i in 0..n {
        // Circular orbits at slightly different radii and longitudes.
        let r = 140_000.0 + 50.0 * f64::from(i);
        let v = circular_speed(&field, r);
        let phi = 0.3 * f64::from(i);
        particles.push(
            i,
            Vec3::new(r * phi.cos(), r * phi.sin(), 0.0),
            Vec3::new(-v * phi.sin(), v * phi.cos(), 0.0),
        );
    }
    InitialState {
        time_s: 100.0,
        particles,
    }
}

fn system() -> System {
    System {
        saturn: saturn(),
        moons: MoonEphemeris::none(),
    }
}

#[test]
fn steps_advance_time_and_match_the_plain_loop() {
    let (cfg, sys, ini) = (config(), system(), init(5));
    let mut b = CpuBackend::new(&cfg, &sys, &ini).unwrap();
    b.step(400).unwrap();
    b.step(100).unwrap();
    assert_eq!(b.time_s(), 100.0 + 500.0 * cfg.time.step_s);
    assert!(b.moons().moons.is_empty());

    let field = SaturnField::<f64>::from_params(&sys.saturn)
        // ICR-011: the backend drifts about mu_eff at a0 (config a0 = 140,220 km).
        .with_drift_radius(140_220.0);
    let mut expected = ini.particles.clone();
    step_particles(&field, &mut expected, cfg.time.step_s, 500).unwrap();
    let view = b.particles().unwrap();
    assert_eq!(view.time_s, 100.0 + 500.0 * cfg.time.step_s);
    // 400 + 100 steps and 500 steps differ by rounding only (the merged half
    // drifts are split differently).
    let (a, e) = (view.particles, &expected);
    for i in 0..e.len() {
        assert!((a.x_km[i] - e.x_km[i]).abs() < 1e-6, "x {i}");
        assert!((a.y_km[i] - e.y_km[i]).abs() < 1e-6, "y {i}");
        assert!((a.vx_km_s[i] - e.vx_km_s[i]).abs() < 1e-11, "vx {i}");
        assert!((a.vy_km_s[i] - e.vy_km_s[i]).abs() < 1e-11, "vy {i}");
    }
}

#[test]
fn stats_report_a_small_energy_error() {
    let (cfg, sys, ini) = (config(), system(), init(5));
    let mut b = CpuBackend::new(&cfg, &sys, &ini).unwrap();
    assert_eq!(b.stats().unwrap().max_rel_energy_error, Some(0.0));
    b.step(1000).unwrap();
    let s = b.stats().unwrap();
    assert_eq!(s.steps_total, 1000);
    let err = s.max_rel_energy_error.unwrap();
    assert!(err > 0.0 && err < 1e-6, "{err:e}");
}

#[test]
fn an_ephemeris_that_is_too_short_is_refused() {
    let mut sys = system();
    sys.moons = MoonEphemeris {
        start_s: 0.0,
        interval_s: 100.0,
        tracks: vec![MoonTrack {
            name: "prometheus".to_string(),
            gm_km3_s2: 10.0,
            samples: vec![BodyState::default(); 2],
            accelerations_km_s2: vec![Vec3::zero(); 2],
        }],
    };
    let r = CpuBackend::new(&config(), &sys, &init(1));
    assert!(matches!(r, Err(Error::Input(_))), "{:?}", r.err());
}

#[test]
fn unbound_particle_is_an_error() {
    let mut ini = init(2);
    ini.particles.vx_km_s[1] = 1.0e3;
    ini.particles.vy_km_s[1] = 1.0e3;
    let mut b = CpuBackend::new(&config(), &system(), &ini).unwrap();
    assert!(matches!(b.step(1), Err(Error::Backend(_))));
}
