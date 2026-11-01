//! The `Backend` trait can be implemented and used through `Box<dyn Backend>`,
//! and the input types check their data.
//!
//! `Drift` is a toy backend for this test only: particles move in straight
//! lines (no forces) and the moons stay at their first sample.

use fring_model::{
    Backend, BodyState, Error, InitialState, MoonEphemeris, MoonState, MoonTrack, ParticleView,
    Particles, Result, RunConfig, SaturnParams, StepStats, System, Vec3,
};

struct Drift {
    step_s: f64,
    time_s: f64,
    steps_total: u64,
    moons: Vec<BodyState>,
    particles: Particles,
}

impl Backend for Drift {
    fn new(config: &RunConfig, system: &System, init: &InitialState) -> Result<Self> {
        config.validate()?;
        system.validate()?;
        init.particles.validate()?;
        Ok(Self {
            step_s: config.time.step_s,
            time_s: init.time_s,
            steps_total: 0,
            moons: system.moons.tracks.iter().map(|t| t.samples[0]).collect(),
            particles: init.particles.clone(),
        })
    }

    fn step(&mut self, n_steps: u64) -> Result<()> {
        let dt_s = n_steps as f64 * self.step_s;
        let p = &mut self.particles;
        for i in 0..p.len() {
            p.x_km[i] += p.vx_km_s[i] * dt_s;
            p.y_km[i] += p.vy_km_s[i] * dt_s;
            p.z_km[i] += p.vz_km_s[i] * dt_s;
        }
        self.time_s += dt_s;
        self.steps_total += n_steps;
        Ok(())
    }

    fn time_s(&self) -> f64 {
        self.time_s
    }

    fn moons(&self) -> MoonState {
        MoonState {
            time_s: self.time_s,
            moons: self.moons.clone(),
        }
    }

    fn particles(&mut self) -> Result<ParticleView<'_>> {
        Ok(ParticleView {
            time_s: self.time_s,
            particles: &self.particles,
        })
    }

    fn stats(&mut self) -> Result<StepStats> {
        Ok(StepStats {
            time_s: self.time_s,
            steps_total: self.steps_total,
            ..Default::default()
        })
    }
}

fn config() -> RunConfig {
    RunConfig::from_toml_str(include_str!("data/example-run.toml")).unwrap()
}

/// Example values from docs/PLAN.md section 5 (not cited; tests only).
fn saturn() -> SaturnParams {
    SaturnParams {
        gm_km3_s2: 3.7931e7,
        reference_radius_km: 60_330.0,
        j2: 1.6291e-2,
        j4: -9.36e-4,
        j6: 8.68e-5,
    }
}

fn one_moon_system() -> System {
    let state = |x_km: f64| BodyState {
        position_km: Vec3::new(x_km, 0.0, 0.0),
        velocity_km_s: Vec3::new(0.0, 16.5, 0.0),
    };
    System {
        saturn: saturn(),
        moons: MoonEphemeris {
            start_s: 0.0,
            interval_s: 600.0,
            tracks: vec![MoonTrack {
                name: "prometheus".to_string(),
                gm_km3_s2: 0.01,
                samples: vec![state(139_380.0), state(139_380.0)],
            }],
        },
    }
}

fn two_particles() -> InitialState {
    let mut particles = Particles::with_capacity(2);
    particles.push(0, Vec3::new(140_220.0, 0.0, 0.0), Vec3::new(0.0, 16.4, 0.0));
    particles.push(
        1,
        Vec3::new(0.0, 140_220.0, 0.0),
        Vec3::new(-16.4, 0.0, 0.1),
    );
    InitialState {
        time_s: 0.0,
        particles,
    }
}

#[test]
fn backend_works_through_a_trait_object() {
    let config = config();
    let mut backend: Box<dyn Backend> =
        Box::new(Drift::new(&config, &one_moon_system(), &two_particles()).unwrap());

    backend.step(10).unwrap();
    assert_eq!(backend.time_s(), 3600.0); // 10 × 360 s

    let view = backend.particles().unwrap();
    assert_eq!(view.time_s, 3600.0);
    assert_eq!(view.particles.len(), 2);
    assert_eq!(view.particles.id, [0, 1]);
    assert_eq!(
        view.particles.position_km(1),
        Vec3::new(-16.4 * 3600.0, 140_220.0, 360.0)
    );

    let moons = backend.moons();
    assert_eq!(moons.moons.len(), 1);

    let stats = backend.stats().unwrap();
    assert_eq!(stats.steps_total, 10);
    assert_eq!(stats.collisions_since_last, 0);
    assert_eq!(stats.max_rel_energy_error, None);
}

#[test]
fn particle_arrays_of_different_length_are_rejected() {
    let mut init = two_particles();
    init.particles.vz_km_s.pop();
    let err = Drift::new(&config(), &one_moon_system(), &init)
        .err()
        .unwrap();
    assert!(matches!(err, Error::Input(msg) if msg.contains("vz_km_s")));
}

#[test]
fn non_finite_particle_value_is_rejected() {
    let mut init = two_particles();
    init.particles.y_km[1] = f64::NAN;
    assert!(matches!(init.particles.validate(), Err(Error::Input(msg)) if msg.contains("y_km[1]")));
}

#[test]
fn ephemeris_checks() {
    let mut system = one_moon_system();
    assert!(system.validate().is_ok());
    assert_eq!(system.moons.sample_count(), 2);
    assert_eq!(system.moons.end_s(), 600.0);

    // Only one sample: Hermite interpolation needs at least two.
    system.moons.tracks[0].samples.pop();
    assert!(matches!(system.validate(), Err(Error::Input(_))));

    // No moons is valid.
    let none = System {
        saturn: saturn(),
        moons: MoonEphemeris::none(),
    };
    assert!(none.validate().is_ok());
    assert_eq!(none.moons.sample_count(), 0);
}

#[test]
fn saturn_checks() {
    let mut s = saturn();
    assert!(s.validate().is_ok());
    s.gm_km3_s2 = -1.0;
    assert!(matches!(s.validate(), Err(Error::Input(_))));
}
