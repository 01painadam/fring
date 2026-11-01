//! f64 reference backend.
//!
//! Owner: A2 (see `AGENTS.md`).
//!
//! C1: Saturn's field (point mass + J2 + J4 + J6) and the fixed-step
//! drift-kick-drift leapfrog. The physics lives in
//! `fring_model::physics`; this crate holds the particle loop and the
//! [`CpuBackend`]. Moons (C3, C4) and rayon (C4) come later.

pub mod integrate;

use fring_model::physics::SaturnField;
use fring_model::{
    Backend, Error, InitialState, MoonState, ParticleView, Particles, Result, RunConfig, StepStats,
    System,
};
use integrate::{particle_energy, step_particles};

/// The CPU reference backend (f64, Saturn's field only, no moons yet).
pub struct CpuBackend {
    field: SaturnField<f64>,
    step_s: f64,
    start_s: f64,
    steps_total: u64,
    particles: Particles,
    /// Energy per unit mass of each particle at the start, km² s⁻².
    energy0: Vec<f64>,
    /// Largest |dE/E| seen at any call of `stats`.
    max_rel_energy_error: f64,
}

impl Backend for CpuBackend {
    fn new(config: &RunConfig, system: &System, init: &InitialState) -> Result<Self> {
        config.validate()?;
        system.validate()?;
        init.particles.validate()?;
        if !system.moons.tracks.is_empty() {
            return Err(Error::Input(
                "CpuBackend does not support moons yet (tasks C3 and C4); \
                 the ephemeris must have no tracks"
                    .to_string(),
            ));
        }
        let field = SaturnField::<f64>::from_params(&system.saturn);
        let energy0 = (0..init.particles.len())
            .map(|i| particle_energy(&field, &init.particles, i))
            .collect();
        Ok(Self {
            field,
            step_s: config.time.step_s,
            start_s: init.time_s,
            steps_total: 0,
            particles: init.particles.clone(),
            energy0,
            max_rel_energy_error: 0.0,
        })
    }

    fn step(&mut self, n_steps: u64) -> Result<()> {
        step_particles(&self.field, &mut self.particles, self.step_s, n_steps);
        self.steps_total += n_steps;
        Ok(())
    }

    fn time_s(&self) -> f64 {
        self.start_s + self.steps_total as f64 * self.step_s
    }

    fn moons(&self) -> MoonState {
        MoonState {
            time_s: self.time_s(),
            moons: Vec::new(),
        }
    }

    fn particles(&mut self) -> Result<ParticleView<'_>> {
        Ok(ParticleView {
            time_s: self.time_s(),
            particles: &self.particles,
        })
    }

    fn stats(&mut self) -> Result<StepStats> {
        // Computed here, not at every step, to keep the step loop cheap.
        for (i, &e0) in self.energy0.iter().enumerate() {
            let e = particle_energy(&self.field, &self.particles, i);
            self.max_rel_energy_error = self.max_rel_energy_error.max(((e - e0) / e0).abs());
        }
        Ok(StepStats {
            time_s: self.time_s(),
            steps_total: self.steps_total,
            max_rel_energy_error: Some(self.max_rel_energy_error),
            ..Default::default()
        })
    }
}
