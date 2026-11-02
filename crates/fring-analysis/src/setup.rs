//! Builds the system and the initial particles from a config file, the same
//! way as `fring-run` does (`apps/fring-run/src/lib.rs`), so that an analysis
//! harness sees the same initial conditions as a production run.

use fring_model::params::{moons_from_toml_str, saturn_from_toml_str};
use fring_model::physics::{
    DEFAULT_INTERVAL_S, MoonInit, SaturnField, build_ephemeris, generate_particles,
};
use fring_model::{Error, InitialState, Result, RunConfig, SaturnParams, System};
use std::path::Path;

/// Everything that a backend needs.
pub struct Setup {
    /// The run config.
    pub config: RunConfig,
    /// Saturn and the moon ephemeris.
    pub system: System,
    /// Particles at t = 0.
    pub init: InitialState,
    /// Saturn's parameters.
    pub saturn: SaturnParams,
    /// Saturn's field.
    pub field: SaturnField<f64>,
}

fn read(path: &Path) -> Result<String> {
    std::fs::read_to_string(path).map_err(|e| Error::Io(format!("{}: {e}", path.display())))
}

/// Loads `params/saturn.toml` and `params/moons.toml` from `params_dir`.
pub fn load_params(params_dir: &Path) -> Result<(SaturnParams, Vec<MoonInit>)> {
    let saturn = saturn_from_toml_str(&read(&params_dir.join("saturn.toml"))?)?;
    let moons = moons_from_toml_str(&read(&params_dir.join("moons.toml"))?)?;
    Ok((saturn, moons))
}

/// Makes the setup for the config file at `config_path`.
pub fn load(config_path: &Path) -> Result<Setup> {
    let config = RunConfig::from_toml_str(&read(config_path)?)?;
    config.validate()?;
    let (saturn, all) = load_params(Path::new(&config.run.params_dir))?;
    let mut moons = Vec::new();
    for name in &config.moons.include {
        let m = all
            .iter()
            .find(|m| &m.name == name)
            .ok_or_else(|| Error::Config(format!("unknown moon {name:?}")))?;
        moons.push(m.clone());
    }
    let n_steps = config.time.n_steps();
    let step_s = config.time.step_s;
    let eph = build_ephemeris(
        &saturn,
        &moons,
        n_steps as f64 * step_s + DEFAULT_INTERVAL_S,
        DEFAULT_INTERVAL_S,
    )?;
    let field = SaturnField::<f64>::from_params(&saturn);
    let particles = generate_particles(
        &field,
        &config.particles.distribution,
        config.particles.count,
        config.run.seed,
    )?;
    Ok(Setup {
        init: InitialState {
            time_s: 0.0,
            particles,
        },
        system: System { saturn, moons: eph },
        saturn,
        field,
        config,
    })
}
