//! The [`Backend`] trait: the interface that the CPU and GPU backends implement.

use crate::Result;
use crate::config::RunConfig;
use crate::particles::{InitialState, ParticleView};
use crate::system::{MoonState, System};

/// A simulation engine: `fring-cpu` (f64 reference) or `fring-gpu` (wgpu).
///
/// `fring-run` makes a backend from the config, calls [`step`](Backend::step)
/// in blocks, and reads particles and statistics between blocks. All state
/// that crosses this interface is inertial and f64; a backend may use other
/// frames and precisions inside.
///
/// The trait is dyn-compatible, so `fring-run` can hold a `Box<dyn Backend>`
/// and choose the backend from `[run] backend` at run time.
pub trait Backend {
    /// Makes a backend for one run.
    ///
    /// `config` must be valid ([`RunConfig::validate`]), `system` and `init`
    /// must pass their `validate` checks, and the moon ephemeris must cover
    /// the whole run. The backend returns [`Error::Config`](crate::Error::Config)
    /// or [`Error::Input`](crate::Error::Input) if they do not.
    fn new(config: &RunConfig, system: &System, init: &InitialState) -> Result<Self>
    where
        Self: Sized;

    /// Advances the simulation by `n_steps` time steps of `[time] step_s`.
    fn step(&mut self, n_steps: u64) -> Result<()>;

    /// Current simulation time in seconds since the epoch.
    fn time_s(&self) -> f64;

    /// Moon states at the current time (inertial, f64), interpolated from
    /// the ephemeris.
    fn moons(&self) -> MoonState;

    /// Particle states at the current time (inertial, f64). A GPU backend
    /// copies its data to the host here, so do not call this every step.
    fn particles(&mut self) -> Result<ParticleView<'_>>;

    /// Statistics since the previous call (see [`StepStats`]).
    fn stats(&mut self) -> Result<StepStats>;

    /// The chaos indicator of every particle since the start of the run
    /// (ICR-003, ADR 0008), or `None` when the config has no
    /// `[diagnostics.chaos]` table or the backend does not support it. A GPU
    /// backend reads its accumulators back here, so do not call this every step.
    fn chaos(&mut self) -> Result<Option<ChaosReport>> {
        Ok(None)
    }
}

/// Chaos indicator per particle (ICR-003, ADR 0008), in particle order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ChaosReport {
    /// Simulation time in seconds since the epoch.
    pub time_s: f64,
    /// Number of renormalisations done (the last is at `n_renorm * K * h`
    /// after the start).
    pub n_renorm: u64,
    /// Finite-time Lyapunov exponent L / t in s⁻¹ (t = `n_renorm * K * h`).
    pub ftle_per_s: Vec<f64>,
    /// L: sum of ln(d / d0) over all renormalisations (dimensionless).
    pub log_growth: Vec<f64>,
    /// Mean MEGNO: the mean of Y_j over all renormalisations. About 2 for a
    /// regular orbit; about λ t / 2 for a chaotic one.
    pub megno_mean: Vec<f64>,
    /// Bit 0: a separation was zero or not finite (the interval was skipped).
    pub flags: Vec<u32>,
}

/// Diagnostics that a backend reports (`docs/PLAN.md` section 7.7).
///
/// More diagnostics will be added as fields. To stay compatible, build this
/// struct with `..Default::default()` for the fields that you do not set.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StepStats {
    /// Simulation time in seconds since the epoch.
    pub time_s: f64,
    /// Steps done since the start of the run.
    pub steps_total: u64,
    /// Collisions since the previous call to [`Backend::stats`].
    pub collisions_since_last: u64,
    /// Largest |ΔE / E| of any particle since the start of the run, where E
    /// is the energy per unit mass in Saturn's field. `None` when the backend
    /// does not track it (energy is not conserved when moons are present).
    pub max_rel_energy_error: Option<f64>,
}
