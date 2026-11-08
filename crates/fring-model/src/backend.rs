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

    /// Lost particles (ICR-004): the reason and the step of each loss, or
    /// `None` when the config has no `[loss]` with `on_unbound = "freeze"`
    /// or the backend does not support it. A GPU backend reads its status
    /// buffer back here, so do not call this every step.
    fn dropped(&mut self) -> Result<Option<DropReport>> {
        Ok(None)
    }
}

/// Reason 0 of [`DropReport::reason`]: the particle is live.
pub const DROP_LIVE: u8 = 0;
/// Reason 1: unbound or invalid orbit (the Kepler drift fails).
pub const DROP_UNBOUND: u8 = 1;
/// Reason `2 + k`: inside moon `k` (moon order of the `System`, `k` = 0 to 5).
pub const fn drop_inside_moon(k: usize) -> u8 {
    2 + k as u8
}

/// Lost particles (ICR-004 section 2). With `on_unbound = "freeze"` a lost
/// particle is flagged and frozen. The frozen position is unspecified within
/// one block: use the reason and `drop_step`, not the position, for physics.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DropReport {
    /// Simulation time of the report, s.
    pub time_s: f64,
    /// Per particle, in `ParticleView` order: 0 live, 1 unbound, `2 + k`
    /// inside moon `k`.
    pub reason: Vec<u8>,
    /// Per particle: the 0-based global step (counted from the start of the
    /// run) in which the loss was found. `u64::MAX` for a live particle.
    pub drop_step: Vec<u64>,
    /// Particles lost as unbound, cumulative since the start of the run.
    pub n_unbound: u64,
    /// Particles lost inside each moon, cumulative, in `System` order.
    pub n_inside_moon: Vec<u64>,
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
    /// Collisions (accepted pairs) since the previous call to
    /// [`Backend::stats`]. This is the only count of accepted pairs
    /// (ICR-009, amendment A1).
    pub collisions_since_last: u64,
    /// Largest |ΔE / E| of any particle since the start of the run, where E
    /// is the energy per unit mass in Saturn's field. `None` when the backend
    /// does not track it (energy is not conserved when moons are present).
    pub max_rel_energy_error: Option<f64>,
    /// Collision diagnostics since the previous call to [`Backend::stats`]
    /// (ICR-009). All zero when the run has no collisions.
    pub collision: CollisionStats,
}

/// Collision diagnostics (`crates/fring-collide/SPEC.md` section 9, ICR-009).
///
/// Every field counts since the previous call to [`Backend::stats`]. "Over
/// collision steps" means a sum over each collision step in that interval
/// (one collision step every `every_steps` integrator steps). A field that
/// does not apply to the collision model stays zero.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CollisionStats {
    /// Kinetic energy lost in collisions, Σ ¼ (1 − ε²) gₙ² over the accepted
    /// pairs, per unit superparticle mass, in km² s⁻² (gₙ: normal component
    /// of the relative velocity, km/s).
    pub energy_loss_km2_s2: f64,
    /// Stochastic: candidate pairs tested.
    pub candidates: u64,
    /// Stochastic: cells with at least one particle, summed over collision
    /// steps.
    pub occupied_cells: u64,
    /// Stochastic: Σ N_c over occupied cells and collision steps (N_c: the
    /// number of superparticles in a cell).
    pub occupancy_sum: u64,
    /// Stochastic: Σ N_c² over occupied cells and collision steps.
    pub occupancy_sq_sum: u64,
    /// Stochastic: particles in cells with N_c < 2 (so no partner), summed
    /// over collision steps.
    pub lonely_particles: u64,
    /// Stochastic: cells with W σ g_max Δt_c / V_c > 1, summed over collision
    /// steps (SPEC 5.6).
    pub overfull_cells: u64,
    /// Stochastic: candidates with |g| > g_max, accepted with probability 1
    /// (SPEC 5.3; ICR-009, amendment A2).
    pub gmax_exceeded: u64,
    /// Hard-sphere: collisions with overlap depth δ > 0.2 s, with s the
    /// particle radius (SPEC 8).
    pub deep_collisions: u64,
}
