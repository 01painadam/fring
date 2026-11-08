//! Run configuration: one TOML file per run (`docs/PLAN.md` section 4.4).
//!
//! The schema is a shared interface. An example with every section is in
//! `crates/fring-model/tests/data/example-run.toml`.
//!
//! Rules:
//!
//! - Every field has its unit in its name (`step_s`, `width_deg`). Angles are in
//!   degrees here and in radians in code.
//! - Unknown fields are errors, so a typing error cannot be ignored silently.
//! - Physical constants (GM, J2, moon masses) are not in the run config. They
//!   come from `params/` (directory set by `[run] params_dir`).
//! - [`RunConfig::from_toml_str`] parses and then checks every range. Code that
//!   builds a `RunConfig` in another way must call [`RunConfig::validate`].
//!
//! ```
//! use fring_model::config::RunConfig;
//!
//! let text = include_str!("../tests/data/example-run.toml");
//! let config = RunConfig::from_toml_str(text).unwrap();
//! assert_eq!(config.particles.count, 10_000);
//! ```

use crate::units::SECONDS_PER_JULIAN_YEAR;
use crate::{Error, Result};
use serde::{Deserialize, Serialize};

/// The whole run config. Each field is one `[section]` of the TOML file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunConfig {
    /// `[run]`: name, backend, seed, parameter directory.
    pub run: RunSection,
    /// `[time]`: time step and duration.
    pub time: TimeConfig,
    /// `[frame]`: the reference orbit for the rotating and local frames.
    pub frame: FrameConfig,
    /// `[domain]`: full ring or co-moving arc.
    pub domain: Domain,
    /// `[moons]`: which moons are present.
    pub moons: MoonsConfig,
    /// `[particles]`: count and initial distribution.
    pub particles: ParticlesConfig,
    /// `[collisions]`: collision model and its parameters.
    pub collisions: Collisions,
    /// `[output]`: output cadence.
    pub output: OutputConfig,
    /// `[diagnostics]`: optional extra diagnostics (ICR-003). Absent means
    /// none, and a backend then does no extra work.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diagnostics: Option<DiagnosticsConfig>,
    /// `[loss]`: what happens to a lost particle (ICR-004). Absent means a
    /// lost particle is an error, as before.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loss: Option<LossConfig>,
    /// `[substep]`: per-particle sub-stepping near the moons (ICR-004).
    /// Absent means no sub-stepping.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub substep: Option<SubstepConfig>,
}

/// What a backend does with a particle with an unbound or invalid orbit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OnUnbound {
    /// `Backend::step` returns an error (the default).
    #[default]
    Error,
    /// The particle is flagged and frozen (ICR-004 section 2).
    Freeze,
}

/// Where the moon radius for the "inside a moon" rule comes from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MoonRadiusRule {
    /// No radius rule (the default).
    #[default]
    Off,
    /// The radii of `<params_dir>/moons.toml`; needs `on_unbound = "freeze"`.
    Params,
}

/// `[loss]` section (ICR-004 section 1).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LossConfig {
    /// `"error"` (default) or `"freeze"`.
    #[serde(default)]
    pub on_unbound: OnUnbound,
    /// `"off"` (default) or `"params"`.
    #[serde(default)]
    pub moon_radius: MoonRadiusRule,
}

/// `[substep]` section (ICR-004 section 3).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SubstepConfig {
    /// Target value of `h v_eff / d_eff` per sub-step, 0.07 to 1.0 (default 0.1).
    #[serde(default = "default_substep_accuracy")]
    pub accuracy: f64,
    /// Highest level L: a step is split into at most 2^L sub-steps. 0 to 6
    /// (default 4).
    #[serde(default = "default_substep_max_level")]
    pub max_level: u8,
}

fn default_substep_accuracy() -> f64 {
    0.1
}

fn default_substep_max_level() -> u8 {
    4
}

/// Lowest `[substep] accuracy` (ICR-004 section 1: below it every particle
/// would be sub-stepped everywhere).
pub const SUBSTEP_ACCURACY_MIN: f64 = 0.07;
/// Highest `[substep] accuracy`.
pub const SUBSTEP_ACCURACY_MAX: f64 = 1.0;
/// Highest `[substep] max_level` (n = 64 sub-steps).
pub const SUBSTEP_MAX_LEVEL: u8 = 6;

/// `[diagnostics]` section (ICR-003).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiagnosticsConfig {
    /// `[diagnostics.chaos]`: shadow particle and MEGNO per particle.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chaos: Option<ChaosConfig>,
}

/// The method of the chaos indicator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChaosMethod {
    /// A second particle with a small position offset, renormalised
    /// periodically (ADR 0008).
    Shadow,
}

/// `[diagnostics.chaos]` section (ICR-003, ADR 0008).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChaosConfig {
    /// Method (only `"shadow"`).
    pub method: ChaosMethod,
    /// Initial and renormalised position separation d0 of the shadow, km (> 0).
    pub offset_km: f64,
    /// Renormalise at every global step number that is a multiple of this
    /// (at least 1).
    pub renorm_steps: u32,
}

/// `[run]` section.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunSection {
    /// Short name of the run. Letters, digits, `-` and `_` only, because it is
    /// used as a directory name.
    pub name: String,
    /// Which backend runs the simulation.
    pub backend: BackendKind,
    /// Seed for the Philox RNG ([`crate::rng::Philox::from_seed`]). Range 0 to
    /// 2^63 − 1, because TOML integers are signed 64-bit.
    pub seed: u64,
    /// Directory with the parameter files, relative to the working directory.
    /// Default: `"params"`.
    #[serde(default = "default_params_dir")]
    pub params_dir: String,
}

fn default_params_dir() -> String {
    "params".to_string()
}

/// Backend choice: `"cpu"` (f64 reference) or `"gpu"` (wgpu).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    /// f64 reference backend (`fring-cpu`).
    Cpu,
    /// wgpu backend (`fring-gpu`).
    Gpu,
}

/// `[time]` section.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TimeConfig {
    /// Fixed time step in seconds. About 360 s gives 150 steps per orbit at
    /// the F ring core (orbital period about 53,600 s).
    pub step_s: f64,
    /// Simulated duration in Julian years (365.25 days).
    pub duration_years: f64,
}

impl TimeConfig {
    /// Simulated duration in seconds.
    pub fn duration_s(&self) -> f64 {
        self.duration_years * SECONDS_PER_JULIAN_YEAR
    }

    /// Number of steps: duration divided by step, rounded up, so the run
    /// covers at least the full duration.
    pub fn n_steps(&self) -> u64 {
        (self.duration_s() / self.step_s).ceil() as u64
    }
}

/// `[frame]` section: the reference orbit that defines the rotating frame and
/// the local coordinates (`docs/PLAN.md` section 4.2).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrameConfig {
    /// Semi-major axis a₀ of the reference circular orbit, in km. The rotating
    /// frame turns at the mean motion n(a₀) of the J2 to J6 field
    /// (`physics::frames::reference_mean_motion`), so a circular orbit at a₀
    /// is at rest in it.
    pub reference_semi_major_axis_km: f64,
}

/// `[domain]` section, selected by `kind`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Domain {
    /// All longitudes (`kind = "full_ring"`).
    FullRing,
    /// An azimuthal arc that rotates at n₀ (`kind = "arc"`). A particle that
    /// leaves one end comes in at the other end (`docs/PLAN.md` section 7.6).
    Arc {
        /// Arc width Δθ in degrees, in (0, 360].
        width_deg: f64,
        /// Inertial longitude of the arc centre at time 0, in degrees, from
        /// the x axis.
        centre_longitude_deg: f64,
    },
}

/// `[moons]` section.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MoonsConfig {
    /// Names of the moons to include, as in `params/moons.toml` (for example
    /// `["prometheus", "pandora"]`). An empty list gives Saturn only.
    pub include: Vec<String>,
}

/// `[particles]` section.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParticlesConfig {
    /// Number of test particles or superparticles, at least 1.
    pub count: u32,
    /// Initial distribution (`[particles.distribution]`).
    pub distribution: Distribution,
}

/// Initial particle distribution (`docs/PLAN.md` section 7.4), selected by
/// `kind`. The mean longitude is uniform in [0°, 360°) unless stated.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Distribution {
    /// Narrow core (`kind = "core"`): Gaussian in semi-major axis, with one
    /// eccentricity and one apse longitude for all particles.
    Core {
        /// Mean semi-major axis in km.
        semi_major_axis_km: f64,
        /// Standard deviation of the semi-major axis in km (≥ 0).
        sigma_semi_major_axis_km: f64,
        /// Eccentricity, in [0, 1). This is the epicyclic amplitude (radial
        /// excursion `a e` about the guiding circle, first-order epicycle with
        /// the exact n and kappa), not an osculating Keplerian eccentricity.
        eccentricity: f64,
        /// Longitude of pericentre in degrees.
        apse_longitude_deg: f64,
    },
    /// Uniform band in semi-major axis with circular orbits (`kind = "band"`),
    /// for example for chaos maps across the region between the moons.
    Band {
        /// Inner edge in km.
        min_semi_major_axis_km: f64,
        /// Outer edge in km (greater than the inner edge).
        max_semi_major_axis_km: f64,
    },
    /// Uniform band in semi-major axis with one eccentricity and one apse
    /// longitude for all particles (`kind = "eccentric_band"`, ICR-007), for
    /// example for chaos maps of core-like eccentric orbits. The draws of `a`
    /// and of the mean longitude are those of [`Distribution::Band`] with the
    /// same seed, so `eccentricity = 0` gives the same states as `band`
    /// (equal as numbers; a zero velocity component can change sign).
    EccentricBand {
        /// Inner edge (guiding semi-major axis) in km.
        min_semi_major_axis_km: f64,
        /// Outer edge in km (greater than the inner edge).
        max_semi_major_axis_km: f64,
        /// Eccentricity, in [0, 1): the epicyclic amplitude, as for
        /// [`Distribution::Core`].
        eccentricity: f64,
        /// Longitude of pericentre in degrees.
        apse_longitude_deg: f64,
    },
    /// Localised cloud (`kind = "clump"`): Gaussian offsets in position around
    /// one point on a circular orbit, for example to mimic a collision event.
    /// Every particle has the circular velocity of the centre point (no
    /// velocity dispersion), so the cloud shears out and its particles
    /// oscillate in the epicycles that the position offsets imply.
    Clump {
        /// Semi-major axis of the centre in km.
        semi_major_axis_km: f64,
        /// Inertial longitude of the centre at time 0, in degrees.
        longitude_deg: f64,
        /// Standard deviation of each position component in km (> 0).
        sigma_km: f64,
    },
    /// Loosely packed moonlet (`kind = "moonlet"`, ICR-008): particles
    /// uniform in a sphere of radius `packing_radius_km` about one point on a
    /// circular orbit. The particles have no self-gravity, so the moonlet is
    /// not bound and comes apart at once; `velocity` selects how it moves at
    /// time 0 ([`MoonletVelocity`]).
    Moonlet {
        /// Radius of the circular orbit of the centre in km.
        semi_major_axis_km: f64,
        /// Inertial longitude of the centre at time 0, in degrees.
        longitude_deg: f64,
        /// Radius of the sphere that holds the particles in km (> 0 and less
        /// than `semi_major_axis_km`).
        packing_radius_km: f64,
        /// Start velocities.
        velocity: MoonletVelocity,
    },
}

/// Start velocities of a [`Distribution::Moonlet`] (ICR-008).
///
/// In the frame that turns with the circular orbit of the centre (x radial,
/// y along the orbit), Hill's equations put the guiding centre of a particle
/// that starts at radial offset `x0` with velocity `vy0` at
/// `4 x0 + 2 vy0 / n`. With the exact n and kappa of Saturn's field the
/// factors below get a factor `(n / kappa)^2`, about 1.009 at the F ring.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MoonletVelocity {
    /// At rest in the frame that turns about Saturn at the mean motion n of
    /// the centre: `v = n z x r` (`velocity = "synchronous"`), like a moonlet
    /// that spins once per orbit. Guiding centres at about `4 x0`.
    Synchronous,
    /// Every particle has the circular velocity of the centre point, with no
    /// spin, as in [`Distribution::Clump`] (`velocity = "centre"`). Guiding
    /// centres at about `2 x0`.
    Centre,
    /// Every particle has the circular velocity at its own cylindrical radius
    /// (`velocity = "circular"`): pure Keplerian shear, guiding centres at
    /// `x0`.
    Circular,
}

/// `[collisions]` section, selected by `model` (`docs/PLAN.md` section 7.5,
/// `crates/fring-collide/SPEC.md`, ICR-009).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "model", rename_all = "snake_case", deny_unknown_fields)]
pub enum Collisions {
    /// No collisions: test particles (`model = "none"`).
    None,
    /// Stochastic superparticle model (`model = "stochastic"`). The cells are
    /// boxes in local guiding-centre coordinates (SPEC 5.1).
    Stochastic {
        /// Radius s of one real particle in km.
        particle_radius_km: f64,
        /// Number W of real particles that one superparticle represents.
        weight: f64,
        /// Radial cell size L_x in km.
        cell_radial_km: f64,
        /// Azimuthal cell size L_y in km. The backend uses D / round(D / L_y),
        /// with D the azimuthal length of the domain, so that whole cells
        /// fill the domain (SPEC 5.1).
        cell_azimuthal_km: f64,
        /// Vertical cell size L_z in km.
        cell_vertical_km: f64,
        /// One collision step every this many integrator steps (≥ 1, default
        /// 1). The collision time step is `every_steps` × `time.step_s`
        /// (SPEC 7).
        #[serde(default = "default_every_steps")]
        every_steps: u64,
        /// Normal coefficient of restitution (`[collisions.restitution]`).
        restitution: Restitution,
    },
    /// Hard-sphere model with real overlap detection (`model = "hard_sphere"`).
    /// It acts at every integrator step.
    HardSphere {
        /// Particle radius in km.
        particle_radius_km: f64,
        /// Normal coefficient of restitution (`[collisions.restitution]`).
        restitution: Restitution,
    },
}

fn default_every_steps() -> u64 {
    1
}

/// Normal coefficient of restitution ε(vₙ), selected by `law`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "law", rename_all = "snake_case", deny_unknown_fields)]
pub enum Restitution {
    /// ε is the same for all impact speeds (`law = "constant"`).
    Constant {
        /// ε, in [0, 1].
        epsilon: f64,
    },
    /// ε = (vₙ / v_c)^(−exponent) for vₙ > v_c, else 1 (`law = "power_law"`).
    /// Bridges et al. (1984) has this form; `params/sources.md` gives values.
    PowerLaw {
        /// Speed v_c in km/s (> 0).
        reference_speed_km_s: f64,
        /// Exponent (≥ 0).
        exponent: f64,
    },
}

/// `[output]` section.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputConfig {
    /// Write one particle frame every this many steps (≥ 1).
    pub frame_every_steps: u64,
    /// Write one line of statistics every this many steps (≥ 1).
    pub stats_every_steps: u64,
    /// Also write particle velocities in frames. Default: false.
    #[serde(default)]
    pub velocities: bool,
}

impl RunConfig {
    /// Parses a TOML config and checks all values.
    pub fn from_toml_str(text: &str) -> Result<Self> {
        let config: RunConfig = toml::from_str(text).map_err(|e| Error::Config(e.to_string()))?;
        config.validate()?;
        Ok(config)
    }

    /// Writes the config as TOML.
    pub fn to_toml_string(&self) -> Result<String> {
        toml::to_string(self).map_err(|e| Error::Config(e.to_string()))
    }

    /// True if `[loss]` has `on_unbound = "freeze"`.
    pub fn freeze_unbound(&self) -> bool {
        self.loss.is_some_and(|l| l.on_unbound == OnUnbound::Freeze)
    }

    /// True if `[loss]` has `moon_radius = "params"`.
    pub fn moon_radius_rule(&self) -> bool {
        self.loss
            .is_some_and(|l| l.moon_radius == MoonRadiusRule::Params)
    }

    /// True if any of the ICR-004 keys changes the behaviour of a backend:
    /// `[substep]`, `moon_radius = "params"` or `on_unbound = "freeze"`. A
    /// `[loss]` table with only the defaults does not.
    pub fn loss_or_substep_active(&self) -> bool {
        self.substep.is_some() || self.freeze_unbound() || self.moon_radius_rule()
    }

    fn validate_loss_and_substep(&self) -> Result<()> {
        let moons = self.moons.include.len();
        if self.moon_radius_rule() {
            require(
                self.freeze_unbound(),
                "loss.moon_radius = \"params\" needs loss.on_unbound = \"freeze\"".to_string(),
            )?;
            require(
                moons >= 1,
                "loss.moon_radius = \"params\" needs at least one moon in moons.include"
                    .to_string(),
            )?;
        }
        if let Some(s) = &self.substep {
            require(
                moons >= 1,
                "[substep] needs at least one moon in moons.include".to_string(),
            )?;
            require(
                s.accuracy.is_finite()
                    && (SUBSTEP_ACCURACY_MIN..=SUBSTEP_ACCURACY_MAX).contains(&s.accuracy),
                format!(
                    "substep.accuracy must be in [{SUBSTEP_ACCURACY_MIN}, {SUBSTEP_ACCURACY_MAX}], found {}",
                    s.accuracy
                ),
            )?;
            require(
                s.max_level <= SUBSTEP_MAX_LEVEL,
                format!(
                    "substep.max_level must be at most {SUBSTEP_MAX_LEVEL}, found {}",
                    s.max_level
                ),
            )?;
        }
        if self.substep.is_some() || self.moon_radius_rule() {
            require(
                moons <= 4,
                format!("substep or loss.moon_radius needs at most 4 moons, found {moons}"),
            )?;
        }
        Ok(())
    }

    /// Checks that every value is in its allowed range.
    pub fn validate(&self) -> Result<()> {
        let run = &self.run;
        require(
            !run.name.is_empty()
                && run
                    .name
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'),
            format!(
                "run.name must be non-empty and use only letters, digits, '-' and '_', found {:?}",
                run.name
            ),
        )?;
        require(
            run.seed <= i64::MAX as u64,
            format!("run.seed must be at most 2^63 - 1, found {}", run.seed),
        )?;
        require(
            !run.params_dir.is_empty(),
            "run.params_dir must not be empty".to_string(),
        )?;

        positive("time.step_s", self.time.step_s)?;
        positive("time.duration_years", self.time.duration_years)?;
        positive(
            "frame.reference_semi_major_axis_km",
            self.frame.reference_semi_major_axis_km,
        )?;

        if let Domain::Arc {
            width_deg,
            centre_longitude_deg,
        } = self.domain
        {
            require(
                width_deg.is_finite() && width_deg > 0.0 && width_deg <= 360.0,
                format!("domain.width_deg must be in (0, 360], found {width_deg}"),
            )?;
            finite("domain.centre_longitude_deg", centre_longitude_deg)?;
        }

        for (i, name) in self.moons.include.iter().enumerate() {
            require(
                !name.is_empty(),
                "moons.include has an empty name".to_string(),
            )?;
            require(
                !self.moons.include[..i].contains(name),
                format!("moons.include has {name:?} twice"),
            )?;
        }

        require(
            self.particles.count >= 1,
            "particles.count must be at least 1".to_string(),
        )?;
        self.particles.distribution.validate()?;
        self.collisions.validate()?;

        if let Some(chaos) = self.diagnostics.as_ref().and_then(|d| d.chaos) {
            positive("diagnostics.chaos.offset_km", chaos.offset_km)?;
            require(
                chaos.renorm_steps >= 1,
                "diagnostics.chaos.renorm_steps must be at least 1".to_string(),
            )?;
        }
        self.validate_loss_and_substep()?;
        require(
            self.output.frame_every_steps >= 1,
            "output.frame_every_steps must be at least 1".to_string(),
        )?;
        require(
            self.output.stats_every_steps >= 1,
            "output.stats_every_steps must be at least 1".to_string(),
        )?;
        Ok(())
    }
}

/// Checks the edges of a band: both > 0 and min < max.
fn band_edges(p: &str, min_semi_major_axis_km: f64, max_semi_major_axis_km: f64) -> Result<()> {
    positive(
        &format!("{p}.min_semi_major_axis_km"),
        min_semi_major_axis_km,
    )?;
    positive(
        &format!("{p}.max_semi_major_axis_km"),
        max_semi_major_axis_km,
    )?;
    require(
        min_semi_major_axis_km < max_semi_major_axis_km,
        format!(
            "{p}.min_semi_major_axis_km ({min_semi_major_axis_km}) must be less than \
             max_semi_major_axis_km ({max_semi_major_axis_km})"
        ),
    )
}

impl Distribution {
    fn validate(&self) -> Result<()> {
        const P: &str = "particles.distribution";
        match *self {
            Distribution::Core {
                semi_major_axis_km,
                sigma_semi_major_axis_km,
                eccentricity,
                apse_longitude_deg,
            } => {
                positive(&format!("{P}.semi_major_axis_km"), semi_major_axis_km)?;
                non_negative(
                    &format!("{P}.sigma_semi_major_axis_km"),
                    sigma_semi_major_axis_km,
                )?;
                require(
                    eccentricity.is_finite() && (0.0..1.0).contains(&eccentricity),
                    format!("{P}.eccentricity must be in [0, 1), found {eccentricity}"),
                )?;
                finite(&format!("{P}.apse_longitude_deg"), apse_longitude_deg)
            }
            Distribution::Band {
                min_semi_major_axis_km,
                max_semi_major_axis_km,
            } => band_edges(P, min_semi_major_axis_km, max_semi_major_axis_km),
            Distribution::EccentricBand {
                min_semi_major_axis_km,
                max_semi_major_axis_km,
                eccentricity,
                apse_longitude_deg,
            } => {
                band_edges(P, min_semi_major_axis_km, max_semi_major_axis_km)?;
                require(
                    eccentricity.is_finite() && (0.0..1.0).contains(&eccentricity),
                    format!("{P}.eccentricity must be in [0, 1), found {eccentricity}"),
                )?;
                finite(&format!("{P}.apse_longitude_deg"), apse_longitude_deg)
            }
            Distribution::Clump {
                semi_major_axis_km,
                longitude_deg,
                sigma_km,
            } => {
                positive(&format!("{P}.semi_major_axis_km"), semi_major_axis_km)?;
                finite(&format!("{P}.longitude_deg"), longitude_deg)?;
                positive(&format!("{P}.sigma_km"), sigma_km)
            }
            Distribution::Moonlet {
                semi_major_axis_km,
                longitude_deg,
                packing_radius_km,
                velocity: _,
            } => {
                positive(&format!("{P}.semi_major_axis_km"), semi_major_axis_km)?;
                finite(&format!("{P}.longitude_deg"), longitude_deg)?;
                positive(&format!("{P}.packing_radius_km"), packing_radius_km)?;
                require(
                    packing_radius_km < semi_major_axis_km,
                    format!(
                        "{P}.packing_radius_km ({packing_radius_km}) must be less than \
                         semi_major_axis_km ({semi_major_axis_km})"
                    ),
                )
            }
        }
    }
}

impl Collisions {
    fn validate(&self) -> Result<()> {
        match *self {
            Collisions::None => Ok(()),
            Collisions::Stochastic {
                particle_radius_km,
                weight,
                cell_radial_km,
                cell_azimuthal_km,
                cell_vertical_km,
                every_steps,
                restitution,
            } => {
                positive("collisions.particle_radius_km", particle_radius_km)?;
                positive("collisions.weight", weight)?;
                positive("collisions.cell_radial_km", cell_radial_km)?;
                positive("collisions.cell_azimuthal_km", cell_azimuthal_km)?;
                positive("collisions.cell_vertical_km", cell_vertical_km)?;
                require(
                    every_steps >= 1,
                    "collisions.every_steps must be at least 1".to_string(),
                )?;
                restitution.validate()
            }
            Collisions::HardSphere {
                particle_radius_km,
                restitution,
            } => {
                positive("collisions.particle_radius_km", particle_radius_km)?;
                restitution.validate()
            }
        }
    }
}

impl Restitution {
    fn validate(&self) -> Result<()> {
        match *self {
            Restitution::Constant { epsilon } => require(
                epsilon.is_finite() && (0.0..=1.0).contains(&epsilon),
                format!("collisions.restitution.epsilon must be in [0, 1], found {epsilon}"),
            ),
            Restitution::PowerLaw {
                reference_speed_km_s,
                exponent,
            } => {
                positive(
                    "collisions.restitution.reference_speed_km_s",
                    reference_speed_km_s,
                )?;
                non_negative("collisions.restitution.exponent", exponent)
            }
        }
    }
}

fn require(ok: bool, message: String) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(Error::Config(message))
    }
}

fn finite(name: &str, value: f64) -> Result<()> {
    require(
        value.is_finite(),
        format!("{name} must be a finite number, found {value}"),
    )
}

fn positive(name: &str, value: f64) -> Result<()> {
    require(
        value.is_finite() && value > 0.0,
        format!("{name} must be a finite number > 0, found {value}"),
    )
}

fn non_negative(name: &str, value: f64) -> Result<()> {
    require(
        value.is_finite() && value >= 0.0,
        format!("{name} must be a finite number >= 0, found {value}"),
    )
}
