//! The live mode (task W2): set-up, step planning and the projection maths.
//!
//! The integrator itself is `fring_gpu::GpuStepper` (in `web.rs`, wasm32 only).
//! Everything here is plain Rust with no web or GPU types, so the unit tests run
//! on the host:
//!
//! - [`Setup`]: the `params/` files, the moon ephemeris, the `System` (with or
//!   without Pandora) and the initial particles for an initial band.
//! - [`StepPlanner`] and [`Governor`]: how many 360 s steps to run in a frame.
//! - Views: [`ViewFrame`] and [`view_transform`] work in the inertial frame, the
//!   frame of the integrator.
//! - [`ProjectionUniform`] and [`project_twin`]: the data and the Rust twin of
//!   `shaders/project.wgsl`, which turns the double-single state into the
//!   camera-relative instances that the render pass draws.
//!
//! Units: km and s. Angles in radians. Time `t_s` is seconds since the epoch of
//! `params/` (2008-01-01 TDB).

use crate::view::ViewTransform;
use fring_model::config::{Distribution, MoonletVelocity};
use fring_model::params::{
    RingParams, moon_radii_from_toml_str, moons_from_toml_str, ring_from_toml_str,
    saturn_from_toml_str,
};
use fring_model::physics::{
    DEFAULT_INTERVAL_S, MoonInit, SaturnField, build_ephemeris, generate_particles,
    interpolate_moons, reference_mean_motion,
};
use fring_model::{Error, InitialState, MoonEphemeris, Result, SaturnParams, System};

const SATURN_TOML: &str = include_str!("../../../params/saturn.toml");
const MOONS_TOML: &str = include_str!("../../../params/moons.toml");
const RING_TOML: &str = include_str!("../../../params/ring.toml");

/// Number of test particles.
pub const PARTICLE_COUNT: u32 = 100_000;
/// Seed of the initial conditions (Philox key).
pub const SEED: u64 = 20_081_101;
/// Integration step, s (about 150 steps per orbit at the core).
pub const STEP_S: f64 = 360.0;
/// Steps per dispatch of the stepper.
pub const BLOCK_STEPS: usize = 128;
/// Length of the moon ephemeris, s (10 Julian years).
pub const SPAN_S: f64 = 10.0 * 365.25 * 86_400.0;
/// The run restarts this long before the end of the ephemeris, s.
pub const END_MARGIN_S: f64 = 2.0 * 86_400.0;
/// Slowest speed, simulated hours per real second.
pub const MIN_HOURS_PER_S: f64 = 1.0;
/// Fastest speed, simulated hours per real second.
pub const MAX_HOURS_PER_S: f64 = 500.0;
/// Most steps in one frame (30 fps budget, see the handoff).
pub const MAX_STEPS_PER_FRAME: u64 = 200;
/// Frames of GPU work that may be queued before the page stops stepping.
pub const MAX_IN_FLIGHT: u32 = 2;
/// Compute budget of the integrator, particle-steps per second. Source: the
/// W2-DROP-GPU benchmark (mode Dp with the moon-radius rule on, 7.9e8 to 8.3e8
/// particle-steps/s on the Mac mini), with 30 % of the frame kept for
/// rendering, so about 5.5e8. The speed limit for a particle count follows from
/// it ([`max_hours_for`]).
pub const COMPUTE_BUDGET_PARTICLE_STEPS_PER_S: f64 = 5.5e8;
/// The particle counts that the page offers.
pub const PARTICLE_COUNTS: [u32; 4] = [10_000, 100_000, 300_000, 1_000_000];
/// Added to the colour id of an instance that belongs to the faint reference
/// (overlay mode); the render shader draws it with a low alpha.
pub const COLOUR_FAINT_OFFSET: f32 = 10.0;
/// Instances before the particles of one picture: Saturn and up to three moons.
pub const HEAD_SLOTS: usize = 4;

/// Size of the disc of a particle, pixels.
pub const PARTICLE_SIZE_PX: f32 = 2.0;
/// Size of the disc of a moon, pixels.
pub const MOON_SIZE_PX: f32 = 10.0;
/// Colour ids of the render shader.
pub const COLOUR_PARTICLE: f32 = 0.0;
/// Colour id of Saturn.
pub const COLOUR_SATURN: f32 = 1.0;
/// Colour id of the first moon; moon `k` has `COLOUR_MOON_FIRST + k`.
pub const COLOUR_MOON_FIRST: f32 = 2.0;
/// Floats in one instance (x, y, size in pixels, colour id).
pub const FLOATS_PER_INSTANCE: usize = 4;

/// Name of the moon that the Pandora switch controls.
pub const PANDORA: &str = "pandora";

/// The initial band of the particles.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Band {
    /// Narrow core: Gaussian in a (display width from the dust strands), with
    /// the eccentricity and apse of the F ring.
    Core,
    /// Circular orbits, uniform in a between `min_km` and `max_km`.
    Circular {
        /// Smallest semi-major axis, km.
        min_km: f64,
        /// Largest semi-major axis, km.
        max_km: f64,
    },
    /// Loosely packed moonlet (ICR-008): uniform in a sphere of `radius_km`
    /// about a point on the circle of radius `a_km`, 180° from Prometheus at
    /// t = 0 ([`Setup::moonlet_longitude_deg`]).
    Moonlet {
        /// Orbital radius of the centre, km.
        a_km: f64,
        /// Packing radius, km.
        radius_km: f64,
        /// Start velocities.
        velocity: MoonletVelocity,
    },
}

/// Packing radius of the moonlet when the page opens, km.
pub const MOONLET_RADIUS_KM: f64 = 20.0;

/// The moonlet velocity mode of the page control: 0 synchronous, 1 centre,
/// 2 circular.
pub fn moonlet_velocity(index: u32) -> Option<MoonletVelocity> {
    match index {
        0 => Some(MoonletVelocity::Synchronous),
        1 => Some(MoonletVelocity::Centre),
        2 => Some(MoonletVelocity::Circular),
        _ => None,
    }
}

/// The circular orbit that the moonlet view follows: radius `a_km`, longitude
/// `lon0_rad + n_rad_s t`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Guide {
    /// Orbital radius, km.
    pub a_km: f64,
    /// Inertial longitude at t = 0, rad.
    pub lon0_rad: f64,
    /// Mean motion (exact, Saturn's field), rad/s.
    pub n_rad_s: f64,
}

impl Guide {
    /// Inertial (x, y) of the guide point at time `t_s`, km.
    pub fn position_km(&self, t_s: f64) -> [f64; 2] {
        let th = self.lon0_rad + self.n_rad_s * t_s;
        [self.a_km * th.cos(), self.a_km * th.sin()]
    }
}

/// What the view follows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ViewFrame {
    /// Fixed camera, centred on Saturn: everything orbits anticlockwise.
    Inertial,
    /// The camera turns at n0 with the ring core, so the core stays still.
    Rotating,
    /// Centred on moon `k` (index in the full ephemeris) and turning with it:
    /// Saturn is straight down and the orbital motion goes to the left.
    Moon(usize),
    /// Centred on the circular orbit of the moonlet centre from t = 0 and
    /// turning with it (Hill's frame), oriented as `Moon`: Saturn straight
    /// down, orbital motion to the left.
    Moonlet(Guide),
}

/// The `params/` files, parsed. Cheap to build; the ephemeris is separate.
pub struct Setup {
    /// Saturn's parameters.
    pub saturn: SaturnParams,
    /// The F ring.
    pub ring: RingParams,
    /// The moons at t = 0.
    pub moons: Vec<MoonInit>,
    field: SaturnField<f64>,
    n0_rad_s: f64,
}

/// Mean radii (km) of the moons of `system`, in the order of its tracks, read
/// from the text of `params/moons.toml` and matched BY NAME. The result goes to
/// `GpuStepper::set_moon_radii_km` (ICR-006).
///
/// # Errors
/// A track whose name has no radius in the text, or a bad text.
pub fn moon_radii_for(system: &System, moons_toml: &str) -> Result<Vec<f64>> {
    let table = moon_radii_from_toml_str(moons_toml)?;
    system
        .moons
        .tracks
        .iter()
        .map(|t| {
            table
                .iter()
                .find(|(name, _)| *name == t.name)
                .map(|(_, r)| *r)
                .ok_or_else(|| Error::Config(format!("no radius_km for moon {:?}", t.name)))
        })
        .collect()
}

/// The moon radii of `system` from the built-in `params/moons.toml`.
pub fn builtin_moon_radii_for(system: &System) -> Result<Vec<f64>> {
    moon_radii_for(system, MOONS_TOML)
}

impl Setup {
    /// Reads the `params/` files that are compiled into the crate.
    pub fn builtin() -> Result<Self> {
        Self::from_texts(SATURN_TOML, MOONS_TOML, RING_TOML)
    }

    /// Reads the three `params/` files from their text.
    pub fn from_texts(saturn_toml: &str, moons_toml: &str, ring_toml: &str) -> Result<Self> {
        let saturn = saturn_from_toml_str(saturn_toml)?;
        let ring = ring_from_toml_str(ring_toml)?;
        let moons = moons_from_toml_str(moons_toml)?;
        let field = SaturnField::<f64>::from_params(&saturn);
        let n0_rad_s = reference_mean_motion(&field, ring.semi_major_axis_km);
        Ok(Self {
            saturn,
            ring,
            moons,
            field,
            n0_rad_s,
        })
    }

    /// The reference semi-major axis a0, km.
    pub fn a0_km(&self) -> f64 {
        self.ring.semi_major_axis_km
    }

    /// The mean motion n0 of the ring core, rad/s.
    pub fn n0_rad_s(&self) -> f64 {
        self.n0_rad_s
    }

    /// Orbit radius of moon `k` at t = 0 (equatorial plane), km.
    pub fn moon_radius_km(&self, k: usize) -> f64 {
        let p = self.moons[k].state.position_km;
        p.x.hypot(p.y)
    }

    /// Distance from the orbit of moon `k` to the ring core, km.
    pub fn moon_gap_km(&self, k: usize) -> f64 {
        (self.moon_radius_km(k) - self.a0_km()).abs()
    }

    /// Inertial longitude of the moonlet centre at t = 0: 180° from
    /// Prometheus (AJ, 2026-11-03), in [0°, 360°). 180° if there is no
    /// Prometheus.
    pub fn moonlet_longitude_deg(&self) -> f64 {
        let lon = self.moon_index("prometheus").map_or(0.0, |k| {
            let p = self.moons[k].state.position_km;
            p.y.atan2(p.x).to_degrees()
        });
        (lon + 180.0).rem_euclid(360.0)
    }

    /// The circular orbit of a moonlet centre at radius `a_km`.
    pub fn moonlet_guide(&self, a_km: f64) -> Guide {
        Guide {
            a_km,
            lon0_rad: self.moonlet_longitude_deg().to_radians(),
            n_rad_s: reference_mean_motion(&self.field, a_km),
        }
    }

    /// Index of the moon called `name`, if there is one.
    pub fn moon_index(&self, name: &str) -> Option<usize> {
        self.moons.iter().position(|m| m.name == name)
    }

    /// Integrates the moons for `duration_s` and tabulates them (IAS15, f64).
    /// This takes about 0.07 s per year natively.
    pub fn build_ephemeris(&self, duration_s: f64) -> Result<MoonEphemeris> {
        build_ephemeris(&self.saturn, &self.moons, duration_s, DEFAULT_INTERVAL_S)
    }

    /// The system for the particles: Saturn and the moons of `full`. With
    /// `pandora_on == false` the track of Pandora is removed. Prometheus keeps
    /// the path it has in `full`, which includes Pandora's pull on it.
    pub fn system(&self, full: &MoonEphemeris, pandora_on: bool) -> System {
        let mut moons = full.clone();
        if !pandora_on {
            moons.tracks.retain(|t| t.name != PANDORA);
        }
        System {
            saturn: self.saturn,
            moons,
        }
    }

    /// The band of the preset `index`: 0 core, 1 circular a0 +- a0 e, 2 circular
    /// between the moons (150 km inside the two outer orbits).
    pub fn preset_band(&self, index: u32) -> Band {
        let a0 = self.a0_km();
        match index {
            1 => {
                let w = a0 * self.ring.eccentricity;
                Band::Circular {
                    min_km: a0 - w,
                    max_km: a0 + w,
                }
            }
            2 => {
                let radii: Vec<f64> = (0..self.moons.len())
                    .map(|k| self.moon_radius_km(k))
                    .collect();
                let lo = radii.iter().copied().fold(f64::INFINITY, f64::min);
                let hi = radii.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                Band::Circular {
                    min_km: lo + 150.0,
                    max_km: hi - 150.0,
                }
            }
            _ => Band::Core,
        }
    }

    /// The `Distribution` of a band.
    pub fn distribution(&self, band: Band) -> Distribution {
        match band {
            Band::Core => Distribution::Core {
                semi_major_axis_km: self.a0_km(),
                sigma_semi_major_axis_km: self.ring.dust_strand_width_min_km,
                eccentricity: self.ring.eccentricity,
                apse_longitude_deg: self.ring.apse_longitude_deg,
                inclination_deg: 0.0,
            },
            Band::Circular { min_km, max_km } => Distribution::Band {
                min_semi_major_axis_km: min_km,
                max_semi_major_axis_km: max_km,
            },
            Band::Moonlet {
                a_km,
                radius_km,
                velocity,
            } => Distribution::Moonlet {
                semi_major_axis_km: a_km,
                longitude_deg: self.moonlet_longitude_deg(),
                packing_radius_km: radius_km,
                velocity,
            },
        }
    }

    /// The initial particles (inertial, t = 0) of `count` particles in `band`.
    ///
    /// # Errors
    /// [`fring_model::Error::Input`] if the band is empty or not finite, if a
    /// moonlet does not have 0 < packing radius < orbital radius, or if a
    /// particle state is not valid.
    pub fn initial(&self, band: Band, count: u32) -> Result<InitialState> {
        if let Band::Circular { min_km, max_km } = band
            && !(min_km.is_finite() && max_km.is_finite() && 0.0 < min_km && min_km < max_km)
        {
            return Err(fring_model::Error::Input(format!(
                "the band {min_km} to {max_km} km is empty or not finite"
            )));
        }
        if let Band::Moonlet {
            a_km, radius_km, ..
        } = band
            && !(a_km.is_finite() && radius_km.is_finite() && 0.0 < radius_km && radius_km < a_km)
        {
            return Err(fring_model::Error::Input(format!(
                "a moonlet needs 0 < packing radius ({radius_km} km) < orbital radius ({a_km} km)"
            )));
        }
        let particles = generate_particles(&self.field, &self.distribution(band), count, SEED)?;
        Ok(InitialState {
            time_s: 0.0,
            particles,
        })
    }

    /// Inertial (x, y) of every moon of `full` at time `t_s`, km.
    ///
    /// # Errors
    /// [`fring_model::Error::Input`] if `t_s` is outside the ephemeris.
    pub fn moon_positions(&self, full: &MoonEphemeris, t_s: f64) -> Result<Vec<[f64; 2]>> {
        let m = interpolate_moons(full, t_s)?;
        Ok(m.moons
            .iter()
            .map(|b| [b.position_km.x, b.position_km.y])
            .collect())
    }
}

/// The map from the inertial frame to the view plane at one time.
///
/// `moons` holds the inertial (x, y) of every moon of the full ephemeris.
///
/// # Panics
/// If `frame` is `Moon(k)` and `k` is not an index of `moons`.
pub fn view_transform(
    frame: ViewFrame,
    n0_rad_s: f64,
    t_s: f64,
    moons: &[[f64; 2]],
) -> ViewTransform {
    match frame {
        ViewFrame::Inertial => ViewTransform::IDENTITY,
        // The rotating frame is the inertial frame turned by -n0 t.
        ViewFrame::Rotating => ViewTransform::new([0.0, 0.0], -n0_rad_s * t_s),
        ViewFrame::Moon(k) => {
            let m = moons[k];
            ViewTransform::new(m, std::f64::consts::FRAC_PI_2 - m[1].atan2(m[0]))
        }
        ViewFrame::Moonlet(g) => ViewTransform::new(
            g.position_km(t_s),
            std::f64::consts::FRAC_PI_2 - (g.lon0_rad + g.n_rad_s * t_s),
        ),
    }
}

/// Writes the instances before the particles: Saturn, then the moons that are
/// drawn (all of `moons`, except Pandora when `pandora_on` is false), as
/// `FLOATS_PER_INSTANCE` floats each, relative to the eye. Returns the number
/// of instances. The transform and the eye subtraction run in f64 first.
#[allow(clippy::too_many_arguments)] // plain data in, plain data out
pub fn head_instances(
    setup: &Setup,
    view: &ViewTransform,
    eye_km: [f64; 2],
    px_per_km: f64,
    moons: &[[f64; 2]],
    pandora_on: bool,
    out: &mut Vec<f32>,
) -> usize {
    head_instances_styled(
        setup, view, eye_km, px_per_km, moons, pandora_on, true, 0.0, out,
    )
}

/// As [`head_instances`], with a choice: `with_saturn` false leaves Saturn out
/// (the faint overlay draws the moons only), and `colour_offset` is added to
/// every colour id (`COLOUR_FAINT_OFFSET` for the faint reference).
#[allow(clippy::too_many_arguments)] // plain data in, plain data out
pub fn head_instances_styled(
    setup: &Setup,
    view: &ViewTransform,
    eye_km: [f64; 2],
    px_per_km: f64,
    moons: &[[f64; 2]],
    pandora_on: bool,
    with_saturn: bool,
    colour_offset: f32,
    out: &mut Vec<f32>,
) -> usize {
    out.clear();
    let mut push = |p: [f64; 2], size_px: f32, colour: f32| {
        let v = view.apply(p);
        out.extend_from_slice(&[
            (v[0] - eye_km[0]) as f32,
            (v[1] - eye_km[1]) as f32,
            size_px,
            colour + colour_offset,
        ]);
    };
    if with_saturn {
        push(
            [0.0, 0.0],
            (2.0 * setup.saturn.reference_radius_km * px_per_km) as f32,
            COLOUR_SATURN,
        );
    }
    for (k, m) in moons.iter().enumerate() {
        if !pandora_on && setup.moons[k].name == PANDORA {
            continue;
        }
        push(*m, MOON_SIZE_PX, COLOUR_MOON_FIRST + k as f32);
    }
    out.len() / FLOATS_PER_INSTANCE
}

/// Splits `x` into a double-single pair `(hi, lo)` with `hi = x as f32`.
pub fn ds_split(x: f64) -> [f32; 2] {
    let hi = x as f32;
    [hi, (x - f64::from(hi)) as f32]
}

/// Word index of the high part of coordinate plane `plane` (0..6: x, y, z, vx,
/// vy, vz) of particle `i` in the state buffer of `GpuStepper` (ICR-005 item 5).
/// The low part is at the next index.
pub fn ds_word(plane: usize, i: usize, capacity: usize) -> usize {
    2 * (plane * capacity + i)
}

/// The per-frame uniform of `shaders/project.wgsl` (48 bytes).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProjectionUniform {
    /// High part of `q`, the inertial point that is under the eye, km.
    pub q_hi: [f32; 2],
    /// Low part of `q`.
    pub q_lo: [f32; 2],
    /// Cosine and sine of the turn angle of the view.
    pub rot: [f32; 2],
    /// Diameter of a particle disc, pixels.
    pub size_px: f32,
    /// A runtime 1.0 (the guard against Metal fast math, see the shader).
    pub one: f32,
    /// Number of particles.
    pub n: u32,
    /// Capacity of the state buffer (particles per plane).
    pub cap: u32,
    /// Index of the first particle instance in the instance buffer.
    pub first: u32,
}

impl ProjectionUniform {
    /// The uniform for a view, a camera centre (the eye, view-plane km) and
    /// the particle layout.
    pub fn new(view: &ViewTransform, eye_km: [f64; 2], n: u32, cap: u32, first: u32) -> Self {
        let q = view.source_of(eye_km);
        let (qx, qy) = (ds_split(q[0]), ds_split(q[1]));
        let [c, s] = view.cos_sin();
        Self {
            q_hi: [qx[0], qy[0]],
            q_lo: [qx[1], qy[1]],
            rot: [c as f32, s as f32],
            size_px: PARTICLE_SIZE_PX,
            one: 1.0,
            n,
            cap,
            first,
        }
    }

    /// The 48 bytes for `queue.write_buffer` (the layout of the shader struct).
    pub fn to_bytes(&self) -> [u8; 48] {
        let mut out = [0u8; 48];
        let f = [
            self.q_hi[0],
            self.q_hi[1],
            self.q_lo[0],
            self.q_lo[1],
            self.rot[0],
            self.rot[1],
            self.size_px,
            self.one,
        ];
        for (k, x) in f.iter().enumerate() {
            out[4 * k..4 * k + 4].copy_from_slice(&x.to_le_bytes());
        }
        let u = [self.n, self.cap, self.first, 0];
        for (k, x) in u.iter().enumerate() {
            out[32 + 4 * k..36 + 4 * k].copy_from_slice(&x.to_le_bytes());
        }
        out
    }
}

/// The Rust twin of `project` in `shaders/project.wgsl`: the view position
/// (km, relative to the eye) of a particle whose x and y are the
/// double-single pairs `x` and `y`. The same f32 operations in the same order.
pub fn project_twin(x: [f32; 2], y: [f32; 2], u: &ProjectionUniform) -> [f32; 2] {
    let dx = (x[0] - u.q_hi[0]) + (x[1] - u.q_lo[0]);
    let dy = (y[0] - u.q_hi[1]) + (y[1] - u.q_lo[1]);
    [u.rot[0] * dx - u.rot[1] * dy, u.rot[1] * dx + u.rot[0] * dy]
}

/// Turns the time since the last frame and the speed into whole steps.
#[derive(Clone, Debug, Default)]
pub struct StepPlanner {
    acc_steps: f64,
}

impl StepPlanner {
    /// A planner with no steps owed.
    pub fn new() -> Self {
        Self::default()
    }

    /// Steps to run in a frame of `dt_s` real seconds at `hours_per_s`
    /// (clamped to the slider range). A fraction of a step is carried to the
    /// next frame. At most `cap` steps are returned; the rest is dropped (the
    /// simulation runs slower than asked). Returns `(steps, limited)`.
    pub fn plan(&mut self, dt_s: f64, hours_per_s: f64, cap: u64) -> (u64, bool) {
        let h = hours_per_s.clamp(MIN_HOURS_PER_S, MAX_HOURS_PER_S);
        self.acc_steps += dt_s.max(0.0) * h * 3600.0 / STEP_S;
        let whole = self.acc_steps.floor();
        let n = (whole as u64).min(cap);
        let limited = whole as u64 > cap;
        self.acc_steps -= whole;
        (n, limited)
    }
}

/// What the compare control shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompareMode {
    /// Only the live ring.
    Off,
    /// The canvas is split: reference on the left, live on the right.
    Side,
    /// The reference is drawn faint under the live dots.
    Overlay,
}

impl CompareMode {
    /// 0 = off, 1 = side by side, 2 = overlay.
    pub fn from_index(index: u32) -> Option<Self> {
        match index {
            0 => Some(Self::Off),
            1 => Some(Self::Side),
            2 => Some(Self::Overlay),
            _ => None,
        }
    }
}

/// The fastest speed (simulated hours per real second) that the compute budget
/// allows for `count` particles: `min(500, budget * 360 s / 3600 s / count)`.
/// This is 500 h/s at 10^5, 183 h/s at 3x10^5 and 55 h/s at 10^6.
pub fn max_hours_for(count: u32) -> f64 {
    let h = COMPUTE_BUDGET_PARTICLE_STEPS_PER_S * STEP_S / 3600.0 / f64::from(count.max(1));
    h.clamp(MIN_HOURS_PER_S, MAX_HOURS_PER_S)
}

/// Particle capacity of the stepper: `count` rounded up to a multiple of the
/// workgroup size (as `GpuStepper` does).
pub fn capacity_for(count: u32, workgroup: u32) -> u64 {
    u64::from(count)
        .next_multiple_of(u64::from(workgroup))
        .max(u64::from(workgroup))
}

/// Bytes of the state buffer of the stepper: 6 planes of `cap` double-single
/// values, 8 bytes each.
pub fn state_bytes(cap: u64) -> u64 {
    6 * cap * 8
}

/// Bytes of the reference copy: the planes x and y only (planes 0 and 1), 8
/// bytes per value. A view mode that reads z would need plane 2 in the copy as
/// well, and a larger `ref_bytes`.
pub fn ref_bytes(cap: u64) -> u64 {
    2 * cap * 8
}

/// Bytes of the status buffer of the stepper (ICR-006): 8 header words and one
/// drop-reason word for each of the `cap` particles.
pub fn status_bytes(cap: u64) -> u64 {
    (8 + cap) * 4
}

/// Steps to run in a frame so that the run does not pass the end of the moon
/// data. `left_s` is the time left, `step_s` the step. Returns the steps and
/// `true` if the run has ended (not more than one step left): then no step runs.
pub fn limit_to_end(steps: u64, left_s: f64, step_s: f64) -> (u64, bool) {
    if left_s <= step_s {
        (0, true)
    } else {
        (steps.min((left_s / step_s) as u64), false)
    }
}

/// The line under the time readout: the drop counts (ICR-006: unbound, inside
/// Prometheus, inside Pandora). The Pandora part is left out if Pandora is off.
pub fn drop_text(counts: [u32; 3], pandora_on: bool) -> String {
    let group = |n: u32| {
        let s = n.to_string();
        let mut out = String::new();
        for (k, c) in s.chars().enumerate() {
            if k > 0 && (s.len() - k).is_multiple_of(3) {
                out.push(',');
            }
            out.push(c);
        }
        out
    };
    let mut t = format!(
        "Dropped: {} unbound \u{b7} {} inside Prometheus",
        group(counts[0]),
        group(counts[1])
    );
    if pandora_on {
        t.push_str(&format!(" \u{b7} {} inside Pandora", group(counts[2])));
    }
    t
}

/// Instances in the instance buffer for `count` particles: the live picture
/// (`HEAD_SLOTS` + `count`) and the reference picture, one after the other.
pub fn instance_slots(count: u32) -> u64 {
    2 * (HEAD_SLOTS as u64 + u64::from(count))
}

/// Index of the first instance of the reference picture (its head).
pub fn ref_base_slot(count: u32) -> u32 {
    HEAD_SLOTS as u32 + count
}

/// Bytes of the instance buffer for `count` particles.
pub fn instance_bytes(count: u32) -> u64 {
    instance_slots(count) * (FLOATS_PER_INSTANCE * 4) as u64
}

/// Checks that `count` particles fit the limits of the device, before any
/// buffer is made. Returns a message for the page if they do not.
pub fn count_fits(
    count: u32,
    workgroup: u32,
    max_binding_bytes: u64,
    max_buffer_bytes: u64,
    max_workgroups_per_dim: u32,
) -> std::result::Result<(), String> {
    let cap = capacity_for(count, workgroup);
    let needs = [
        ("state", state_bytes(cap)),
        ("reference", ref_bytes(cap)),
        ("instances", instance_bytes(count)),
    ];
    let mb = |b: u64| b as f64 / 1e6;
    for (what, bytes) in needs {
        if bytes > max_binding_bytes.min(max_buffer_bytes) {
            return Err(format!(
                "{count} particles need {:.0} MB for the {what} buffer; this GPU allows {:.0} MB. Choose fewer particles.",
                mb(bytes),
                mb(max_binding_bytes.min(max_buffer_bytes))
            ));
        }
    }
    if cap.div_ceil(64) > u64::from(max_workgroups_per_dim) {
        return Err(format!(
            "{count} particles need more workgroups than this GPU allows in one dispatch. Choose fewer particles."
        ));
    }
    Ok(())
}

/// Lowers the cap on steps per frame when frames are slow and raises it again
/// when they are fast.
#[derive(Clone, Debug)]
pub struct Governor {
    ema_dt_s: f64,
    cap: u64,
}

impl Default for Governor {
    fn default() -> Self {
        Self {
            ema_dt_s: 1.0 / 60.0,
            cap: MAX_STEPS_PER_FRAME,
        }
    }
}

impl Governor {
    /// Takes the real time of the last frame.
    pub fn observe(&mut self, dt_s: f64) {
        self.ema_dt_s = 0.9 * self.ema_dt_s + 0.1 * dt_s.clamp(0.0, 0.25);
        if self.ema_dt_s > 1.0 / 28.0 {
            self.cap = ((self.cap as f64 * 0.9) as u64).max(4);
        } else if self.ema_dt_s < 1.0 / 40.0 {
            self.cap = (self.cap + 2).min(MAX_STEPS_PER_FRAME);
        }
    }

    /// The most steps for the next frame.
    pub fn cap(&self) -> u64 {
        self.cap
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fring_model::Vec3;

    /// Two days of ephemeris: enough for the tests, and fast in a debug build.
    fn setup_and_eph() -> (Setup, MoonEphemeris) {
        let s = Setup::builtin().unwrap();
        let e = s.build_ephemeris(2.0 * 86_400.0).unwrap();
        (s, e)
    }

    #[test]
    fn ring_params_come_from_the_params_file() {
        let s = Setup::builtin().unwrap();
        assert_eq!(s.a0_km(), 140222.43);
        assert_eq!(s.ring.eccentricity, 0.00235);
        assert_eq!(s.moons.len(), 2);
        assert_eq!(s.moons[0].name, "prometheus");
        assert_eq!(s.moon_index(PANDORA), Some(1));
    }

    #[test]
    fn presets_give_the_expected_distributions() {
        let s = Setup::builtin().unwrap();
        match s.distribution(s.preset_band(0)) {
            Distribution::Core {
                semi_major_axis_km,
                sigma_semi_major_axis_km,
                eccentricity,
                ..
            } => {
                assert_eq!(semi_major_axis_km, 140222.43);
                assert_eq!(sigma_semi_major_axis_km, 10.0);
                assert_eq!(eccentricity, 0.00235);
            }
            d => panic!("{d:?}"),
        }
        let Band::Circular { min_km, max_km } = s.preset_band(1) else {
            panic!()
        };
        assert!((max_km - min_km - 2.0 * 140222.43 * 0.00235).abs() < 1e-9);
        let Band::Circular { min_km, max_km } = s.preset_band(2) else {
            panic!()
        };
        // Between the two moons, with 150 km clearance.
        assert!(min_km > s.moon_radius_km(0) && max_km < s.moon_radius_km(1));
        assert!(min_km < s.a0_km() && s.a0_km() < max_km);
    }

    #[test]
    fn initial_particles_have_the_band_and_the_count() {
        let s = Setup::builtin().unwrap();
        let init = s.initial(Band::Core, 500).unwrap();
        assert_eq!(init.particles.len(), 500);
        assert_eq!(init.time_s, 0.0);
        let band = Band::Circular {
            min_km: 140_000.0,
            max_km: 140_100.0,
        };
        let p = s.initial(band, 500).unwrap().particles;
        for i in 0..p.len() {
            let r = p.position_km(i).norm();
            assert!((140_000.0..=140_100.0).contains(&r), "r = {r}");
        }
        let bad = Band::Circular {
            min_km: 2.0,
            max_km: 1.0,
        };
        assert!(s.initial(bad, 10).is_err());
    }

    #[test]
    fn pandora_off_keeps_prometheus_unchanged() {
        let (s, e) = setup_and_eph();
        let on = s.system(&e, true);
        let off = s.system(&e, false);
        assert_eq!(on.moons.tracks.len(), 2);
        assert_eq!(off.moons.tracks.len(), 1);
        assert_eq!(off.moons.tracks[0], e.tracks[0]);
        assert_eq!(off.moons.tracks[0].name, "prometheus");
        off.validate().unwrap();
    }

    #[test]
    fn moons_stay_near_their_start_radius() {
        let (s, e) = setup_and_eph();
        for t in [0.0, 3.0e4, 1.7e5] {
            let m = s.moon_positions(&e, t).unwrap();
            for (k, p) in m.iter().enumerate() {
                // The radius changes by up to a e (about 600 km) over an orbit; 1500 km is a loose bound.
                assert!((p[0].hypot(p[1]) - s.moon_radius_km(k)).abs() < 1500.0);
            }
        }
        assert!(s.moon_positions(&e, 1.0e9).is_err());
    }

    #[test]
    fn planner_carries_fractions_and_clamps() {
        let mut p = StepPlanner::new();
        // 6 h/s = 60 steps/s: a 1/60 s frame owes exactly 1 step.
        let mut total = 0;
        for _ in 0..60 {
            total += p.plan(1.0 / 60.0, 6.0, 200).0;
        }
        assert!((59..=60).contains(&total), "{total}");
        // 1 h/s = 10 steps/s: short frames carry the fraction.
        let mut p = StepPlanner::new();
        let n: Vec<u64> = (0..25).map(|_| p.plan(0.01, 1.0, 200).0).collect();
        assert_eq!(n.iter().sum::<u64>(), 2); // 0.25 s x 10 steps/s = 2.5
        // Clamps: the speed to the slider range, the steps to the cap.
        let mut p = StepPlanner::new();
        assert_eq!(p.plan(1.0, 1e6, 200), (200, true));
        let mut p = StepPlanner::new();
        assert_eq!(p.plan(1.0, 0.0, 200), (10, false));
        // Nothing for a negative dt.
        assert_eq!(StepPlanner::new().plan(-1.0, 6.0, 200), (0, false));
    }

    #[test]
    fn governor_lowers_the_cap_when_slow_and_recovers() {
        let mut g = Governor::default();
        assert_eq!(g.cap(), MAX_STEPS_PER_FRAME);
        for _ in 0..60 {
            g.observe(0.05);
        }
        assert!(g.cap() < 20, "{}", g.cap());
        for _ in 0..400 {
            g.observe(1.0 / 60.0);
        }
        assert_eq!(g.cap(), MAX_STEPS_PER_FRAME);
    }

    #[test]
    fn state_words_follow_the_layout_of_icr_005() {
        // Particle i of plane c: high at 2 (c cap + i), low next to it.
        assert_eq!(ds_word(0, 0, 64), 0);
        assert_eq!(ds_word(0, 5, 64), 10);
        assert_eq!(ds_word(1, 5, 64), 2 * (64 + 5));
        assert_eq!(ds_word(5, 63, 64), 2 * (5 * 64 + 63));
    }

    #[test]
    fn uniform_bytes_have_the_shader_layout() {
        let v = ViewTransform::new([1.0, 2.0], 0.5);
        let u = ProjectionUniform::new(&v, [3.0, 4.0], 7, 64, 3);
        let b = u.to_bytes();
        let f = |k: usize| f32::from_le_bytes(b[4 * k..4 * k + 4].try_into().unwrap());
        let w = |k: usize| u32::from_le_bytes(b[4 * k..4 * k + 4].try_into().unwrap());
        assert_eq!(f(0), u.q_hi[0]);
        assert_eq!(f(3), u.q_lo[1]);
        assert_eq!(f(4), u.rot[0]);
        assert_eq!(f(6), PARTICLE_SIZE_PX);
        assert_eq!(f(7), 1.0);
        assert_eq!((w(8), w(9), w(10)), (7, 64, 3));
    }

    /// `source_of` is the inverse of the view map for the eye.
    #[test]
    fn eye_source_goes_back_to_the_eye() {
        let v = ViewTransform::new([1.0e5, -3.0e4], 1.1);
        let eye = [250.0, -40.0];
        let q = v.source_of(eye);
        let back = v.apply(q);
        assert!((back[0] - eye[0]).abs() < 1e-9 && (back[1] - eye[1]).abs() < 1e-9);
    }

    /// The projection of a double-single position, relative to the eye, is
    /// accurate to a small part of a pixel at any zoom, in every view.
    ///
    /// Bound: the subtraction `hi - q_hi` is exact (Sterbenz) near the eye, the
    /// low parts add up to 1e-9 km, and the sum and the rotation round by 6e-8
    /// of |d| each, so the error is at most about 3e-7 |d| + 1e-9 km. A
    /// particle in the window has |d| < 2000 px / px_per_km. So the error is at
    /// most 2000 * 3e-7 = 6e-4 px, and the tolerance is 2e-3 px. A projection
    /// that lost the low parts would be wrong by up to 0.008 km (80 px at
    /// 10 km wide), so this test catches that.
    #[test]
    fn projection_twin_is_exact_to_a_pixel_fraction_in_every_view() {
        let (s, e) = setup_and_eph();
        let t = 1.0e5;
        let moons = s.moon_positions(&e, t).unwrap();
        let pts: Vec<Vec3<f64>> = (0..500)
            .map(|i| {
                let a = 0.37 * f64::from(i);
                let r = 140_000.0 + 0.9 * f64::from(i);
                Vec3::new(r * a.cos(), r * a.sin(), 0.0)
            })
            .collect();
        let height_px = 1000.0;
        for frame in [
            ViewFrame::Inertial,
            ViewFrame::Rotating,
            ViewFrame::Moon(0),
            ViewFrame::Moon(1),
        ] {
            let view = view_transform(frame, s.n0_rad_s(), t, &moons);
            let exact: Vec<[f64; 2]> = pts.iter().map(|p| view.apply([p.x, p.y])).collect();
            let eye = exact[7];
            for half_height_km in [1.8e5, 1.0e4, 300.0, 5.0] {
                let px_per_km = 0.5 * height_px / half_height_km;
                let u = ProjectionUniform::new(&view, eye, 500, 512, 3);
                let mut checked = 0;
                for (p, ex) in pts.iter().zip(&exact) {
                    let (rx, ry) = (ex[0] - eye[0], ex[1] - eye[1]);
                    if rx.abs() * px_per_km > 2000.0 || ry.abs() * px_per_km > 2000.0 {
                        continue;
                    }
                    let got = project_twin(ds_split(p.x), ds_split(p.y), &u);
                    let err_px = (f64::from(got[0]) - rx).hypot(f64::from(got[1]) - ry) * px_per_km;
                    assert!(err_px < 2e-3, "{frame:?} {half_height_km} km: {err_px} px");
                    checked += 1;
                }
                assert!(checked >= 1, "{frame:?}: nothing in the window");
            }
        }
    }

    /// Circular orbit at a0 with the mean motion n0: fixed in the rotating
    /// view, going round once per 2 pi / n0 in the inertial view.
    #[test]
    fn rotating_view_holds_the_core_still() {
        let (s, e) = setup_and_eph();
        let n0 = s.n0_rad_s();
        for t in [0.0, 1.0e4, 1.0e5] {
            let moons = s.moon_positions(&e, t).unwrap();
            let a = s.a0_km();
            let p = [a * (n0 * t).cos(), a * (n0 * t).sin()];
            let v = view_transform(ViewFrame::Rotating, n0, t, &moons).apply(p);
            assert!((v[0] - a).abs() < 1e-7 && v[1].abs() < 1e-7, "{v:?}");
            let i = view_transform(ViewFrame::Inertial, n0, t, &moons).apply(p);
            assert_eq!(i, p);
        }
    }

    /// In a moon view the moon is at the origin and Saturn is straight down at
    /// the moon's distance. Orbital motion goes to the left: a point 1 mrad
    /// ahead of the moon on its circle is at x = -r sin(1e-3), y =
    /// -r (1 - cos(1e-3)). Tolerance 1e-6 km: the inputs are 1.4e5 km, so the
    /// f64 rounding is about 3e-11 km.
    #[test]
    fn moon_view_puts_the_moon_at_the_centre_saturn_below_motion_left() {
        let (s, e) = setup_and_eph();
        let t = 1.5e5;
        let moons = s.moon_positions(&e, t).unwrap();
        for (k, m) in moons.iter().enumerate() {
            let view = view_transform(ViewFrame::Moon(k), s.n0_rad_s(), t, &moons);
            let mv = view.apply(*m);
            assert!(mv[0].abs() < 1e-6 && mv[1].abs() < 1e-6);
            let r = m[0].hypot(m[1]);
            let sat = view.apply([0.0, 0.0]);
            assert!(sat[0].abs() < 1e-6 && (sat[1] + r).abs() < 1e-6, "{sat:?}");
            let lon = m[1].atan2(m[0]) + 1e-3;
            let v = view.apply([r * lon.cos(), r * lon.sin()]);
            assert!((v[0] + r * 1e-3_f64.sin()).abs() < 1e-6, "{v:?}");
            assert!((v[1] + r * (1.0 - 1e-3_f64.cos())).abs() < 1e-6, "{v:?}");
        }
    }

    fn moonlet_band(velocity: MoonletVelocity) -> Band {
        Band::Moonlet {
            a_km: 140_222.43,
            radius_km: MOONLET_RADIUS_KM,
            velocity,
        }
    }

    /// The moonlet starts 180° from Prometheus, inside its packing radius.
    #[test]
    fn moonlet_starts_opposite_prometheus_inside_its_radius() {
        let s = Setup::builtin().unwrap();
        let k = s.moon_index("prometheus").unwrap();
        let pr = s.moons[k].state.position_km;
        let g = s.moonlet_guide(140_222.43);
        let c = g.position_km(0.0);
        // Opposite: the unit vectors add to zero.
        let (rp, rc) = (pr.x.hypot(pr.y), c[0].hypot(c[1]));
        assert!((pr.x / rp + c[0] / rc).abs() < 1e-12 && (pr.y / rp + c[1] / rc).abs() < 1e-12);
        let init = s
            .initial(moonlet_band(MoonletVelocity::Synchronous), 2000)
            .unwrap();
        let p = &init.particles;
        for i in 0..p.len() {
            let d = (p.x_km[i] - c[0]).hypot(p.y_km[i] - c[1]).hypot(p.z_km[i]);
            assert!(d <= MOONLET_RADIUS_KM * (1.0 + 1e-12), "{d}");
        }
        for (a_km, radius_km) in [(140_222.43, 0.0), (100.0, 100.0), (f64::NAN, 20.0)] {
            let band = Band::Moonlet {
                a_km,
                radius_km,
                velocity: MoonletVelocity::Centre,
            };
            assert!(s.initial(band, 1).is_err(), "{a_km} {radius_km}");
        }
        assert_eq!(moonlet_velocity(2), Some(MoonletVelocity::Circular));
        assert_eq!(moonlet_velocity(3), None);
    }

    /// The moonlet view holds the guide point at the origin, puts Saturn
    /// straight down, and holds still any point that turns rigidly with the
    /// guide: so a synchronous moonlet starts at rest in this view.
    /// Tolerance 1e-6 km, as in the moon view test.
    #[test]
    fn moonlet_view_turns_with_the_moonlet_centre() {
        let s = Setup::builtin().unwrap();
        let g = s.moonlet_guide(140_222.43);
        let init = s
            .initial(moonlet_band(MoonletVelocity::Synchronous), 200)
            .unwrap();
        let p = &init.particles;
        let frame = ViewFrame::Moonlet(g);
        let v0 = view_transform(frame, s.n0_rad_s(), 0.0, &[]);
        for t in [0.0, 3.6e3, 5.0e4, 1.0e6] {
            let view = view_transform(frame, s.n0_rad_s(), t, &[]);
            let c = view.apply(g.position_km(t));
            assert!(c[0].abs() < 1e-6 && c[1].abs() < 1e-6, "{t}: {c:?}");
            let sat = view.apply([0.0, 0.0]);
            assert!(
                sat[0].abs() < 1e-6 && (sat[1] + g.a_km).abs() < 1e-6,
                "{sat:?}"
            );
            let (sn, cs) = (g.n_rad_s * t).sin_cos();
            for i in 0..p.len() {
                // Rigid turn by n t about Saturn: the motion of v = n z x r.
                let (x, y) = (p.x_km[i], p.y_km[i]);
                let moved = view.apply([cs * x - sn * y, sn * x + cs * y]);
                let start = v0.apply([x, y]);
                assert!((moved[0] - start[0]).hypot(moved[1] - start[1]) < 1e-6);
                // The start velocity is that rigid turn.
                let v = p.velocity_km_s(i);
                assert!((v.x + g.n_rad_s * y).abs() < 1e-15 && (v.y - g.n_rad_s * x).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn head_instances_hide_pandora_when_it_is_off() {
        let (s, e) = setup_and_eph();
        let moons = s.moon_positions(&e, 0.0).unwrap();
        let mut out = Vec::new();
        let n = head_instances(
            &s,
            &ViewTransform::IDENTITY,
            [0.0, 0.0],
            1e-3,
            &moons,
            true,
            &mut out,
        );
        assert_eq!(n, 3);
        assert_eq!(out[3], COLOUR_SATURN);
        assert_eq!(out[7], COLOUR_MOON_FIRST);
        assert_eq!(out[11], COLOUR_MOON_FIRST + 1.0);
        let n = head_instances(
            &s,
            &ViewTransform::IDENTITY,
            [0.0, 0.0],
            1e-3,
            &moons,
            false,
            &mut out,
        );
        assert_eq!(n, 2);
        assert_eq!(out.len(), 2 * FLOATS_PER_INSTANCE);
        // Saturn's diameter in pixels: 2 R * scale.
        assert!((f64::from(out[2]) - 2.0 * s.saturn.reference_radius_km * 1e-3).abs() < 1e-3);
    }

    #[test]
    fn moon_radii_follow_the_tracks_by_name() {
        let s = Setup::builtin().unwrap();
        let full = s.build_ephemeris(5.0 * 86_400.0).unwrap();
        let on = builtin_moon_radii_for(&s.system(&full, true)).unwrap();
        assert_eq!(on, vec![43.09, 40.63]);
        let off = builtin_moon_radii_for(&s.system(&full, false)).unwrap();
        assert_eq!(off, vec![43.09]);
        // A name without a radius is an error.
        let text = "[[moon]]\nname = \"prometheus\"\nradius_km = 43.09\n";
        assert!(moon_radii_for(&s.system(&full, true), text).is_err());
    }

    #[test]
    fn drop_text_shows_the_counts_and_leaves_out_pandora_when_off() {
        assert_eq!(
            drop_text([12, 45, 3], true),
            "Dropped: 12 unbound \u{b7} 45 inside Prometheus \u{b7} 3 inside Pandora"
        );
        assert_eq!(
            drop_text([0, 0, 0], false),
            "Dropped: 0 unbound \u{b7} 0 inside Prometheus"
        );
        assert!(drop_text([1234, 0, 0], true).starts_with("Dropped: 1,234 unbound"));
    }

    #[test]
    fn the_run_stops_at_the_end_of_the_moon_data() {
        assert_eq!(limit_to_end(100, 1.0e6, STEP_S), (100, false));
        // 10.5 steps left: at most 10 steps.
        assert_eq!(limit_to_end(100, 10.5 * STEP_S, STEP_S), (10, false));
        assert_eq!(limit_to_end(100, 1.5 * STEP_S, STEP_S), (1, false));
        assert_eq!(limit_to_end(100, STEP_S, STEP_S), (0, true));
        assert_eq!(limit_to_end(100, -5.0, STEP_S), (0, true));
    }

    #[test]
    fn speed_limit_follows_the_compute_budget() {
        assert_eq!(max_hours_for(10_000), MAX_HOURS_PER_S);
        assert_eq!(max_hours_for(100_000), MAX_HOURS_PER_S);
        assert!((max_hours_for(300_000) - 550.0 / 3.0).abs() < 1e-9);
        assert!((max_hours_for(1_000_000) - 55.0).abs() < 1e-9);
        // 1 h/s is 10 steps/s, so the limit in particle-steps/s is the budget.
        let steps_per_s = max_hours_for(1_000_000) * 3600.0 / STEP_S;
        assert!((steps_per_s * 1e6 - COMPUTE_BUDGET_PARTICLE_STEPS_PER_S).abs() < 1.0);
    }

    #[test]
    fn buffer_sizes_at_one_million_particles() {
        let cap = capacity_for(1_000_000, 64);
        assert_eq!(cap, 1_000_000);
        assert_eq!(state_bytes(cap), 48_000_000);
        assert_eq!(ref_bytes(cap), 16_000_000);
        assert_eq!(ref_bytes(cap) % 4, 0, "copy_buffer_to_buffer alignment");
        assert_eq!(instance_bytes(1_000_000), 32_000_128);
        assert_eq!(ref_base_slot(1_000_000), 1_000_004);
        // A count that is not a multiple of the workgroup size is padded.
        assert_eq!(capacity_for(100_001, 64), 100_032);
        let (bind, buf) = (128 << 20, 256 << 20);
        assert!(count_fits(1_000_000, 64, bind, buf, 65_535).is_ok());
    }

    #[test]
    fn count_that_does_not_fit_gives_a_message() {
        let msg = count_fits(1_000_000, 64, 32_000_000, 256 << 20, 65_535).unwrap_err();
        assert!(
            msg.contains("1000000 particles") && msg.contains("state"),
            "{msg}"
        );
        assert!(count_fits(10_000, 64, 32_000_000, 256 << 20, 65_535).is_ok());
        // Too many workgroups for one dispatch.
        assert!(count_fits(1_000_000, 64, u64::MAX, u64::MAX, 1000).is_err());
    }

    #[test]
    fn compare_modes_from_index() {
        assert_eq!(CompareMode::from_index(0), Some(CompareMode::Off));
        assert_eq!(CompareMode::from_index(1), Some(CompareMode::Side));
        assert_eq!(CompareMode::from_index(2), Some(CompareMode::Overlay));
        assert_eq!(CompareMode::from_index(3), None);
    }

    #[test]
    fn faint_head_has_offset_ids_and_no_saturn() {
        let (s, e) = setup_and_eph();
        let moons = s.moon_positions(&e, 0.0).unwrap();
        let mut out = Vec::new();
        let n = head_instances_styled(
            &s,
            &ViewTransform::IDENTITY,
            [0.0, 0.0],
            1e-3,
            &moons,
            true,
            false,
            COLOUR_FAINT_OFFSET,
            &mut out,
        );
        assert_eq!(n, 2);
        assert_eq!(out[3], COLOUR_MOON_FIRST + COLOUR_FAINT_OFFSET);
        assert_eq!(out[7], COLOUR_MOON_FIRST + 1.0 + COLOUR_FAINT_OFFSET);
    }
}
