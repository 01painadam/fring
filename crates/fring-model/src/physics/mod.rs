//! Physics shared by the CPU, GPU and web code (owner: A2, reviewed by A0).
//!
//! Everything here is generic over [`Real`](crate::Real), has no I/O and no
//! threads, and compiles for `wasm32-unknown-unknown`. Units: km, s.
//!
//! - [`gravity`]: Saturn's field (point mass + J2 + J4 + J6): acceleration,
//!   potential and energy.
//! - [`circular`]: circular-orbit speed and the frequencies n, kappa, nu.
//! - [`kepler`]: exact Kepler drift about the point mass.
//! - [`moon_dynamics`], [`ias15`], [`ephemeris`], [`hermite`]: the moons (task C3):
//!   equations of motion (ADR 0004), the IAS15 integrator, the ephemeris table
//!   and its quintic Hermite interpolation (ICR-002).
//! - [`wisdom_holman`]: the baseline integrator (ADR 0003).
//! - [`moon_kick`]: the moons' direct and indirect pull in the kick, and the
//!   Wisdom-Holman loop with moons (task C4).
//! - [`initial`]: initial conditions from the config distributions (task C4).
//! - [`leapfrog`]: plain drift-kick-drift step, kept as a reference.
//! - [`frames`]: inertial, rotating and local coordinates.

pub mod chaos;
pub mod circular;
pub mod ephemeris;
pub mod frames;
pub mod gravity;
pub mod hermite;
pub mod ias15;
pub mod initial;
pub mod kepler;
pub mod leapfrog;
pub mod moon_dynamics;
pub mod moon_kick;
pub mod wisdom_holman;

pub use chaos::{ChaosAccum, shadow_offset_km};
pub use circular::{CircularFrequencies, circular_speed, frequencies};
pub use ephemeris::{
    DEFAULT_EPSILON, DEFAULT_INTERVAL_S, MoonInit, build_ephemeris, build_ephemeris_with,
    interpolate_moons, interpolate_moons_into,
};
pub use frames::{
    LocalState, inertial_to_rotating, local_to_rotating, reference_mean_motion,
    rotating_to_inertial, rotating_to_local,
};
pub use gravity::SaturnField;
pub use hermite::{hermite3, hermite5};
pub use ias15::Ias15;
pub use initial::{EpicycleElements, epicycle_state, generate_particles, initial_particle};
pub use kepler::kepler_drift;
pub use leapfrog::{dkd_step, leapfrog_circular_speed};
pub use moon_dynamics::MoonSystem;
pub use moon_kick::{
    KickTable, indirect_acceleration, kick_acceleration, moon_direct_acceleration, wh_steps_moons,
};
pub use wisdom_holman::{wh_step, wh_steps};
