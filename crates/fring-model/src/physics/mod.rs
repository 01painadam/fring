//! Physics shared by the CPU, GPU and web code (owner: A2, reviewed by A0).
//!
//! Everything here is generic over [`Real`](crate::Real), has no I/O and no
//! threads, and compiles for `wasm32-unknown-unknown`. Units: km, s.
//!
//! - [`gravity`]: Saturn's field (point mass + J2 + J4 + J6): acceleration,
//!   potential and energy.
//! - [`circular`]: circular-orbit speed and the frequencies n, kappa, nu.
//! - [`kepler`]: exact Kepler drift about the point mass.
//! - [`wisdom_holman`]: the baseline integrator (ADR 0003).
//! - [`leapfrog`]: plain drift-kick-drift step, kept as a reference.
//! - [`frames`]: inertial, rotating and local coordinates.

pub mod circular;
pub mod frames;
pub mod gravity;
pub mod kepler;
pub mod leapfrog;
pub mod wisdom_holman;

pub use circular::{CircularFrequencies, circular_speed, frequencies};
pub use frames::{
    LocalState, inertial_to_rotating, local_to_rotating, reference_mean_motion,
    rotating_to_inertial, rotating_to_local,
};
pub use gravity::SaturnField;
pub use kepler::kepler_drift;
pub use leapfrog::{dkd_step, leapfrog_circular_speed};
pub use wisdom_holman::{wh_step, wh_steps};
