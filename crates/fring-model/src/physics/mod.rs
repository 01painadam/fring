//! Physics shared by the CPU, GPU and web code (owner: A2, reviewed by A0).
//!
//! Everything here is generic over [`Real`](crate::Real), has no I/O and no
//! threads, and compiles for `wasm32-unknown-unknown`. Units: km, s.
//!
//! - [`gravity`]: Saturn's field (point mass + J2 + J4 + J6): acceleration,
//!   potential and energy.
//! - [`circular`]: circular-orbit speed and the frequencies n, kappa, nu.
//! - [`leapfrog`]: one drift-kick-drift step of one particle.

pub mod circular;
pub mod gravity;
pub mod leapfrog;

pub use circular::{CircularFrequencies, circular_speed, frequencies};
pub use gravity::SaturnField;
pub use leapfrog::{dkd_step, leapfrog_circular_speed};
