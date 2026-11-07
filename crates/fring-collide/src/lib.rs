//! Collision models: shared specification and CPU implementations.
//!
//! Owner: A4 (see `AGENTS.md`). The specification is `SPEC.md` in this crate.
//!
//! - [`pair`]: the pair operator, the impact geometry and the restitution
//!   laws. Every model uses them.
//! - [`frame`]: peculiar velocities and collision cells of the stochastic
//!   model.
//! - [`testkit`]: the shared statistical tests K1-T4 to K1-T9, and two slow
//!   reference models that pass them.

pub mod frame;
pub mod pair;
pub mod testkit;
