//! Collision models: shared specification and CPU implementations.
//!
//! Owner: A4 (see `AGENTS.md`). The specification is `SPEC.md` in this crate.
//!
//! - [`pair`]: the pair operator, the impact geometry and the restitution
//!   laws. Every model uses them.
//! - [`frame`]: peculiar velocities and collision cells of the stochastic
//!   model.
//! - [`stochastic`]: the stochastic superparticle model (K2): cells, the
//!   no-time-counter method and the production entry point.
//! - [`tally`]: what a collision step reports.
//! - `testkit` (Cargo feature `testkit`): the shared statistical tests K1-T4 to K1-T9, and two slow
//!   reference models that pass them.

pub mod frame;
pub mod pair;
pub mod stochastic;
pub mod tally;
#[cfg(feature = "testkit")]
pub mod testkit;
