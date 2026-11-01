//! Shared types, units, config, Philox RNG, force functions and analytic
//! formulas for `fring`.
//!
//! Owner: A0 (see `AGENTS.md`). The types here are shared interfaces: change
//! them only through an interface change request (`docs/interfaces/`).
//!
//! Rules (`docs/PLAN.md` section 6.2):
//!
//! - No dependency on wgpu, rayon or I/O. The crate compiles to
//!   `wasm32-unknown-unknown`.
//! - Physics functions are generic over [`Real`], so the same code runs in f64
//!   (CPU reference) and f32 (to compare with the GPU shaders).
//!
//! Start with [`units`] for the conventions.

pub mod config;
pub mod error;
pub mod real;
pub mod rng;
pub mod units;
pub mod vec3;

pub use error::{Error, Result};
pub use real::Real;
pub use vec3::Vec3;
