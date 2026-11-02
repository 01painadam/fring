//! Chaos indicators and ring diagnostics.
//!
//! Owner: A8 (see `AGENTS.md`). Task A1-ANALYSIS: the chaos map
//! ([`map`]), the moons' share of the apse rate ([`apse`], [`laplace`]), the
//! closest approach to Prometheus ([`closest`]) and the Lyapunov time of the
//! moons ([`lyapunov`]). The `fring-analyse` binary runs them and writes
//! small CSV files; the plots are made by `python/` from those files.

pub mod apse;
pub mod closest;
pub mod laplace;
pub mod lyapunov;
pub mod map;
pub mod setup;
pub mod stats;

/// Philox stream used by the analysis tools (initial kicks and shadow
/// directions of the Lyapunov runs). Not used by the simulation.
pub const ANALYSIS_STREAM: fring_model::rng::Stream = fring_model::rng::Stream::ANALYSIS;
