//! wgpu backend and WGSL shaders.
//!
//! Owner: A3 (see `AGENTS.md`).
//!
//! Task G1 (GPU precision spike, ADR 0006):
//! - [`gpu`]: open the GPU and run one compute kernel (wgpu).
//! - [`probe`]: precision probe of the GPU (TwoSum, fma, division, square root).
//! - [`wf32`]: f32 with WGSL rules and error models; own sine and cosine.
//! - [`ds`]: double-single numbers, the Rust twin of `shaders/ds.wgsl`.
//! - [`spike`]: the precision options A, B, B+, C, D (Rust twins of
//!   `shaders/spike.wgsl`).

pub mod ds;
pub mod gpu;
pub mod probe;
pub mod spike;
pub mod wf32;
