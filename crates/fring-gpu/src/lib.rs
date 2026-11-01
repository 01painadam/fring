//! wgpu backend and WGSL shaders.
//!
//! Owner: A3 (see `AGENTS.md`).
//!
//! Task G1 (GPU precision spike, ADR 0006):
//! - [`gpu`]: open the GPU and run one compute kernel (wgpu).
//! - [`probe`]: precision probe of the GPU (TwoSum, fma, division, square root).

pub mod gpu;
pub mod probe;
