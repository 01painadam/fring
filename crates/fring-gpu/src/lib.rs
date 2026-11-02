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
//! - [`spike_gpu`]: the same options on the GPU.

#[cfg(not(target_arch = "wasm32"))]
pub mod backend;
pub mod chaos;
pub mod ds;
pub mod gpu;
pub mod probe;
pub mod spike;
pub mod stepper;
#[cfg(not(target_arch = "wasm32"))]
pub mod spike_gpu;
pub mod wf32;
pub mod wh;
pub mod wh_gpu;
