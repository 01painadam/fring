//! WebAssembly app: replay viewer and live test-particle toy.
//!
//! Owner: A7 (see `AGENTS.md`).
//!
//! Task W2: the live test-particle mode. About 10^5 particles run on the GPU with
//! the Wisdom-Holman kernels of `fring-gpu`, and the page draws them.
//! - [`live`]: set-up, step planning, views and the projection maths (host-testable).
//! - [`view`]: camera pan and zoom (host-testable).
//! - `web` (wasm32 only): canvas, render and projection pipelines, `GpuStepper`.

pub mod live;
pub mod view;

#[cfg(target_arch = "wasm32")]
mod web;
