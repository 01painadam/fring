//! WebAssembly app: replay viewer and live test-particle toy.
//!
//! Owner: A7 (see `AGENTS.md`).
//!
//! Task W1: a page that draws particles in the frame that rotates at n0.
//! - [`scene`]: particle set-up and analytic epicycle motion (host-testable).
//! - [`view`]: camera pan and zoom (host-testable).
//! - `web` (wasm32 only): canvas, wgpu render pipeline, page controls.

pub mod scene;
pub mod view;

#[cfg(target_arch = "wasm32")]
mod web;
