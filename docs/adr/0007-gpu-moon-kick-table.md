# 0007: Moons on the GPU from a host-computed kick table (task G2)

Status: accepted by A0 2026-11-01; AJ reviews at gate G2
Date: 2026-11-01
Author: A3  Approved by A0: 2026-11-01

## Context

PLAN section 6.3 said that the GPU kernels interpolate the moon positions from an uploaded ephemeris table (quintic Hermite, ICR-002). A 30-year table for two moons is 76 MB (about 1.05 million nodes of 9 f64 values). In f32 the interpolation needs the node-relative form and DS basis functions: with f32 basis functions the position error is about 10⁻³ km (ADR 0006: f32 resolves 0.016 km at 1.4 × 10⁵ km). The CPU backend does something simpler: for each block of steps it fills a `KickTable` (moon positions and the indirect term at the kick times, in f64) and every particle reads it.

## Options

1. Quintic Hermite in WGSL (PLAN 6.3): the node table (76 MB in DS) on the GPU, a "moon pass" kernel in DS that writes the kick table, a Rust twin. More code and one more place where the GPU can differ from the CPU.
2. Host-computed kick table: the CPU fills the `KickTable` in f64 for each block, rounds it to DS and uploads it (`queue.write_buffer`), as it does for the CPU backend.

## Decision

Option 2. For each block of steps, `GpuBackend` calls `KickTable::<f64>::fill`, the same function and the same ephemeris as `CpuBackend`. It rounds the moon positions to DS (error 5 × 10⁻¹⁰ km) and computes the indirect term in f64 before rounding it (DS for option C, f32 for option D). The table has one row per step: 144 B (36 f32; room for 4 moons).

- The GPU and CPU see the same moon positions and the same indirect term, so no interpolation error enters the GPU-CPU comparison (PLAN section 10).
- No extra kernel and no Hermite code in WGSL. The 76 MB table stays on the host (the `MoonEphemeris` already holds it); the GPU never holds more than a few blocks of the kick table (ring of 3 buffers).
- Cost: one interpolation of two moons per step on the CPU (well below 1 µs) and 144 B per step uploaded. The CPU fills block k+1 while the GPU runs block k.
- The web toy (W2) can do the same in wasm, because `fring_model::physics` compiles for wasm32. So quintic Hermite in WGSL is not planned.

**The step h in the f32 kick (found in G2).** Option D applies the kick with `h` as one f32 (`prm_h().x`). G1 used a step that is exact in f32 (356.25 s). A general `step_s` is not, and the rounding of h is then a systematic scale error of the whole kick, up to 6 × 10⁻⁸, which gives a mean-motion error of about 10⁻⁶ rad at 10³ orbits (10 % of the PLAN section 10 budget). The host removes it at no cost on the GPU: it multiplies the f32 constants that the D kick uses (J2, J4, J6, GM of the moons, and the f32 indirect term of every row) by h / f32(h) before rounding them. Option C carries h in DS and needs no correction. Tests use a step that is not exact in f32.

## Consequences

- PLAN section 6.3 points here. ICR-002 stays valid (the quintic Hermite is used on the host, by `fring_model::physics`).
- `fring-gpu` does not need the ephemeris on the GPU, so the GPU memory use is the particle state (48 B per particle) plus three kick-table buffers.
- To reverse: add the moon pass kernel (option 1); the kernels read the same table layout.

## Evidence

See the pull request for G2 and `docs/handoffs/G2.md`. Code: `crates/fring-gpu/src/{wh,backend}.rs`, `crates/fring-gpu/shaders/wh.wgsl`.
