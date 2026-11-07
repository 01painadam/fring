# 0001: fring-model interfaces (task F2)

Status: accepted (AJ, gate G0, 2026-11-01)
Date: 2026-11-01
Author: A0  Approved by A0: 2026-11-01

## Context

Every backend, the runner, the web app and the analysis use the types in `fring-model`. Changes after G0 need an interface change request and touch several roles, so the first version must be right in the parts that are expensive to change: the `Backend` trait, the config schema, the RNG layout and the float trait. `docs/PLAN.md` section 6.2 gives a sketch of the trait; this ADR records where the code differs from it and why.

## Decisions

### 1. Backend trait

```rust
pub trait Backend {
    fn new(config: &RunConfig, system: &System, init: &InitialState) -> Result<Self> where Self: Sized;
    fn step(&mut self, n_steps: u64) -> Result<()>;
    fn time_s(&self) -> f64;
    fn moons(&self) -> MoonState;
    fn particles(&mut self) -> Result<ParticleView<'_>>;
    fn stats(&mut self) -> Result<StepStats>;
}
```

Differences from the sketch in PLAN section 6.2:

- `new` has a third argument, `system: &System` (Saturn's J2/J4 field and the pre-computed moon ephemeris). PLAN section 6.3 says the runner computes the ephemeris before the run; the sketch had no place to pass it. Keeping it apart from `InitialState` makes clear that it does not evolve with the particles.
- `particles()` and `stats()` return `Result`, because a GPU read-back can fail (device lost).
- `time()` is `time_s()`: the unit is in the name (`AGENTS.md`, Conventions).

Everything that crosses the trait is **inertial and f64**. A GPU backend converts from its f32 local coordinates when the host asks. Reason: one frame and one precision for every consumer (I/O, analysis, comparisons); the conversion cost is paid only at output times.

### 2. Particles: structure of arrays with ids

`Particles` has one `Vec` per component plus `id: Vec<u32>`. Structure of arrays matches the GPU buffers (PLAN section 11, item 2). The id lets a backend sort particles by cell (item 6) and still lets analysis follow one particle. `u32` ids allow 4 × 10⁹ particles, above the 10⁷ stretch target.

### 3. Float trait: own `Real`, not `num_traits::Float`

`Real` has about 15 items (constants, conversion, `sqrt`, `sin`, `cos`, `atan2`, `ln`, `floor`, `min`, `max`, `mul_add`). It is implemented for `f32` and `f64`.

- No extra dependency.
- A double-single type (task G1) can implement it. `num_traits::Float` has more than 50 methods, most of which a double-single type would not need.
- Rule: add a method only when a physics function needs it and WGSL has the same function, so every generic function can have a WGSL twin.

### 4. Philox4x32-10 layout

- Key = run seed, `[low 32 bits, high 32 bits]`.
- Counter = `[index, step low, step high, stream << 16 | draw]`.
- Streams: 0 tests, 1 initial conditions, 2 collisions. A0 adds new streams; numbers are never reused.
- Floats: `u01_f64` uses 52 bits and `u01_f32` uses 23 bits, both `(k + 0.5) / 2^bits`, so values are in the open interval (0, 1) and exact in their type. The WGSL twin then gives the same bits as Rust.

Verified against all six `philox4x32` vectors of Random123 (`crates/fring-model/tests/philox.rs`).

### 5. Config schema

- One TOML file with sections `run`, `time`, `frame`, `domain`, `moons`, `particles`, `collisions`, `output` (`crates/fring-model/tests/data/example-run.toml`).
- `deny_unknown_fields` everywhere: a typing error such as `step_sec` stops the run instead of being ignored.
- Choices between models are tagged sections (`kind = "arc"`, `model = "stochastic"`, `law = "power_law"`). New models add a variant without changes to the others.
- Output cadence is in steps (`frame_every_steps`), not seconds, so there is no rounding of output times.
- The step count is `ceil(duration / step)`: the run covers at least the requested duration.
- No physical constants in the run config. They come from `params/` (`[run] params_dir`).
- The collision fields follow PLAN section 7.5. Task K1 may change them with an ICR. ICR-009 (accepted 2026-11-07) did: `model = "stochastic"` has three cell sizes (`cell_radial_km`, `cell_azimuthal_km`, `cell_vertical_km`) and `every_steps` (default 1) instead of one `cell_size_km` (`crates/fring-collide/SPEC.md` sections 5.1 and 7).

### 6. Dependencies

`serde` (derive) and `toml` only, approved by AJ on 2026-11-01. Both compile for `wasm32-unknown-unknown`. The error type is hand-written (no `thiserror`).

## Consequences

- `fring-cpu` and `fring-gpu` implement one trait with three inputs. `fring-run` (R1) loads params, computes the ephemeris (C3), builds the initial state, then calls `Backend::new`.
- GPU read-back includes a local-to-inertial conversion. This is cheap at output cadence but must not happen every step.
- `StepStats` will get more fields. Code must build it with `..Default::default()`.
- To change any of this after G0: an ICR in `docs/interfaces/`.

## Evidence

- `cargo test -p fring-model`: 39 tests pass (14 unit, 5 backend, 12 config, 6 Philox, 2 doc tests).
- `cargo check -p fring-model --target wasm32-unknown-unknown` passes.
- Pull request for task F2.
