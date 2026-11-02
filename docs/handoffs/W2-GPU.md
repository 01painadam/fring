# W2-GPU: GpuStepper, SelfTest, native-only blocking items
Role: A3  Branch: ws/a3-w2-gpu  Status: doing
Updated: 2026-11-02 17:10

## Goal and acceptance criteria
ICR-005 items 1 to 5 in `crates/fring-gpu`. Done when: `GpuStepper` gives the same states, bit for bit, as `GpuBackend::step` (with and without moons and chaos); `SelfTest::submit` and `poll` give the same result as `self_test`; all existing tests pass; `cargo check -p fring-gpu --target wasm32-unknown-unknown` passes; `fring-bench` shows no slow-down.

## Done
- Baseline `fring-bench` (before any code change): `runs/bench/w2gpu-before-1.csv`, `-2.csv` (binary copy `runs/bench/fring-bench-before`).
- Code written (not yet committed or verified): target cfg, `Gpu::from_parts`, `WhKernels::compile`, `SelfTest`, `stepper.rs`, `GpuBackend` on top of `GpuStepper`, tests `gpu_stepper.rs` and `gpu_selftest.rs`.

## Next step (one concrete action)
Build, fix compile errors, run fmt, clippy and the tests; then commit in microcommits.

## How to verify the current state
```sh
export PATH="/opt/homebrew/opt/rustup/bin:$HOME/.cargo/bin:$PATH"
cargo fmt --all -- --check && cargo clippy --workspace --all-targets -- -D warnings
cargo test -p fring-gpu --release -- --ignored      # needs the GPU
cargo check -p fring-gpu --target wasm32-unknown-unknown
```

## Open questions / blockers
- None.

## Decisions made
- ICR-005 amendment: `mode` in `GpuStepper::new`, `GpuBackend::stepper()`, one slot per block.
- Also added (additive, small): `GpuBackend::gpu()` and `GpuStepper::chaos_buffer()` so that the test can read both sides' buffers; `WhKernels` is `Clone`.
