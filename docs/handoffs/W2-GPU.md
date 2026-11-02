# W2-GPU: GpuStepper, SelfTest, native-only blocking items
Role: A3  Branch: ws/a3-w2-gpu  Status: review
Updated: 2026-11-02 17:13

## Goal and acceptance criteria
ICR-005 items 1 to 5 in `crates/fring-gpu`. Done when: `GpuStepper` gives the same states, bit for bit, as `GpuBackend::step` (with and without moons and chaos); `SelfTest::submit` and `poll` give the same result as `self_test`; all existing tests pass; `cargo check -p fring-gpu --target wasm32-unknown-unknown` passes; `fring-bench` shows no slow-down.

## Done
- Items 1 to 5 of ICR-005 in `crates/fring-gpu`; commits df2f417, f8834ef, f1696b0, 7c65b51.
- Tests: `tests/gpu_stepper.rs` (7 configs: moons on/off, chaos off/on with `renorm_steps` 5, modes Dp, D, C, block 7 and default; calls 1, 60, 89; after each call `time_s` bits, state, status and chaos accumulator bytes equal), `tests/gpu_selftest.rs`, unit test in `probe.rs`.
- `cargo test --workspace`: 215 passed, 0 failed, 21 ignored. fmt and clippy clean.
- `cargo test -p fring-gpu --release -- --ignored` (log `runs/logs/w2gpu-ignored.log`): all suites `ok`: chaos_done_when 1, done_when 1, gpu_backend 5, gpu_chaos 1, gpu_selftest 1, gpu_stepper 1, gpu_twins 1, gpu_wh 1.
- `cargo check -p fring-gpu --target wasm32-unknown-unknown`: ok, also with `RUSTFLAGS="-D warnings"`.
- A0 review (2026-11-02 17:13): commit 46fc3f1 runs `cargo fmt` (order of `pub mod stepper;` in `lib.rs`; `cargo fmt --check` failed before it). A0 ran the Definition of done on the pushed head: fmt, clippy, `cargo test --workspace` and the `--ignored` GPU tests give 227 passed, 0 failed (log `runs/logs/a0-review-w2gpu.log`); wasm32 check passes with and without `RUSTFLAGS="-D warnings"`. ICR-005 amendment item 4 lists the extra public items.
- Bench (particle-steps/s, best of 3; CPU rows left out; `runs/bench/w2gpu-*.csv`):

| backend | N | chaos | before 1 | before 2 | after | % vs mean before |
|---|---|---|---|---|---|---|
| gpu_d | 10000 | false | 1.004e+09 | 9.983e+08 | 1.004e+09 | +0.3 |
| gpu_d | 10000 | true | 4.867e+08 | 4.868e+08 | 4.873e+08 | +0.1 |
| gpu_dp | 10000 | false | 8.661e+08 | 8.629e+08 | 8.672e+08 | +0.3 |
| gpu_dp | 10000 | true | 4.212e+08 | 4.215e+08 | 4.207e+08 | -0.2 |
| gpu_c | 10000 | false | 4.088e+08 | 4.057e+08 | 4.088e+08 | +0.4 |
| gpu_c | 10000 | true | 2.015e+08 | 2.013e+08 | 2.014e+08 | +0.0 |
| gpu_d | 100000 | false | 1.098e+09 | 1.083e+09 | 1.099e+09 | +0.8 |
| gpu_d | 100000 | true | 5.389e+08 | 5.378e+08 | 5.388e+08 | +0.1 |
| gpu_dp | 100000 | false | 9.487e+08 | 9.359e+08 | 9.495e+08 | +0.8 |
| gpu_dp | 100000 | true | 4.659e+08 | 4.647e+08 | 4.659e+08 | +0.1 |
| gpu_c | 100000 | false | 4.685e+08 | 4.657e+08 | 4.698e+08 | +0.6 |
| gpu_c | 100000 | true | 2.319e+08 | 2.317e+08 | 2.32e+08 | +0.1 |
| gpu_d | 1000000 | false | 1.109e+09 | 1.107e+09 | 1.109e+09 | +0.1 |
| gpu_d | 1000000 | true | 5.44e+08 | 5.45e+08 | 5.456e+08 | +0.2 |
| gpu_dp | 1000000 | false | 9.532e+08 | 9.558e+08 | 9.566e+08 | +0.2 |
| gpu_dp | 1000000 | true | 4.702e+08 | 4.716e+08 | 4.716e+08 | +0.1 |
| gpu_c | 1000000 | false | 4.721e+08 | 4.755e+08 | 4.756e+08 | +0.4 |
| gpu_c | 1000000 | true | 2.348e+08 | 2.357e+08 | 2.356e+08 | +0.1 |
| gpu_d | 10000000 | false | 1.064e+09 | 1.066e+09 | 1.067e+09 | +0.1 |
| gpu_d | 10000000 | true | 5.332e+08 | 5.264e+08 | 5.264e+08 | -0.6 |
| gpu_dp | 10000000 | false | 9.205e+08 | 9.219e+08 | 9.228e+08 | +0.2 |
| gpu_dp | 10000000 | true | 4.613e+08 | 4.573e+08 | 4.565e+08 | -0.6 |
| gpu_c | 10000000 | false | 4.603e+08 | 4.619e+08 | 4.618e+08 | +0.1 |
| gpu_c | 10000000 | true | 2.304e+08 | 2.233e+08 | 2.236e+08 | -1.4 |

All changes are within run-to-run noise. Before 1 and before 2 differ by up to 3.2% (gpu_c, 10⁷, chaos: 2.304e+08 against 2.233e+08; A0 review corrected this from "1.3%"). The largest change, -1.4% on that row, is the mean of the two baselines against "after"; "after" is +0.1% against before 2.

## Next step (one concrete action)
A0 merges the W2-GPU pull request when CI is green on its head commit, then sets W2-GPU to `done` and starts W2 (A7).

## How to verify the current state
```sh
export PATH="/opt/homebrew/opt/rustup/bin:$HOME/.cargo/bin:$PATH"
cargo fmt --all -- --check && cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo test -p fring-gpu --release -- --ignored      # needs the GPU
cargo check -p fring-gpu --target wasm32-unknown-unknown
```

## Open questions / blockers
- None.

## Decisions made
- ICR-005 amendment: `mode` in `GpuStepper::new`, `GpuBackend::stepper()`, one slot per block.
- `encode_block` (crate-private) is the block loop: `GpuBackend` calls it once per block (one submit per block, poll after the table fill); `encode_steps` calls it for all blocks with one slot per block. `GpuBackend` sets `steps_total` back on failure because the stepper counts blocks when it encodes them.
- Also added (additive, small): `GpuBackend::gpu()` and `GpuStepper::chaos_buffer()` so that the test can read both sides' buffers; `WhKernels` is `Clone`.
