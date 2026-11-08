# E1-GPU: flag and freeze and the radius rule on the GPU (no sub-stepping)
Role: A3  Branch: ws/a3-e1-gpu  Status: review
Updated: 2026-11-08

## Goal and acceptance criteria
ICR-004 amendment 2, part 1, items 3 and 4. GPU: drop reasons, exact `drop_step` words (`8 + cap + i`), frozen state = block start state, the radius rule through the generic functions of `fring-model`, `GpuBackend::dropped`, `set_moon_radii_km`; `[substep]` is a `Config` error. Tests G1 to G6. PLAN: GPU agrees with CPU (section 10) and with E1-REF; with the `[loss]` keys absent `GpuBackend` is bit for bit unchanged; `fring-bench` shows no slow-down.

## Done
- Kernels (`shaders/wh.wgsl`, `arc.wgsl`, `chaos.wgsl`): `drop_particle(i, reason, seg)` writes `drop_step = first_step + max(seg - 1, 0)` at `status[8 + cap + i]` (only on the drop path). `first_step` is in a second `vec4<u32>` of the control uniform (`ctl` is now 32 bytes, `wh::control_at`, `wh::CONTROL_BYTES`). `status_words = 8 + 2 cap`; the arc log moved to `8 + 2 cap`. Words 0 to `8 + cap - 1` are unchanged (the web reads them).
- `chaos_renorm` has the status buffer (binding 4): a lost particle (main or shadow) skips the renorm, its accumulators stop and `flags` gets bit 1 (value 2). The shadow dispatch runs the plain kernel when the radius rule is on, so the pair is frozen together (main lost: the shadow kernel returns at once; shadow lost: reason 1 for the pair).
- `GpuBackend` (`src/backend.rs`): reads `[loss]` (`freeze_unbound`, `moon_radius_rule`); `[substep]` is `Error::Config`; `[loss]` keys with mode C are `Error::Config`; the rule with the arc is `Error::Config`; freeze with chaos and with the arc is allowed. `set_moon_radii_km` (as `CpuBackend`; `Error::Config` after the first step; with the rule on and the radii never set, the first `step` is `Error::Config`). With `freeze`, `step` does not return the unbound error. `dropped()` builds the `DropReport` from the status words.
- `GpuStepper::set_moon_radii_km` now also works with chaos (not with the arc, not in mode C).
- `DropOutcome::hit_rel_km` (twin) for the generic-function check.
- Tests: `tests/twins_loss.rs` (G1, normal tests, 4); `tests/gpu_loss.rs` (G2 backend, G3, 6 ignored GPU tests); `tests/gpu_vs_cpu_loss.rs` (G4, ignored); examples `e1_bench` and `e1_bench_off` (G6). Two older tests changed because the layout and a rule changed on purpose (not tolerances): `status_buffer_has_8_plus_2_cap_words...` (buffer length) and `set_moon_radii_checks_its_arguments` (rule with chaos is now allowed).

## Convention of `drop_step` (written here as A0 asked)
GPU segment `s`: 0 for the half drift before kick 0, `k + 1` for the drift after kick `k` (the last one is the half drift to the end of the block). `drop_step = first_step + max(s - 1, 0)`. So a loss in the end half drift of a block gets `first + n - 1`. The CPU merges the end half drift with the start half drift of the next block, so for a loss there it can give one step more. G3 uses the GPU convention exactly; G4 compares distributions (KS). The doc comments are at `drop_particle` in `wh.wgsl` and at `GpuStepper::status_buffer`.

## Evidence (logs in `runs/logs/`, git-ignored)
- G1 (`cargo test -p fring-gpu --test twins_loss -- --nocapture`): `segment_min_d2` twin against `fring_model::...::segment_min_d2` with `T = f32` and `T = f64`, 20,000 random cases: relative error 3.0e-7 (f32) and 3.9e-7 (f64) for the 18,952 cases without heavy cancellation (limit 1e-6); error scaled by `|a|^2 + |b|^2` 1.2e-7 and 9.3e-8 (limit 1e-6) for all. **Reading of "relative < 1e-6":** `d2` cancels when the segment passes close to the centre, and then even the generic f32 against the generic f64 gives 2.5e-6 relative (d2 = 4 km^2, |a|^2 = 8000 km^2), so the plain relative limit is used where `d2 >= 0.1 max(|a|^2, |b|^2)` (this contains every pass near the radii) and the scaled limit everywhere. A0 to confirm. `drift_hits_moon` at radius +-1 km for both moons: as expected. `advance_dropping`: 116 hits confirmed by the generic f32 and f64 functions, 128 clear, none within 1e-5 of the radius.
- G2: `gpu_drop.rs` golden hashes (`rule_off_state_bits_are_the_golden_bits`) and all ignored fring-gpu tests pass after the kernel change (`e1_gpu_final_ignored.log`); web GPU tests (`cargo test -p fring-web --release -- --ignored`) pass; `g2_off_means_off_on_the_backend`: `[loss]` absent, `[loss]` defaults, freeze with no loss, and the rule with tiny radii give the same state bits, modes D and Dp.
- G3 (`gpu_loss.rs`): unbound (two from the start: `drop_step` 0; one made unbound by a kick of Prometheus at step 4), crossings of both moons (radius +-1 and +-10 km), block lengths 8, 37 and 500: reason, exact `drop_step` (against the twin and the step of the 1-step-block twin), others bit for bit as in a run without the bad ones, frozen state = state of a run stopped at the block start, `dropped()` counts. Chaos: lost pair has flags bit 1, accumulators equal at steps 30 and 60, live particles bit for bit. Arc: frozen particle keeps its state, 160 of 240 others were wrapped, all as in the run without the loss. Config errors.
- G4 (`runs/logs/e1_gpu_g4.log`, 83 s): 20,000 particles in 139,700 to 141,100 km, 2 years (175,320 steps of 360 s), 1,993 sample times (every 88 steps). Lost [unbound, Prometheus, Pandora]: CPU [0, 615, 110], GPU [0, 646, 106] (z = 0, -0.87, 0.27). `drop_step` KS: 725 CPU and 752 GPU lost, D = 0.0241, p = 0.981. Samples inside 1 R_H of Prometheus 22 and 20 (z 0.31), 3 R_H 503 and 526 (z -0.72); Pandora 1 R_H 9 and 7, 3 R_H 297 and 314; distinct particles agree as well (all |z| < 1). R_H is 63.3 km (Prometheus) and 61.6 km (Pandora). Power, GPU radii x 1.1: 825 lost against 725 (z of the total -2.54), so a 10 % radius mistake shows at about 2.5 sigma; the 1 R_H samples have too few events (7 to 22) to test.
- G6 (`runs/logs/e1_gpu_g6.log`): baseline 224cb92 against this tree, keys absent: 9.564e8, 9.586e8 (base) and 9.561e8 (this tree) particle-steps/s at N = 10^6, 9.486e8, 9.499e8 and 9.480e8 at 10^5, 8.670e8, 8.680e8 and 8.659e8 at 10^4: no slow-down (-0.0 to -0.2 %, inside the noise). Freeze alone: ratio 0.999 to 1.001. **Freeze + radius rule: ratio 0.838 and 0.837 (N = 10^4), 0.884 and 0.883 (10^5), 0.888 and 0.887 (10^6): 11 to 16 % slower, the limit of 5 % is NOT met.** This is the cost of the rule that W2-DROP-GPU measured (-12 to -13 %). I tried a skip of far moons (hi-only prefilter): slower (0.77 to 0.82), reverted. The cost is not the arithmetic of the segment test alone. Needs A0 (accept 12 % as the price of the rule, or a new task to restructure the kernel, for example the rule test in a separate pass).
- G5 (GPU against E1-REF): **open**. E1-RUN (A5) is not merged.

## Next step (one concrete action)
A0: decide the G6 result (rule cost 11 to 16 %), confirm the reading of the G1 limit, then merge. G5 when E1-RUN is merged: run `GpuBackend` through the E1-RUN runner with a backend choice, then `e1_compare.py`.

## How to verify the current state
```sh
export PATH="/opt/homebrew/opt/rustup/bin:$HOME/.cargo/bin:$PATH"
cargo test -p fring-gpu --test twins_loss -- --nocapture                          # G1, 4 tests, a second
cargo test -p fring-gpu --release -- --ignored --test-threads=1                   # all GPU tests, about 2 min (includes G2, G3)
cargo test -p fring-gpu --release --test gpu_vs_cpu_loss -- --ignored --nocapture # G4, 83 s; E1_G4_N, E1_G4_YEARS for a quick run
cargo run -p fring-gpu --release --example e1_bench                               # G6, three variants, about 1.5 min
cargo run -p fring-gpu --release --example e1_bench_off                           # keys absent only; also runs on the tree before E1-GPU
cargo check -p fring-gpu --target wasm32-unknown-unknown
```
For B2 (A5): `e1_bench` is the model for the `[loss]` rows of `fring-bench` (it needs `GpuBackend::set_moon_radii_km` after `new`).

## Open questions / blockers
- G6: the rule costs 11 to 16 %, not 5 %.
- Not tested: a chaos shadow that is lost while its main particle is live (rare; reason 1 for the pair by design).
- G5 waits for E1-RUN.

## Decisions made (link ADRs)
- A0 2026-11-08: `ctl` 32 bytes and arc log move accepted; end half drift convention; G6 as an example in `fring-gpu`; `freeze` allowed with chaos and with the arc (A0 message).
- A3: the radius rule is also allowed with chaos (the shadow runs the plain kernel). The ICR says "all backends"; A3-ANALYSIS needs it. Reverse it if A0 disagrees: remove the `shadow_plain` branch and restore the `Config` error in `GpuStepper::set_moon_radii_km`.
