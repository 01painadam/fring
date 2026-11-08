# K5-WIRE: stochastic collisions in GpuBackend
Role: A3  Branch: ws/a3-k5-wire  Status: review
Updated: 2026-11-08

## Goal and acceptance criteria
Connect K5 (WGSL collisions) to the GPU run: `GpuBackend::step` bins the particles (K5-BIN) and calls `GpuCollider` every `every_steps` steps, and `stats()` fills `StepStats::collision` from `read_tally`. The `fring-run` refusal of `model = "stochastic"` on the GPU is A5's.
(a) non-zero `CollisionStats`; (b) GPU against CPU statistics, difference <= 4 combined SE; (c) same seed, bit-identical; (d) `model = "none"` unchanged; (e) chaos with collisions refused; (f) cost of one collision step at 10^5 and 10^6.

## Done
- `crates/fring-gpu/src/backend.rs`: `GpuCollision` (binner, collider, `every`, summed `Tally`). Built in `with_options` after `set_arc`. `step` caps each block at the next multiple of `every_steps`; after such a block, bin + collide are encoded in the same encoder and submitted once; then `read_tally` is read (one wait per collision step) and added on the host in pass order. `stats()` takes the sum (`mem::take`), sets `max_rel_energy_error = None` (as on the CPU). On any error `steps_total` is set back.
- Frozen or dropped particles do not collide: `bin.wgsl:122` gives them no cell.
- `tests/gpu_backend_collide.rs`: 7 tests (1 without a GPU).
- `examples/k5_wire_bench.rs`: cost of a collision step.

## Refusals (all in `check_collisions`, before any GPU is opened) and why
- `hard_sphere`: K3 exists on the CPU only.
- `domain.kind = "arc"` with collisions: the CPU refuses it too, and no test runs the collider on an arc (ARC-CELLS-GPU). A test of the collider on the arc must come first.
- `[diagnostics.chaos]` with `stochastic`: a shadow particle has no meaning under random kicks; the separation (MEGNO) would measure the noise (A0, 2026-11-08). Same message as the CPU (names the shadow).
- GPU mode C with collisions: not tested (A0, 2026-11-08). Modes dp and d are accepted.

## Notes
- The shader uses the sorted particle INDEX as the "smallest id" for the random numbers; the CPU uses the id. They are equal when the ids are 0..n in order (as `generate_particles` makes them). Otherwise the draws differ, the statistics do not.
- A split of a `step` call inside a block (not at a multiple of `every_steps`) adds a synchronisation (a rounding change, as on the CPU), so the bits differ; splits at multiples are bit-identical. The cadence counts are exact in both cases.
- `read_tally` clears per pass, so it is read after every collision pass (accepted by A0 for now).

## Evidence (logs in `runs/logs/`, git-ignored)
Test log: `k5-wire-tests.log`. Statistics test (3000 particles, eccentric band e = 1e-4, 120 steps, 12 seeds per backend, the same initial state; SE is over the seeds):

| statistic | GPU mean (SE/mean) | CPU mean (SE/mean) | difference |
|---|---|---|---|
| collisions | 8.661825e4 (7.7e-4) | 8.661692e4 (7.7e-4) | 0.01 SE |
| candidates | 2.146118e5 (9.2e-4) | 2.146156e5 (9.2e-4) | 0.01 SE |
| energy loss km2/s2 | 1.235489e-5 (1.3e-3) | 1.235496e-5 (1.3e-3) | 0.00 SE |

Bound: 4 combined SE (SPEC, ICR-010). Reported only: <N^2>/<N> 8.7467 against 8.7468; occupied cells 49663.5 on both (test bound 1e-3 relative, my choice for a deterministic statistic; A0 may drop it).
Config choice: e = 1e-4 at a = 140,000 km gives a peculiar speed about a e n = 3e-3 km/s, 2000 times the stepper noise of circular orbits (1.6e-6 km/s). A circular band would measure the noise. SE is below 0.13% of the mean for each statistic, so the test sees a 0.5% error. The seeds change only the collision draws, so the SE does not include the spread of initial states.

Cost of one collision step (bin + collide + tally read), release, M4, `k5-wire-bench.log`. Baseline = the same calls with `model = "none"`:

| N | every_steps | baseline min / median ms | with collisions min / median ms | cost per collision step min / median ms |
|---|---|---|---|---|
| 1e5 | 1 (10 x step(1)) | 6.37 / 6.70 | 19.75 / 19.95 | 1.34 / 1.32 |
| 1e5 | 10 (1 x step(10)) | 2.37 / 2.40 | 3.75 / 3.76 | 1.38 / 1.36 |
| 1e6 | 1 | 24.24 / 24.73 | 118.61 / 120.89 | 9.44 / 9.62 |
| 1e6 | 10 | 10.70 / 10.75 | 20.33 / 20.38 | 9.63 / 9.63 |

The two rows of each N agree within 3%, so the per-pass wait for the tally costs little. One collision step is about 6 WH steps at 10^5 (0.24 ms per step) and about 9 at 10^6 (1.07 ms per step). Open question for A4: the cost is mainly `cell.wgsl` (one thread per sorted position); a compaction of the run starts (K5 handoff) is the first thing to try if it matters. I did not split bin from collide here; K5-BIN has the bin pass alone.

## Next step (one concrete action)
A0: review the pull request. A5: remove the `fring-run` refusal of `model = "stochastic"` on the GPU and add the B2 benchmark.

## How to verify the current state
```sh
cargo test -p fring-gpu --release --test gpu_backend_collide -- --include-ignored --nocapture --test-threads=1   # 7 tests, 2 s
cargo run -p fring-gpu --release --example k5_wire_bench                                                          # about 15 s
cargo test -p fring-gpu --release -- --ignored                                                                    # existing GPU tests (model = "none" unchanged)
cargo check -p fring-gpu --target wasm32-unknown-unknown
```

## Open questions / blockers
- None for K5-WIRE. Mode C and the arc with collisions are refused until tested.

## Decisions made (link ADRs)
- A0 2026-11-08: the refusals above; bound 4 combined SE; per-pass tally read accepted.
