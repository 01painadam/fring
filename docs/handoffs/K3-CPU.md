# K3-CPU: hard-sphere collisions in CpuBackend (A2 part)
Role: A2  Branch: ws/a2-k3-cpu  Status: review
Updated: 2026-11-08

## Goal and acceptance criteria
PLAN section 9, K3-CPU: the CPU backend runs the K3 hard-sphere model at every integrator step on the synchronised state and reports `CollisionStats` in `StepStats`. Done when a `fring-run` config with `model = "hard_sphere"` runs and writes non-zero collision columns to `stats.csv`. The `fring-run` part (removing its `hard_sphere` refusal) is A5's.

## Done
- `crates/fring-cpu/src/collide.rs`: `CollisionState` holds a `Kind` (stochastic or hard sphere). The hard-sphere step: mask `reason == 0`, `find_pairs`, copy the velocities of the pair members into a scratch, `resolve_pairs`, write back. Time `t_s = start_s + step * step_s`, `h = step_s`. `every` is 1. `StageTimes` has new fields `mask_s`, `find_s`, `resolve_s`, `write_s` (A0 accepted).
- `lib.rs` is not changed: `every = 1` already gives blocks of one step and a collision step after each.
- Refusals (`Error::Config`): `hard_sphere` is no longer refused. Chaos is refused for both models (message names the model). Arc with collisions stays refused (D1-CPU). New: a radius below about 1.03e-4 km (see Open questions) is refused with a clear message.
- No ICR: no change to `Backend`, `StepStats`, `CollisionStats` or the config. `max_rel_energy_error` is `None` with collisions.
- K3 uses no random numbers. The seed has no effect (tested).

## Evidence (logs in `runs/logs/`, git-ignored)
- `cargo test -p fring-cpu --test collisions_hs`: 7 passed (debug, about 4 s), `runs/logs/k3cpu-tests.log`. Setup: a 600-particle clump (Gaussian width 6 km), radius 50 m (much larger than real particles, so that collisions happen).
  - `hs_nonzero_stats`: 250 steps: 6 collisions, loss 8.08e-8 km2/s2, 3 deep.
  - `hs_energy_balance_of_the_collision_step`: tally loss 8.099726e-8 = kinetic energy lost across the collision step (7 collisions); limit 1e-9 relative.
  - `hs_same_input_bit_identical`, `hs_threads_1_4_8`, `hs_call_splits` (including 200 calls of 1 step): bit-identical (exact). Unlike K2, any split is identical, because every step is a collision step.
  - `hs_frozen_do_not_collide`: frozen particles placed in front of free ones: 5 collisions if all take part, 0 with the frozen left out; the 600 free particles are bit-identical to a run without the frozen ones.
  - `hs_refusals`: chaos and arc refused; radius 1 cm refused.
- `cargo test -p fring-cpu --test collisions`: 7 passed (K2; `refusals` updated). `--test golden`: 1 passed, hashes unchanged.
- Timing, release, 8 threads (`hs_step_timing`, `runs/logs/k3cpu-timing.log`), one step, 360 s step, 87 661 steps per model year:

| N | mask | find_pairs | resolve + copy | integrator, block 1 | integrator, block 4096 |
|---|---|---|---|---|---|
| 10^5 | 0.01 ms | 122 to 131 ms | 0.00 ms | 2.7 ms | 1.75 ms |
| 10^6 | 0.01 ms | 1425 to 1690 ms | 0.01 ms | 25.3 ms | 17.9 ms |

  **Cost per model year** (steps per year x ms per step): 10^5 particles: 10 700 to 11 500 s (about 3.0 to 3.2 hours) for `find_pairs`, plus about 240 s of integrator. 10^6 particles: 125 000 to 148 000 s (about 35 to 41 hours), plus about 2200 s of integrator. So `find_pairs` is more than 98 % of the step. Plan K4 and B2 with this: a hard-sphere run of the full ring is not practical above about 10^5 particles until `find_pairs` is faster.
  - Blocks of one step make the integrator about 50 % (10^5) and 41 % (10^6) slower per step (fork/join per step), which is small next to `find_pairs`.
  - `find_pairs` is 1.2 to 1.7 us per particle. It is serial. The likely cause (inferred from the K2 sort time of 0.5 ms per 10^5 and from the code, not measured per stage) is the neighbour search in `pairs_in_cells`: for every occupied cell (almost one per particle in a ring) it does a binary search over all cell keys for each neighbour cell. A parallel search, a merge over the sorted keys or a hash would be the fix. That is `fring-collide` (A4).

## Next step (one concrete action)
A0 opens the PR. Then A5 removes the `hard_sphere` refusal in `fring-run`.

## How to verify the current state
```sh
cargo test -p fring-cpu --test collisions_hs      # 7 passed
cargo test -p fring-cpu --test collisions         # 7 passed
cargo test -p fring-cpu --test golden             # unchanged hashes
cargo test -p fring-cpu --release --test collisions_hs -- --ignored --nocapture   # timing table
```

## Open questions / blockers
- **K3 cannot run the full ring with real particles.** The cell grid counts cells along the ring in a `u32` (`CellGrid::new`): 2 pi a0 / (2 s) must be below 2^32, so s must be at least 1.03e-4 km (10 cm). A radius of 1 cm is refused by `CpuBackend::new` with a clear message. The arc domain (D1, D1-CPU) removes the limit (12 237 km / 2e-5 km = 6e8 cells). K4 will use the arc.
- `find_pairs` speed (see above): A4 decision.
- Pairs are found at step ends only; the missed fraction is 2.09 (<g>h / 4s)^2 (SPEC 8). In the test clump, half of the collisions were deep (radius 50 m is large against the relative speed times the step); real runs must check `deep_collisions`.

## Decisions made (link ADRs)
- A0 (2026-11-08): `find_s` and `resolve_s` in the public `StageTimes`: accepted. Chaos refused for both models. Skip the wasm check for `fring-cpu`.
