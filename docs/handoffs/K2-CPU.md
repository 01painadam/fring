# K2-CPU: stochastic collisions in CpuBackend (A2 part)
Role: A2  Branch: ws/a2-k2-cpu  Status: review
Updated: 2026-11-08

## Goal and acceptance criteria
PLAN section 9, K2-CPU: the CPU backend calls the K2 stochastic step every `every_steps` integrator steps on the synchronised state (cell y from the rotating-frame longitude), and reports `CollisionStats` and the accepted count in `StepStats`. Done when a `fring-run` config with `model = "stochastic"` runs and writes non-zero collision columns to `stats.csv`; the same seed gives bit-identical results. The `fring-run` part is A5's.

## Done
- `crates/fring-cpu/src/collide.rs`: `CollisionState` (model, domain, tally) and the parallel step: cell keys in parallel, `CellList::group` (serial sort), cells in parallel in chunks of 256 with one `CellScratch` per chunk, tallies summed in chunk order, velocity changes added in cell order.
- `lib.rs`: blocks are cut at multiples of `every_steps` of the global step; after a block that ends at a multiple, the collision step runs with `step = first_step + block`, `t_s = start_s + step * step_s`, `dt = every_steps * step_s`. Particles with `ext.reason != 0` (dropped, frozen) do not take part. `stats()` returns the sums since the previous call and resets them; `max_rel_energy_error` is `None` with collisions. `every_steps = 1` gives a collision step after every step; `step(0)` does nothing.
- Refusals (`Error::Config` in `CpuBackend::new`): `hard_sphere` (K3-CPU), `domain.kind = "arc"` with collisions (D1-CPU), and `[diagnostics.chaos]` with `stochastic`.
- No ICR: no change to `Backend`, `StepStats`, `CollisionStats` or the config.

## Decisions made
- **A0 decision (2026-11-08):** chaos with stochastic collisions is refused. A MEGNO shadow has no meaning under random velocity kicks: the main particle and its shadow get different kicks, so the separation measures noise. K5-WIRE does the same on the GPU.
- Call splits: splits at multiples of `every_steps` give bit-identical results. A split at another step adds a synchronisation of the integrator, which changes its rounding by about 1e-9 km (also without collisions: 9.9e-10 km measured). With collisions the state differs by the same amount (1.0e-9 to 2.1e-9), the counts and all integer stats are equal, and the energy loss is equal to 1e-5 relative. So bit-identity for any split is not possible, and does not come from the collision step.

## Evidence (logs in `runs/logs/`, git-ignored)
- `cargo test -p fring-cpu --test collisions`: 7 tests pass (debug, about 3 s). Example: 200 steps, 3000 particles: 1723 collisions, 5303 candidates, loss 3.3e-11 km2/s2, 0 overfull cells.
- Timing of one collision step, release, 8 threads (`collision_step_timing`, `runs/logs/k2cpu-timing.log`), about 0.05 candidates per particle per step, 8820 occupied cells:

| N | keys | group (serial sort) | cells | apply | total |
|---|---|---|---|---|---|
| 10^5 | 1.3 ms | 4.8 ms | 0.6 ms | 0.2 ms | 6.9 ms |
| 10^6 | 9.0 ms | 36.5 ms | 9.1 ms | 5.6 ms | 60 ms |

  The serial `CellList::group` is 60 to 70 % of the step at both sizes: more than half, as A0 asked to be told. `fring-collide` is not changed. A parallel sort or radix sort inside `fring-collide` (A4) would be the first fix; B2 should use `every_steps` > 1 until then.

## Next step (one concrete action)
A0 opens the PR. Then A5 removes the `stochastic` refusal in `fring-run`.

## How to verify the current state
```sh
cargo test -p fring-cpu --test collisions          # 7 passed
cargo test -p fring-cpu --test golden              # unchanged hashes
cargo test -p fring-cpu --release --test collisions -- --ignored --nocapture   # timing table
```

## Open questions / blockers
- Serial sort in `CellList::group` (see above).
