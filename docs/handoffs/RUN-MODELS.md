# RUN-MODELS: fring-run accepts the collision models
Role: A5  Branch: ws/a5-run-models  Status: review
Updated: 2026-11-08

## Goal and acceptance criteria
`fring-run` stops refusing what a backend now does, and never lets a backend ignore a setting without an error. K2-CPU and K3-CPU acceptance: a config with `model = "stochastic"` or `"hard_sphere"` runs on the CPU and writes non-zero collision columns to `stats.csv`.

## Done
- `check_supported` (`apps/fring-run/src/lib.rs`) now looks at the backend.
  - CPU: stochastic and hard_sphere allowed.
  - CPU + arc: refused, task D1-CPU (the CPU backend would ignore the arc).
  - GPU + stochastic: refused, task K5-WIRE (the GPU backend on main ignores `[collisions]`).
  - GPU + hard_sphere: refused for good ("K3 is CPU only; use backend = cpu").
  - GPU + arc: allowed (D1 works on the GPU).
- `tests/run.rs`: `refuses_what_a_backend_would_ignore` replaces the old refusal test.
- `tests/models.rs`: two CPU runs (600-particle clump, sigma 6 km, radius 50 m, no moons, 250 steps, stats every 50 steps). Measured (log: `runs/logs/run-models-first.log`, not committed):
  - stochastic: 6 rows, 4 collisions, loss 8.619e-7 km2/s2, 12 candidates, 29367 occupied cells (sum over rows).
  - hard_sphere: 6 rows, 6 collisions, loss 8.082e-8 km2/s2, 0 candidates, 0 occupied cells.
  - Both tests take about 2 s together in debug.

## Next step (one concrete action)
When A0 says K5-WIRE and D1-CPU are on main: remove the GPU-stochastic and CPU-arc refusals in `check_supported`. Add a GPU stochastic test (accept "backend error" with no GPU, as `gpu_backend_is_selected_not_refused` does) and a CPU arc run test. Update the refusal test (GPU hard_sphere stays).

## How to verify the current state
`cargo test -p fring-run` : all pass (models: 2 tests, run: refusal test).
`cargo clippy -p fring-run --all-targets -- -D warnings` : clean.

## Open questions / blockers
- Waiting for K5-WIRE and D1-CPU on main.

## Decisions made (link ADRs)
- None.
