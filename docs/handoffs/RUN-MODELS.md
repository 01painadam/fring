# RUN-MODELS: fring-run accepts the collision models
Role: A5  Branch: ws/a5-run-models  Status: review
Updated: 2026-11-09 12:00

## Goal and acceptance criteria
`fring-run` stops refusing what a backend now does, and never lets a backend ignore a setting without an error. K2-CPU and K3-CPU acceptance: a config with `model = "stochastic"` or `"hard_sphere"` runs on the CPU and writes non-zero collision columns to `stats.csv`.

## Done
- `check_supported` (`apps/fring-run/src/lib.rs`) now looks at the backend.
  - CPU: stochastic and hard_sphere allowed.
  - Only GPU + hard_sphere is refused now ("K3 is CPU only; use backend = cpu").
  - Step 3 (K5-WIRE and D1-CPU on main): the CPU arc refusal and the GPU stochastic refusal are removed. The GPU backend refuses arc or mode C with collisions, and chaos with collisions; `setup()` turns those into a setup error (exit code 2), so `fring-run` keeps no second list.
- `tests/run.rs`: `refuses_hard_sphere_on_the_gpu` replaces the old refusal test. New: `gpu_stochastic_is_selected_not_refused`, `gpu_stochastic_with_chaos_is_a_clean_setup_error`, `cpu_arc_run_keeps_particles_in_the_arc` (band a0 +- 300 km, 0.02 years, bound check plus a wrap count; 136 wraps; all fast in debug).
- `tests/models.rs`: two CPU runs (600-particle clump, sigma 6 km, radius 50 m, no moons, 250 steps, stats every 50 steps). Measured (log: `runs/logs/run-models-first.log`, not committed):
  - stochastic: 6 rows, 4 collisions, loss 8.619e-7 km2/s2, 12 candidates, 29367 occupied cells (sum over rows).
  - hard_sphere: 6 rows, 6 collisions, loss 8.082e-8 km2/s2, 0 candidates, 0 occupied cells.
  - Both tests take about 2 s together in debug.

## Next step (one concrete action)
A0 re-runs the checks and opens the PR.

## How to verify the current state
`cargo test -p fring-run` : all pass (run.rs: 11 passed, 2 ignored; models.rs: 2).
`cargo clippy -p fring-run --all-targets -- -D warnings` : clean.

## Open questions / blockers
- None.

## Decisions made (link ADRs)
- None.
