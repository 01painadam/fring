# ICR-009-IMPL: implement ICR-009 with amendments A1 to A3
Role: A0  Branch: ws/a0-icr-009-impl  Status: doing
Updated: 2026-11-07 16:30

## Goal and acceptance criteria
From `docs/status/BOARD.md` and `docs/interfaces/ICR-009.md` (section "A0 decision"): three cell sizes and `every_steps` in `[collisions]`, `CollisionStats` in `StepStats`, `fring-collide` as a dependency of `fring-cpu` and `fring-gpu`, the CI wasm32 line, the column list in `docs/snapshot-format.md`. A5 reviews the `fring-io` and `fring-run` test changes. Blocks K2, K3 and K5-BIN.

## Done
- `crates/fring-model/src/config.rs`: `Collisions::Stochastic` has `cell_radial_km`, `cell_azimuthal_km`, `cell_vertical_km` (each finite and > 0) and `every_steps` (≥ 1, default 1). `cell_size_km` is gone; it is now an unknown field (error).
- `crates/fring-model/tests/config.rs`: the ICR example parses and round-trips with `every_steps` = 1 by default; new test `stochastic_cells_and_cadence_are_checked` (every_steps = 3; zero, negative and infinite sizes; every_steps = 0; old `cell_size_km`).
- `crates/fring-model/src/backend.rs`: `StepStats::collision: CollisionStats` (Copy, Default), with `energy_loss_km2_s2`, `candidates`, `occupied_cells`, `occupancy_sum`, `occupancy_sq_sum`, `lonely_particles`, `overfull_cells`, `gmax_exceeded` (A2), `deep_collisions`. No `collisions` field (A1). Exported from `fring_model`.
- A3, A5 files: `crates/fring-io/tests/stats.rs` (4 literals) and `crates/fring-io/tests/run_dir.rs` (1 literal) use `..StepStats::default()`. `apps/fring-run/src/lib.rs` refusal now says "K2 to K4"; `apps/fring-run/tests/run.rs` uses the three cell sizes and expects "K2".
- `fring-collide.workspace = true` in `fring-cpu` and `fring-gpu` `[dependencies]`. CI wasm32 line and `AGENTS.md` Commands list include `fring-collide`.
- `docs/snapshot-format.md` section 8: collision columns (names equal to the field names, in struct order, at the end). `docs/adr/0001-fring-model-interfaces.md`: one line that ICR-009 replaced `cell_size_km`.

## Next step (one concrete action)
When the full Definition of done passes (log `runs/logs/icr-009-impl-dod.log`), push the branch, open the PR (plain-language summary, A5 asked to review the `fring-io` and `fring-run` test changes), set the board row to `review`.

## How to verify the current state
```sh
cargo test -p fring-model --test config      # 15 pass, including stochastic_cells_and_cadence_are_checked
cargo test -p fring-io -p fring-run          # all pass (refuses_features_that_are_not_built_yet expects K2)
cargo check -p fring-model -p fring-io -p fring-collide -p fring-gpu -p fring-web --target wasm32-unknown-unknown
grep -rn cell_size_km crates apps            # only crates/fring-collide/SPEC.md line 70 (A4 updates it in K2)
```

## Open questions / blockers
- Column names in `stats.csv`: A0 chose "column name = field name" (no `collision_` prefix). Cheap to change until ICR-009-IO writes files; AJ can change it in the PR.
- For A4 (K2): `crates/fring-collide/SPEC.md` line 70 still says "The config today has one `cell_size_km`; ICR-009 asks for three". Not edited here (A4's crate). Review nits N1 (`testkit` feature) and N5 (`Tally` holds a `CollisionStats`) also wait for K2.

## Decisions made (link ADRs)
- `every_steps` default by `#[serde(default = "default_every_steps")]`, so a config without it is valid and means every step.
- A5 test literals use `..StepStats::default()` (the rule of ADR 0001 line 73), not an explicit `collision:` field, so later `StepStats` fields do not break them again.
