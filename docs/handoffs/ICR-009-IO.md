# ICR-009-IO: `fring-run` writes the `CollisionStats` fields to `stats.csv`
Role: A5  Branch: ws/a5-icr-009-io  Status: done
Updated: 2026-11-07 18:50

## Goal and acceptance criteria
From `docs/interfaces/ICR-009.md` and `docs/snapshot-format.md` section 8 ("Collision columns"):

- `stats.csv` has nine more columns after `max_rel_energy_error`, in the field order of `fring_model::CollisionStats`, with column name = field name: `energy_loss_km2_s2, candidates, occupied_cells, occupancy_sum, occupancy_sq_sum, lonely_particles, overfull_cells, gmax_exceeded, deep_collisions`.
- `energy_loss_km2_s2` is a float (shortest round-trip text); the others are u64.
- A file without these columns (written before this task) reads as zero.
- A5 review notes from PR #39 (comment 6708448317):
  1. Give the literals in `crates/fring-io/tests/stats.rs` and `run_dir.rs` non-zero `CollisionStats` values, and extend `same()` in `stats.rs`.
  2. Test that a version 1 file without the collision columns reads as zero.
  3. Optional: the refusal in `fring-run` names the task per model (stochastic K2, hard sphere K3), and the test covers `hard_sphere`.

## Done
- `crates/fring-io/src/stats.rs`: `STATS_COLUMNS` has 13 names. `write_stats_row` writes the nine collision values; it builds the row in memory and writes it with one `write_all`. `read_stats_csv_report` reads them by name; a missing column or an empty field gives zero. `counters()` and `counters_mut()` use a struct pattern without `..`, so a new `CollisionStats` field stops the build until it has a column.
- Tests (`crates/fring-io/tests/stats.rs`, `run_dir.rs`): notes 1 and 2. Non-zero collision values (including `u64::MAX`, `5e-324`, `0.1 + 0.2`) round-trip bit-exact. New: `stats_collision_columns_in_struct_order` (pins the row text `2.0,3,4,,0.5,11,...,18` and reads a subset of columns in another order) and `stats_file_without_collision_columns_reads_as_zero`. `stats_rejects_bad_input` has three collision cases.
- Note 3: `apps/fring-run/src/lib.rs` refuses `stochastic` with task K2 and `hard_sphere` with task K3. `refuses_features_that_are_not_built_yet` covers both.
- Real run: a 100-particle, 0.01-year copy of `acceptance-1y-1e4.toml` writes the 13-column header and 8 rows, all collision values 0 (model `none`).
- Definition of done: all pass. Log `runs/logs/icr-009-io-dod.log` (git-ignored): fmt, clippy, `cargo test --workspace` (261 passed, 0 failed, 44 ignored), the CI wasm32 line, fring-web wasm32 clippy.

## Next step (one concrete action)
None. Task done: PR #41 merged (3fc452f) after the A0 review (PR #41 comment 6708830313) and green CI on d5affa7. K4 now waits only for K2 and K3.

## How to verify the current state
```sh
git log --oneline -1 main                         # includes 3fc452f (merge of PR #41)
cargo test -p fring-io --test stats               # 9 passed
cargo test -p fring-run refuses                   # 1 passed
```

## Open questions / blockers
- None.

## Decisions made (link ADRs)
- Column name = field name, no `collision_` prefix (A0, ICR-009-IMPL; AJ confirmed "No prefix" on 2026-11-07).
- `STATS_VERSION` stays 1: the columns are additive (`docs/snapshot-format.md` section 8).
