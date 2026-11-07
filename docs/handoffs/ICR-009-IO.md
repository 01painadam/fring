# ICR-009-IO: `fring-run` writes the `CollisionStats` fields to `stats.csv`
Role: A5  Branch: ws/a5-icr-009-io  Status: doing
Updated: 2026-11-07 17:45

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
- Branch made from `main` at c179845. Board row `doing`.

## Next step (one concrete action)
Extend `STATS_COLUMNS`, `write_stats_row` and `read_stats_csv_report` in `crates/fring-io/src/stats.rs` with the nine collision columns.

## How to verify the current state
```sh
git status -sb                      # ws/a5-icr-009-io
cargo test -p fring-io              # all pass
cargo test -p fring-run             # all pass
```

## Open questions / blockers
- AJ: keep column names without a `collision_` prefix? Default yes (A0 decision, as in `docs/snapshot-format.md`).

## Decisions made (link ADRs)
- Column name = field name, no prefix (A0, ICR-009-IMPL).
