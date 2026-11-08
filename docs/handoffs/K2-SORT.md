# K2-SORT: faster cell grouping and hard-sphere pair search
Role: A4  Branch: ws/a4-k2-sort  Status: review
Updated: 2026-11-08

## Goal and acceptance criteria
Make `CellList::group` and the K3 `find_pairs` faster (serial only). Results
stay bit for bit the same: same groups, same pairs in the same order, same
stats, same golden hashes. `fring-collide` has no thread dependency and must
build for wasm32. Parallel work is a later decision (A0, AJ).

## Done
- `crates/fring-collide/examples/k2_sort_bench.rs`: per-stage timing.
- `crates/fring-collide/tests/k2_sort_regression.rs`: copies of the old group
  and find_pairs as references; edge cases (1 to 5 cells along the arc, seam,
  negative keys, one cell, empty, one particle, all inactive).
- `pairs_in_cells`: for grids with 3 or more cells along the ring and no
  vertical wrap, 9 column cursors (galloping search) replace 27 binary
  searches per cell. Other grids use the old path.

## Next step (one concrete action)
Report to A0. Open choice (needs A0): group is now 71 of 293 ms at 10^6. A
radix sort or a packed key would need a non-generic `group` (u32 cell limit and
unbounded radial/vertical range must stay correct). Not done.

## How to verify the current state
    cargo test -p fring-collide --release
    cargo run -p fring-collide --release --example k2_sort_bench
Expected at 10^6 (serial): find_pairs about 290 ms (was 1496 ms).

## Open questions / blockers
- K3-CPU (PR #67) not on main yet. Its timing test was run on a temporary merge (not committed): find at 10^5 12.6 ms, at 10^6 about 162 ms (was 122-131 and 1425-1690). Log: `runs/logs/k2sort_hs_timing.txt`. Re-run after PR #67 merges.
- `fring-cpu --release --test golden` fails on origin/main too (release float rounding); it passes in debug. Not caused by this task.

## Decisions made (link ADRs)
- No new dependency, no threads.
