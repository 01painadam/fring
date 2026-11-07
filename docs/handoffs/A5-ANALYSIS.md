# A5-ANALYSIS: eccentric core orbits, closest approach, apse and 120 deg
Role: A8  Branch: ws/a8-a5-analysis  Status: doing
Updated: 2026-11-07

## Goal and acceptance criteria
See docs/results/a1-analysis.md and a2-analysis.md for the model. Parts: (a) step test of eccentric_band at the zone ends, (b) closest approach to both moons for ecc and circ bins, (c) apse and 120 deg at 30 years.

## Done
- (a) step test data, (c) apse and 120 deg data (commit ca26ac7).
- (b) 12 `closest` runs (2500 particles, 6 bins each for ecc and circ): docs/results/data/a5_closest_ecc.csv.
- (b) check runs, 5000 particles, ecc bins 00, 10, 20, 21: docs/results/data/a5/a5-cl5k-ecc-b*.csv. Chaotic fraction (all) 2.5k vs 5k: 0.862/0.869, 0.830/0.829, 0.853/0.859, 0.847/0.846. With an encounter within 6.3 R_H: 0.988/0.987, 0.938/0.930, 0.969/0.970, 0.973/0.970. Without: 0.269/0.282, 0.803/0.804, 0.258/0.267, 0.221/0.223. So the 2.5k numbers are stable.
- Key result of (b): ecc bins 0 to 2 and 20, 21 almost all have a 6.3 R_H encounter (2061 to 2160 of 2500) and 97 to 99 % of those are chaotic. Circ bins have none (0 to 27).

## Next step (one concrete action)
Write docs/results/a5-analysis.md (follow a2-analysis.md layout) from the three data sets above; add the Reproduce commands (docs/results/configs/a5/run_closest.sh <ecc|circ> <bin> [5k]).

## How to verify the current state
`python3 crates/fring-analysis/python/a5_closest.py runs docs/results/data/a5_closest_ecc.csv` (needs the runs/ CSVs, git-ignored); `cargo fmt --all -- --check` passes.

## Open questions / blockers
- None. Do not edit BOARD.md; no PR until A0 asks.

## Decisions made
- 5k check runs only for four ecc bins, as a sample-size test.
