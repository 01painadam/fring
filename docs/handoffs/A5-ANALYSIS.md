# A5-ANALYSIS: eccentric core orbits, closest approach, apse and 120 deg
Role: A8  Branch: ws/a8-a5-analysis  Status: review
Updated: 2026-11-07 (final)

## Goal and acceptance criteria
See docs/results/a1-analysis.md and a2-analysis.md for the model. Parts: (a) step test of eccentric_band at the zone ends, (b) closest approach to both moons for ecc and circ bins, (c) apse and 120 deg at 30 years.

## Done
- (a) step test data, (c) apse and 120 deg data (commit ca26ac7).
- (b) 12 `closest` runs (2500 particles, 6 bins each for ecc and circ): docs/results/data/a5_closest_ecc.csv.
- (b) check runs, 5000 particles, ecc bins 00, 10, 20, 21: docs/results/data/a5/a5-cl5k-ecc-b*.csv. Chaotic fraction (all) 2.5k vs 5k: 0.862/0.869, 0.830/0.829, 0.853/0.859, 0.847/0.846. With an encounter within 6.3 R_H: 0.988/0.987, 0.938/0.930, 0.969/0.970, 0.973/0.970. Without: 0.269/0.282, 0.803/0.804, 0.258/0.267, 0.221/0.223. So the 2.5k numbers are stable.
- Key result of (b): ecc bins 0 to 2 and 20, 21 almost all have a 6.3 R_H encounter (2061 to 2160 of 2500) and 97 to 99 % of those are chaotic. Circ bins have none (0 to 27).

## Next step (one concrete action)
A0 reviews docs/results/a5-analysis.md and the figures. No PR until A0 asks.

## How to verify the current state
- `python3 crates/fring-analysis/python/a5_closest.py /Users/adampain/Projects/fring/runs docs/results/data/a5_closest_ecc.csv` (needs runs/ CSVs): ecc bin 0 row has 2154 chaotic of 2500, 2061 within 6.3 R_H; ecc bin 10 has 487.
- `cd crates/fring-analysis/python && uv run python plots.py a5` rewrites figures/a5_closest.* and a5_phase120_e30.*.
- `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test -p fring-analysis` pass.
- Main numbers: h/4 step test dfrac -0.0002 and -0.0026 (sd 0.0044, 0.0050); quiet zone 30 yr chaotic fraction 0.9915 (apse 113.78) against 0.9936 (353.78), control 0.8837.

## Open questions / blockers
- None. Do not edit BOARD.md; no PR until A0 asks.

## Decisions made
- 5k check runs only for four ecc bins, as a sample-size test.
