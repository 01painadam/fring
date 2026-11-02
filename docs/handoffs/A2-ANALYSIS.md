# A2-ANALYSIS: Pandora close approaches; chaos map of eccentric orbits
Role: A8  Branch: ws/a8-a2-analysis  Status: doing
Updated: 2026-11-02

## Goal and acceptance criteria
PLAN section 9 (A2-ANALYSIS): (a) closest approach to Pandora for the particles of the 10-year chaos map; (b) chaos map of core-like eccentric orbits (ICR-007 `eccentric_band`, e = 0.00235 and apse 353.78 deg from `params/ring.toml`) across 140,150 to 140,700 km, 10 and 30 years, against the e = 0 control. Done when `docs/results/a2-analysis.md` says whether the outer edge needs sub-stepping (h/4 test) and whether eccentric core orbits are more or less chaotic.

## Done
- Code: `closest` (several moons, segment minimum, window, id join), `Point.a_km`, stats (z, KS, Holm, bootstrap), `ecc-configs`.
- Configs in `docs/results/configs/` (`ecc/`, `edge-h*.toml`, `closest-pandora.toml`).

## Next step (one concrete action)
Background jobs started 2026-11-02 21:53 (do NOT start them again, M13): (1) CPU `fring-analyse closest` -> `runs/closest-pandora.csv` (+ `_prometheus_cum/hist.csv`, `_pandora_*`), log `runs/logs/closest-pandora.log`; (2) GPU chain `scratchpad/chain1.sh`: chaos-map-main10, edge-h, edge-h4, then `ecc/run_list.sh list_e10.txt e10`; status in `runs/logs/chain1-status.csv` (ends with a line `done`) and `runs/logs/ecc-status-e10.csv`. When both are finished: (a) run `closest-bins` for both moons and copy the small CSVs to `docs/results/data/`; (b) `ecc-compare runs e10 10 docs/results/data`, `edge-compare runs/edge-h runs/edge-h4 10 docs/results/data/edge_step.csv`; (c) start 30 yr for the bins that completed in BOTH ecc and circ at 10 yr (write a list file from ecc-status-e10.csv, run `ecc/run_list.sh <list> e30` in the background), then `ecc-compare runs e30 30 ...`; (d) `uv run python plots.py closest_pandora ecc`; (e) write `docs/results/a2-analysis.md`; (f) fmt, clippy, test --workspace and report.

## How to verify the current state
`cargo test -p fring-analysis --release` (12+ tests). Runs: `runs/logs/ecc-status-e10.csv` (name, exit, seconds).

## Open questions / blockers
- None yet.

## Decisions made
- A0 and AJ (2026-11-02): see the go-ahead message; paired e = 0 control, 25 km bins, 5,000 particles, h/4 test decides sub-stepping.
