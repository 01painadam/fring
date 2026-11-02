# A2-ANALYSIS: Pandora close approaches; chaos map of eccentric orbits
Role: A8  Branch: ws/a8-a2-analysis  Status: review
Updated: 2026-11-02

## Goal and acceptance criteria
PLAN section 9 (A2-ANALYSIS): (a) closest approach to Pandora for the particles of the 10-year chaos map; (b) chaos map of core-like eccentric orbits (ICR-007 `eccentric_band`, e = 0.00235 and apse 353.78 deg from `params/ring.toml`) across 140,150 to 140,700 km, 10 and 30 years, against the e = 0 control. Done when `docs/results/a2-analysis.md` says whether the outer edge needs sub-stepping (h/4 test) and whether eccentric core orbits are more or less chaotic.

## Done
- Code: `closest` (several moons, segment minimum, window, id join), `Point.a_km`, stats (z, KS, Holm, bootstrap), `ecc-configs`.
- Configs in `docs/results/configs/` (`ecc/`, `edge-h*.toml`, `closest-pandora.toml`).

## Next step (one concrete action)
None for A8: review by A0. Note: `docs/results/a2-analysis.md`. All runs done (exit 0).

## How to verify the current state
`cargo test -p fring-analysis --release` (12+ tests). Runs: `runs/logs/ecc-status-e10.csv` (name, exit, seconds).

## Open questions / blockers
- None yet.

## Decisions made
- A0 and AJ (2026-11-02): see the go-ahead message; paired e = 0 control, 25 km bins, 5,000 particles, h/4 test decides sub-stepping.
