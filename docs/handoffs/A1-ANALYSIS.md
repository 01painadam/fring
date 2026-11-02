# A1-ANALYSIS: first science result
Role: A8  Branch: ws/a8-a1-analysis  Status: review
Updated: 2026-11-02 11:45

## Goal and acceptance criteria
PLAN section 9: chaos map (indicator against initial a and longitude relative to Prometheus) between the moons; figure and short note on whether a quiet band appears near the core. Gates add: (2) the moons' share of the apse rate at the core (G0 question 4); (3) a converged Lyapunov time of the moons (G1).

## Done
- `crates/fring-analysis`: lib (`map`, `apse`, `laplace`, `lyapunov`, `closest`, `setup`, `stats`) and bin `fring-analyse` (subcommands in its header). Python project `crates/fring-analysis/python/` (Altair, `plots.py`).
- Runs (all in `runs/`, git-ignored; configs in `docs/results/configs/`): chaos maps main5, main10, main30 (GPU), zoom10; apse runs none, prom, pand, both, both-phase, both-half (CPU); closest approach (CPU harness); Lyapunov pairs (tangent method).
- Results and numbers: `docs/results/a1-analysis.md`. Figures: `docs/results/figures/`. Small data: `docs/results/data/`.
- Definition of done: fmt, clippy -D warnings, `cargo test --workspace --release` all pass.

## Next step (one concrete action)
A0 reviews the PR. Optional follow-up: closest approach to Pandora (same harness, `moon` argument is fixed to Prometheus in `cmd_closest`), and a 2 km radial map of the 30-year run with the full band after sub-stepping exists.

## How to verify the current state
`cargo test -p fring-analysis`; `cd crates/fring-analysis/python && uv run python plots.py` (needs `docs/results/data/`).

## Open questions / blockers
- Pandora close approaches are not measured (outer edge of the map).
- Philox stream 8 is used by the analysis tools (`fring_analysis::ANALYSIS_STREAM`); A0 should register it in `rng.rs` (not mine to edit).
- Candidate `mistakes.md` entries for A0: (1) a shadow-orbit Lyapunov estimate with a small offset measures rounding noise that the Kepler shear turns into growth; use the tangent vector. (2) `Ias15::new` restarted with a large first step is wrong by about 1e-4; use 1 % of the dynamical time. (3) A finite-time growth rate of a separation that includes Kepler shear is biased by about ln(t)/t; G1's 0.14 to 0.49 per year rates were mostly shear. (4) `chaos.csv` a0_km is the point-mass Keplerian a, 640 km above the orbital radius for circular orbits in J2 to J6.

## Decisions made
- A0 (message 2026-11-02): Altair + vl-convert (uv project `crates/fring-analysis/python/`), band 139,700 to 141,100 km, class "bounded" for mean Y < 1.5, CPU harness for closest approach, 100 independent base trajectories for Part 3.
