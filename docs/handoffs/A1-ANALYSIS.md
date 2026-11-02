# A1-ANALYSIS: first science result
Role: A8  Branch: ws/a8-a1-analysis  Status: doing
Updated: 2026-11-02

## Goal and acceptance criteria
PLAN section 9: chaos map (indicator against initial a and longitude relative to Prometheus) between the moons; figure and short note on whether a quiet band appears near the core. Gates add: (2) the moons' share of the apse rate at the core (G0 question 4); (3) a converged Lyapunov time of the moons (G1).

## Done
- `crates/fring-analysis`: lib (`map`, `apse`, `laplace`, `lyapunov`, `closest`, `setup`, `stats`) and bin `fring-analyse` (subcommands in its header). Python project `crates/fring-analysis/python/` (Altair, `plots.py`).
- Runs (all in `runs/`, git-ignored; configs in `docs/results/configs/`): chaos maps main5, main10, main30 (GPU), zoom10; apse runs none, prom, pand, both, both-phase, both-half (CPU); closest approach (CPU harness); Lyapunov pairs (tangent method).
- Findings so far: no-moon apse rate 2.698412 deg/d (matches PLAN 7.2); both moons 2.6997 to 2.6999; Lyapunov: shadow orbits are not usable (d0 dependent, noise amplified by Kepler shear, IAS15 restart needs a small first step), tangent vector gives two populations.

## Next step (one concrete action)
Collect the last numbers (closest approach, Lyapunov final, robustness runs, CPU against GPU 30 years), write `docs/results/a1-analysis.md`, run the Definition of done, open the PR.

## How to verify the current state
`cargo test -p fring-analysis`; `cd crates/fring-analysis/python && uv run python plots.py` (needs `docs/results/data/`).

## Open questions / blockers
- None.

## Decisions made
- A0 (message 2026-11-02): Altair + vl-convert (uv project `crates/fring-analysis/python/`), band 139,700 to 141,100 km, class "bounded" for mean Y < 1.5, CPU harness for closest approach, 100 independent base trajectories for Part 3.
