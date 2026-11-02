# A1-ANALYSIS: first science result
Role: A8  Branch: ws/a8-a1-analysis  Status: doing
Updated: 2026-11-02

## Goal and acceptance criteria
PLAN section 9: chaos map (indicator against initial a and longitude relative to Prometheus) between the moons; figure and short note on whether a quiet band appears near the core. Gates add: (2) the moons' share of the apse rate at the core (G0 question 4); (3) a converged Lyapunov time of the moons (G1).

## Done
- Configs in `docs/results/configs/`. Run `chaos-map-main5` started (GPU, 2e5 particles, 5 years) as a timing test, output `runs/test-main5`.

## Next step (one concrete action)
Write `crates/fring-analysis` (bin `fring-analyse`, subcommands map, apse, closest, lyapunov, laplace), then run the four apse runs and the Lyapunov pair timing.

## How to verify the current state
`cargo test -p fring-analysis`

## Open questions / blockers
- None.

## Decisions made
- A0 (message 2026-11-02): Altair + vl-convert (uv project `crates/fring-analysis/python/`), band 139,700 to 141,100 km, class "bounded" for mean Y < 1.5, CPU harness for closest approach, 100 independent base trajectories for Part 3.
