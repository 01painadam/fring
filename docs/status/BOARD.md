# Task board

Current phase: **1 (validated orbits, CPU)**. Gate G0 was approved on 2026-11-01 (`docs/gates/G0.md`).

Rules (full text in `AGENTS.md` and `CLAUDE.md`):

- Each role edits only its own rows. A0 adds tasks.
- Status: `todo`, `doing`, `blocked`, `review`, `done`. For a gate: `todo` (evidence not ready), `review` (`docs/gates/G<n>.md` is ready for AJ), `done` (AJ wrote "Approved").
- "Blocked by": a task can start only when every item in this column is `done`.
- "Gate G<n>" is a gate. "G1", "G2" and "G3" without "Gate" are GPU tasks.
- Pick the next task: the first `todo` row in the current phase whose blockers are all `done`.
- Branch name: `ws/<role>-<task>`, for example `ws/a2-c1`.
- Task details and acceptance criteria: `docs/PLAN.md`, section 9. State of a task: `docs/handoffs/<ID>.md`.

## Phase 0: foundation

| ID | Task | Role | Status | Branch | Blocked by |
|---|---|---|---|---|---|
| F1 | Workspace, CI, coordination files | A0 | done | ws/a0-f1 | |
| F2 | `fring-model`: units doc, config structs, `Backend` trait, float trait, Philox RNG | A0 | done | ws/a0-f2 | F1 |
| F3 | Snapshot format spec, `fring-io` read and write with round-trip tests | A5 | done | ws/a5-f3 | F1 |
| L1 | Literature survey of F ring dynamics since about 2000 | A1 | done | ws/a1-l1 | F1 |
| L2 | Cited parameter set at a stated epoch, confirmed analytical formulas; `params/saturn.toml` uses the `SaturnParams` field names | A1 | done | ws/a1-l2 | F1 |
| Gate G0 | Approve interfaces, parameters and the observations to reproduce | AJ | done | ws/a0-g0 | F2, F3, L1, L2 |

## Phase 1: validated orbits (CPU)

| ID | Task | Role | Status | Branch | Blocked by |
|---|---|---|---|---|---|
| C1 | Saturn point mass + J2/J4/J6 forces (J6: ICR-001) and the leapfrog | A2 | done | ws/a2-c1 | Gate G0 |
| C2 | Wisdom-Holman integrator (ADR 0003); precession test at three radii; frame conversions inertial, rotating, local (`fring-model/src/physics/`) | A2 | done | ws/a2-c2 | C1 |
| C3 | Moon ephemeris integrator and Hermite interpolation | A2 | done | ws/a2-c3 | C1, V2 |
| C4 | Test particles with moons, rayon parallel; initial conditions from the config distributions (`fring-model/src/physics/`) | A2 | done | ws/a2-c4 | C3, V3 |
| V1 | REBOUND reference: single-particle precession | A6 | done | ws/a6-v1 | Gate G0 |
| V2 | REBOUND reference: Prometheus and Pandora over 20 years | A6 | done | ws/a6-v2 | Gate G0 |
| V3 | REBOUND reference: test particles near the core | A6 | done | ws/a6-v3 | Gate G0 |
| R1 | `fring-run` CLI: config, backend, snapshots, `meta.json`, progress | A5 | done | ws/a5-r1 | C4 |
| Gate G1 | CPU backend agrees with REBOUND; moon chaos looks right | AJ | todo | | C1, C2, C3, C4, V1, V2, V3, R1 |

## Phase 2: GPU backend

| ID | Task | Role | Status | Branch | Blocked by |
|---|---|---|---|---|---|
| G1 | GPU precision spike: f32, f32 + double-single accumulation, double-single (ADR) | A3 | todo | | Gate G1 |
| G2 | wgpu setup, kick and drift kernels, moon table in WGSL, Rust twin tests | A3 | todo | | G1 |
| G3 | Chaos indicator on GPU | A3 | todo | | G2 |
| B1 | `fring-bench`: particle-steps per second, CPU and GPU, N = 10⁴ to 10⁷ | A5 | todo | | G2 |
| A1-ANALYSIS | First chaos map of the region between the moons; measure the moons' share of ϖ̇ at the core (G0) | A8 | todo | | G3 |
| Gate G2 | GPU agrees with CPU; first chaos map reviewed | AJ | todo | | G1, G2, G3, B1, A1-ANALYSIS |

## Phase 3: first web toy

| ID | Task | Role | Status | Branch | Blocked by |
|---|---|---|---|---|---|
| W1 | `fring-web` scaffold: wasm32 + WebGPU, particles in the rotating frame | A7 | todo | | Gate G2 |
| W2 | Live test-particle mode, about 10⁵ particles, controls, static deploy | A7 | todo | | W1 |
| Gate G3 | First public demo | AJ | todo | | W1, W2 |

## Phase 4: collisions

| ID | Task | Role | Status | Branch | Blocked by |
|---|---|---|---|---|---|
| K1 | `fring-collide/SPEC.md` and the shared statistical tests | A4 | todo | | Gate G3 |
| K2 | Stochastic collision model on CPU | A4 | todo | | K1 |
| K3 | Hard-sphere collision model on CPU (cell list) | A4 | todo | | K1 |
| K4 | Cross-check stochastic and hard-sphere models in a small arc | A4 + A6 | todo | | K2, K3 |
| K5 | GPU binning and stochastic collisions in WGSL | A4 + A3 | todo | | K2 |
| D1 | Co-moving arc domain with azimuthal wrapping | A3 | todo | | Gate G3 |
| D2 | Arc error test, full ring versus arc, Δθ in {5°, 10°, 30°} | A6 | todo | | D1 |
| Gate G4 | Collision models validated; arc size chosen | AJ | todo | | K1, K2, K3, K4, K5, D1, D2 |

## Phase 5: science runs

| ID | Task | Role | Status | Branch | Blocked by |
|---|---|---|---|---|---|
| S1 | Run matrix for overnight runs (30 years each) | A8 + A5 | todo | | Gate G4 |
| S2 | Run the matrix: core width, survival fraction, eccentricity over time | A8 | todo | | S1 |
| Gate G5 | Results reviewed; choose the explainer story | AJ | todo | | S1, S2 |

## Phase 6: replay and showcase

| ID | Task | Role | Status | Branch | Blocked by |
|---|---|---|---|---|---|
| W3 | Replay viewer with time slider, WebGL2 fallback | A7 | todo | | Gate G5 |
| E1 | Figures and numbers for the explainer, each from a script | A8 | todo | | Gate G5 |
| E2 | README, docs site, licence, citation file, release v1.0 | A0 | todo | | W3, E1 |
| Gate G6 | Publish | AJ | todo | | W3, E1, E2 |
