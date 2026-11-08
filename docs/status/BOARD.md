# Task board

Current phase: **4 (collisions)**. Gates G0 and G1 were approved on 2026-11-01, gate G2 on 2026-11-02, gate G3 on 2026-11-03 (`docs/gates/`).

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
| Gate G1 | CPU backend agrees with REBOUND; moon chaos looks right | AJ | done | ws/a0-g1 | C1, C2, C3, C4, V1, V2, V3, R1 |

## Phase 2: GPU backend

| ID | Task | Role | Status | Branch | Blocked by |
|---|---|---|---|---|---|
| G1 | GPU precision spike: f32, f32 + double-single accumulation, double-single (ADR); includes the Wisdom-Holman Kepler drift, and decides GPU or CPU for science runs (gate G1) | A3 | done | ws/a3-g1 | Gate G1 |
| G2 | wgpu setup, kick and drift kernels, moon kick table (host, ADR 0007), Rust twin tests; production mode D+ (ADR 0006 addendum) | A3 | done | ws/a3-g2 | G1 |
| G3 | Chaos indicator on GPU and CPU: shadow particle + MEGNO (ADR 0008, ICR-003) | A3 | done | ws/a3-g3 | G2 |
| B1 | `fring-bench`: particle-steps per second, CPU and GPU, N = 10⁴ to 10⁷ (`docs/benchmarks.md`) | A5 | done | ws/a5-b1 | G2 |
| A1-ANALYSIS | First chaos map of the region between the moons; measure the moons' share of ϖ̇ at the core (G0); converged Lyapunov time of the moons (G1) | A8 | done | ws/a8-a1-analysis | G3 |
| Gate G2 | GPU agrees with CPU; first chaos map reviewed | AJ | done | ws/a0-gate-g2 | G1, G2, G3, B1, A1-ANALYSIS |

## Phase 3: first web toy

| ID | Task | Role | Status | Branch | Blocked by |
|---|---|---|---|---|---|
| W1 | `fring-web` scaffold: wasm32 + WebGPU, particles in the rotating frame | A7 | done | ws/a7-w1 | Gate G2 |
| W2-GPU | ICR-005 in `fring-gpu`: native-only blocking items, `GpuStepper`, two-phase `SelfTest` | A3 | done | ws/a3-w2-gpu | W1 |
| W2 | Live test-particle mode, about 10⁵ particles, controls, static deploy | A7 | done | ws/a7-w2 | W2-GPU |
| W2-COMPARE | Compare the live ring with a stored reference state (t = 0 or 'set to now'): side by side or overlay; particle count as a control | A7 | done | ws/a7-w2-compare | W2 |
| W2-DROP-GPU | ICR-006 in `fring-gpu`: drop reason per particle, early return for dropped particles, opt-in moon-radius rule | A3 | done | ws/a3-w2-drop-gpu | W2-COMPARE |
| W2-DROP | Web: the run goes on when particles are dropped (hidden, counts shown); pause at the end of the run, no restart | A7 | done | ws/a7-w2-drop | W2-DROP-GPU |
| A2-ANALYSIS | Pandora close approaches; chaos map of core-like eccentric orbits (gate G2 follow-up; does not block gate G3) | A8 | done | ws/a8-a2-analysis | Gate G2 |
| L3 | Moon masses and elements in `params/` against GR03: do they explain the factor 1.4 in the Lyapunov rate? (gate G2 follow-up; does not block gate G3) | A1 | done | ws/a1-l3 | Gate G2 |
| Gate G3 | First public demo | AJ | done | ws/a0-gate-g3 | W1, W2, W2-COMPARE, W2-DROP-GPU, W2-DROP |

## Phase 4: collisions

| ID | Task | Role | Status | Branch | Blocked by |
|---|---|---|---|---|---|
| W2-MOONLET | Web: moonlet scenario (orbital radius, packing radius, velocity mode; 180° from Prometheus) and a view that turns with the moonlet centre (ICR-008; added by AJ) | A7 | done | ws/a7-w2-moonlet | Gate G3 |
| K1 | `fring-collide/SPEC.md` and the shared statistical tests (A0 approves, AJ confirms; Opus) | A4 | done | ws/a4-k1 | Gate G3 |
| ICR-009-IMPL | Implement ICR-009 with amendments A1 to A3: three cell sizes and `every_steps` in `[collisions]`, `CollisionStats` in `StepStats`, `fring-collide` as a dependency of `fring-cpu` and `fring-gpu`, CI wasm32 line, column list in `docs/snapshot-format.md` (A5 reviews the `fring-io` and `fring-run` test changes) | A0 | done | ws/a0-icr-009-impl | K1 |
| ICR-009-IO | `fring-run` writes the `CollisionStats` fields to `stats.csv` (ICR-009) | A5 | done | ws/a5-icr-009-io | ICR-009-IMPL |
| K1-REFS | Open the five references of `fring-collide/SPEC.md` (Bird 1994, Bridges et al. 1984, Carnahan and Starling 1969, Haff 1983, van Noije and Ernst 1998); cite the Bridges et al. power-law values in `params/sources.md` | A1 | done | ws/a1-k1-refs | K1 |
| A5-ANALYSIS | A2-ANALYSIS follow-ups: h/4 test for `eccentric_band`, `closest` on `eccentric_band`, second apse phase at 30 years | A8 | done | ws/a8-a5-analysis | Gate G3 |
| E1-ICR | Write ICR-004: flag and freeze on the CPU, radius rule twin to `fring-model/src/physics/`, sub-stepping trigger and scheme, REBOUND comparison set | A2 | done | ws/a2-e1-icr (PR #54; ICR-004 Accepted 2026-11-08) | Gate G3, A5-ANALYSIS |
| L3-PARAMS | GR03b moon masses as a variant file in `params/` (nominal values unchanged) | A1 | done | ws/a1-l3-params | Gate G3 |
| A4-ANALYSIS | L3 deciding run: moon Lyapunov rate with the GR03b masses; GR03 method on our masses | A8 | done | ws/a8-a4-analysis (PR #58; masses do not explain the factor 1.4) | L3-PARAMS |
| K2 | Stochastic collision model on CPU | A4 | done | ws/a4-k2 | K1, ICR-009-IMPL |
| E1-REF | REBOUND (IAS15) reference for particles within 1 R_H of Prometheus (from the spec only) | A6 | done | ws/a6-e1-ref (PR #59; WHFast stand-in: no power at h) | E1-ICR |
| E1-REF-2 | `e1_compare.py` for ICR-004 amendment 2, part 3: the limits apply to the run without sub-stepping, a run with sub-stepping is reported with no limit, no power check; `AGENTS.md` command line updated by A0 (added by A0 2026-11-08) | A6 | done | ws/a6-e1-ref-2 (PR #63) | E1-REF |
| E1-CPU | ICR-004 and per-particle sub-stepping in the CPU backend | A2 | done | ws/a2-e1-cpu (PR #60; ICR-004 amendment 2: sub-stepping off by default, not on the GPU) | E1-ICR |
| K3 | Hard-sphere collision model on CPU (cell list) | A4 | done | ws/a4-k3 | K1, ICR-009-IMPL |
| K2-CPU | CPU backend runs the stochastic model (K2) every `every_steps` steps and reports `CollisionStats`; `fring-run` accepts `model = "stochastic"` (A5 part) | A2 + A5 | doing | ws/a2-k2-cpu (A2 part merged, PR #64; A5 part after E1-RUN; the serial `CellList::group` sort is 60 to 70 % of a collision step: 60 ms at 10⁶) | K2 |
| K3-CPU | CPU backend runs the hard-sphere model (K3) at every step and reports `CollisionStats`; `fring-run` accepts `model = "hard_sphere"` (A5 part) | A2 + A5 | doing | ws/a2-k3-cpu (A2 part) | K3 |
| D1 | Co-moving arc domain with azimuthal wrapping | A3 | done | ws/a3-d1 | Gate G3 |
| D1-CPU | CPU backend wraps particles into the arc after each step (D1 `physics::arc`), with an energy test; `fring-run` accepts `domain.kind = "arc"` (A5 part) | A2 + A5 | todo | | D1 |
| ARC-CELLS | Cell index in the arc from the signed longitude to the arc centre, clamped to the first and last cell (D1 finding 1): K2 `cell_key` and K3 pairs across the arc seam (A4); K5-BIN `cell_f64` and `cell_of` with the twin (A3) | A3 + A4 | done | ws/a3-arc-cells, ws/a4-arc-cells | D1 |
| K5-BIN | GPU binning: counting sort by cell at each step, with a Rust twin | A3 | done | ws/a3-k5-bin | K1, ICR-009-IMPL |
| E1-RUN | Rust runner for the E1-REF file contract: reads `validation/reference/e1_l1_states.csv`, `e1_l2_states.csv` and `e1_moons_t0.csv`, runs `CpuBackend` without `[substep]` (`--run`, limits) and with it (`--substep`, reported only), writes the WH result files; then `e1_compare.py` for layers 1 and 2; backend-generic so that E1-GPU can use it; the limits apply without sub-stepping (ICR-004 amendment 2, part 3; the script of E1-REF-2, needed only for the last step) (closes the E1-CPU acceptance; added by A0 2026-11-08) | A5 | doing | ws/a5-e1-run | E1-CPU, E1-REF |
| E1-GPU | ICR-004 flag and freeze and the radius rule in WGSL, with Rust twins; no sub-stepping on the GPU (`[substep]` is a config error; ICR-004 amendment 2, part 1) | A3 | review | ws/a3-e1-gpu (G5 against E1-REF follows through the E1-RUN runner; radius rule costs 11 to 16 %, accepted by AJ) | E1-CPU |
| K5 | Stochastic collisions in WGSL (`shaders/collide/`) | A4 | done | ws/a4-k5 (PR #57; ICR-010 accepted 2026-11-08 with T8a 1e-11 km/s) | K2, K5-BIN, ARC-CELLS |
| K2-SORT | Parallel `CellList::group` in `fring-collide` (today a serial sort: 36 of 60 ms of a CPU collision step at 10⁶ particles, K2-CPU); results bit for bit unchanged (added by A0 2026-11-08, AJ) | A4 | todo | | K2-CPU |
| K5-WIRE | Connect K5 to the GPU run: `GpuBackend::step` bins and calls `GpuCollider` every `every_steps` steps and fills `StepStats::collision` from `read_tally` (A3); `fring-run` accepts `model = "stochastic"` on the GPU (A5 part) (added by A0 2026-11-08, AJ) | A3 + A5 | todo | | K5 |
| K4 | Cross-check stochastic and hard-sphere models in a small arc (A6 reviews the report) | A4 | todo | | K2, K3, ICR-009-IO, K2-CPU, K3-CPU, D1, D1-CPU, ARC-CELLS |
| D2 | Arc error test, full ring versus arc, Δθ in {5°, 10°, 30°} | A6 | todo | | D1, D1-CPU |
| A3-ANALYSIS | 30-year chaos map over the full band with flag and freeze and the radius rule on (the run must complete; no sub-stepping, ICR-004 amendment 2) | A8 | todo | | E1-GPU |
| B2 | `fring-bench` again with collisions (GPU: K5-WIRE) and with freeze and the radius rule on; budget check for S1 | A5 | todo | | K5, K5-WIRE, E1-GPU |
| Gate G4 | Collision models validated; arc size chosen; close encounters handled; cost measured | AJ | todo | | K1, K1-REFS, ICR-009-IMPL, ICR-009-IO, K2, K3, K2-CPU, K3-CPU, K4, K5, K5-WIRE, K5-BIN, D1, D1-CPU, ARC-CELLS, D2, E1-CPU, E1-REF-2, E1-RUN, E1-GPU, A3-ANALYSIS, B2 |

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
| X1 | Figures and numbers for the explainer, each from a script (was E1) | A8 | todo | | Gate G5 |
| E2 | README, docs site, licence, citation file, release v1.0 | A0 | todo | | W3, X1 |
| Gate G6 | Publish | AJ | todo | | W3, X1, E2 |
