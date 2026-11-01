# F ring simulation: implementation plan for an agent team

Status: draft v1.1 (2026-11-01). Owner: AJ. Put this file in the repository as `docs/PLAN.md`, next to `AGENTS.md` and `CLAUDE.md` at the repository root.

Change log: v1.1 adds the lean operating mode for a Pro subscription (section 3) and the coordination files in task F1.

## 1. Purpose

Build an open-source Rust code that simulates Saturn's F ring with its two shepherd moons (Prometheus and Pandora) over decades of simulated time, with inelastic collisions, and show the results in a browser. The scientific question: why is the F ring core narrow and long-lived when it sits in a region where Prometheus makes orbits chaotic?

Outputs:

1. A Rust workspace (open source, documented).
2. A web app: replay of overnight runs, plus a live test-particle toy.
3. Data and figures for a written explainer (AJ writes the explainer; agents supply figures and numbers).

There is no previous dissertation or code to port. The physics comes from the literature (workstream A1).

## 2. Scope

In scope:

- Saturn as a point mass plus J2, J4 and J6 (J6 added at gate G0, ICR-001).
- Prometheus and Pandora as massive bodies that interact with Saturn and with each other.
- Ring particles that feel Saturn and the moons but do not affect the moons.
- Collisions between ring particles: a stochastic superparticle model (primary) and a hard-sphere model (check).
- Two domains: full ring and a co-moving arc of the core.
- GPU backend (wgpu on Metal) as the main engine, with a CPU f64 backend as the reference.

Not in scope (v1): particle self-gravity, other moons (Mimas, Janus, Epimetheus etc.), radiation forces, plasma drag, particle growth or fragmentation, J8 and higher.

## 3. Rules for agents

The full rules are in `AGENTS.md` (all agents) and `CLAUDE.md` (Claude Code only). This section is a summary.

**Operating mode: lean (decided 2026-11-01).** AJ uses a Pro subscription, so usage limits are the main constraint and sessions can stop without warning. Thus:

- One main Claude Code session at a time. It takes one role per task, from the task board. The "agents" in section 8 are roles, not parallel sessions.
- At most one subagent at a time: for A6 validation (independence) or a large read-only search.
- Default model Sonnet. Opus only for interface design (F2), gate reviews, the GPU precision spike (G1), the collision specification (K1), and validation failures that are not obvious after one attempt.
- All state is in the repository: `docs/status/BOARD.md` (task board), `docs/handoffs/<task-id>.md` (one per task, with an exact "Next step" line), `docs/interfaces/` (change requests), `docs/gates/` (gate evidence and approval), `docs/adr/` (decisions). Agents commit and update the handoff at each checkpoint, at least every 30 minutes of work.
- If usage increases later, change the "Operating mode" section of `CLAUDE.md` to allow parallel roles in separate worktrees.

- **Ownership.** Each workstream owns its directories (section 8). Do not edit files outside your directories. To change a shared interface, open an interface change request (an issue with label `interface`) and wait for the lead agent (A0).
- **Worktrees and branches.** One git worktree per agent. Branch name: `ws/<id>-<short-task>` (for example `ws/a2-j2-forces`). Small pull requests, one task each.
- **Definition of done for a task.** Code compiles on stable Rust; `cargo fmt`, `cargo clippy -D warnings` and all tests pass; the acceptance criteria in the task are met and shown in the pull request description (numbers, plots or logs).
- **Gates.** At each gate (section 9), work stops until AJ approves. Agents do not start tasks of the next phase before the gate.
- **Units in code.** Every physical quantity has its unit in the name or doc comment. No unitless magic numbers.
- **Sources.** Every physical constant has a citation in `params/sources.md`.
- **No silent tolerance changes.** If a validation test fails, do not relax the tolerance without an issue that explains why, approved by A0.
- **Determinism.** All randomness comes from the seeded counter-based RNG (section 4.5).

## 4. Conventions

### 4.1 Units

Kilometres, seconds, kilograms. Gravitational parameters as GM in km³ s⁻². Angles in radians in code, degrees only in configs and plots.

### 4.2 Frames

- **Inertial frame:** origin at Saturn's centre, x-y plane is Saturn's equatorial plane, z along the spin axis. All forces are defined here.
- **Rotating frame:** rotates about z at the reference mean motion n₀ of the F ring core (semi-major axis a₀). Used for output, collision cells and the arc domain.
- **Local (guiding-centre) coordinates:** relative to a reference circular orbit at a₀ in the rotating frame. x = radial offset, y = azimuthal arc length offset, z = vertical. Used by the GPU (f32) and by collision cells.

### 4.3 Precision policy

- CPU backend: f64 everywhere. It is the reference.
- GPU backend: Metal has no f64. Store state as f32 in local coordinates, and use double-single (two f32) arithmetic where a spike (task G1) shows that it is necessary. The GPU must agree with the CPU within the tolerances in section 10.

### 4.4 Configuration

One TOML file per run, read by `fring-run`. It contains: backend, domain, time step, duration, particle count and initial distribution, collision model and parameters, output cadence, RNG seed. The config is copied into the run output directory unchanged.

### 4.5 Random numbers

Philox4x32-10 (counter-based), implemented in Rust and in WGSL, with the same outputs for the same (seed, counter). Test vectors in `fring-model/tests/philox.rs`.

### 4.6 Snapshot format

Directory per run:

- `config.toml` (copy of input), `meta.json` (code version, git hash, machine, wall time).
- `moons.bin`: moon state at each output time.
- `frames/NNNNNN.bin`: header (time, particle count, frame type) plus particle positions in the rotating frame as f32 (x, y, z). Optional velocities.
- `stats.parquet` or `stats.csv`: per-output diagnostics (section 7.7).

The format has a version number in each header. A0 owns the spec in `docs/snapshot-format.md`.

## 5. Physical parameters

These values are approximate starting points. Task L2 replaces them with cited values (JPL SSD and recent Cassini papers) at a stated epoch.

| Quantity | Approximate value | Note |
|---|---|---|
| Saturn GM | 3.7931 × 10⁷ km³ s⁻² | |
| Saturn reference radius R | 60,330 km | J2, J4, J6 are defined for a specified R |
| J2 | 1.6291 × 10⁻² | |
| J4 | −9.36 × 10⁻⁴ | |
| J6 | 8.68 × 10⁻⁵ | Added at G0 (ICR-001); cited value in `params/saturn.toml` |
| F ring core semi-major axis | about 140,220 km | |
| F ring core eccentricity | about 0.0024 | Apse precesses with J2, J4, J6 |
| Prometheus semi-major axis | about 139,380 km | Mass about 1.6 × 10¹⁷ kg |
| Pandora semi-major axis | about 141,720 km | Mass about 1.4 × 10¹⁷ kg |
| Prometheus–Pandora interaction | near 121:118 resonance, about 6.2 yr period | Chaotic |

## 6. Architecture

### 6.1 Repository layout

```
fring/
  Cargo.toml                 (workspace)
  AGENTS.md                  (rules from section 3, short)
  docs/                      (PLAN.md, ADRs, snapshot-format.md, literature/)
  params/                    (saturn.toml, moons.toml, ring.toml, sources.md)
  crates/
    fring-model/             (types, constants, config, RNG, force functions, analytic formulas)
    fring-cpu/               (f64 reference backend, rayon)
    fring-gpu/               (wgpu backend, WGSL shaders)
    fring-collide/           (collision models: shared spec, CPU implementations)
    fring-io/                (snapshot read/write, stats)
    fring-analysis/          (chaos indicators, ring diagnostics)
  apps/
    fring-run/               (native CLI for overnight runs)
    fring-bench/             (benchmarks)
    fring-web/               (WebAssembly app: replay + live toy)
  validation/                (Python: REBOUND comparisons, plots, notebooks)
  runs/                      (git-ignored output)
```

### 6.2 Core interface

`fring-model` defines the shared trait. Both backends implement it.

```rust
pub trait Backend {
    fn new(config: &RunConfig, init: &InitialState) -> Result<Self> where Self: Sized;
    fn step(&mut self, n_steps: u64) -> Result<()>;
    fn time(&self) -> f64;                       // seconds since epoch
    fn moons(&self) -> MoonState;                // f64, inertial
    fn particles(&mut self) -> ParticleView<'_>; // copies GPU data to host if needed
    fn stats(&mut self) -> StepStats;            // energy error, collisions this interval, etc.
}
```

Rules:

- `fring-model` has no dependency on wgpu, rayon or I/O, and compiles to `wasm32-unknown-unknown`.
- Force functions in `fring-model` are generic over a float trait so that the CPU uses f64 and tests can compare with f32.
- WGSL shaders mirror the Rust force functions. Each shader function has a Rust twin and a test that compares them (section 10).

### 6.3 Moons

Moons are few, so integrate them on the CPU in f64 with a high-accuracy integrator (for example IAS15-style or a high-order symplectic method with a small step). Before a run, `fring-run` pre-computes a moon ephemeris table for the full duration and uploads it to the GPU. Particle kernels interpolate moon positions (Hermite, from positions and velocities). This removes moon integration from the hot loop and makes CPU and GPU see the same moons.

## 7. Physics specification

### 7.1 Gravitational potential

Φ(r, z) = −(GM/r) [1 − J2 (R/r)² P₂(z/r) − J4 (R/r)⁴ P₄(z/r) − J6 (R/r)⁶ P₆(z/r)]

with P₂(u) = (3u² − 1)/2, P₄(u) = (35u⁴ − 30u² + 3)/8 and P₆(u) = (231u⁶ − 315u⁴ + 105u² − 5)/16. Here r is the spherical radius. Acceleration a = −∇Φ in Cartesian coordinates. Test the analytical gradient against a finite-difference gradient.

Moon acceleration on a particle: point mass, a = −GMᵢ (r − rᵢ)/|r − rᵢ|³. Add a softening only if a test shows close encounters cause problems, and record it in an ADR.

### 7.2 Analytical checks (for near-circular equatorial orbits)

With Ω_K² = GM/r³ and q = R/r:

- n² = Ω_K² [1 + (3/2) J2 q² − (15/8) J4 q⁴ + (35/16) J6 q⁶]
- κ² = Ω_K² [1 − (3/2) J2 q² + (45/8) J4 q⁴ − (175/16) J6 q⁶]
- ν² = Ω_K² [1 + (9/2) J2 q² − (75/8) J4 q⁴ + (245/16) J6 q⁶]
- Apsidal precession ϖ̇ = n − κ; nodal regression Ω̇ = −(ν − n).

For a circular equatorial orbit in the potential of section 7.1 these are exact, not approximations. L2 confirmed the J2 and J4 terms by symbolic derivation (`docs/literature/formulas.md`, `docs/literature/derivations/n_kappa_nu.py`); A0 derived the J6 terms in the same way for ICR-001.

### 7.3 Integrators

- **Baseline (ADR 0003):** Wisdom-Holman split: an exact Kepler drift about Saturn alternates with kicks from J2, J4, J6 and the moons, drift-kick-drift, fixed step. Target 150 steps per orbit. At 150 steps per orbit it gives the apse precession at the core to 0.001 %; the plain leapfrog gives −9.7 % (ADR 0003). The former option C5 is now the baseline.
- **Reference:** the drift-kick-drift leapfrog (task C1) stays in the code for comparison and fast tests.
- Close encounters with Prometheus: measure the closest approach distribution. If some particles need smaller steps, use per-particle sub-stepping (block time steps) rather than reducing the global step.

### 7.4 Initial conditions

Configurable distributions: a narrow core (Gaussian in semi-major axis, with given eccentricity and apse), a uniform band across the moons' region (for chaos maps), and a "clump" (a localised cloud, for example to mimic a collision event). Moon initial states come from cited elements at the run epoch.

### 7.5 Collision models

**Stochastic superparticle model (primary).** Each superparticle represents W real particles of radius s. In each step:

1. Bin particles into cells in local coordinates (cell size a fraction of the core width, set in config).
2. In each cell, select candidate pairs with a no-time-counter method (Bird, DSMC). The collision probability of a pair is proportional to W σ v_rel Δt / V_cell, where σ = π (2s)².
3. For each accepted pair, apply an inelastic collision with normal restitution ε(v_n) (configurable: constant, or a velocity-dependent law such as Bridges et al. 1984) and a random impact geometry. Conserve linear momentum exactly.
4. Count collisions and energy loss per cell for diagnostics.

**Hard-sphere model (check).** Real overlap detection with a cell list in local coordinates, impulse response with the same ε(v_n). Used at small N in an arc to validate the stochastic model.

Both models share one specification (`fring-collide/SPEC.md`) and one set of statistical tests.

### 7.6 Domains

- **Full ring:** all longitudes. Used for test particles and for low-density collisional runs.
- **Co-moving arc:** an azimuthal arc of width Δθ that rotates at n₀. Particles that leave one end come in at the other end with the same local x, z and velocity. Moon forces stay exact (moons move in the inertial frame). This is an approximation: wrapped particles change their phase relative to Prometheus. Task D2 measures the error against the full ring.

### 7.7 Diagnostics

Per output interval:

- Single-particle tests: relative energy error, measured precession rates.
- Chaos: a fast Lyapunov indicator or MEGNO per test particle (needs the variational equations or a shadow particle), and diffusion of semi-major axis.
- Ring: core width (radial RMS and 90 % width), mean eccentricity and apse longitude, fraction of particles still in the core band, velocity dispersion, collision rate.

## 8. Workstreams and agents

| ID | Workstream | Owns | Depends on |
|---|---|---|---|
| A0 | Lead and integration | `Cargo.toml`, `docs/`, `AGENTS.md`, `crates/fring-model` (interfaces), CI | |
| A1 | Literature and parameters | `docs/literature/`, `params/` | |
| A2 | CPU reference backend | `crates/fring-cpu` | A0 interfaces |
| A3 | GPU backend | `crates/fring-gpu` | A0 interfaces, A2 for comparison |
| A4 | Collisions | `crates/fring-collide`, collision shaders in `fring-gpu/shaders/collide/` | A2, A3 |
| A5 | I/O, runner, benchmarks | `crates/fring-io`, `apps/fring-run`, `apps/fring-bench` | A0 |
| A6 | Independent validation | `validation/` | A1; reads outputs from A2 to A5 |
| A7 | Web | `apps/fring-web` | A3 (shaders), A5 (format) |
| A8 | Analysis | `crates/fring-analysis`, figures | A5, A6 |

A6 must not read the implementation code of A2 to A4 when it writes oracle tests. It writes tests from the physics specification only, so that the tests are independent.

## 9. Phases, tasks and gates

Task format: ID, owner, description, acceptance criteria. Tasks in the same phase can run in parallel unless a dependency is listed.

### Phase 0: foundation

- **F1 (A0).** Create workspace, CI (fmt, clippy, tests on macOS), ADR template, and the coordination files: `docs/status/BOARD.md` with all tasks from this section, and empty `docs/handoffs/`, `docs/interfaces/`, `docs/gates/` and `docs/adr/`. `AGENTS.md` and `CLAUDE.md` already exist; update their command list. *Done when:* empty crates compile; CI is green; the board lists every task with its role and blockers.
- **F2 (A0).** `fring-model`: units doc, config structs, `Backend` trait, float trait, Philox RNG with published test vectors. *Done when:* RNG test vectors pass in Rust; `fring-model` builds for wasm32.
- **F3 (A5).** Snapshot format spec and `fring-io` read/write with round-trip tests. *Done when:* a 10⁶-particle frame round-trips bit-exact.
- **L1 (A1).** Literature survey of F ring dynamics since about 2000 (shepherding, chaos, the core location, jets and spirals, moonlets, collisional models). *Done when:* `docs/literature/summary.md` lists hypotheses, key results with numbers, and the observations a model must reproduce, each with a citation.
- **L2 (A1).** Cited parameter set at a stated epoch (section 5) and confirmed analytical formulas (section 7.2). *Done when:* `params/*.toml` and `params/sources.md` exist; every number has a source.

**Gate G0 (AJ):** approve interfaces, parameters and the list of observations to reproduce.

### Phase 1: validated orbits (CPU)

- **C1 (A2).** Saturn point mass + J2/J4/J6 forces and the leapfrog. *Done when:* finite-difference gradient test passes (relative error < 10⁻⁸); a circular orbit at a₀ has relative energy error < 10⁻¹⁰ over 10⁴ orbits with no secular drift.
- **C2 (A2).** Precession test. *Done when:* measured ϖ̇ and Ω̇ at three radii agree with section 7.2 to better than 0.1 %.
- **C3 (A2).** Moon ephemeris integrator and Hermite interpolation. *Done when:* interpolation error at mid-interval < 10⁻³ km; moon orbits agree with A6's REBOUND run (V2).
- **C4 (A2).** Test particles with moons, rayon parallel. *Done when:* agrees with REBOUND for 100 particles over 100 orbits (V3).
- **V1 to V3 (A6).** REBOUND (with REBOUNDx for J2/J4, and J6 as an additional force) reference runs for: single-particle precession (V1), Prometheus–Pandora over 20 years, including the resonance period (V2), test particles near the core (V3). *Done when:* scripts produce reference files and comparison plots in `validation/reports/`.
- **R1 (A5).** `fring-run` CLI: reads config, runs a backend, writes snapshots and `meta.json`, shows progress. *Done when:* a CPU run of 10⁴ particles for 1 year completes from one config file.

**Gate G1 (AJ):** CPU backend agrees with REBOUND; moon chaos looks right.

### Phase 2: GPU backend

- **G1 (A3).** Precision spike. Compare three options on one circular orbit and one eccentric orbit over 10⁴ orbits: (a) f32 local coordinates, (b) f32 local coordinates with double-single position accumulation, (c) double-single throughout. *Done when:* an ADR picks the cheapest option that meets the tolerances in section 10.
- **G2 (A3).** wgpu setup, buffers (structure of arrays), kick and drift kernels, moon table upload and interpolation in WGSL. Each WGSL function has a Rust twin test. *Done when:* GPU matches CPU (section 10) for 10⁴ test particles over 10³ orbits.
- **G3 (A3).** Chaos indicator on GPU (shadow particle or variational equations). *Done when:* matches CPU indicator for 1,000 particles.
- **B1 (A5).** `fring-bench`: particle-steps per second for CPU and GPU at N = 10⁴ to 10⁷. *Done when:* results table in `docs/benchmarks.md` with machine details.
- **A1-ANALYSIS (A8).** First science result: chaos map (indicator versus initial semi-major axis and longitude relative to Prometheus) across the region between the moons. *Done when:* figure and short note on whether a quiet band appears near the observed core.

**Gate G2 (AJ):** GPU agrees with CPU; first chaos map reviewed.

### Phase 3: first web toy

- **W1 (A7).** `fring-web` scaffold: Rust to wasm32 with wgpu (WebGPU), renders particles in the rotating frame. *Done when:* it runs in current Chrome and Safari.
- **W2 (A7).** Live test-particle mode using the `fring-gpu` shaders: about 10⁵ particles, controls for time speed, Pandora on/off, and initial band. *Done when:* runs at 30 fps or more on the Mac mini; deploys as a static site.

**Gate G3 (AJ):** first public demo.

### Phase 4: collisions

- **K1 (A4).** `fring-collide/SPEC.md` and the statistical tests (shared by both models): momentum conservation, energy loss per collision versus ε, collision rate in a uniform box versus kinetic theory. *Done when:* A0 approves the spec.
- **K2 (A4).** Stochastic model on CPU. *Done when:* all K1 tests pass; collision rate in a uniform box is within 2 % of theory.
- **K3 (A4).** Hard-sphere model on CPU (cell list). *Done when:* K1 tests pass.
- **K4 (A4, A6).** Cross-check: in a small arc (about 10⁵ particles), compare stochastic and hard-sphere models for velocity dispersion and radial spreading rate. *Done when:* report shows agreement or explains the difference.
- **K5 (A4, A3).** GPU binning (counting sort by cell each step) and stochastic collisions in WGSL. *Done when:* GPU and CPU agree statistically (same distributions within sampling error) on K1 tests.
- **D1 (A3).** Co-moving arc domain with azimuthal wrapping. *Done when:* wrap preserves energy in the local frame; unit tests for boundary cases.
- **D2 (A6).** Arc error test: test particles (no collisions) in full ring versus arc for core width and eccentricity over 5 years, for Δθ in {5°, 10°, 30°}. *Done when:* report recommends the smallest Δθ with error < 5 %.

**Gate G4 (AJ):** collision models validated; arc size chosen.

### Phase 5: science runs

- **S1 (A8, A5).** Run matrix for overnight runs: with and without collisions, with and without Pandora, several core locations and ε values, 30 years each. *Done when:* AJ approves the matrix and it fits the compute budget (section 11).
- **S2 (A8).** Run the matrix; compute core width, survival fraction and eccentricity over time. *Done when:* figures and a results note in `docs/results/`.

**Gate G5 (AJ):** results reviewed; decide the explainer story.

### Phase 6: replay and showcase

- **W3 (A7).** Replay viewer: streams snapshot frames (decimated, compressed) and plays them with a time slider. WebGL2 fallback for replay if WebGPU is not available. *Done when:* a 30-year run plays smoothly from a static host.
- **E1 (A8).** Figures and numbers for the explainer, each reproducible from a script in `validation/` or `fring-analysis`.
- **E2 (A0).** README, docs site, licence, citation file, release v1.0.

**Gate G6 (AJ):** publish.

## 10. Testing and validation

| Test | Tolerance |
|---|---|
| J2/J4/J6 gradient versus finite difference | relative < 10⁻⁸ (f64) |
| Energy, single particle, 10⁴ orbits (CPU) | relative < 10⁻¹⁰, no secular trend |
| Precession rates versus analytical | < 0.1 % |
| CPU versus REBOUND, test particles, 100 orbits | position difference < 1 km (non-chaotic orbits) |
| GPU versus CPU, 10³ orbits, non-chaotic orbits | semi-major axis difference < 0.01 km; phase difference < 10⁻⁵ rad |
| GPU versus CPU, chaotic orbits | statistics only (indicator distribution, diffusion rate) |
| WGSL function versus Rust twin | relative < 10⁻⁶ (f32) |
| Collision rate versus kinetic theory | < 2 % |
| Momentum per collision | exact to rounding |

Chaotic orbits diverge between any two codes, so compare statistics for them, never trajectories.

## 11. Performance plan

Required throughput for the target overnight run (8 h = 28,800 s):

- 30 years ≈ 18,000 orbits. At 100 steps per orbit: 1.8 × 10⁶ steps.
- With 10⁶ superparticles: 1.8 × 10¹² particle-steps / 28,800 s ≈ 6 × 10⁷ particle-steps per second.
- With 10⁷ superparticles: about 6 × 10⁸ particle-steps per second.

Targets: 10⁶ superparticles with collisions (must), 10⁷ (stretch). Measure with `fring-bench` before you optimise.

Optimisation list, in expected order of value (estimates, to be measured):

1. Pre-computed moon ephemeris (removes moon integration from the hot loop).
2. GPU kernels with structure-of-arrays buffers and f32 local coordinates.
3. Stochastic collisions instead of geometric detection.
4. Co-moving arc (more particles per unit of ring for the same N).
5. Larger time step with a split integrator (C5) and per-particle sub-stepping only near Prometheus.
6. Keep particles sorted by cell to improve memory access.
7. Output only decimated frames and on-GPU statistics; avoid copying all particles to the host every step.

## 12. Risks

| Risk | Mitigation |
|---|---|
| f32 on GPU gives secular drift | Precision spike G1 before other GPU work; CPU f64 reference stays in CI |
| Rust beginner owner cannot review GPU code easily | Rust twin for each WGSL function; plain-language summary in each pull request |
| Chaos makes CPU/GPU comparison look like a bug | Compare statistics for chaotic orbits (section 10) |
| Arc wrapping gives false structure | Measure in D2 before science runs |
| Parameters from memory are wrong | L2 replaces all values with cited ones before G0 |
| Agents edit shared code at the same time | Ownership table, interface change requests, small pull requests |

## 13. Open questions for AJ

1. Which Mac mini chip and memory (this sets realistic GPU particle counts)?
2. Licence (MIT/Apache-2.0 dual is common in Rust)?
3. Repository host and name?
4. Which observations matter most for the story (core width, core eccentricity, jets and spirals, or survival time)?
