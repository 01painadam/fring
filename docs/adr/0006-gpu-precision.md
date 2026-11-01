# 0006: GPU number format for the Wisdom-Holman integrator (task G1)

Status: accepted by A0 (2026-11-01); AJ reviews at gate G2
Date: 2026-11-01
Author: A3  Approved by A0: not yet

## Context

PLAN section 4.3 says the GPU stores f32 and uses double-single (DS, two f32: hi + lo, about 48 significant bits) where task G1 shows that it is necessary. Since ADR 0003, the integrator is Wisdom-Holman (WH): an exact Kepler drift on Saturn-centred position and velocity, and a kick with J2, J4, J6, the moons and the indirect term (ADR 0004). At r = 1.4 × 10⁵ km, f32 resolves 0.016 km.

**Tolerances** (PLAN section 10, as written; A0 decision: they decide the option):

- GPU against CPU, 10³ orbits, non-chaotic orbits: semi-major axis difference Δa < 0.01 km; phase difference Δλ < 10⁻⁵ rad.
- WGSL function against its Rust twin: relative < 10⁻⁶ (f32).

**What they require.** Δλ < 10⁻⁵ rad after 10³ orbits (N = 1.5 × 10⁵ steps) means:
- a systematic mean-motion error below 10⁻⁵ / (2π · 10³) = 1.6 × 10⁻⁹ (a constant Δa of 0.15 m);
- a random error in a per step below 4.7 × 10⁻¹² of a (Brouwer's law: Δλ_rms = 1.5 n h (σ_a/a) N^{3/2} / √3).

f32 has ε = 1.2 × 10⁻⁷, so the phase tolerance is about 1/70 of one f32 rounding. The Δa tolerance alone would allow 1.8 × 10⁻¹⁰ per step; the phase tolerance is the hard one.

**For comparison: the CPU's own error.** At 150 steps per orbit the f64 CPU code differs from the true orbit by its splitting error (ADR 0005). Measured here as the CPU at step h against the CPU at h/10, without moons: Δλ = 1.5 × 10⁻⁴ rad at 10³ orbits and 1.5 × 10⁻³ rad at 10⁴ orbits, growing linearly (a mean-motion error of 2.4 × 10⁻⁸). The GPU tolerance is 15 times smaller than this.

**Decision rule (AJ at gate G1, A0):** the GPU runs the science runs only if the cheapest option that meets the tolerances is at least 3 times faster than the CPU backend (6.4 × 10⁷ particle-steps/s; 6.9 × 10⁷ in the benchmark below).

## The GPU (probe, `docs/adr/0006/probe_m4.txt`)

Apple M4, Metal, wgpu 30.0.1; 2²⁰ random values per check.

| Check | Result |
|---|---|
| `+` and `*` correctly rounded | yes (0 failures) |
| `fma` fused | yes (0 failures) |
| Plain TwoSum, Fast2Sum, Dekker split | **fail** for every value with a non-zero error term |
| Guarded TwoSum and Fast2Sum (see below) | exact (0 failures) |
| Guarded DS add, mul, div, sqrt | 2.4, 3.4, 4.8, 5.9 × 2⁻⁴⁸ (published bounds 3, 5, 15 for add, mul, div) |
| DS add, mul, sine, cosine: GPU bits against the Rust twin | identical (0 differences) |
| Division `a / b` | max 1.71 ULP; 28 % not correctly rounded |
| `sqrt` | max 1.73 ULP; 28 % not correctly rounded |
| Built-in `sin`, `cos` on [−π, π] | max 1.3 × 10⁻⁷ absolute (the WGSL bound is 4.9 × 10⁻⁴) |
| Own `sin_cos_f32` | max 9.3 × 10⁻⁸ absolute; GPU bits identical to the Rust twin |

**Cause of the failures.** wgpu-hal 30 builds Metal libraries with the default `MTLCompileOptions`, which has fast math on; naga 30 has no option to change it. Fast math lets the compiler rewrite `(a + b) − a` as `b`, which removes the error terms of TwoSum.

**Guard.** The DS library (`crates/fring-gpu/shaders/ds.wgsl`) multiplies the sums whose error it needs by `ds_one`, a 1.0 read at run time from a uniform buffer. The compiler cannot know that it is 1.0, so it cannot cancel the terms; the multiplication is exact, so the results are those of the plain algorithms. Cost: two extra multiplications per Fast2Sum and a sort by magnitude in TwoSum (DS add from 20 to 32 operations). TwoSum is Fast2Sum after a branch-free sort by magnitude: Knuth's 6-operation TwoSum with guards passed alone but failed inside the DS addition (the compiler re-associated it after inlining). So a check of each primitive alone is not enough.

## Options

All options run the same WH step (half drift, then kick and drift, half drifts merged) with the same moon table. They differ only in the number format. Rust twins: `crates/fring-gpu/src/spike.rs`; WGSL: `shaders/spike.wgsl`.

| Option | State | Kepler drift | Kick |
|---|---|---|---|
| A | f32 | f32 (generic `kepler_drift::<Wf32>`) | f32 |
| B | DS | f32 from f32 copies of the state; increments added to the DS state | f32, added to the DS velocity |
| B+ | DS | DS: 1/a, a, n, e cos E₀, e sin E₀, n dt; f32: Kepler solve and the f, g coefficients | as D |
| D | DS | DS, except the Kepler equation: f32 Newton, then one DS Newton step with a DS small-angle series for sin, cos and 1 − cos | f32 force magnitudes times the DS position; moon separations in DS |
| C | DS | DS throughout (generic `kepler_drift::<Ds>`) | DS throughout |

PLAN option (a), "f32 local coordinates", does not fit WH: the exact drift needs Saturn-centred position and velocity, so local coordinates would be converted back every step and keep the 0.016 km resolution. A is the nearest equivalent. PLAN option (b) is B; PLAN option (c) is C. B+ and D are the "DS only where needed" options of the plan for G1.

Found on the way: in D, `g = dt − (x − sin x)/n` must be DS. An f32 quotient biases every step of a circular orbit in the same way (the position moves along the velocity without the matching velocity change), and a drifts: Δλ = 3.0 × 10⁻⁵ rad at 10³ orbits with f32, 2.2 × 10⁻⁶ with DS.

## Results

All numbers are GPU (M4) against the f64 CPU code, from the same initial state. Ensemble: three orbit types at a = 140,222.43 km, 32 particles each at different longitudes and epicycle phases: circular; core (e = 0.0024, i = 1.1 × 10⁻⁴); stress (e = 0.01, i = 10⁻³; it crosses Prometheus' orbit). Step h = 356.25 s (150.03 steps per orbit). Δλ is the angle between the two position vectors in the equatorial plane; Δa is the difference of the osculating semi-major axes.

**Chaos test.** An f64 shadow particle starts 1 mm further out. A particle is regular up to a time if the shadow's |Δa| stays below 10 times its start value and its Δλ below 10 times the linear Kepler shear. Without moons, all 96 particles are regular to 10⁴ orbits. With both moons: at 10³ orbits 47 are regular (circular 30, core 16, stress 1); at 10⁴ orbits none are. PLAN section 10 applies to the regular particles.

### Maximum over all regular particles (all three orbit types)

| | CPU's own error (h against h/10) | A | B | B+ | **D** | C |
|---|---|---|---|---|---|---|
| Saturn only, 10³ orbits: Δa (km) | 2.7 × 10⁻³ | 23 | 1.4 | 0.59 | **2.2 × 10⁻⁴** | 8.9 × 10⁻⁷ |
| Saturn only, 10³ orbits: Δλ (rad) | 2.1 × 10⁻⁴ | 0.71 | 4.1 × 10⁻² | 1.7 × 10⁻² | **3.8 × 10⁻⁶** | 2.7 × 10⁻⁸ |
| Both moons, 10³ orbits: Δa (km) | 1.8 × 10⁻² | 41 | 5.1 | 6.4 | **1.2 × 10⁻³** | 1.6 × 10⁻⁶ |
| Both moons, 10³ orbits: Δλ (rad) | 2.6 × 10⁻⁴ | 1.5 | 0.26 | 0.26 | **6.1 × 10⁻⁶** | 2.6 × 10⁻⁸ |
| Saturn only, 10⁴ orbits: Δa (km) | 2.2 × 10⁻² | 80 | 33 | 15 | **2.4 × 10⁻³** | 7.6 × 10⁻⁶ |
| Saturn only, 10⁴ orbits: Δλ (rad) | 2.1 × 10⁻³ | 3.1 (saturated) | 3.1 (saturated) | 1.8 | **1.0 × 10⁻⁴** | 1.3 × 10⁻⁶ |
| PLAN 10 at 10³ orbits (Δa < 0.01 km, Δλ < 10⁻⁵ rad) | | fails | fails | fails | **passes** | passes |

Both moons at 10⁴ orbits: every particle is chaotic, so trajectories cannot be compared (PLAN section 10). As a statistic, the median over particles of Δλ(GPU − CPU) / Δλ(1 mm shadow) is 1.0 to 1.4 for every option on the core and stress orbits (all phases are randomised), and on the circular orbits A 143, B 56, B+ 71, D 14, C 0.30. At 10³ orbits the same ratio is 23 to 72 for D and 0.09 to 0.15 for C: D's precision error equals a change of the initial position by a few centimetres, C's by less than 0.2 mm.

With the moons, the CPU's own error at 10³ orbits is of the same order as without them (Δλ up to 2.6 × 10⁻⁴ rad).

### Growth with time (Saturn only, fit of RMS ∝ orbits^p from 100 to 10³ orbits)

| | A | B | B+ | D | C |
|---|---|---|---|---|---|
| p for Δa | 0.45 to 0.61 | 0.08 to 0.95 | 0.0 to 1.10 | 0.77 to 0.93 | 0.48 to 0.70 |
| p for Δλ | 1.37 to 1.68 | 1.06 to 1.89 | 1.01 to 2.01 | 0.99 to 1.13 | 1.51 to 1.74 |

A follows Brouwer's law (random rounding: Δa ∝ t^½, Δλ ∝ t^{3/2}). B and B+ have a systematic drift of a (Δλ ∝ t²) on eccentric orbits. D has a constant mean-motion offset (Δλ ∝ t): its f32 kick magnitudes give a small systematic error in the J2 force (with a DS kick instead, D's phase error after 10 orbits fell from 1.3 × 10⁻⁸ to 7.4 × 10⁻¹¹ rad in a test). C is a random walk at DS level.

### Rust twins and error models

The Rust twins with the M4 division model give the same sizes as the GPU (D, Saturn only, 10³ orbits: Δλ max 4.1 × 10⁻⁶ against 3.8 × 10⁻⁶ on the GPU). With the WGSL worst-case model (division and sqrt ±2 ULP), D reaches 7.4 × 10⁻⁶ rad (Saturn only) and 9.3 × 10⁻⁶ rad (moons, circular) at 10³ orbits: still inside the tolerance, with little margin. All values: `docs/adr/0006/g1_*_summary.csv` (sources `gpu`, `rust_m4`, `rust_wgsl`, `cpu_own`).

### Cost (`docs/adr/0006/bench_m4.csv`)

10⁶ particles in the core, both moons, 150 steps per dispatch, best of 3, kernel time only. CPU: the `fring-cpu` code path (f64, rayon, all 10 cores), 10⁵ particles, same machine.

| | CPU f64 | A | B | B+ | **D** | C |
|---|---|---|---|---|---|---|
| Particle-steps per second | 6.9 × 10⁷ | 6.4 × 10⁹ | 5.1 × 10⁹ | 1.7 × 10⁹ | **1.1 × 10⁹** | 4.8 × 10⁸ |
| Times the CPU | 1 | 92 | 74 | 24 | **16** | 6.9 |
| With the WGSL built-in sin and cos instead of ours | | 6.8 × 10⁹ | 5.4 × 10⁹ | 1.7 × 10⁹ | 1.1 × 10⁹ | 4.8 × 10⁸ |

Our own sine and cosine cost at most 7 % (option A); for D and C the difference is below 1 %. The built-ins are not needed for speed.

**Would a cheaper option be acceptable?** The next cheaper option, B+, has Δλ up to 1.7 × 10⁻² rad at 10³ orbits: 80 times the CPU's own error and 1,700 times the tolerance. B and A are worse. D's error is 1/50 of the CPU's own error. So no cheaper option is acceptable even against the CPU's own error.

## Decision

1. **The science runs use the GPU with option D.** D is the cheapest option that meets PLAN section 10 (Δa ≤ 1.2 × 10⁻³ km and Δλ ≤ 6.1 × 10⁻⁶ rad at 10³ orbits, all regular particles, with and without moons). It runs 1.1 × 10⁹ particle-steps per second, 16 times the CPU backend: well above the 3-times rule of gate G1. It also reaches the 10⁷-particle stretch target of PLAN section 11 (6 × 10⁸) for test particles, before collisions.
2. **Option C stays as the GPU reference mode** (6.9 times the CPU, errors 100 times smaller than D). G2 keeps it as a switch, to check D in production configurations and as a fallback.
3. **The web toy (W2) uses option A** (plain f32; it does not depend on the guard or on a fused multiply-add, so it suits other GPUs and browsers). Its errors (Δa 6 to 8 km RMS and Δλ 0.2 to 0.3 rad at 10³ orbits) do not matter for a live picture of 10⁵ particles, but they blur structure finer than a few km after some hundred orbits. If W2 needs better, B costs 20 % of A's speed and cuts the error by 10 to 100.
4. The CPU backend stays as the f64 reference in CI (PLAN section 4.3).

## Consequences

- **PLAN section 4.3 (A0 to edit):** the GPU state is Saturn-centred, inertial position and velocity in DS (12 f32 per particle, 48 bytes; 10⁷ particles need 480 MB), not f32 local coordinates. Local coordinates (collision cells, K5) are computed from the DS state when needed; f32 is enough for cells.
- `Backend::particles()` converts DS to f64 exactly (hi + lo). The `Backend` trait does not change.
- **G2** builds the backend on `shaders/ds.wgsl`, `shaders/trig.wgsl` and the option-D kernel of `shaders/spike.wgsl`, with persistent buffers and blocks of steps. At start-up it must, and fail with `Error::Backend` otherwise:
  1. call `fring_gpu::probe::self_test` (guarded TwoSum, Fast2Sum, fma error and DS add and mul on 2¹⁶ values; about 0.1 s);
  2. compare its production kernels with the Rust twins for one step of a few particles (relative < 10⁻⁶), because a primitive can pass alone and fail after inlining (the TwoSum case above).
- The step: D and C carry h in DS, so any f64 step works. A needs a step that is exact in f32 (rounding h = T/150 to f32 alone gives 1.6 × 10⁻⁴ rad at 10³ orbits); this matters only for W2.
- Own sine and cosine stay in all shaders (portable; at most 7 % slower than the built-ins on the M4). The DS sine and cosine series is valid for |x| ≤ 0.25 rad; the Kepler drift at 150 steps per orbit needs |ΔE| ≤ 0.05.
- **Work that the GPU path adds** (A0 asked for the list): G2 (backend, moon table, buffers, start-up checks), G3 (chaos indicator on the GPU), K5 (binning and stochastic collisions in WGSL with DS positions), and the GPU part of B1. The CPU backend must keep pace as the reference.
- CI runs the Rust tests of `fring-gpu` (error models, DS bounds, sine and cosine, ten orbits per option; under 1 s). GitHub runners have no usable Metal GPU, so the probe, `gpu_twins`, the matrix and the benchmark run locally; their results are committed in `docs/adr/0006/`.
- To reverse: run the CPU backend for science runs (it meets the must target of PLAN section 11 for test particles). The GPU code then serves only W2.

## Risks

1. **Fast math.** wgpu-hal 30 compiles Metal shaders with fast math on and naga has no switch, so the DS algorithms survive only through the guard (`ds_one`). A later Metal compiler could see through it. Mitigation: the start-up self-test and twin check above (fail closed). Fallbacks, in order: (a) ask wgpu upstream for an option to disable fast math (`MTLCompileOptions.mathMode = .safe`); (b) Metal shading language passthrough shaders, which are not WGSL and need their own twins; (c) exact integer (fixed-point) accumulation of the state, which fast math cannot change, although D needs more than accumulation; (d) the CPU backend.
2. **The f32 parts of D** (Kepler solve before the DS step, kick magnitudes) depend on the GPU's division and square root. With the WGSL worst-case model D still passes, but at 7.4 to 9.3 × 10⁻⁶ rad. On a GPU other than the M4, rerun the probe and the matrix first. If D fails there, use C.
3. **Long runs.** D's phase error grows linearly: 10⁻⁴ rad at 10⁴ orbits, about 2 × 10⁻⁴ for a 30-year run (1.8 × 10⁴ orbits), against 1.6 × 10⁻³ and about 3 × 10⁻³ for the CPU's own splitting error. With the moons, all orbits are chaotic on that time scale, and D's error equals a few centimetres of initial position: far below the physical uncertainties.
4. **Twins match in distribution, not bit for bit,** where division or square root is involved (the model of the M4 hardware is statistical). DS add, mul, sine and cosine and the f32 sine and cosine match bit for bit (0 differences in 2²⁰). Kernel twins agree to 2.3 × 10⁻⁷ after one step (all options; PLAN section 10 asks 10⁻⁶) and to 5.4 × 10⁻¹³ (C) and 4.8 × 10⁻⁹ (D) after one orbit.

## Evidence

Commands (from the repository root; GPU commands need the M4):

```sh
cargo test -p fring-gpu                                                          # Rust tests, also in CI
cargo run -p fring-gpu --release --example g1_probe > docs/adr/0006/probe_m4.txt  # probe and self-test
cargo test -p fring-gpu --release --test gpu_twins -- --ignored --nocapture      # WGSL against Rust twins
cargo run -p fring-gpu --release --example g1_precision -- --orbits 10000 --moons none   # about 3 minutes
cargo run -p fring-gpu --release --example g1_precision -- --orbits 10000 --moons both   # about 3.5 minutes
cargo run -p fring-gpu --release --example g1_bench > docs/adr/0006/bench_m4.csv  # about 1 minute
uv run --no-project --with matplotlib python3 docs/adr/0006/plot_g1.py           # docs/adr/0006/g1_phase.png
uv run --no-project --with numpy python3 docs/adr/0006/remez_sincos.py           # sine and cosine coefficients
```

Files in `docs/adr/0006/`: `probe_m4.txt`, `g1_none_summary.csv` and `g1_both_summary.csv` (10³ and 10⁴ orbits, regular and all particles, growth exponents, ratio to the shadow), `g1_*_series.csv` (time series), `g1_*_particles.csv` (every particle), `bench_m4.csv`, `g1_phase.png` (RMS Δλ against orbits, core orbit), `remez_sincos.py`, `plot_g1.py`.

Code: `crates/fring-gpu/src/{wf32,ds,probe,spike,spike_gpu,gpu}.rs`, `crates/fring-gpu/shaders/{ds,trig,probe,spike}.wgsl`, `crates/fring-gpu/examples/{g1_probe,g1_precision,g1_bench}.rs`, `crates/fring-gpu/tests/gpu_twins.rs`.

Dependencies: wgpu 30.0.1, pollster 1.0.1 (A0, 2026-11-01). A clean workspace build takes 13.9 s instead of 5.4 s (debug) and 20.0 s instead of 7.2 s (release).
