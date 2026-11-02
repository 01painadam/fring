# 0008: Chaos indicator on the CPU and the GPU (task G3)

Status: proposed by A3 for A0; numbers below are from the M4 (2026-11-02)
Date: 2026-11-02
Author: A3 (and A2 for the CPU part)

## Context

PLAN 7.7 and task G3 need a chaos indicator per test particle, on the GPU and as a CPU reference. Task A1-ANALYSIS (A8) draws a chaos map from it. ICR-003 (accepted) defines the interface. The moons come from a fixed ephemeris, so the indicator measures the chaos of a particle in a given moon field, not of the moons.

## Options

1. **Shadow particle with renormalisation (Benettin), MEGNO from the renormalised series.** A second particle, d0 away, takes the same kicks and drifts. State 48 B more per particle (96 B); cost 2 times the step (measured).
2. **Variational equations with the WH tangent map.** A tangent vector (f32 would do) with the Kepler-drift tangent map and the kick Hessian of J2, J4, J6 and the moons. Cost estimate 2.5 to 3 times, about 600 lines of new WGSL with Rust twins, and a second integrator that A6 would need to validate.

## Decision

Option 1. It reuses the production kernel (the indicator measures the orbit that the GPU computes), needs about 150 lines of WGSL and gives the CPU twin for free (same algorithm on `wh_steps`). Option 2 stays possible later.

### Why a plain sum of logarithms is not the classifier (the shear argument)

On a regular orbit a radial offset delta grows linearly, because of the Kepler shear: dn/n = -1.5 delta/a. After N orbits the along-track offset is 1.5 * 2 pi N * delta. At N = 10^3 that is 9,400 delta, so ln(d/d0) is about 9.2 for a regular orbit. A threshold on the log growth cannot separate regular from chaotic orbits. MEGNO removes the linear part: a linear growth gives Y -> 2, an exponential growth d ~ exp(lambda t) gives Y = lambda t (mean about lambda t / 2).

### Definition (discrete MEGNO)

- K = `renorm_steps`, h = `step_s`. Renormalise at global step numbers that are multiples of K (`j = 1, 2, ...`; time is counted from the start of the run). The result does not depend on how `fring-run` or the GPU splits the steps into blocks: both backends cut their blocks at multiples of K. (Calls that end between multiples of K change the rounding of the half-step drifts only.)
- l_j = ln(d_j / d0), with d_j the position separation of shadow and particle just before renormalisation. The whole 6D separation is then scaled by d0 / d_j.
- L = sum l_j (the FTLE is L / t, t = j K h); S = sum (j - 1/2) l_j; Y_j = 2 S / j; mean MEGNO = (sum Y_j) / j. (Time is in units of K h, so no unit enters.)
- Checks (`physics/chaos.rs` tests): l_j = ln(j/(j-1)) gives Y -> 2; l_j = c gives mean Y = c j / 2 = lambda t / 2.
- The shadow starts at d0 along the radial direction (position only, zero velocity offset). Same start for CPU and GPU. No random stream is used. (A random direction was tried first: the results of the sweep were the same within noise, so the simpler rule won.)
- GPU: `shaders/chaos.wgsl`, kernel `chaos_renorm`. The separation comes from the DS difference (`ds_sub`), never from hi parts alone. l_j in f32 (error about 1e-7). L, S and sum Y are DS accumulators (`ds_add_f`, with the uniform-1.0 guard of M16). Rust twin: `fring_gpu::chaos::renorm`; twin test `tests/gpu_chaos.rs`: shadow state 9e-10 km, L 6e-7, S 1e-6, Y 8e-7 (limits 1e-9 km and 5e-6; `docs/adr/0008/twin_test.txt`).
- Memory: shadow state 48 B + accumulators 32 B per particle (128 B in all). Throughput on the M4 (100,000 particles, `docs/adr/0008/throughput_m4.txt`): D+ 9.3e8 -> 4.6e8 particle-steps/s, C 4.7e8 -> 2.3e8 (ratio 0.50 in both). A run without `[diagnostics.chaos]` allocates nothing extra and runs the same code as before.

## The sweep and the choice of d0 and K

Set: 1,000 circular particles, a uniform in 139,600 to 140,600 km, uniform longitude, both moons, h = 356.2537 s, 150,000 steps (about 1000 orbits). Baseline: the same CPU run with every particle moved by 1e-7 km. Files: `docs/adr/0008/sweep_*.txt`. D+ column: GPU against CPU, regular on both (557 of 1,000 at d0 = 1 km).

| d0 (km) | CPU vs CPU: regular dL p95 / max | GPU D+: regular dL p95 / max | GPU D+ class agreement |
|---|---|---|---|
| 1e-4 | 1.76 / 4.1 | 3.57 / 5.5 | 99.6 % |
| 1e-3 | 0.30 / 2.7 | 1.98 / 4.9 | 100 % |
| 1e-2 | 0.029 / 2.0 | 0.34 / 3.1 | 100 % |
| 1e-1 | 0.0032 / 0.21 | 0.044 / 1.2 | 100 % |
| **1** | **0.00036 / 0.050** | **0.0042 / 0.074** | **100 %** |
| 10 | 0.0001 / 0.098 | 0.0012 / 0.30 | 100 % (but the class counts and the <Y> median move: 572 regular, median 1.96 against 2.10; the tangent map is no longer linear) |

**Why a small d0 fails.** About 20 % of the particles have a bounded deviation (resonant or libration orbits, <Y> well below 2). For them the sum of logarithms has no large linear signal, and rounding noise in a (about 3e-11 km per step, random walk) grows by the shear into an along-track error of about 6e-5 km. That is 6 % of d0 = 1e-3 km, and it does not depend on the size of the perturbation (1e-10 to 1e-4 km gave the same dL). The error is proportional to 1/d0. The GPU noise is larger than the f64 noise by a factor of about 15. Radial offset (instead of a random direction) and a phase-space norm (positions plus velocity times 1/n) were both tried and did not change this.

**Choice: d0 = 1 km.** It is on the plateau 0.01 to 1 km (the Y quantiles 10/50/90 % are 0.38/2.12/8.18 at 0.1 km and 0.38/2.10/8.18 at 1 km; FTLE medians agree to 0.3 %). The band starts 220 km from Prometheus, so d0 / distance is about 5e-3 or less, except in close encounters. 10 km is already outside the linear range. K = 50, 100, 200 (d0 = 1 km) agree: the same classes within 2 particles, dL p95 0.0042 to 0.0051 (`sweep_k_*.txt`). **K = 100** (0.67 orbit) stays.

## Classes

Regular: mean Y <= 2.5. Chaotic: mean Y >= 5. In between: uncertain. At 1000 orbits and d0 = 1 km the CPU gives 557 regular, 161 uncertain and 282 chaotic of 1,000. The CPU baseline gives 557/161/282 as well, and the class agreement is 100 % of the 839 particles that are not uncertain. The Y quantiles (10/50/90 %) are 0.38, 2.10 and 8.18. The thresholds are unchanged from the A0 decision.

## Done-when result (GPU against CPU, 1,000 particles)

| | D+ | C | D (reported) |
|---|---|---|---|
| class agreement (839 particles compared; 836 for D) | 100 % | 100 % | 100 % |
| regular on both: dL median / p95 / max | 4.3e-4 / 4.2e-3 / 0.074 | 4.4e-4 / 5.4e-3 / 0.13 | 1.3e-2 / 0.15 / 2.0 |
| chaotic on both: FTLE quantile difference 10/50/90 % | 0.4 / 0.5 / 1.3 % | 0.0 / 0.6 / 1.9 % | 1.5 / 0.3 / 0.2 % |
| baseline (CPU, 1e-7 km): same | 100 %; 0.0000243 / 0.00036 / 0.050; 0.1 / 0.2 / 0.5 % | | |

Rules 1, 2 and 4 pass for D+ and C, and the limit of 15 % in rule 3 passes with a factor of 8. **Rule 3 "at most 2 times the baseline difference" fails** for D+ (0.42 / 0.47 / 1.26 % against 2 x 0.09 / 0.19 / 0.48 %) and for C (50 / 90 % quantile). The baseline difference is below 3 %, which is the case that A0 named: the rule is not changed here (`done_when_spec_baseline_1e-7.txt`). The cause is that a 1e-7 km displacement is too small to decorrelate the chaotic orbits within 1000 orbits (growth e^10 about 2e4 turns 1e-7 km into 2e-3 km), while the GPU orbits differ from the CPU by 1e-4 km and more (G2: dA up to 0.01 km). With a baseline displacement of 1e-4 km (`FRING_G3_BASE_KM=1e-4`) all rules pass (`done_when_baseline_1e-4.txt`). A0 decides.

## Consequences

- `fring-model` (config table, `ChaosReport`, `Backend::chaos`, `physics/chaos.rs`), `fring-cpu`, `fring-gpu`, `fring-io` (`chaos.csv`, `meta.json` field) and `fring-run` change as in ICR-003.
- A8 reads `chaos.csv` (`apps/fring-run/configs/chaos-map.toml` is a ready config: 10^4 particles, 5 years, GPU). 10^4 particles for 5 years (about 2,900 orbits, 443,000 steps of 356 s) is 4.4e9 particle-steps, twice that with the shadow, so about 20 s of GPU time in D+ at 4.6e8 particle-steps/s.
- Limits: the indicator depends on the moon field and the length of the run; <Y> grows with t for chaotic orbits, so compare runs of equal length. Close encounters (d0 comparable to the moon distance) leave the linear range.
