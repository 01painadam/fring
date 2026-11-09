# 0003: Wisdom-Holman split as the baseline particle integrator

Status: accepted
Date: 2026-11-01
Author: A0  Approved by A0: 2026-11-01 (AJ approved at gate G1, 2026-11-01)

## Context

`docs/PLAN.md` section 7.3 made the drift-kick-drift leapfrog the baseline, with 100 to 200 steps per orbit, and a Wisdom-Holman split as an option (C5) "only if it passes the same validation tests and is faster in wall time".

In the C2 plan, A2 found that the leapfrog cannot meet the C2 criterion (precession within 0.1 % of PLAN section 7.2) at 150 steps per orbit. A0 checked this with an independent script (`docs/adr/0003/precession_integrators.py`). Setup: J2, J4 and J6 field from `params/saturn.toml`; a = 140222.43 km; e = 10⁻³; 300 orbits; ϖ̇ from a fit to the pericentre longitudes.

| Integrator | Steps per orbit | ϖ̇ (°/d) | Error |
|---|---|---|---|
| PLAN 7.2 (oracle) | | 2.69841 | |
| Leapfrog (DKD) | 150 | 2.43643 | −9.71 % |
| Leapfrog (DKD) | 600 | 2.68200 | −0.61 % |
| Yoshida 4th order (3 kicks per step) | 150 | 2.69657 | −0.068 % |
| Wisdom-Holman (Kepler drift, J kick) | 150 | 2.69838 | −0.0011 % |

ϖ̇ = n − κ is about 0.5 % of n, so a small relative error in n or κ becomes a large error in ϖ̇. The leapfrog's −9.7 % is −0.26 °/d. That is more than 100 times the effect of J6 (ICR-001), and it changes the 17-year apse cycle relative to Prometheus. The leapfrog also shifts κ, and κ sets the location of the m = 110 resonance. The leapfrog error scales as h², so 0.1 % needs about 1,500 steps per orbit.

## Options

1. Leapfrog with about 1,500 steps per orbit: 10 times the cost.
2. Leapfrog at 150 steps per orbit, with Richardson extrapolation in the test only. The test would pass, but production runs would keep the 10 % error. Rejected.
3. Yoshida 4th order: −0.07 % at 150 steps per orbit, with 3 force evaluations per step.
4. Wisdom-Holman: an exact Kepler drift about Saturn (GM) alternates with kicks from the non-Kepler forces (J2, J4, J6, and later the moons). −0.001 % at 150 steps per orbit. Cost per step: one Kepler solve (a few Newton iterations, with sin and cos) and one force evaluation. Consecutive half drifts merge into one full drift.

## Decision

Option 4. The Wisdom-Holman split, drift-kick-drift with a fixed step, is the baseline particle integrator for the CPU and the GPU. It meets the PLAN's own rule for C5: it passes the validation test, and it is faster at equal accuracy. Task C2 implements it in `crates/fring-model/src/physics/` (generic over `Real`) and validates it. The C1 leapfrog stays in the code as a reference and for fast tests.

## Consequences

- PLAN section 7.3 now names Wisdom-Holman as the baseline. C5 is no longer a separate task.
- The kick uses the acceleration without the point-mass term (`SaturnField` J terms, later plus the moons).
- **GPU (G1, G2):** the Kepler drift in f32 is part of the precision spike G1. Check the cancellation in 1/a = 2/r − v²/GM and the Kepler-equation solve near e ≈ 10⁻³.
- **Moons (C3, C4):** in a Saturn-centred frame, each moon also gives an indirect acceleration −GMᵢ rᵢ/|rᵢ|³ (Saturn is accelerated by the moon). PLAN 7.1 lists only the direct term. A0 decides this before C4, against the REBOUND reference runs (V2, V3).
- Close encounters with Prometheus make the kick large. Per-particle sub-stepping (PLAN 7.3) stays the remedy.
- To reverse: switch the step function back to the leapfrog in the backends. No interface changes.

## Evidence

- `docs/adr/0003/precession_integrators.py`, run with `uv run --no-project --with numpy python3 docs/adr/0003/precession_integrators.py`. Output as in the table above.
- C2 tests in the pull request for C2.

C2 measurements (A2, debug build). Method: e = i = 10⁻³, 300 orbits, 150 steps per orbit, rates from a least-squares fit to the pericentre and ascending-node longitudes. Error = measured / PLAN 7.2 − 1.

| r (km) | WH ϖ̇ error | WH Ω̇ error | Leapfrog ϖ̇ error | Leapfrog Ω̇ error |
|---|---|---|---|---|
| 100000 | −2.6 × 10⁻⁵ | −7.4 × 10⁻⁶ | −4.96 × 10⁻² | −1.39 × 10⁻³ |
| 140222.43 | −1.7 × 10⁻⁵ | −7.0 × 10⁻⁶ | −9.71 × 10⁻² | −1.35 × 10⁻³ |
| 180000 | −1.3 × 10⁻⁵ | −7.2 × 10⁻⁶ | −1.60 × 10⁻¹ | −1.34 × 10⁻³ |

The leapfrog value at a₀ agrees with the script above (−9.71 %). Other C2 numbers:

- Kepler drift, one period: position 9.5 × 10⁻¹⁶, velocity 6.4 × 10⁻¹⁶ (relative). Against a 20000-step leapfrog: 4.8 × 10⁻¹². f32 against f64 (one step of 1/150 period): position 4.1 × 10⁻⁸, velocity 7.0 × 10⁻⁹.
- WH energy, circular start at a₀, 150 steps per orbit, 10⁴ orbits: max |ΔE/E₀| = 8.3 × 10⁻¹³, drift 2.3 × 10⁻¹³. With e = i = 0.01, 10³ orbits: 7.0 × 10⁻⁹, drift 8.3 × 10⁻¹³.
- WH one orbit in f32 against f64: 0.20 km.
- Debug run time: precession tests 0.06 s, WH energy tests 0.8 s.

## Amendment 1: drift about an effective GM (ICR-011, 2026-11-09)

Accepted by AJ on 2026-11-09. The Kepler drift is about μ_eff = n(a₀)² a₀³ = GM (1 + 3/2 J2 q² − 15/8 J4 q⁴ + 35/16 J6 q⁶), q = R/a₀, with a₀ = `[frame] reference_semi_major_axis_km`. The kick adds (μ_eff − GM) r/|r|³, so the total force does not change. At a₀ = 140,220 km, μ_eff/GM − 1 = 4.5848 × 10⁻³.

Reason: a circular orbit of the J2 field is a Kepler ellipse with e of about μ_eff/GM − 1 in the drift about GM. The split error is then secular in the mean longitude: −12.8 km per year along-track at h = 356.25 s (E1-DIAG, ICR-011). With μ_eff the error over one year at h is −0.072, +0.0001 and +0.109 km at r = 139,500, 140,220 and 141,500 km (test T1, `crates/fring-cpu/tests/wh_eff.rs`). The E1 layer 2 count inside 1 R_H of Prometheus went from 29 to 24 (IAS15: 24). The cost per step does not change.

`SaturnField::from_params` keeps the drift about GM (old behaviour, bit for bit); `with_drift_radius(a0)` selects μ_eff. The precession numbers above (C2) were measured with the drift about GM and stay valid for that split.
