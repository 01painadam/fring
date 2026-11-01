# 0002: Energy test for the leapfrog (task C1)

Status: accepted by A0
Date: 2026-11-01
Author: A2  Approved by A0: 2026-11-01

## Context

`docs/PLAN.md` section 10 asks for a relative energy error below 10⁻¹⁰ for one particle on a circular orbit at a₀ over 10⁴ orbits, with no secular trend. With 150 steps per orbit and a start at the exact continuous circular speed v = sqrt(r |a_r|), the leapfrog energy oscillates with an amplitude of 1.985 × 10⁻⁷ (measured). That is bounded and does not drift, but it is 2000 times above 10⁻¹⁰. Reaching 10⁻¹⁰ by a smaller step would need about 6700 steps per orbit.

## Options

1. Weaken the tolerance. Not allowed without an ADR, and it hides nothing useful.
2. Use 6700 steps per orbit. About 45 times more work; slow in debug.
3. Start on the leapfrog's own circular orbit (E1), and keep the physical start as a second test (E2) with its own limits.

## Decision

Option 3 (A0 decision).

- **E1:** the drift-kick-drift map commutes with rotations about z. So there is a speed v_h at a₀ for which one step maps (a₀, 0, 0), (0, v_h, 0) to the same state turned by an angle. `physics::leapfrog_circular_speed` finds v_h by bisection on |x'| = a₀. Then every step is a pure rotation. Limits: max |E − E₀| / |E₀| < 10⁻¹⁰ at every step, and drift below 10⁻¹⁰.
- **E2:** start at v = sqrt(r |a_r|), 150 steps per orbit, 10⁴ orbits: max |ΔE/E₀| < 5 × 10⁻⁷, drift below 10⁻¹⁰. Also 300 steps per orbit for 10³ orbits.
- **Drift** is the least-squares slope, over the orbits, of the per-orbit maximum (and of the per-orbit minimum) of the signed error, times the span. Sampling only at the orbit ends does not work for E2: the orbit-end values follow the slow beat between the step phase and the epicycle phase, and the fitted slope times the span was 1.9 × 10⁻⁹ although the envelope is flat. The envelope slope is 2.6 × 10⁻¹³.

## Consequences

- C1 meets the PLAN criterion literally in E1 and shows the physical behaviour in E2.
- The expected h² scaling of E2 does not hold. The measured amplitude ratio for 150 and 300 steps per orbit is 15.97, so it scales as h⁴. Reason: the continuous circular start is off the leapfrog circular orbit by a relative amount δ ∝ h² (v_h − v_c = −1.82 × 10⁻³ km/s, relative −1.11 × 10⁻⁴), and the energy error is of second order in δ. The test checks the ratio 16 ± 20 %.
- Later tests that need a clean circular orbit (C2 precession) can start from `leapfrog_circular_speed` too.

## Evidence

`crates/fring-cpu/tests/energy.rs` (run `cargo test -p fring-cpu --test energy -- --nocapture`), debug build, total 0.3 s:

| Test | max abs(dE/E0) | drift |
|---|---|---|
| E1, 150 steps/orbit, 10⁴ orbits, start at v_h | 2.19 × 10⁻¹³ | 2.8 × 10⁻¹⁴ |
| E2, 150 steps/orbit, 10⁴ orbits, start at v_c | 1.985 × 10⁻⁷ | 2.6 × 10⁻¹³ |
| E2, 300 steps/orbit, 10³ orbits | 1.243 × 10⁻⁸ | |
