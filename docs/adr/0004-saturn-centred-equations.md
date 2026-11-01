# 0004: Saturn-centred equations of motion with indirect terms

Status: accepted
Date: 2026-11-01
Author: A0  Approved by A0: 2026-11-01 (AJ approved at gate G1, 2026-11-01)

## Context

All states in `fring` are Saturn-centred (`params/moons.toml` comes from JPL Horizons with centre 500@699; `docs/PLAN.md` section 4.2). A Saturn-centred frame is not inertial: the moons accelerate Saturn. PLAN section 7.1 gave only the direct moon term for particles. ADR 0003 left this open for A0 to decide before the moon and particle tasks. A6 needs the decision now, for the REBOUND references V2 and V3.

## Decision

The equations of motion in the Saturn-centred frame (G·M as GM; J terms from PLAN 7.1 with J2, J4, J6; a_J is the J part of `SaturnField`):

- Moon i:
  r̈ᵢ = −(GM + GMᵢ) rᵢ/|rᵢ|³ + a_J(rᵢ) + Σ_{j≠i} GMⱼ [ (rⱼ − rᵢ)/|rⱼ − rᵢ|³ − rⱼ/|rⱼ|³ ]
- Massless particle:
  r̈ = −GM r/|r|³ + a_J(r) + Σⱼ GMⱼ [ (rⱼ − r)/|rⱼ − r|³ − rⱼ/|rⱼ|³ ]

The last term in each bracket is the indirect term. It is the same for all particles at one time, so it costs one vector per step.

Neglected, with reason:
- The reaction of Saturn to the J forces of the moons. It is of relative order GMᵢ/GM ≈ 3 × 10⁻¹⁰.
- Particle masses (particles are test particles; PLAN section 2).

## Size of the effect

The indirect acceleration from Prometheus is GMᵢ/|rᵢ|² = 0.0107/139,400² ≈ 5.5 × 10⁻¹³ km s⁻². For comparison, the direct pull of Prometheus on a core particle 840 km away is about 1.5 × 10⁻⁸ km s⁻².

The indirect term is a uniform field that rotates at Prometheus' mean motion. It forces the epicycle of a core particle near resonance (κ_F − n_P ≈ 8 °/d), with an amplitude of about 5.5 × 10⁻¹³ / (2 n Δ) ≈ 10⁻³ km. So it is small, but it is the correct physics, and it costs almost nothing. Leaving it out would also make the Rust model differ from any N-body reference.

## Reference model for A6 (V2, V3)

REBOUND with Saturn as a free massive body. The J2, J4, J6 force acts on moons and particles relative to Saturn's position. Outputs are Saturn-centred: subtract Saturn's state. Within the neglected terms above, this is the same dynamical system as the equations above.

## Consequences

- PLAN section 7.1 is updated with these equations.
- C3 (moon integrator) and C4 (particles with moons) implement them, including the indirect terms and the (GM + GMᵢ) two-body term for moons.
- The particle integrator (ADR 0003) keeps the Kepler drift with GM; the moon and indirect terms go in the kick.
- To reverse: remove the indirect sums. Reference runs would then need Saturn fixed at the origin.

## Evidence

- Orders of magnitude above. V2 and V3 provide the numerical reference.
