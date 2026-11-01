# 0005: Test definition for CPU against REBOUND (task C4)

Status: accepted
Date: 2026-11-01
Author: A2  Approved by A0: 2026-11-01

## Context

`docs/PLAN.md` section 10 asks for a position difference below 1 km between the CPU backend and REBOUND for non-chaotic test particles over 100 orbits. Task C4 is done when the CPU backend meets this for the 100 particles of V3. The PLAN gives no step size.

The Wisdom-Holman integrator (ADR 0003) has a second-order error in the splitting of the Kepler drift and the J kick. It does not change the orbit shape noticeably, but it shifts the mean motion by a relative amount of about 7 × 10⁻⁸ at 150 steps per orbit (measured below). Over 100 orbits that is a position difference along the orbit of several kilometres. A0 measured −2.2 km at 150 steps per orbit for a circular start. This is physically negligible: Δa is about 0.006 km. But a strict 1 km test at 150 steps per orbit cannot pass, and the error scales as h².

The V3 initial states must be read bit-exactly from `validation/reference/v3_ic.csv`: 1 m in a gives 1.8 km after 100 orbits.

## Options

1. Run the strict test at 150 steps per orbit with a 6 km limit. Changes the tolerance. Not allowed.
2. Run the strict test (< 1 km) at a smaller step, and test the production step against a bound derived from measurement. Same pattern as ADR 0002.
3. Run only at 1500 steps per orbit. Slow in debug, and the production step is not tested.

## Decision

Option 2.

- **Strict PLAN section 10 test:** 600 steps per orbit (h = T0/600 = 89.1 s), 100 particles, 100 orbits, position difference below 1 km at every full orbit. This runs in CI (`crates/fring-cpu/tests/v3.rs`).
- **Production step:** 150 steps per orbit runs in CI with the bound 12 km (about 2 times the measured 5.81 km). The bound only guards against the error growing; the h² scaling below explains its size.
- **Small step:** 1500 steps per orbit, bound 0.1 km, `#[ignore]` (long). A bug in the moon terms would show here as an error of kilometres.
- The step is T0/N with T0 = 53445.914 s from the reference file, so the output grid T0/10 holds N/10 whole steps.

## Consequences

- C4 meets PLAN section 10 literally at 600 steps per orbit, and the report states that the production step is 16 times worse, from the h² error of the splitting.
- Moon encounters are not the limit: the smallest b / (v_rel h), with b the distance to a moon, is 19.9 at 150 and 79.6 at 600 steps per orbit (the closest approach in V3 is 459 km).
- If a later task changes the production step, the 150 test and its bound change with it (new ADR or an edit of this one, approved by A0).

## Evidence

Maximum position difference over all particles and the 101 full orbits (`cargo test -p fring-cpu --test v3 -- --nocapture`, and `--release -- --ignored` for 1500). The worst particle is id 9 at orbit 100 in all three runs.

| Steps per orbit | max abs(dr) (km) | at 50 orbits (km) | max abs(dv) (km/s) |
|---|---|---|---|
| 150 | 5.810 | 1.230 | 6.87 × 10⁻⁴ |
| 600 | 0.3633 | 0.0768 | 4.29 × 10⁻⁵ |
| 1500 | 0.0579 | 0.01224 | 6.84 × 10⁻⁶ |

The ratio 150 to 600 is 15.99, and 600 to 1500 is 6.27 (h² gives 16 and 6.25). So the whole difference is the h² splitting error. 5.81 km after 100 orbits is a mean-motion error of 6.6 × 10⁻⁸ (Δa about 0.006 km). `validation/scripts/v3_compare.py` gives the same 3.633 × 10⁻¹ km for the 600 run.

The moons of the Rust ephemeris (IAS15, interval 1800 s) against `v3_moons.csv`, 1001 times: max 3.07 × 10⁻⁴ km.
