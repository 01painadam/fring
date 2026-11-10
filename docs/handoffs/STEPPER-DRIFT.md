# STEPPER-DRIFT: explain the along-track error at a0 through GpuStepper
Role: A3  Branch: ws/a3-stepper-drift  Status: review
Updated: 2026-11-10

## Goal and acceptance criteria
Explain the +0.009 to +0.010 km per year along-track error at a0 = 140,222.43 km through `GpuStepper` (WH-EFF-WEB finding, PR #86) against +0.0002 km in the T1 test (`gpu_wh_eff`, 140,220 km).

## Done
- E1 (T1 harness, one `apply`, zero table, D+ and C, 88,582 steps), along-track km:

  | drift radius | start r | D+ | C | CPU f64 |
  |---|---|---|---|---|
  | 140220 | 140220 | +0.0002 | +0.0013 | +0.0001 |
  | 140222.43 | 140220 | +0.0007 | -0.0013 | -0.0002 |
  | 140222.43 | 140222.43 | +0.0095 | -0.0006 | +0.0000 |
  | 140220 | 140222.43 | +0.0118 | +0.0008 | +0.0003 |

  The T1 harness reproduces the stepper value. The start radius carries it, not the drift radius, and not the stepper path (blocks, table fill, `kick_scale`). E3 was not needed.
- E2: 32 start radii from 140,220 km in steps of 0.00390625 km (a quarter of the f32 ulp, 0.015625 km), drift about mu_eff(140,220). D+ error swings between -0.011 and +0.021 km, quasi-periodically in r over about 0.07 km, not as a one-ulp sawtooth. C stays within 0.0022 km of the CPU. Maximum |GPU - CPU|: D+ 0.0207 km, C 0.0022 km. The CPU f64 value is below 0.0002 km at every radius.
- Tests added to `crates/fring-gpu/tests/gpu_wh_eff.rs` (`#[ignore]`, need a GPU): `e1_start_and_drift_radius`, `e2_scan_start_radius`. Bounds (A0, 2026-11-10): D+ 0.05 km, C 0.01 km. T1 bounds are unchanged.

## Explanation
- Cause: a bias from f32 arithmetic in the f32 part of `kick_dir_plus` (J4 and J6 terms and the f32 radius, `wh.wgsl`). The J2 term and the epsilon term are in DS. C uses DS throughout and is clean. Which f32 term dominates is not isolated.
- 0.0207 km per year at a0: delta n / n = 0.0207 / 140222 / 3701 rad = 4.0e-11. This equals a semi-major-axis offset of (2/3) a delta n / n = 3.7e-6 km (about 4 mm). Over 30 years: 0.62 km along track, against 2 to 3 km of split error (0.07 to 0.11 km per year at the band edge).
- It is a fixed bias per particle that depends on its start radius. It is not a random walk, because the same rounding repeats at every step.
- T1 passes only because 140,220 km is near a zero of the curve. The T1 GPU-against-CPU bound (0.02 km) holds at 140,220 km but not at every radius (0.0207 km at 140,220.059 km).
- Not the cause: block size, step, `kick_scale`, drift radius, table fill, reference n.

## Fix note (A0 and AJ decide; an ADR is needed if done)
Move the J4 and J6 terms of `kick_dir_plus` (P4, P6, q4, q6 and the radius ratio) to double-single, as the J2 term already is. The result would be close to option C for the kick. Cost not measured; the kick is the heavier part of the step, so expect a step time between D+ and C. Measure with `fring-bench` before a decision. The current bias is about 5 times below the split error, so no fix may be needed.

## Next step (one concrete action)
A0 reviews and opens the PR.

## How to verify the current state
`cargo test -p fring-gpu --release --test gpu_wh_eff -- --ignored --nocapture e1_ e2_` prints the E1 and E2 tables; E2 ends with `E2 max |GPU - CPU|: Dp 0.0207 km, C 0.0022 km`.

## Open questions / blockers
- Whether to fix (see above).

## Decisions made (link ADRs)
- Bounds D+ 0.05 km and C 0.01 km for the scan (A0, 2026-11-10).
