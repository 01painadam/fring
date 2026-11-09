# 0009: G3 chaos rule over four seeds; mode D reported only in the arc test

Status: accepted (A0, 2026-11-09)
Date: 2026-11-09
Author: A3
Decided by AJ 2026-11-09; A0 approves.

## Context

ICR-011 moved the Wisdom-Holman drift to mu_eff (WH-EFF-GPU). Two ignored GPU tests then failed, with the GPU and the CPU both about mu_eff. The investigation (`docs/handoffs/WH-EFF-GPU.md`) found no split mismatch: every kernel and twin uses the same drift and kick words; with no moons, 100 particles and one year, the FTLE of GPU and CPU agree to 4e-6 per day (D+) and 2.5e-6 (C) on a value of 2.545e-2.

**1. `chaos_done_when` (G3, ADR 0008).** Rule: for chaotic-on-both particles, each FTLE quantile difference (10, 50, 90 %) of GPU against CPU is at most 2 times the same difference of one baseline run (CPU against CPU displaced by 1e-4 km). Rule: for regular-on-both particles, max |dL| <= 0.3. Measured at the 90 % quantile (the rule fails on the old tree too: seed 101, quantile 0, D+ 0.0032 and C 0.0013 against 0.0006):

| seed | tree | baseline q90 | D+ q90 | C q90 |
|---|---|---|---|---|
| 20261102 | GM drift (3ebacb9) | 0.026 | 0.013 | 0.019 |
| 101 | GM drift | 0.019 | 0.006 | 0.027 |
| 202 | GM drift | 0.022 | 0.007 | 0.004 |
| 303 | GM drift | 0.005 | 0.001 | 0.008 |
| 20261102 | mu_eff drift | 0.018 | 0.058 | 0.032 |
| 101 | mu_eff drift | 0.027 | 0.019 | 0.005 |
| 202 | mu_eff drift | 0.004 | 0.007 | 0.016 |
| 303 | mu_eff drift | 0.035 | 0.014 | 0.012 |

The baseline q90 alone ranges from 0.004 to 0.035, so a limit of 2 x one baseline run is a noisy limit. The max |dL| of 0.341 in mode C (seed 20261102) is one particle, no. 145, a weak regular orbit (L = 1.04). The CPU baseline moves it by 2.42 under a start offset of 1e-4 km, so the GPU is 7 times closer to the CPU than the baseline is.

**2. `gpu_arc`, mode D, 5 degree arc.** Rule: position difference of the arc run against the rotated full-ring run is at most 4 times the rounding floor. Ratio arc / floor over four ensembles:

| ensemble seed | GM drift | mu_eff drift |
|---|---|---|
| 0 | 2.5 | 4.6 |
| 1 | 3.3 | 3.6 |
| 2 | 2.9 | 3.0 |
| 3 | 3.5 | 4.0 |

In absolute terms mode D is better with mu_eff (arc against ring 4.2e-3 km, was 5.5e-3 to 7.5e-3 km). The floor fell by a factor of 1.5 to 2.4 because the f32 radial factor of the kick is small near a0 (the J2 and epsilon terms cancel), so the ratio rose without a worse result. Mode D+ is unchanged and passes.

## Decision

1. `chaos_done_when` runs four seeds (20261102, 101, 202, 303; `FRING_G3_SEED` runs one). For each seed and for D+ and C:
   - the "same class" rules are unchanged (agreement >= 0.97, and not more than 1 point below the baseline);
   - the 95 % |dL| limit 0.1 and the absolute FTLE limit 0.15 per quantile are unchanged;
   - the max |dL| rule is replaced by a per-particle rule: for every particle regular on both, GPU |dL| <= max(0.3, 2 x the |dL| of the same particle in the CPU baseline run).
   
   Across the seeds, for each FTLE quantile (10, 50, 90 %): the median over the seeds of the GPU difference is at most 2 x the median over the seeds of the baseline difference. The test stays `#[ignore]` (about 2 minutes).
2. `gpu_arc`: mode D is reported only (the ratio is printed, no assertion). D+ and C keep the bound of 4.

## Result on the merged head

A0 re-ran both tests on b09399b (WH-EFF-GPU merged with main 9f134b1, which includes GOLDEN-REL; `runs/logs/a0-review-wh-eff-gpu.log`). GOLDEN-REL changed release bits, so the per-seed values differ a little from the context table (seed 20261102 baseline q90: 0.025, not 0.018). Median over the four seeds:

| FTLE quantile | baseline | limit | D+ | C |
|---|---|---|---|---|
| 10 % | 0.0083 | 0.0165 | 0.0011 | 0.0009 |
| 50 % | 0.0070 | 0.0139 | 0.0016 | 0.0033 |
| 90 % | 0.0266 | 0.0532 | 0.0163 | 0.0137 |

Every seed: same class 100 %. `chaos_done_when` passes in 62 s. `gpu_arc` mode D (5 degree arc): position ratio 4.63, velocity 4.35 (printed only); mode dp 1.71 / 1.97 (5 degrees) and 2.69 / 2.93 (30 degrees), bound 4.

## Reasons

- The single-baseline rule fails on the old tree as well, so it measures the noise of one baseline run and not an error of the GPU.
- Four seeds and the median remove the dependence on one sample, and the per-particle rule removes the dependence on one sensitive orbit, without a looser bound for the physics.
- Mode D is not a science mode (ADR 0006 addendum). Its arc ratio is a ratio of two rounding noises.

## Consequences

The run takes about 2 minutes, not 1. No tolerance for D+ or C in `gpu_arc` changes. A0 approves this ADR in the pull request of WH-EFF-GPU.
