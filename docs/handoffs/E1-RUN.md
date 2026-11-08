# E1-RUN: run the CPU backend on the E1-REF files
Role: A5  Branch: ws/a5-e1-run  Status: review
Updated: 2026-11-08

## Goal and acceptance criteria
Rust runner for the E1-REF file contract (`docs/handoffs/E1-REF.md`): read the inputs, run the backend, write the WH result files, compare with `validation/scripts/e1_compare.py`. ICR-004 amendment 2, part 3: the limits apply to the run without sub-stepping. Closes the E1-CPU acceptance "the CPU statistics agree with E1-REF".

## Done
- `apps/fring-run/src/e1.rs` (library, tests) and `apps/fring-run/src/bin/fring-e1.rs`. `layer1<B: Backend>` and `layer2<B: Backend>` take a closure that makes the backend; only the CPU is wired (`cpu_backend`). A3 adds a GPU arm for E1-GPU G5.
- CSV readers and writers are in `e1.rs` (a validation protocol, not the run format).
- Tests: `cargo test -p fring-run e1` (4 unit tests, 4 integration tests in `apps/fring-run/tests/e1.rs`).
- No change in `fring-model`, `fring-cpu`, the config schema or `validation/`.

## Design
- One moon ephemeris from the epoch, covering 352 d (layer 1) or 365.25 d (layer 2). Layer 1: one backend per `t0_idx` (`InitialState.time_s = t0_s`). Particles are independent, so grouping changes nothing (test: one id alone equals the same id in its group, bit for bit).
- **Exact end time (A0 accepted).** Layer 1: 485 steps of 172800/485 = 356.2887 s (nominal 356.2537 s gives 485.05 steps; 486 steps would end 340 s late). Layer 2: 88,582 steps of 31,557,600/88,582 = 356.2527 s (a 365.25 d run with 356.2537 s would end 63 s late, about 1000 km along the orbit). The config duration is (n - 0.5) steps so that `ceil` gives exactly n (asserted). The WH error scales as h^2, so these changes are 2e-4 and 6e-6 relative.
- Variants. `plain` (file for `--run`): `[loss] on_unbound = "freeze"`, no radius rule, no `[substep]`. Bound particles are bit for bit as without `[loss]` (E1-CPU test C2); an unbound particle is frozen and listed. `substep`: `plain` plus `[substep]` accuracy 0.1, max_level 4 (`--substep`, reported only). `loss`: `plain` plus `moon_radius = "params"` (information only). IAS15 has no radius rule, so `plain` is the like-for-like test.
- Layer 2 closest distance: after every step, a cubic Hermite of the relative position (particle minus Prometheus, with velocities) between step ends; 32 samples and a parabola refine. The reference uses Hermite on 593.756 s samples; the stride study below shows the definition does not change the count.
- Layer 2 chaos indicator: `[diagnostics.chaos]` offset 1 km, renorm every 100 steps (ADR 0008); `megno_mean` from `Backend::chaos()`.
- Files: `runs/e1/<variant>/l1_wh.csv`, `l2_wh.csv`, `frozen_l1.csv`, `frozen_l2.csv`, `runs/e1/moons_diff.csv`. Rows are sorted by id. Runs are deterministic: two runs give identical files (cmp).

## Results (release, M-series Mac; logs in `runs/logs/`, git-ignored)
Moon positions against `e1_moons_t0.csv` at all 36 epochs (`runs/e1/moons_diff.csv`, `runs/logs/e1-run.log`): max |dr| 2.0e-6 km, max |dv| 2.4e-10 km/s (both moons). Far below 0.1 km: the effect on b (at least 43 km) is below 5e-8 relative. No action.

Wall time (`runs/logs/e1-run.log`): ephemeris under 1 s; layer 1 0.1 s (plain, substep), 0.2 s (loss); layer 2 11.9 s (plain), 29.4 s (substep), 30.8 s (loss).

Frozen or dropped (`frozen_l*.csv`): `plain` 0 and `substep` 0 in both layers; `loss` 0 in layer 1 and 20 in layer 2, all reason 2 (inside Prometheus, ids 67, 91, 101, 132, 145, 151, 176, 184, 185, 187, 189, 191, 195, 210, 218, 237, 248, 263, 287, 299). The radius rule changes layer 2: chaotic fraction 0.277 against 0.337 (still within 3 sigma = 0.082). Layer 1 stratum 0 starts at 43.04 km, below the radius 43.09 km, but no layer 1 particle was dropped.

### Layer 1 (`runs/logs/e1-compare-l1.log`, exit 0)
|Delta a_WH - Delta a_IAS15| / |Delta a_IAS15|, `plain` (limits for strata 1 to 3: median < 5 %, p95 < 25 %):
| stratum | median | p95 | max |
|---|---|---|---|
| [0.68,1) (reported) | 1.25e-3 | 1.00e-2 | 0.41 |
| [1,2) | 9.0e-4 | 3.2e-3 | 0.24 |
| [2,4.1) | 5.5e-4 | 1.9e-3 | 0.058 |
| [4.1,6.3] | 3.7e-4 | 2.8e-3 | 0.054 |

All limits pass by a factor 15 to 130. `substep` (reported): medians 6.4e-4, 4.2e-4, 3.3e-4, 3.4e-4; p95 6.0e-3, 3.9e-3, 1.9e-3, 2.8e-3.

### Layer 2 (`runs/logs/e1-compare-l2.log`, exit 1: **FAILED by 0.1**)
| | IAS15 | plain | substep (no limit) | loss (no limit) |
|---|---|---|---|---|
| inside 1 R_H | 24 | **29** (limit +-4.9, FAIL) | 28 | 29 |
| inside 3 R_H | 65 | 70 (+-8.1) | 70 | 70 |
| inside 6.3 R_H | 153 | 158 (+-12.4) | 158 | 158 |
| KS of Delta a | | D 0.037, p 0.99 | D 0.040, p 0.97 | D 0.043, p 0.94 |
| chaotic fraction (3 sigma 0.082) | 0.337 | 0.320 | 0.320 | 0.277 |

The only failed limit is the count inside 1 R_H: WH 29 against 24, a difference of 5 against a limit of max(1, sqrt(24)) = 4.9. I did not change the limit. What I checked (logs `runs/logs/e1-l2-noise.log`, `e1-l2-div4.log`, `e1-l2-div16.log`):
- Individual trajectories differ (chaos): of 24 and 29 particles inside 1 R_H, 17 are in both sets. The 20-day check against `e1_l2_ckpt20d.csv` agrees (closest distances equal to 4 digits for 8 of 10 ids, 10 km apart for the other 2; the final states differ by 2800 km only because the reference file ends at the last 593.756 s sample, 170 s before 20 d).
- The definition of the closest distance is not the cause: measuring from every step, every 2nd or every 3rd step gives 29, 29, 29.
- Step size: nominal steps of 340, 350, 360, 370 s give 29, 28, 31, 29 inside 1 R_H (3 R_H: 66, 67, 70, 68; 6.3 R_H: 158, 158, 159, 159). So the excess of 4 to 7 is stable at the production step, not a lucky draw of one step size.
- It converges with h: h/4 gives 26, 65, 153 (chaotic fraction 0.337); h/16 gives 26, 65, 153 (0.330). Both pass. Time: 40 s and 113 s.
- The REBOUND WHFast stand-in at h gave 23, 63, 154 (E1-REF handoff), so the Rust CPU WH at h differs from the stand-in by +6 inside 1 R_H. Possible cause (not tested): the Rust code kicks with the exact tabulated moons, while WHFast evolves the moons with the same step. A0 and AJ decide: the limit is AJ's.

### Paired realisation study of the layer 2 count (A0 follow-up)
Option `--l2-perturb-km x --l2-seed k`: a random offset of length x km (isotropic; Philox stream `ANALYSIS`, counter index = particle id, key = seed) is added to each layer 2 start position. With x = 0 nothing changes: the output file is bit for bit equal to the run without the option (`cmp`), and a unit test checks it. Study: seeds 1 to 10, x = 1e-6 km, three settings on the same perturbed states: `plain` at h (88,582 steps), `substep` at h, `plain` at h/4 (`--l2-step 89.0634 --l2-stride 4`). 30 runs, about 15 min. Scripts: `apps/fring-run/scripts/e1_realisations.sh` and `e1_realisations_analyse.py`; logs `runs/logs/e1-l2-realisations.log` and `e1-l2-realisations-summary.log`.

Counts inside 1 / 3 / 6.3 R_H and chaotic fraction:
| seed | plain h | substep h | plain h/4 |
|---|---|---|---|
| 1 | 29 70 158, 0.320 | 29 70 158, 0.320 | 26 65 153, 0.337 |
| 2 | 29 70 158, 0.320 | 29 70 158, 0.320 | 26 65 153, 0.337 |
| 3 | 29 70 158, 0.320 | 29 70 158, 0.320 | 25 65 153, 0.337 |
| 4 | 29 70 158, 0.320 | 29 70 158, 0.320 | 25 65 153, 0.337 |
| 5 | 29 70 158, 0.320 | 29 70 158, 0.320 | 25 65 153, 0.337 |
| 6 | 29 70 158, 0.317 | 29 70 158, 0.320 | 26 65 153, 0.337 |
| 7 | 29 70 158, 0.317 | 29 70 158, 0.320 | 25 65 153, 0.337 |
| 8 | 29 70 158, 0.320 | 28 70 158, 0.320 | 26 65 153, 0.337 |
| 9 | 29 70 158, 0.317 | 28 70 158, 0.320 | 26 65 153, 0.337 |
| 10 | 29 70 158, 0.317 | 28 70 158, 0.320 | 26 65 153, 0.337 |

Mean and standard error of the mean over 10 seeds (inside 1 R_H; 3 R_H and 6.3 R_H have zero spread):
| setting | 1 R_H | 3 R_H | 6.3 R_H | chaotic fraction |
|---|---|---|---|---|
| plain h | 29.00 +- 0.00 | 70 | 158 | 0.319 +- 0.001 |
| substep h | 28.70 +- 0.15 | 70 | 158 | 0.320 |
| plain h/4 | 25.60 +- 0.16 | 65 | 153 | 0.337 |

Paired differences (mean +- standard error of the mean difference): plain h minus h/4: 3.40 +- 0.16 (1 R_H), 5 +- 0 (3 R_H), 5 +- 0 (6.3 R_H), -0.018 +- 0.001 (chaotic fraction). Substep minus h/4: 3.10 +- 0.28, 5, 5, -0.017.

Particles inside 1 R_H in all 10 realisations: plain h 29 (of 29 in any), substep 28 (of 29 in any), h/4 25 (of 26 in any).

**Reading.** A start offset of 1e-6 km does not change the set of particles inside 1 R_H: 25 to 29 of the 29 are the same in every realisation. So on this ensemble the counts are almost deterministic. The chaotic part is small at this offset (the Lyapunov rate of the chaotic particles is about 0.22 per year, so 1e-6 km grows by a factor of a few in one year, not enough to change an encounter). The h against h/4 difference is therefore a step-size effect on the same orbits and not noise: +3.4 +- 0.16 inside 1 R_H, +5 inside 3 and 6.3 R_H. The stated error is small because the realisations are nearly identical, so it measures the spread of this offset only. It does not measure how a different, independent sample of 300 particles would count. h/4 itself gives 25.6 against IAS15 24. The sub-stepped run removes only 0.3 of the 3.4.
Open: a larger offset (for example 1 km, which is the MEGNO shadow size) would decorrelate the chaotic orbits and give the true realisation spread. About 7 min for h and h/4 with 10 seeds.

## Next step (one concrete action)
A0: decide on the layer 2 count inside 1 R_H (WH 29 against IAS15 24, limit +-4.9; passes with h/4 and with sub-stepping at 28). Then merge. A3 (E1-GPU G5): call `fring_run::e1::layer1/layer2` with a GPU backend maker, compare the same way.

## How to verify the current state
```sh
cargo build --release -p fring-run --bin fring-e1
target/release/fring-e1 all --variants plain,substep,loss --out runs/e1      # about 75 s; prints the moon difference
cd validation && uv run python scripts/e1_compare.py layer1 --run ../runs/e1/plain/l1_wh.csv --substep ../runs/e1/substep/l1_wh.csv --json ../runs/e1/layer1.json   # exit 0
cd validation && uv run python scripts/e1_compare.py layer2 --run ../runs/e1/plain/l2_wh.csv --substep ../runs/e1/substep/l2_wh.csv --json ../runs/e1/layer2.json   # exit 1: 29 against 24 inside 1 R_H
cargo test -p fring-run e1
```
Noise studies: `--l2-step <s>` (nominal step; adjusted for an exact end), `--l2-stride <k>` (closest distance from every k-th step), for example `--l2-step 89.0634 --l2-stride 4` is h/4.

## Open questions / blockers
- The layer 2 count (above).
- Layer 2 reference final states are at the last 593.756 s sample before 365.25 d (up to 656 s early); ours are exactly at 365.25 d. Delta a is not affected (a is conserved away from encounters); the apse and phase are not used for layer 2.

## Decisions made (link ADRs)
- A0 2026-11-08: exact-end step sizes; `plain` = freeze without radius rule; CSV code in `e1.rs`; no `--compare` flag. No ADR.
