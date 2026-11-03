# Phase 4 planning: proposal

Written 2026-11-03 by A0. Status: **accepted**. AJ (2026-11-03, with the gate G3 approval): "yes to your recommendations", so all six decisions of section 5 are "yes". A0 copied the plan into `docs/PLAN.md` section 9 and `docs/status/BOARD.md`; those files are now the reference, and this file is the record of the reasons. One addition while copying: decision 2 needs a file in `params/`, which A1 owns, so the board has a small A1 row L3-PARAMS, and A4-ANALYSIS is blocked by it.

## Summary

- Keep the nine phase 4 tasks of `PLAN.md` section 9.
- Add three tasks: A4-ANALYSIS (the L3 deciding run), A5-ANALYSIS (the A2-ANALYSIS follow-ups) and B2 (the benchmark rerun with collisions).
- Split E1 into four tasks and K5 into two, so that each task has one role.
- Put the rows in an order that gives a good sequence for one session at a time.

## 1. New tasks

### A4-ANALYSIS (A8): the L3 deciding run

Source: `docs/literature/l3-moon-parameters-gr03.md`, section 6.

1. 100 runs of 100 years with the GR03b masses: GM(Prometheus) = 0.02200 and GM(Pandora) = 0.01301 km³ s⁻². These are our values (0.01071 and 0.00926) times 2.054 and 1.405. All else as in A1-ANALYSIS: seed 20261105, 2 km kick, tangent method, renormalisation every 0.25 yr, rate over years 30 to 100.
2. The same 100 runs with our masses, with L(t) written for each run. `docs/results/data/lyapunov_pairs.csv` keeps only the window slopes of each run, so the GR03 method (rate over years 0 to 100, shear subtracted with the (log t)/t curve) needs this new output. Then the mean over all 100 runs, not only the chaotic group.
3. Optional: one run of 3000 years with our masses, to compare with the single GR03 trajectory.

*Done when:* a note in `docs/results/` gives, for both mass sets, the chaotic-group rate (with the error of the mean), the chaotic fraction (26 % now) and the mean over all runs, and applies the L3 rule: a chaotic-group rate of 0.30 to 0.35 per year means that the masses explain the factor of 1.4; a rate near 0.22 means that they do not.

*Cost:* A1-ANALYSIS ran 100 runs of 100 years in 11 min on 8 threads. Items 1 and 2: about 22 min. Item 3: 10⁴ run-years took 8 × 11 = 88 thread-minutes, which is 0.53 thread-seconds per run-year, so 3000 years take about 26 min on one thread (if the larger masses do not change the IAS15 step much).

*Needs:* the GR03b masses in `params/` (decision 2). *Blocked by:* Gate G3. Does not block Gate G4.

### A5-ANALYSIS (A8): the A2-ANALYSIS follow-ups

Source: `docs/handoffs/A2-ANALYSIS.md`, "Open questions".

- (a) A step test with h/4 for `eccentric_band` near the two ends of the quiet zone. Eccentric orbits have b_crit 6.3 R_H, circular orbits 4.1 R_H, so the circular result ("no sub-stepping needed") does not cover them. The rule is the same as in A2-ANALYSIS: a change of more than 3 sigma of the noise floor means that sub-stepping is needed.
- (b) `closest` on `eccentric_band`: does the higher chaos of the eccentric orbits come from close encounters at the orbit ends, or from eccentricity resonances?
- (c) A second apse phase (+120°) at 30 years. Only one phase was run at 30 years.

*Done when:* an addendum to `docs/results/a2-analysis.md` gives the three answers with numbers.

*Cost:* (a) about 5 min of GPU (A2 estimate). (b) About 25 min of CPU (estimate from the Pandora check, G2 question 5). (c) 22 eccentric runs: 846 s of GPU in A2-ANALYSIS (`runs/logs/ecc-status-e30.csv`). The e = 0 control does not depend on the apse phase, so it is not run again. Total: about 20 min of GPU and 25 min of CPU.

*Blocked by:* Gate G3. Does not block Gate G4. Result (a) is an input to E1-ICR: if eccentric orbits need sub-stepping, the trigger in ICR-004 must reach 6.3 R_H.

### B2 (A5): benchmark rerun with collisions and sub-stepping

`docs/handoffs/B1.md` and G2 question 4 say: run `fring-bench` again when the collision kernels exist. Add rows with stochastic collisions on (K5) and with sub-stepping on (E1-GPU), for N = 10⁴ to 10⁷. Update the stretch target (10⁷ particles for 30 years: 8.1 h with D+ at G2, against a budget of 8 h).

*Done when:* the new table is in `docs/benchmarks.md`, and one line says whether a 30-year run matrix (S1) fits the budget of PLAN section 11. *Cost:* about 2.5 min per run. *Blocked by:* K5, E1-GPU. Blocks Gate G4 (decision 5).

## 2. Changed tasks

### E1: four tasks with one role each

E1 has two roles (A2 and A3) and five parts. A session takes one role per task (`CLAUDE.md`), and sessions stop without warning, so smaller tasks give cleaner handoffs.

| ID | Role | Task | Done when | Blocked by |
|---|---|---|---|---|
| E1-ICR | A2 | Write ICR-004 (the number is reserved; there is no file yet): flag and freeze on the CPU with the drop reasons of ICR-006; the Rust twin of the radius rule moves to `fring-model/src/physics/`; trigger and scheme of per-particle sub-stepping (block time steps, PLAN 7.3) for both backends; the set of particles and the statistics for the REBOUND comparison; whether `fring-run` uses the radius rule | A0 writes "Accepted", after AJ's decision (ICR-004 changes what `Backend::step` does when a particle is lost) | Gate G3, A5-ANALYSIS |
| E1-REF | A6 | REBOUND (IAS15) reference for particles that pass within 1 R_H of Prometheus, from the spec only | Reference files and comparison script in `validation/` | E1-ICR |
| E1-CPU | A2 | ICR-004 and sub-stepping in the CPU backend | The CPU statistics agree with E1-REF (PLAN section 10); all existing tests pass | E1-ICR |
| E1-GPU | A3 | ICR-004 and sub-stepping in WGSL, with Rust twins | GPU agrees with CPU (section 10) and with E1-REF; with sub-stepping off, `GpuBackend` results are bit for bit unchanged; `fring-bench` shows no slow-down | E1-CPU |

The last part of E1 ("a 30-year chaos-map run over 139,700 to 141,100 km completes") is the same run as A3-ANALYSIS. It moves there. A3-ANALYSIS is then blocked by E1-GPU and blocks Gate G4. The Gate G4 line "close encounters handled (E1)" becomes "(E1-CPU, E1-GPU, A3-ANALYSIS)".

### K5: two tasks with one role each

The ownership splits by directory: binning is in `fring-gpu` (A3), the collision shader in `fring-gpu/shaders/collide/` (A4).

| ID | Role | Task | Done when | Blocked by |
|---|---|---|---|---|
| K5-BIN | A3 | GPU binning: counting sort by cell at each step, with a Rust twin | The GPU sort agrees with a CPU sort (random cases and edge cases); `fring-bench` shows its cost | K1 |
| K5 | A4 | Stochastic collisions in WGSL | As in PLAN: GPU and CPU agree statistically on the K1 tests | K2, K5-BIN |

K4 keeps two roles: A4 does the runs and writes the report; A6 reviews the report without the A4 code.

### K1: AJ confirms the specification

The collision specification is a shared interface (`AGENTS.md`). K1 is done when A0 approves and AJ confirms. K1 runs on Opus (`CLAUDE.md`).

## 3. Row order

A session picks the first `todo` row whose blockers are `done`. While a specification waits for review, the next sessions take the rows below it.

| # | Row | Role | Why here |
|---|---|---|---|
| 1 | K1 | A4 | Starts the longest chain (K1, K2, K5, B2); the review by A0 and AJ takes time |
| 2 | A5-ANALYSIS | A8 | About 45 min of runs; result (a) is an input to ICR-004 |
| 3 | E1-ICR | A2 | Starts the second chain; also needs AJ |
| 4 | A4-ANALYSIS | A8 | About 22 min of runs; closes a question open since G2 |
| 5 | K2 | A4 | |
| 6 | E1-REF | A6 | One subagent, independent of the code |
| 7 | E1-CPU | A2 | |
| 8 | K3 | A4 | |
| 9 | D1 | A3 | |
| 10 | K5-BIN | A3 | |
| 11 | E1-GPU | A3 | |
| 12 | K5 | A4 | |
| 13 | K4 | A4 + A6 | |
| 14 | D2 | A6 | |
| 15 | A3-ANALYSIS | A8 | |
| 16 | B2 | A5 | |
| 17 | Gate G4 | AJ | Blocked by K1 to K5, K5-BIN, D1, D2, E1-CPU, E1-GPU, A3-ANALYSIS, B2 |

## 4. Not in this proposal

- No change to the nominal values in `params/`. If A4-ANALYSIS points to a mass change, that is a separate AJ decision.
- No collisions in the web app. Phase 6 (W3) at the earliest.
- The step review (150 against 300 steps per orbit) stays at S1, as G2 decided.

## 5. Decisions for AJ

1. Add A4-ANALYSIS, A5-ANALYSIS and B2? **A0 recommends yes.**
2. The GR03b masses: A1 adds them to `params/` as a variant file (for example `params/variants/gr03b-moons.toml`, cited in `params/sources.md`), and `fring-analyse lyapunov` gets an optional argument to load it. The nominal values do not change. The other option, a mass factor on the command line, breaks the rule "constants only from `params/`". **A0 recommends the variant file.**
3. Split E1 into E1-ICR, E1-REF, E1-CPU and E1-GPU, and move "the 30-year run completes" into A3-ANALYSIS, which then blocks Gate G4? **A0 recommends yes.**
4. Split K5 into K5-BIN (A3) and K5 (A4)? **A0 recommends yes.**
5. B2 blocks Gate G4, so that AJ sees the cost of collisions before phase 5? **A0 recommends yes.**
6. The row order of section 3? **A0 recommends yes.**
