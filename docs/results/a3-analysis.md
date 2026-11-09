# A3-ANALYSIS: 30-year chaos map over the full band with flag and freeze and the radius rule

Role A8. Date 2026-11-09. This section was written before the final runs were read (commit order: this section, then the numbers).

## 0. Decision rule (fixed before the final numbers)

What I compare, and with what noise. All comparisons are of statistics, never of single trajectories.

- **Chaotic fraction of a radius window.** Two-proportion z test (`stats::two_proportion`, pooled sd). "Confirmed" if |z| < 3. "Replaced" if |z| >= 3; then the new value and its counting error are given.
- **Which fraction.** The survivors (live particles) and the bound (every lost particle counted as chaotic: a moon scattered or absorbed it). A1 kept every particle in the sample, hits included, so the bound is the like-for-like value for an inner-edge number. Both are reported. The verdict uses the bound for windows with losses, and the survivors where nothing was lost.
- **Lambda of a window.** lambda = 2 median(MEGNO) / years of the survivors. Noise: one sigma from a bootstrap of the median (200 resamples, 16 to 84 %), written to `lambda_sigma_per_yr`. "Confirmed" if the old and new values differ by less than 3 sigma of the new value; the old raw main10 run also gets its own sigma, and the two are added in quadrature. For main30 only the committed window values exist (no raw run), so only the new sigma is used.
- **Measured before I set any limit (rule M27).** Pilot `full10-freeze` against A1 main10 on the same 200,000 ids (`docs/handoffs/A3-ANALYSIS.md`): the chaotic fraction of every window differs by less than 0.003 and lambda by less than 1.0 % (largest: outer flank 2.103 to 2.124; the others 0.25 % or less: 7.97 to 7.99, 0.829 to 0.828, 0.864 to 0.862, 5.09 to 5.07; corrected after A0 review, it said 3 %). The limits above (3 sigma) are the usual ones of A2-ANALYSIS and are not tightened to this.
- **Hits.** Lost inside Prometheus among the first 20,000 ids at 10 years (`full10-rule`) against 911 of 20,000 = 4.6 % in A1 (`a1-analysis.md`, closest approach). Confirmed if the counting sd of the new count (sqrt of the count) covers the difference within 3 sigma. This is not the same quantity (A1: a segment distance below 43 km at CPU f64 with the old drift; here: the rule with the radius of `params/moons.toml` on the GPU), and I say so.
- **Not covered.** Encounters within 1 R_H of Prometheus (5.4 %) need the CPU closest-approach harness and are not rerun here (K4 holds the CPU). The hits above are the comparable number.
- **Differences between old and new runs.** main10 against `full10-freeze` differs in more than mu_eff: it also has DET-SINCOS (release initial states may differ by a few ulp, which does not change the statistics) and freeze (no effect unless a particle becomes unbound). `full10-freeze` against `full10-rule` isolates the radius rule. There is no raw main30: only `docs/results/data/main30_{hist,profile}.csv` and the window values in `a1-analysis.md`, so the 30-year comparison uses those values with binomial errors, and the lambda bootstrap of the old value exists for main10 only.

## 1. What was run

GPU (D+), WH-EFF-GPU included, `fring-run` from main d91d482 (RUN-LOSS). Seed 20261102, N = 200,000, band 139,700 to 141,100 km, step 356.2537 s, chaos indicator as in ADR 0008 (offset 1 km, K = 100), `[loss] on_unbound = "freeze"`, no sub-stepping. Configs: `docs/results/configs/chaos-map-full10-freeze.toml`, `-full10-rule.toml`, `-full30.toml` (the last two add `moon_radius = "params"`). Tables: `fring-analyse windows` (`data/a3_windows_*.csv`, `data/a3_full30_losses_by_year.csv`). The lost list comes from `drops.csv`; flag bit 1 of `chaos.csv` is checked against it (the sets are equal in all three runs, else the tool stops).

| Run | Rule | Wall time | Lost |
|---|---|---|---|
| pilot `full10-freeze` (before RUN-LOSS, K4 running) | off | 404.6 s | 2 (unbound) |
| `full10-freeze` | off | 395 s | 2 (unbound) |
| `full10-rule` | on | 416 s (+5.3 % on freeze) | 9,053 Prometheus, 734 Pandora, 0 unbound |
| `full30` | on | 1,237 s (20.6 min) | 19,177 Prometheus, 3,873 Pandora, 0 unbound |

- **The 30-year run completes.** The first try in A1 stopped at "unbound or invalid orbit". With the rule on, 23,050 of 200,000 particles (11.5 %) are frozen, and none of them as unbound. A1 main10 took 384 s.
- Estimate from the plan: 21 to 23 min (main30 rate 4.4e8 particle-steps/s and G6 cost 11 to 16 %). Measured: 20.6 min. The rule cost at N = 2e5 is 5.3 % on the 10-year run (416 s against 395 s), below the G6 range because the shadow runs the plain kernel. Three times the rule run is 1,248 s, within 1 % of the measured 1,237 s.
- **Determinism.** `chaos.csv` of the final `full10-freeze` and of the pilot are identical (`cmp`), so RUN-LOSS changed no result. The first 10 years of `full30` lose exactly the same particles as `full10-rule` (9,053 and 734).
- **Old outputs unchanged.** `fring-analyse map` on the raw main10 run gives `main10_hist.csv` and `main10_profile.csv` equal (`cmp`) to the committed files (arguments `139700 141100 7 10 2`).

## 2. Losses

Lost by window (`data/a3_windows_full30.csv`, `a3_windows_full10-rule.csv`):

| Window (km) | N | Prometheus 10 yr | Prometheus 30 yr | Pandora 10 yr | Pandora 30 yr |
|---|---|---|---|---|---|
| 139,700 to 139,950 | 35,575 | 9,053 (25.4 %) | 18,987 (53.4 %) | 0 | 0 |
| 139,950 to 140,150 | 28,609 | 0 | 190 (0.7 %) | 0 | 0 |
| 140,150 to 140,900 | 107,284 | 0 | 0 | 0 | 1 |
| 140,900 to 141,100 | 28,532 | 0 | 0 | 734 (2.6 %) | 3,872 (13.6 %) |

- Below 140,000 km: 9,053 of 42,688 (21.2 %) are in Prometheus after 10 years and 19,166 (44.9 %) after 30 years. Nothing is lost between 140,150 and 140,900 km. The core window loses no particle in 30 years.
- Losses per year (`data/a3_full30_losses_by_year.csv`). Prometheus: 979, 956, 848, 743, 444, 341, 309 in years 1, 5, 10, 15, 20, 25, 30 (the supply from the inner edge dries up). Pandora: 85, 65, 99, 138, 167, 177, 141 (it rises, then falls).
- Moon hits among the same sets as in A1 (counting sd = sqrt of the counts):

| Quantity | A1 / A2 | New (rule, 10 yr) | z |
|---|---|---|---|
| Prometheus, first 20,000 ids | 911 (4.6 %) | 879 (4.4 %) | -0.76 |
| Pandora, 140,850 to 141,100 km | 376 of 20,000 (1.9 %) | 734 of 35,783 (2.05 %) | +1.4 |

  Both are confirmed by the rule of section 0. They are not the same quantity (A1 counted a segment distance below the body radius with the old drift and CPU f64; here the GPU rule with the radius of `params/moons.toml`), and both agree to within 1.4 sigma. Over 30 years the first 20,000 ids lose 1,875 particles in Prometheus (9.4 %).
- **Not rerun:** the 5.4 % within 1 R_H and the 12.8 % within 3 R_H of A1 (CPU closest-approach harness; K4 holds the CPU).

## 3. Inner-edge numbers of A1-ANALYSIS: confirmed or replaced

Chaotic fraction (z of the two-proportion test, section 0; "bound" counts lost particles as chaotic). 10 years, A1 main10 against `full10-rule` on the same ids:

| Window (km) | A1 | Survivors | Bound | z (bound) | Verdict |
|---|---|---|---|---|---|
| 139,700 to 139,950 | 0.9789 | 0.9713 (26,522 live) | 0.9786 | -0.23 | confirmed |
| 139,950 to 140,150 | 0.7321 | 0.7311 | 0.7311 | -0.2 | confirmed |

30 years, A1 main30 (committed values, 139,950 to 141,050 km, different particles, N from `window_summary.csv`) against `full30`:

| Window (km) | A1 (N) | Survivors | Bound | z (rule) | Verdict |
|---|---|---|---|---|---|
| 139,950 to 140,150 inner flank | 0.911 (19,934) | 0.9131 | 0.9137 | +1.02 (bound) | confirmed |
| 140,197 to 140,247 core | 0.909 (4,778) | 0.9049 | 0.9049 | -0.76 | confirmed |
| 140,150 to 140,700 quiet zone | 0.885 (55,145) | 0.8849 | 0.8849 | -0.08 | confirmed |
| 140,700 to 140,900 outer flank | 0.875 (20,093) | 0.8789 | 0.8789 | +1.30 | confirmed |
| 140,900 to 141,050 outer edge | 0.900 (14,828) | 0.8899 | 0.8982 | -0.56 (bound); survivors alone: -3.01 | confirmed (see below) |
| 139,700 to 139,950 inner edge | not run in A1 | 0.9878 (16,588 live) | 0.9943 | | new: 53.4 % lost in Prometheus |

Lambda of the survivors (1/yr). Definition here: 2 median(MEGNO) / years over the survivors of the window (per-particle median). A1 values are rounded to 0.01 and have no error; sigma is the bootstrap sigma of the new value. The last three columns were **added after A0 review; the verdicts stay as written before the numbers.** sigma_comb = sigma x sqrt(1 + N_new / N_old), with the N_old of `window_summary.csv` (30 yr) because the old run also has a sampling error. "Selection bound" is the lambda of the survivors if every lost particle had a MEGNO above the median of the full window (`lambda_selection_bound_per_yr`): an upper limit of what the loss can do, not an estimate.

| Window | A1 | New | sigma | shift / sigma | Verdict by the rule | sigma_comb | shift / sigma_comb | Selection bound |
|---|---|---|---|---|---|---|---|---|
| 139,700 to 139,950, 10 yr | 8.4 | 7.41 (rule), 7.99 (freeze only) | 0.015 | | replaced, see below | | | 8.20 |
| 139,950 to 140,150, 30 yr | 2.23 | 2.172 | 0.010 | 5.6 | replaced (-2.6 %) | 0.016 | 3.6 | 2.184 |
| core, 30 yr | 0.88 | 0.878 | 0.006 | 0.3 | confirmed | 0.010 | 0.2 | 0.878 |
| quiet zone, 30 yr | 0.91 | 0.889 | 0.002 | 10.6 | replaced (-2.3 %) | 0.003 | 6.8 | 0.889 |
| outer flank, 30 yr | 2.15 | 2.132 | 0.008 | 2.4 | confirmed | 0.012 | 1.6 | 2.132 |
| 139,700 to 139,950, 30 yr | not run | 5.93 | 0.017 | | new | | | 11.26 |
| 140,900 to 141,050, 30 yr | 4.75 | 4.424 | 0.013 | 24 | replaced (-6.9 %) | 0.021 | 15.6 | 4.604 |
| 140,900 to 141,100, 30 yr (the full new window) | | 4.594 | 0.015 | | | | | 4.958 |

The window 140,900 to 141,050 km is the one of the A1 main30 run (its band ended at 141,050 km); the verdict row above already uses it. The full new window 140,900 to 141,100 km is shown for completeness.

**Selection at the outer edge at 10 years.** In 140,900 to 141,050 km the rule removes only 41 of 21,354 particles in Pandora at 10 years, and lambda is 4.812 with the rule and 4.818 with freeze only (-0.1 %). In 140,900 to 141,100 km, 734 particles (2.6 %) are removed and lambda is 5.011 against 5.066 (-1.1 %). For the inner edge the rule gives 7.41 against 7.99 (-7.4 %; 25.4 % removed; bound 8.20). So the selection is small where few particles are lost. At 30 years 1,601 of 21,354 (7.5 %) are removed in 140,900 to 141,050 km; there is no freeze-only run at 30 years, so the size is only bounded: at most 4.424 to 4.604 (+4.1 %).

**Like-for-like definition of A1 (added after A0 review).** The A1 window lambda (`plots.py`, `window_summary`) is 2 x the median over the **per-bin medians** of the 2 km profile bins that lie fully inside the window, not the median over particles. The script `crates/fring-analysis/python/window_a1.py` applies that definition to a `*_profile.csv`. Check: on the raw main10 it reproduces the committed A1 values (inner edge 8.4, flank 2.0, core 0.84, quiet zone 0.90, outer flank 2.08, outer edge 5.26). The same definition on the new runs:

| Window | A1 main30 | `full30` | Shift | `full10-freeze` 10 yr | `full10-rule` 10 yr |
|---|---|---|---|---|---|
| inner flank | 2.227 | 2.220 | -0.3 % | 2.04 | 2.04 |
| core | 0.880 | 0.880 | 0 | 0.82 | 0.82 |
| quiet zone | 0.907 | 0.907 | 0 | 0.90 | 0.90 |
| outer flank | 2.147 | 2.180 | +1.6 % | 2.12 | 2.12 |
| outer edge to 141,050 km | 4.747 | 4.547 | -4.2 % | 5.06 | 5.06 |
| inner edge | not run | 6.77 | | 8.40 | 8.18 |

- **The 2 to 3 % shifts of the inner flank and the quiet zone are a difference of definition, not of the dynamics.** With the A1 definition the 30-year values agree to 0.3 % or better. The "replaced" verdicts of those two rows were reached with a per-particle median against a median of bin medians. They stay as written (rule fixed before the numbers), and this paragraph is the correction of their meaning. (The core agreed in both definitions.)
- **Which "replaced" verdicts are selection effects of the rule:**
  - Inner edge (8.4 to 7.41 per particle, 8.40 to 8.18 with the A1 definition): selection. The freeze-only run keeps the particles and gives the A1 value.
  - Outer edge, 10 years: no selection effect (41 particles lost; -0.1 %).
  - Outer edge, 30 years (4.75 to 4.55 like-for-like, -4.2 %): not shown to be selection, and not excluded. The loss is 7.5 % of the window; the extreme selection bound lifts the per-particle value by 4.1 % (4.424 to 4.604), which is the size of the shift, but that bound assumes that every absorbed particle was above the median. A freeze-only 30-year run would settle it (about 20 min of GPU, not done).
  - Inner flank (2.23 to 2.172), quiet zone (0.91 to 0.889): not selection (190 and 0 lost; the bound moves the flank by +0.5 %). They are the definition difference above.
  - Outer flank (+1.6 % like-for-like, 2.4 sigma per particle): confirmed, not selection (1 lost).
- The pilot test of section 0 shows that the code change (WH-EFF, DET-SINCOS, freeze) moves lambda by 1.0 % at most at 10 years, so 2.3 to 6.9 % at 30 years is larger than that effect, and the definition difference above is the reason for the two small ones.
- The conclusions of A1 do not change: the quiet zone is at lambda 0.89 to 0.91 per year, the core is on its inner side, the zone is chaotic.

Reading:
- **Chaotic fractions: all A1 numbers are confirmed.** The inner edge is 98 % chaotic at 10 years (bound 0.9786 against 0.9789) and about 99 % at 30 years. The sampling noise is below 0.01 everywhere. Over the band as a whole the chaotic fraction of the survivors at 30 years is 0.899 and the bound is 0.911 (A1, narrower band: 0.890).
- **The lambda of the inner edge is replaced, by selection.** The rule removes the particles that hit Prometheus, and these have the largest MEGNO. So the median of the survivors falls: 7.41 against 7.99 with freeze only (10 yr), and 5.93 at 30 years (bound 11.26 because 53 % are lost). The freeze-only run, which keeps those particles, gives 7.99 (A1 7.97 on the same ids, sigma 0.013). Use the freeze-only value for "how chaotic the unselected edge is", and the rule value for "how chaotic the surviving ring is".
- The A1 statement "above about 140,250 km the result does not depend on the missing sub-stepping" is supported: the core and quiet zone lose no particle, and the numbers agree.

## 4. Limits

- No sub-stepping (ICR-004 amendment 2). A frozen particle is a removed particle: its chaos indicator stops at the loss, so lambda and class are for the survivors only.
- No raw main30: the 30-year comparison is with the committed window values and binomial errors, on different particles, and the lambda of A1 has no error. The per-particle lambda verdicts of 2 to 3 % compare a different definition (see the like-for-like paragraph in section 3), a rounding (0.005) and a code change, not two independent estimates; use the like-for-like table.
- The 5.4 % within 1 R_H of A1 is not reproduced (needs the CPU harness).
- No figure was made for this note; the tables are in `data/a3_windows_*.csv`.

## 5. Commands

```sh
export PATH="/opt/homebrew/opt/rustup/bin:$HOME/.cargo/bin:$PATH"
cargo build --release -p fring-run -p fring-analysis
for n in full10-freeze full10-rule full30; do target/release/fring-run --config docs/results/configs/chaos-map-$n.toml --out runs/final-$n --quiet; done   # 395 s, 416 s, 1237 s
target/release/fring-analyse windows runs/final-full30 docs/results/data/a3_windows_full30.csv 30 139700-140000 140850-141100 140900-141050
target/release/fring-analyse windows runs/final-full10-rule docs/results/data/a3_windows_full10-rule.csv 10 139700-140000 140850-141100
```
