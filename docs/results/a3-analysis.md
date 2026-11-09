# A3-ANALYSIS: 30-year chaos map over the full band with flag and freeze and the radius rule

Role A8. Date 2026-11-09. This section was written before the final runs were read (commit order: this section, then the numbers).

## 0. Decision rule (fixed before the final numbers)

What I compare, and with what noise. All comparisons are of statistics, never of single trajectories.

- **Chaotic fraction of a radius window.** Two-proportion z test (`stats::two_proportion`, pooled sd). "Confirmed" if |z| < 3. "Replaced" if |z| >= 3; then the new value and its counting error are given.
- **Which fraction.** The survivors (live particles) and the bound (every lost particle counted as chaotic: a moon scattered or absorbed it). A1 kept every particle in the sample, hits included, so the bound is the like-for-like value for an inner-edge number. Both are reported. The verdict uses the bound for windows with losses, and the survivors where nothing was lost.
- **Lambda of a window.** lambda = 2 median(MEGNO) / years of the survivors. Noise: one sigma from a bootstrap of the median (200 resamples, 16 to 84 %), written to `lambda_sigma_per_yr`. "Confirmed" if the old and new values differ by less than 3 sigma of the new value; the old raw main10 run also gets its own sigma, and the two are added in quadrature. For main30 only the committed window values exist (no raw run), so only the new sigma is used.
- **Measured before I set any limit (rule M27).** Pilot `full10-freeze` against A1 main10 on the same 200,000 ids (`docs/handoffs/A3-ANALYSIS.md`): the chaotic fraction of every window differs by less than 0.003 and lambda by less than 3 %. The limits above (3 sigma) are the usual ones of A2-ANALYSIS and are not tightened to this.
- **Hits.** Lost inside Prometheus among the first 20,000 ids at 10 years (`full10-rule`) against 911 of 20,000 = 4.6 % in A1 (`a1-analysis.md`, closest approach). Confirmed if the counting sd of the new count (sqrt of the count) covers the difference within 3 sigma. This is not the same quantity (A1: a segment distance below 43 km at CPU f64 with the old drift; here: the rule with the radius of `params/moons.toml` on the GPU), and I say so.
- **Not covered.** Encounters within 1 R_H of Prometheus (5.4 %) need the CPU closest-approach harness and are not rerun here (K4 holds the CPU). The hits above are the comparable number.
- **Differences between old and new runs.** main10 against `full10-freeze` differs in more than mu_eff: it also has DET-SINCOS (release initial states may differ by a few ulp, which does not change the statistics) and freeze (no effect unless a particle becomes unbound). `full10-freeze` against `full10-rule` isolates the radius rule. There is no raw main30: only `docs/results/data/main30_{hist,profile}.csv` and the window values in `a1-analysis.md`, so the 30-year comparison uses those values with binomial errors, and the lambda bootstrap of the old value exists for main10 only.
