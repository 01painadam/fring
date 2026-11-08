# A4-ANALYSIS: the moons' Lyapunov rate with the GR03b masses

Written 2026-11-08 (task A4-ANALYSIS, role A8). Question (from `docs/literature/l3-moon-parameters-gr03.md`, section 6): does the mass difference between `params/` and GR03b explain the factor 1.4 between our chaotic-group rate (0.22 per year, A1-ANALYSIS) and GR03 (0.30 to 0.35 per year)?

## Summary

- **The masses do not explain the factor.** With the GR03b masses (Prometheus 0.02200, Pandora 0.01301 km³ s⁻²; our values times 2.054 and 1.405) the chaotic-group rate is 0.2145 ± 0.0109 per year (error of the mean, 34 runs). That is 0.5 standard errors below 0.22, 7.8 below 0.30 and 12.4 below 0.35. It is not between 0.22 and the band, and the L3 rule gives "near 0.22".
- **The masses change the chaotic fraction, not the rate.** The fraction goes from 26 % to 34 % (difference 0.08 ± 0.065, 1.2 standard errors, so not significant alone). The mean over all 100 runs goes from 0.0825 to 0.0951.
- **The GR03 method (shear subtracted with the (log t)/t curve, years 0 to 100) gives the same picture.** Chaotic group 0.2217 (ours) and 0.2121 (GR03b). So the method does not close the gap either.
- **One 3000-year trajectory from the nominal state (no kick) gives 0.194 per year**, also near 0.22 (section on the 3000-year trajectories).
- **The open question moves to the definition and the state.** What is left: the window (nine and twelve runs change class between years 30 to 60 and 60 to 100, so the chaos is intermittent), the single trajectory of GR03 (one run of 3000 years), and the epoch and elements (GR03 use a 1981 Voyager state). This note does not decide between them.

## Method

All numbers come from `fring-analyse lyapunov` (tangent method, `crates/fring-analysis/src/lyapunov.rs`), with the settings of A1-ANALYSIS: 100 runs of 100 years, seed 20261105, Gaussian position kick of 2 km (Philox stream 8), d0 = 1 km for the tangent vector, renormalisation every 0.25 yr, IAS15 with epsilon 1e-9. Only the moon file differs between the two sets.

- **Our masses:** `params/moons.toml`. Log line: `moons: params/moons.toml; GM prometheus = 0.01071 km3/s2, GM pandora = 0.00926 km3/s2`.
- **GR03b masses:** `params/variants/gr03b-moons.toml`. Log line: `moons: params/variants/gr03b-moons.toml; GM prometheus = 0.022 km3/s2, GM pandora = 0.01301 km3/s2`. The states are the nominal ones (test `gr03b_variant_has_gr03b_masses_and_nominal_states`).
- **Trust check.** The re-run with our masses is byte for byte equal to the committed A1 files: `diff` of `lyapunov_pairs.csv` and `lyapunov_curve.csv` against `docs/results/data/` gives no difference. The new options do not change the default output (6 runs of 8 years: `diff -r` of the old and the new binary, no difference). So the two sets differ only by the masses.
- **Rate (A1 definition).** Slope of L = ln(d/d0) over years 30 to 100 (OLS). A run is chaotic if the slope is above 0.1 per year. Error of the mean = standard deviation / √n.
- **Rate (GR03 definition).** For every run, least-squares fit of L(t) = λ t + b ln t + a. The b ln t term is the Kepler shear. In terms of L/t this is the "constant + (log t)/t" curve of GR03a (section 2.2), with λ the constant. **Main number:** all points with t > 0 (from 0.25 yr) to 100 yr, because ln t is defined there and no point is chosen by hand. **Sensitivity:** points from 1 yr to 100 yr. **Anchor variant:** c = L(T)/T − ln(T)/T, the constant of the curve that passes through the last point (as GR do). I use the fit as the main number because the anchor depends on one point and on the offset a, which the fit takes out (the anchor is 0.04 higher for the chaotic groups).

## Results (100 runs of 100 years each)

| Quantity | Our masses | GR03b masses |
|---|---|---|
| Chaotic runs (slope above 0.1 per year) | 26 | 34 |
| Chaotic fraction | 0.26 | 0.34 |
| Chaotic-group rate, years 30 to 100 (per year) | 0.2233 ± 0.0120 (std 0.0612; range 0.129 to 0.372) | 0.2145 ± 0.0109 (std 0.0637; range 0.118 to 0.365) |
| Lyapunov time of the chaotic group | 4.5 yr | 4.7 yr |
| Regular-group rate, years 30 to 100 | 0.0331 ± 0.0012 | 0.0337 ± 0.0018 |
| Mean over all 100 runs, years 30 to 100 | 0.0825 ± 0.0090 (median 0.0312) | 0.0951 ± 0.0094 (median 0.0339) |
| GR03 fit, t > 0 to 100 yr: chaotic group | 0.2217 ± 0.0132 | 0.2121 ± 0.0114 |
| GR03 fit, t > 0 to 100 yr: all runs | 0.0708 ± 0.0096 | 0.0842 ± 0.0101 |
| GR03 fit, 1 to 100 yr: chaotic group / all runs | 0.2262 ± 0.0130 / 0.0726 ± 0.0098 | 0.2184 ± 0.0120 / 0.0868 ± 0.0104 |
| GR anchor c: chaotic group / all runs | 0.2635 ± 0.0115 / 0.1432 ± 0.0078 | 0.2388 ± 0.0101 / 0.1485 ± 0.0075 |
| Plain FTLE at 100 yr, all runs | 0.1892 ± 0.0078 | 0.1945 ± 0.0075 |
| Runs that change class between years 30 to 60 and 60 to 100 | 9 | 12 |
| Mean slope, years 30 to 60 / 60 to 100, all runs | 0.0878 / 0.0775 | 0.0939 / 0.0945 |

Checks of the arithmetic (the mean over all runs is the fraction-weighted mean of the two groups):
- Ours: 0.26 × 0.2233 + 0.74 × 0.0331 = 0.0581 + 0.0245 = 0.0826 (table: 0.0825).
- GR03b: 0.34 × 0.2145 + 0.66 × 0.0337 = 0.0729 + 0.0222 = 0.0951 (table: 0.0951).

So the larger mean over all runs comes from the larger fraction of chaotic runs, not from a faster rate.

## The L3 rule

Rule (L3, section 6): the masses explain the factor if the chaotic-group rate with the GR03b masses is between 0.30 and 0.35 per year. They do not explain it if the rate stays near 0.22.

- Value: 0.2145 ± 0.0109 per year (34 runs).
- Distance to 0.22: (0.2145 − 0.22) / 0.0109 = −0.5 standard errors. Ratio 0.2145 / 0.22 = 0.975, and 0.2145 / 0.2233 (the A1 value) = 0.961.
- Distance to the band: (0.30 − 0.2145) / 0.0109 = 7.8 standard errors to the lower edge, (0.35 − 0.2145) / 0.0109 = 12.4 to the upper edge. A rate of 0.30 would need a factor 0.30 / 0.2145 = 1.40.
- The two chaotic groups differ by 0.2233 − 0.2145 = 0.0088, with an error sqrt(0.0120² + 0.0109²) = 0.0162 (0.5 standard errors).
- 23 runs are chaotic with both mass sets, 14 runs change group. In the 23 common runs the rate is 0.2269 (ours) and 0.2218 (GR03b).
- The paired difference of the slopes (GR03b minus ours, all 100 runs) is 0.0126 ± 0.0067 (1.9 standard errors), the sign of a small rise of the mean, as expected from the fraction.

**Result: the rate is near 0.22. The mass difference does not explain the factor 1.4.** The same holds with the GR03 definition (0.2121 ± 0.0114 for GR03b, 7.7 standard errors below 0.30). The run excludes the factor from the masses at a level far beyond the error of the mean. It does not exclude a smaller effect: the chaotic fraction may rise with mass (0.26 to 0.34, 1.2 standard errors).

Caution on the window: nine (ours) and twelve (GR03b) of 100 runs change class between years 30 to 60 and 60 to 100. A chaotic rate over 70 years is therefore partly a rate over the chaotic part of the run. The rate of the chaotic group depends on this window by an amount that I did not measure.

## One trajectory of 3000 years (our masses)

This is one sample of a chaotic system and not a statistic. Trajectory A is run 0 of the set above; trajectory B has no position kick.

Two single trajectories were run. Both are runs of one thread, with our masses.

| Trajectory | Start | L(3000 yr) | Slope of L, years 30 to 3000 | GR fit, t > 0 to 3000 yr | GR fit, t > 0 to 100 yr | Slope in 500-yr windows |
|---|---|---|---|---|---|---|
| A: run 0 of the set above | nominal state + 2 km kick (seed 20261105) | 95.16 | 0.0284 | 0.0284 | 0.0139 | 0.0282, 0.0284, 0.0284, 0.0284, 0.0284, 0.0284 |
| B: nominal state | no kick (`kick_km` = 0), same seed for the tangent vector | 580.24 | 0.1903 | 0.1939 | 0.1993 | 0.209, 0.155, 0.217, 0.178, 0.206, 0.139 |

- **B is the single trajectory that compares with GR03.** It is chaotic from the start. Its rate is 0.194 per year over 3000 years (Lyapunov time 5.2 yr), and 0.199 over the first 100 years. That is near our 100-year chaotic group (0.22) and below the 0.30 to 0.35 of GR03. The 500-yr windows go from 0.139 to 0.217, so one trajectory has a spread of of about ±0.03 around its mean (standard deviation of the six windows 0.032). The anchor c is 0.191.
- **A is regular by the A1 threshold** (0.0284, below 0.1). Its L grows linearly with the same slope in every 500-yr window, so its tangent vector grows slowly and steadily. I did not run a MEGNO check on it, so I do not call it chaotic or regular beyond the threshold.
- B is the pair-0 tangent run with kick 0 and was not planned in PLAN (an extra 805 s). It answers the question in the 3000-year item: a long trajectory from the nominal state gives about 0.19 to 0.20 per year with our masses, not 0.30.
- Both are one sample each. They support the result of the 100-run sets and do not replace it.

## Files

Small files in `docs/results/data/` (the raw output of the runs is in `runs/a4/`, git-ignored):
- `lyapunov_gr_ours.csv`, `lyapunov_gr_gr03b.csv`: one row per run: slope 30 to 100, GR fit (t > 0), GR fit (1 yr), anchor, chaotic flag.
- `lyapunov_Lmean_ours.csv`, `lyapunov_Lmean_gr03b.csv`: mean L(t) over all runs and over the chaotic group.
- `lyapunov_pairs_gr03b.csv`, `lyapunov_curve_gr03b.csv`: the same files as `lyapunov_pairs.csv` and `lyapunov_curve.csv` of A1-ANALYSIS, for the GR03b masses.
- `lyapunov_gr_long3000.csv`, `lyapunov_L_long3000_every10yr.csv`: trajectory A (L every 10 yr). `lyapunov_gr_long3000k0.csv`, `lyapunov_L_long3000k0_every10yr.csv`: trajectory B.

## Reproduce

The runs are deterministic. Wall times on the Mac with 8 threads (other jobs ran at the same time).

```sh
export PATH="/opt/homebrew/opt/rustup/bin:$HOME/.cargo/bin:$PATH"
cargo build --release -p fring-analysis
# argument order: out n years threads d0_km renorm_yr kick_km seed tag method moons_file L
./target/release/fring-analyse lyapunov runs/a4/ours  100 100 8 1.0 0.25 2 20261105 "" tangent params/moons.toml L                      # 537 s
./target/release/fring-analyse lyapunov runs/a4/gr03b 100 100 8 1.0 0.25 2 20261105 "" tangent params/variants/gr03b-moons.toml L      # 644 s
./target/release/fring-analyse lyapunov runs/a4/long3000 1 3000 1 1.0 0.25 2 20261105 "" tangent params/moons.toml L                  # 1065 s, one thread (A)
./target/release/fring-analyse lyapunov runs/a4/long3000k0 1 3000 1 1.0 0.25 0 20261105 "" tangent params/moons.toml L                # 805 s, one thread (B)
cargo test -p fring-analysis                                                                                                            # 22 tests
```

The full L(t) files (`runs/a4/*/lyapunov_L.csv`, 40,000 lines each) come from these commands. The `lyapunov_pairs.csv` and `lyapunov_curve.csv` of the run with our masses are equal to the A1-ANALYSIS files.
