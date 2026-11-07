# A5-ANALYSIS: step test, close encounters and apse phase of eccentric core orbits

Role A8. Date 2026-11-07. Model as in `a1-analysis.md` and `a2-analysis.md`: Saturn J2 to J6, Prometheus and Pandora, no collisions, one moon realisation (O7). Step h = 356.2537 s (150 steps per orbit), GPU chaos runs with the shadow method (offset 1 km). Eccentric orbits: `eccentric_band` (ICR-007), e = 0.00235, apse longitude 353.78 deg from `params/ring.toml`, guiding a in 25 km bins from 140,150 to 140,700 km (bins 0 to 21), seeds 20261200 + bin. e = 0 is the control.

## Summary

1. **The step h is good enough for the eccentric orbits at the ends of the zone.** With h/4 the chaotic fraction changes by -0.0002 (inner 50 km) and -0.0026 (outer 50 km), with counting sd 0.0044 and 0.0050 (z = -0.05 and -0.52). The median lambda changes by 0.1 to 1.3 sigma. No bin shows a change above 0.6 sd (fraction) or 1.3 sigma (lambda). So the A2 maps of eccentric orbits do not depend on the step.
2. **Eccentric orbits near the zone ends meet the moons all the time.** In 10 years 82 to 86 % of the particles in the three inner bins (Prometheus) and 83 to 84 % in the two outer bins (Pandora) come within 6.3 R_H, and 54 to 70 % (inner) and 63 to 70 % (outer) within 4.1 R_H. With e = 0 the same bins give 0 to 27 of 2,500 within 6.3 R_H and none within 4.1 R_H. In the mid-zone bin (140,400 km) 19 % come within 6.3 R_H.
3. **The chaos of the eccentric orbits is almost all in the particles that have an encounter.** Chaotic fraction with an encounter within 6.3 R_H: 0.94 to 0.99. Without: 0.22 to 0.42 in the end bins (the same as the control, 0.38 to 0.60), and 0.80 in the mid-zone bin. The 2.5k and 5k samples agree to 0.01 or better.
4. **The second apse phase (113.78 deg, +120 deg) changes the 30-year result very little.** Quiet zone: chaotic fraction 0.9915 against 0.9936 (difference -0.0022, sd 0.0004), median lambda 1.669 against 1.585 per year (+5 %). Both are far from the control (0.884, 0.886 per year). At 10 years A2 found a larger effect (-0.09 to -0.19); by 30 years both phases are at the ceiling.

## 1. Part (a): the h/4 step test of `eccentric_band` at the zone ends

Question: A2 left open whether the step h is small enough for eccentric orbits, which reach the moons at larger relative speed than the circular orbits that A2 tested at h/4 (b_crit 6.3 R_H against 4.1 R_H).

Method: bins 0, 1, 20 and 21 (5,000 particles each), 10 years, the A2 config with h/4 = 89.063425 s and renorm 400 steps (`configs/a5/a5-h4-ecc-b00.toml`, b01, b20, b21), same seed as the A2 run of the bin (which has h and renorm 100). Each run took 45 s (`runs/logs/a5-status-h4.csv`, exit 0 for all four).

```sh
./target/release/fring-run --config docs/results/configs/a5/a5-h4-ecc-b00.toml --out $R/a5-h4-ecc-b00 --quiet   # and b01, b20, b21 (run_list.sh list_a5-h4.txt h4)
# then: fring-analyse pair-compare, A = the h run of the A2 e10 set, B = the h/4 run (usage in the header of src/bin/fring_analyse.rs)
```

Data: `data/a5_step_ecc.csv` (A is the h run, B the h/4 run).

| Window (km) | n each | chaotic A | chaotic B | difference (sd) | z | median lambda A, B (1/yr) | difference (sigma) | KS D (p) |
|---|---|---|---|---|---|---|---|---|
| 140,150 to 140,200 | 10,000 | 0.8888 | 0.8890 | -0.0002 (0.0044) | -0.05 | 3.741, 3.733 | +0.009 (0.042) | 0.0142 (0.26) |
| 140,650 to 140,700 | 10,000 | 0.8499 | 0.8525 | -0.0026 (0.0050) | -0.52 | 3.262, 3.275 | -0.013 (0.033) | 0.0123 (0.43) |
| bin 0 | 5,000 | 0.8690 | 0.8688 | +0.0002 (0.0068) | 0.03 | 4.017, 4.024 | -0.006 (0.055) | 0.0174 (0.43) |
| bin 1 | 5,000 | 0.9086 | 0.9092 | -0.0006 (0.0058) | -0.10 | 3.512, 3.516 | -0.004 (0.049) | 0.0124 (0.83) |
| bin 20 | 5,000 | 0.8572 | 0.8586 | -0.0014 (0.0070) | -0.20 | 3.104, 3.116 | -0.011 (0.043) | 0.0142 (0.69) |
| bin 21 | 5,000 | 0.8426 | 0.8464 | -0.0038 (0.0073) | -0.52 | 3.416, 3.462 | -0.046 (0.036) | 0.0146 (0.66) |

Conclusion: no difference above 0.6 sd in the chaotic fraction, none above 1.3 sigma in lambda (bin 21), and the KS test of ln Y does not reject at p = 0.05 in any row (smallest p = 0.26). The step h is enough. Limit: the test is at 10 years and at the four end bins, where encounters are most frequent.

## 2. Part (b): closest approach on `eccentric_band`

Question: how many eccentric particles come within 1, 4.1 and 6.3 Hill radii of each moon in 10 years, and is the chaos in the particles that have such an encounter?

Method: `fring-analyse closest` (CPU f64, one step at a time, the minimum distance over the straight segment between two steps, as in A2) with both moons, 2,500 particles per run, for e = 0.00235 and for e = 0, in six bins: 0, 1, 2 (inner end), 10 (140,400 to 140,425, middle) and 20, 21 (outer end). The distances are joined by particle id with the classes of the e10 GPU run of the same bin (statistical relation, as in A2). Hill radii: Prometheus 63.40 km, Pandora 61.41 km. Configs `configs/a5/a5-cl-{ecc,circ}-bNN.toml`, script `configs/a5/run_closest.sh`, summary `crates/fring-analysis/python/a5_closest.py`. Each run took about 16 min (CPU, 4 in parallel). All 12 finished (`runs/logs/a5-status-closest.csv`; the first attempts were stopped by the usage limit and run again).

Data: `data/a5_closest_ecc.csv`. Figure: `figures/a5_closest.png`. Counts are of the 2,500 particles. The error of a count k is sqrt(N p (1 - p)), 17 to 25 for the large counts (column `*_err` in the file). The error of a chaotic fraction is sqrt(p (1 - p) / n).

| Orbits | Bin | a_lo (km) | Prometheus < 4.1 R_H | Prometheus < 6.3 R_H | Pandora < 4.1 R_H | Pandora < 6.3 R_H | Either < 6.3 R_H | Chaotic, all | Chaotic, with encounter | Chaotic, without (sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| e | 0 | 140,150 | 1694 | 2061 | 0 | 0 | 2061 | 0.862 | 0.988 | 0.269 (0.021) |
| e | 1 | 140,175 | 1634 | 2160 | 0 | 0 | 2160 | 0.906 | 0.982 | 0.418 (0.027) |
| e | 2 | 140,200 | 1358 | 2107 | 0 | 0 | 2107 | 0.885 | 0.973 | 0.410 (0.025) |
| e | 10 | 140,400 | 0 | 320 | 0 | 183 | 487 | 0.830 | 0.938 | 0.803 (0.009) |
| e | 20 | 140,650 | 0 | 0 | 1567 | 2093 | 2093 | 0.853 | 0.969 | 0.258 (0.022) |
| e | 21 | 140,675 | 0 | 0 | 1740 | 2083 | 2083 | 0.847 | 0.973 | 0.221 (0.020) |
| 0 | 0 | 140,150 | 0 | 27 | 0 | 0 | 27 | 0.450 | 1.0 | 0.444 (0.010) |
| 0 | 1 | 140,175 | 0 | 0 | 0 | 0 | 0 | 0.402 | none | 0.402 (0.010) |
| 0 | 2 | 140,200 | 0 | 0 | 0 | 0 | 0 | 0.382 | none | 0.382 (0.010) |
| 0 | 10 | 140,400 | 0 | 0 | 0 | 0 | 0 | 0.323 | none | 0.323 (0.009) |
| 0 | 20 | 140,650 | 0 | 0 | 0 | 0 | 0 | 0.558 | none | 0.558 (0.010) |
| 0 | 21 | 140,675 | 0 | 0 | 0 | 1 | 1 | 0.595 | 1.0 | 0.595 (0.010) |

No run has a hit (distance below the body radius) counted here; the 1 R_H columns are all 0 for the e runs too (`*_1.0RH_n` in the file). The chaotic fractions "all" agree with the A2 values for the same bins (A2 `ecc_bins_e10.csv`: 0.80 to 0.92 for e, 0.32 to 0.58 for the control).

### Sample size check (2,500 against 5,000)

To test that 2,500 particles are enough, bins 0, 10, 20 and 21 (e) were run again with 5,000 particles (`configs/a5/a5-cl5k-ecc-bNN.toml`, `run_closest.sh ecc NN 5k`; about 18 min each; outputs `data/a5/a5-cl5k-ecc-bNN.csv`). The 2.5k numbers come from `data/a5_closest_ecc.csv`; the 5k numbers from the files in `data/a5/` (same rule: chaotic class, minimum distance to either moon below 6.3 R_H).

| Bin | Chaotic, all (2.5k / 5k) | With encounter (2.5k / 5k) | Without encounter (2.5k / 5k) | n with encounter (2.5k / 5k) |
|---|---|---|---|---|
| 0 | 0.862 / 0.869 | 0.988 / 0.987 | 0.269 / 0.282 | 2061 / 4163 |
| 10 | 0.830 / 0.829 | 0.938 / 0.930 | 0.803 / 0.804 | 487 / 1014 |
| 20 | 0.853 / 0.859 | 0.969 / 0.970 | 0.258 / 0.267 | 2093 / 4210 |
| 21 | 0.847 / 0.846 | 0.973 / 0.970 | 0.221 / 0.223 | 2083 / 4172 |

The largest difference is 0.013 (bin 0, without encounter), which is inside one counting sd of the 2.5k sample (0.021). The share of particles with an encounter is the same to within 0.01 (for example bin 10: 0.195 and 0.203). So the 2.5k results are stable.

Conclusion: at the ends of the zone, an eccentric particle meets the nearest moon within 6.3 R_H in about 82 to 86 % of the cases, and within 4.1 R_H in 54 to 70 %. A circular particle in the same bins almost never does (0 to 27 of 2,500). The particles with no encounter are chaotic at the same rate as the circular control (end bins: 0.22 to 0.42 against 0.38 to 0.60; bin 0 is lower, 0.27 against 0.45, but its no-encounter sample is only 439). So the large extra chaos of the eccentric orbits comes from the encounters. In the middle bin only 19 % have an encounter, and the chaotic fraction without one is 0.80 (control 0.32); that bin is chaotic for another reason (not tested here; possibly resonances).

## 3. Part (c): the second apse phase at 30 years

Question: A2 showed at 10 years that moving the apse by +120 deg lowers the chaotic fraction (by 0.09 to 0.19). Is this still so at 30 years?

Method: all 22 bins, 5,000 particles per bin, 30 years, the A2 30-year config with the apse longitude 113.78 deg (`configs/a5/a5-p120-ecc-phase120-bNN.toml`, list `list_a5-p120.txt`, run by `run_list.sh`; 801 s in total, exit 0 for all 22 runs in `runs/logs/a5-status-p120.csv`). Compared with the A2 30-year runs of the same bins (`data/ecc_bins_e30.csv`, apse 353.78 deg, e and control; same seeds, so same a and mean longitude).

Data: `data/a5_phase120_e30.csv` (A is the new run; B is the A2 base run for `*_vs_base`, the control for `*_vs_circ`). Figure: `figures/a5_phase120_e30.png`.

| Window (km) | n each | chaotic: apse 113.78 / apse 353.78 / control | difference to 353.78 (sd) | median lambda (1/yr): 113.78 / 353.78 / control | lambda difference (sigma) |
|---|---|---|---|---|---|
| quiet zone 140,150 to 140,700 | 110,000 | 0.9915 / 0.9936 / 0.8837 | -0.0022 (0.0004) | 1.669 / 1.585 / 0.886 | +0.084 (0.004) |
| inner 140,150 to 140,425 | 55,000 | 0.9931 / 0.9951 / 0.9101 | -0.0020 (0.0005) | 1.719 / 1.629 / 0.865 | +0.091 (0.006) |
| outer 140,425 to 140,700 | 55,000 | 0.9898 / 0.9922 / 0.8573 | -0.0024 (0.0006) | 1.616 / 1.549 / 0.914 | +0.067 (0.005) |
| core bin 140,200 to 140,225 | 5,000 | 0.9936 / 0.9934 / 0.8870 | +0.0002 (0.0016) | 2.265 / 2.172 / 0.861 | +0.093 (0.023) |

Per bin (rows `bNN_p120_vs_base` in the file): the difference in the chaotic fraction is between -0.021 (bin 9) and +0.011 (bin 4) with counting sd 0.0004 to 0.003; the largest are bin 9 (-0.021), bin 12 (-0.013), bin 0 (-0.010), bin 4 (+0.011) and bin 5 (+0.009). The median lambda is higher for the second phase in 21 of 22 bins (+0.005 to +0.137 per year; bin 21 is -0.063, 2.9 sigma); the median difference is about +0.08. The difference to the control is 0.03 to 0.21 in the chaotic fraction in every bin, with z above 11.

Conclusion: at 30 years the apse phase makes a difference of 0.2 % in the chaotic fraction and 5 % in lambda in the zone as a whole, where the effect of the eccentricity (0.11 in the fraction, 0.7 per year in lambda) is 50 times larger. Both phases are at the ceiling (0.99), so the fraction hides the difference at 10 years: A2 found -0.09 to -0.19 at 10 years in three bins (`data/ecc_repeats_e10.csv`). The KS test of ln Y rejects equality of the two phases in most bins (the distributions differ slightly because of the small shift in lambda), but the size is small. The sign of the A2 main result does not depend on the phase.

## What this means for E1-ICR and gate G4

I did not read E1-ICR; this section only gives the numbers it can use.

- **Encounter radius.** For the eccentric ring the encounter regime is the usual case, not the exception: in the end bins 82 to 86 % of the particles come within 6.3 R_H in 10 years, and 54 to 70 % within 4.1 R_H. For the circular ring the same numbers are 0 to 1 % (27 of 2,500 in bin 0, 1 in bin 21, 0 in the others). So a b_crit of 6.3 R_H is the right value for the eccentric ring (it counts nearly all particles that matter, as 97 to 99 % of them are chaotic), and 4.1 R_H is enough for the circular ring because in the quiet zone nothing comes closer than 4.1 R_H.
- **Step.** h is enough for the eccentric orbits (part a), at least for the statistics of the chaos. For collisions inside the ring (not tested here) the step of the collision code is a separate matter.
- **Apse phase.** The first-order choice of 353.78 deg from `params/ring.toml` gives results within 0.2 % (fraction) and 5 % (lambda) of the +120 deg case at 30 years. For G4, a single phase is enough for the 30-year statistics; at 10 years the phase matters (A2), so a short run needs both.
- **Sample size.** 2,500 particles per bin are enough for the closest-approach statistics (part b check).

## Limits

- One moon realisation (O7). Counting noise does not include the moons' chaos (A1).
- One e (0.00235) and two apse longitudes. No other values of e.
- The h/4 test is at 10 years and in four bins. The closest-approach runs use CPU trajectories of 2,500 particles and the GPU classes joined by id (a statistical relation, as in A2: the CPU and GPU trajectories diverge after a Lyapunov time, so the join says only that the particle set is the same).
- The `closest` runs take the minimum over the straight segment between steps; the A2 criterion h v_rel / b < 0.1 is not checked per bin here.
- The bin 10 (middle) result, a chaotic fraction of 0.80 without a close encounter, has no explanation yet.
- No collisions and no ring self-gravity.

## Reproduce

```sh
cargo build --release -p fring-run -p fring-analysis
R=/Users/adampain/Projects/fring/runs
docs/results/configs/a5/run_list.sh docs/results/configs/a5/list_a5-h4.txt h4        # 4 runs, 45 s each (part a)
docs/results/configs/a5/run_list.sh docs/results/configs/a5/list_a5-p120.txt p120    # 22 runs, 801 s (part c)
for k in ecc circ; do for b in 00 01 02 10 20 21; do docs/results/configs/a5/run_closest.sh $k $b; done; done   # 12 runs, about 16 min each (part b); needs the e10 runs of A2
for b in 00 10 20 21; do docs/results/configs/a5/run_closest.sh ecc $b 5k; done      # 5k check
python3 crates/fring-analysis/python/a5_closest.py $R docs/results/data/a5_closest_ecc.csv
cd crates/fring-analysis/python && uv run python plots.py a5
```

The step-test and apse comparisons (`data/a5_step_ecc.csv`, `data/a5_phase120_e30.csv`) come from `fring-analyse pair-compare` (see the header of `src/bin/fring_analyse.rs`). Raw runs are in `/Users/adampain/Projects/fring/runs/` (git-ignored); the run status is in `runs/logs/a5-status-h4.csv`, `a5-status-p120.csv` and `a5-status-closest.csv`.
