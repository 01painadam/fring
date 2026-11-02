# A2-ANALYSIS: Pandora close approaches and eccentric core orbits

Role A8. Date 2026-11-02. Model as in `a1-analysis.md`: Saturn J2 to J6, Prometheus and Pandora, no collisions, one moon realisation (O7). Step h = 356.2537 s (150 steps per orbit), GPU mode D+, indicator ADR 0008 (shadow d0 = 1 km, renorm 100 steps).

## Summary

1. **The outer edge of the circular map does not need sub-stepping.** A test with the step h/4 = 89.0634 s (renorm 400 steps) on 30,000 particles from 140,850 to 141,100 km gives the same statistics as the step h. The chaotic fraction changes by -0.0003 (counting sd 0.0032). The median lambda changes by +0.009 per year (sigma 0.023). The criterion was 3 sigma. Many particles do pass close to Pandora (section 1), but the map statistics do not depend on it.
2. **Eccentric core orbits are much more chaotic than circular ones.** With e = 0.00235 and the apse longitude 353.78 deg, over the whole quiet zone 140,150 to 140,700 km the chaotic fraction (mean MEGNO 5 or more) is 0.856 against 0.419 at 10 years, and 0.994 against 0.884 at 30 years. The median lambda is 2.01 against 0.86 per year at 10 years and 1.59 against 0.89 per year at 30 years. In the core window (140,197 to 140,247 km) it is 0.902 against 0.405 at 10 years. All 22 bins differ: the smallest per-bin z is 29.0 at 10 years and 11.5 at 30 years (Holm-adjusted p at most 4.5e-185 and 2.1e-30, `holm_p` in `ecc_bins_e10.csv` and `ecc_bins_e30.csv`).
3. **The result depends on the apse phase.** Moving the apse by +120 deg lowers the eccentric chaotic fraction by 0.09 (inner and core bins) and 0.19 (outer bin) at 10 years. The effect is still "more chaotic than circular" in these bins. The value of the excess is therefore tied to the one apse longitude of `params/ring.toml` (which is itself uncertain by a few degrees).

e and the apse come from `params/ring.toml` (0.00235 and 353.78 deg); PLAN's 0.0024 is that value rounded. All 100 GPU runs of part (b) and the h/4 test finished without a failed particle (exit 0). So no bin was lost.

## 1. Part (a): closest approach to Pandora

Method: `fring-analyse closest` (CPU f64, one step at a time, the minimum distance over the straight segment between two steps), both moons, 20,000 particles of the main10 set (seed 20261102, 2e5 particles, band 139,700 to 141,100 km) with an initial radius from 140,850 to 141,100 km, 10 years. Hill radius of Pandora 61.41 km (mean radius 141,713.6 km); body radius 40.63 km. Data: `data/closest_pandora_pandora_cum.csv`, `closest_pandora_pandora_hist.csv`, `closest_pandora_rbins.csv`, `closest_pandora_classes.csv`.

| Closer than | km | Particles | Share | of which chaotic (main10 class) |
|---|---|---|---|---|
| body radius (hit) | 40.63 | 376 | 1.9 % | 375 |
| 1 R_H | 61.4 | 919 | 4.6 % | 915 |
| 4.1 R_H | 251.8 | 13,207 | 66.0 % | 13,096 |
| 6.3 R_H | 386.9 | 16,142 | 80.7 % | 15,920 |

- By radius (25 km bins, `closest_pandora_rbins.csv`): the share closer than 4.1 R_H rises from 0.08 at 140,850 km to 0.87 at 141,100 km. Hits are 0 below 140,975 km and 0.133 in the bin 141,075 to 141,100 km. Median closest distance 205 km (3.3 R_H).
- Prometheus (same particles): no hit, smallest distance 61.6 km, median 1,242 km. The outer window is far from Prometheus.
- Criterion for "the step resolves an encounter": h v_rel / b < 0.1, with v_rel from the shear 1.5 n b and the epicycle speeds. The epicycle speeds dominate: Pandora a e kappa = 0.070 km/s, an eccentric particle 0.039 km/s. For a circular particle b_crit = 356 x 0.070 / 0.1 = 249 km (4.1 R_H). For an eccentric particle with Pandora, v_rel up to 0.109 km/s gives b_crit = 388 km (6.3 R_H). By this criterion most of the particles in the window have an unresolved encounter (66 % and 81 %).
- The class columns of `closest_pandora_pandora_cum.csv` are empty (the file was written before the join); the classes in the table come from `closest_pandora_classes.csv`.
- Join with the classes: by particle id. The CPU trajectories and the GPU class run (main10 rerun, 388 s) diverge after a Lyapunov time, so the join is a statistical relation, not a trajectory match. 20,000 of 20,000 particles matched.
- The criterion is a screen. The decision comes from the step test (next).

### Step test at the outer edge

`configs/edge-h.toml` and `edge-h4.toml` are identical except for the step (h, renorm 100; h/4, renorm 400), 30,000 particles, 140,850 to 141,100 km, 10 years (64 s and 254 s on the GPU). Data: `data/edge_step.csv` (A = h/4, B = h).

| Window (km) | n each | chaotic A | chaotic B | difference (sd) | median lambda A, B (1/yr) | difference (sigma) | KS D |
|---|---|---|---|---|---|---|---|
| 140,850 to 141,100 | 30,000 | 0.8132 | 0.8135 | -0.0003 (0.0032) | 4.488, 4.479 | +0.009 (0.023) | 0.0055 |
| 140,850 to 140,900 | 6,004 | 0.7783 | 0.7803 | -0.0020 (0.0076) | 3.139, 3.106 | +0.033 (0.046) | 0.016 |
| 140,900 to 141,000 | 12,066 | 0.7874 | 0.7877 | -0.0003 (0.0053) | 4.458, 4.470 | -0.012 (0.035) | 0.0125 |
| 141,000 to 141,100 | 11,930 | 0.8569 | 0.8563 | +0.0006 (0.0045) | 5.824, 5.837 | -0.013 (0.043) | 0.0074 |

All differences are below 1 sigma. The statistics of the map (chaotic fraction, median lambda, the distribution of ln MEGNO) do not change when the step is divided by 4, even though two thirds of the particles pass within 4.1 R_H. **The outer edge does not need sub-stepping for the map.** This is a statement about the statistics, not about single orbits. A single close encounter (the 376 hits) is not resolved by the step h, and a statement about one orbit would need sub-stepping (task E1).

## 2. Part (b): eccentric against circular orbits

### Set-up

- Distribution `eccentric_band` (ICR-007): guiding a uniform in 25 km bins from 140,150 to 140,700 km (22 bins), 5,000 particles per bin, e = 0.00235 (epicycle amplitude, a e is about 330 km), apse longitude 353.78 deg. The control is the same config with e = 0 and the same seed, so each particle id has the same a and mean longitude. The map axis is the guiding a (M22, `Point.a_km`).
- Moons: the same for all runs (built from `params/`, not from the seed). Seeds 20261200 + bin; repeats use seed + 1000.
- Runs: 22 bins x 2 (e, control) at 10 years (643 s in total) and at 30 years (1,691 s in total). Extra: apse + 120 deg and a seed repeat for bins 0 (140,150 to 140,175), 2 (core, 140,200 to 140,225) and 21 (140,675 to 140,700), at 10 years.
- Statistics (M19): the chaotic fraction (mean MEGNO 5 or more) and the median lambda = 2 Y / t with a bootstrap one-sigma interval, never a fixed class threshold across run lengths. At 30 years the control is near its ceiling (0.88), so the median lambda and the KS test of ln Y are the main statistics there. Per bin: two-proportion z test with Holm correction over 22 bins. The test is unpaired, although the e run and its control share the particle ids (same a and mean longitude). Per window: z test, KS. Minimum 1,000 valid particles per set (M17; each bin has 5,000, each window 10,000 to 110,000).
- Noise floor: the seed repeats differ from the base runs by 0.0038, -0.0036 and -0.0050 in the chaotic fraction (counting sd 0.006 to 0.007; z between -0.7 and 0.6) and by 0.06, -0.02 and -0.05 in the median lambda (sigma 0.04 to 0.06). The floor is the counting noise.

### Results

Data: `data/ecc_bins_e10.csv`, `ecc_bins_e30.csv`, `ecc_windows_e10.csv`, `ecc_windows_e30.csv`, `ecc_repeats_e10.csv`. Figures: `figures/ecc_profile_e10.png`, `ecc_profile_e30.png`.

| Window (km) | years | chaotic e / control | difference (sd) | median lambda e / control (1/yr) |
|---|---|---|---|---|
| core 140,197 to 140,247 | 10 | 0.902 / 0.405 | 0.497 (0.006) | 2.82 / 0.85 |
| core | 30 | 0.996 / 0.912 | 0.084 (0.003) | 2.10 / 0.89 |
| inner 140,150 to 140,425 | 10 | 0.866 / 0.395 | 0.472 (0.003) | 2.01 / 0.84 |
| outer 140,425 to 140,700 | 10 | 0.846 / 0.444 | 0.402 (0.003) | 2.02 / 0.90 |
| quiet zone 140,150 to 140,700 | 10 | 0.856 / 0.419 | 0.437 (0.002) | 2.01 / 0.86 |
| quiet zone | 30 | 0.994 / 0.884 | 0.110 (0.001) | 1.59 / 0.89 |

- The control agrees with the A1 map (quiet zone 42 % at 10 years, 89 % at 30 years), so the rerun and the new bins reproduce A1.
- Every bin is more chaotic with e: the chaotic fraction is 0.80 to 0.92 at 10 years in every bin (control 0.32 to 0.58), and the median lambda per bin is 1.47 to 4.02 per year (control 0.75 to 1.34). The KS distance of ln Y is 0.47 to 0.72 per bin at 10 years.
- The lambda of the eccentric orbits falls from 10 to 30 years (2.0 to 1.6 per year in the zone), the control does not (0.86 to 0.89). So the eccentric orbits are chaotic in the early years (probably through close encounters with the moons: at the inner end the pericentre, 140,150 x (1 - 0.00235) = 139,821 km, is 132 km or 2.1 R_H above the apocentre of Prometheus, 139,689 km; at the outer end the apocentre, 141,031 km, is 82 km or 1.3 R_H below the pericentre of Pandora, 141,113 km) and the finite-time value contains this. I did not separate the causes. A test with the distance to the moons (the `closest` tool on `eccentric_band`) would do that.
- Apse phase (10 years, ecc vs base run): chaotic fraction -0.087 (bin 0), -0.094 (core bin), -0.186 (bin 21), all with z below -11. The median lambda goes from 4.02 to 3.29, from 2.92 to 2.42, and from 3.46 to 1.37 (base run to apse + 120 deg, `lam_b` and `lam_a` of the `phase120` rows). The sign of the main result does not change, but the size does.

## Limits

- One moon realisation (O7), and the moons are chaotic in about a quarter of the cases (A1). Counting noise does not include this.
- One e (0.00235) and two apse longitudes (353.78 and 113.78 deg), the second only at 10 years in three bins.
- Orbits that cross the moons: with e = 0.00235 the radial excursion is about 330 km, so orbits at the ends of the zone come within about 2.1 R_H of Prometheus (inner end) and 1.3 R_H of Pandora (outer end). Hits are not removed (as in A1). No run stopped on a failed particle, so there is no dropped bin.
- The h/4 test covers the outer window of the circular map only. It was not repeated for the eccentric orbits, which have larger relative speeds at encounters (b_crit 6.3 R_H against 4.1 R_H).
- Part (a) uses CPU trajectories and the GPU classes joined by id (statistical relation).
- No collisions and no ring self-gravity.

## Reproduce

```sh
cargo build --release -p fring-run -p fring-analysis
R=/Users/adampain/Projects/fring/runs
./target/release/fring-run --config docs/results/configs/chaos-map-main10.toml --out $R/chaos-map-main10 --quiet          # 388 s
./target/release/fring-analyse closest docs/results/configs/closest-pandora.toml $R/closest-pandora.csv prometheus,pandora 140850 141100 20000   # CPU, about 11 min
./target/release/fring-analyse closest-join $R/closest-pandora.csv $R/chaos-map-main10 pandora 61.41 40.63 docs/results/data
./target/release/fring-analyse closest-bins $R/closest-pandora.csv docs/results/data/closest_pandora_rbins.csv 25 61.41 4 40.63
./target/release/fring-run --config docs/results/configs/edge-h.toml  --out $R/edge-h  --quiet    # 64 s
./target/release/fring-run --config docs/results/configs/edge-h4.toml --out $R/edge-h4 --quiet    # 254 s
./target/release/fring-analyse edge-compare $R/edge-h $R/edge-h4 10 docs/results/data/edge_step.csv
./target/release/fring-analyse ecc-configs docs/results/configs/ecc 10 e10    # and: 30 e30 (already committed)
docs/results/configs/ecc/run_list.sh docs/results/configs/ecc/list_e10.txt e10   # 643 s; same with list_e30.txt e30, 1,691 s
./target/release/fring-analyse ecc-compare $R e10 10 docs/results/data           # and: e30 30
cd crates/fring-analysis/python && uv run python plots.py ecc closest_pandora
```

Raw runs are in `/Users/adampain/Projects/fring/runs/` (git-ignored); the status of each run is in `runs/logs/ecc-status-e10.csv`, `ecc-status-e30.csv` and `chain1-status.csv` (exit 0 for all).
