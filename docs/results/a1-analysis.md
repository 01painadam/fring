# A1-ANALYSIS: first science result

Role A8. Date 2026-11-02. Model: Saturn J2 to J6, Prometheus and Pandora (PLAN section 2), no collisions, no other moons. All runs use 150 steps per orbit of the core (356.2537 s) unless said otherwise.

## Summary

1. **There is no quiet band, but there is a quiet zone.** Between the moons, almost every circular orbit is chaotic. The chaos is weakest in a broad zone from 140,150 to 140,700 km. The core (140,222.43 km) sits in the inner part of this zone. In that zone the Lyapunov rate of a particle is about 0.9 per year (a Lyapunov time of about 1.1 years, or 650 orbits). Near the moons it is 2 per year (flanks) and 5 to 8 per year (edges). The number 0.9 per year does not change when the run grows from 5 to 10 to 30 years. So the zone is not "regular but slow to show it". It is genuinely chaotic, only less so.
2. **The moons close most of the apse-rate gap at the core.** Saturn alone (J2 to J6) gives 2.69841 deg/day. With both moons, 50-year runs give 2.69981 deg/day (spread 0.00009 over three runs, so the moons add +0.00140 deg/day). The observed rate is 2.70025 ± 0.00029 deg/day. The remaining gap is 0.00044 deg/day, or 1.3 sigma (both errors). The G0 gap of 6 sigma was Saturn alone.
3. **The moons are chaotic in about a quarter of the cases, with a Lyapunov time of 4.5 years.** 100 independent runs of 100 years: 26 % are chaotic with a rate of 0.223 ± 0.061 per year. The other 74 % show only the linear (Kepler shear) growth. The nominal start state is chaotic (0.175 ± 0.015 per year). The finite-time rates of G1 (0.14 to 0.49 per year) were mostly Kepler shear, not chaos (section 3).

## 1. Chaos map

### Set-up

- Indicator: shadow particle, d0 = 1 km, K = 100 (ADR 0008). Mean MEGNO ⟨Y⟩ classes: **bounded** below 1.5, **regular** 1.5 to 2.5, **uncertain** 2.5 to 5, **chaotic** 5 and above. "Bounded" is the 20 % group of G1 whose separation stays small. I did not test whether these orbits librate, so I only call them bounded.
- Particles: circular, uniform in radius and in longitude. The radius axis is the orbital radius. `chaos.csv` holds the Keplerian a (point mass only), which is about 640 km larger because of J2, so the map does not use it.
- **Band 139,700 to 141,100 km.** Prometheus has e = 0.00223 (from its 50-year track), so its radius goes from 139,067 to 139,689 km. Pandora has e = 0.00424, so its radius goes from 141,113 to 142,314 km. Outside 139,700 to 141,100 km, a particle can cross a moon's orbit. Prometheus Hill radius R_H = 63.4 km.
- Runs (GPU, D+, on the M4):

| Run | Particles | Band (km) | Length | Time |
|---|---|---|---|---|
| main5 | 2 × 10⁵ | 139,700 to 141,100 | 5 yr | 189 s |
| main10 | 2 × 10⁵ | 139,700 to 141,100 | 10 yr | 6.4 min |
| main30 | 1.1 × 10⁵ | 139,950 to 141,050 | 30 yr | 11 min |
| zoom10 | 5 × 10⁴ | 140,000 to 140,450 | 10 yr | 1.7 min |

The 30-year run has a narrower band because the first try with the full band stopped at "unbound or invalid orbit" (a particle scattered by Prometheus; there is no sub-stepping yet).

### Run length matters

⟨Y⟩ of a chaotic orbit grows as λt/2. A fixed class threshold therefore catches more orbits as the run gets longer. Fractions of all particles:

| Run | bounded | regular | uncertain | chaotic |
|---|---|---|---|---|
| 5 yr | 24.6 % | 10.4 % | 18.5 % | 46.5 % |
| 10 yr | 12.8 % | 6.5 % | 14.4 % | 66.3 % |
| 30 yr | 3.4 % | 2.5 % | 5.1 % | 89.0 % |

Between 5 and 10 years, 99.7 % of the particles that were chaotic at 5 years stay chaotic. But only 50 % of the "bounded" ones stay bounded. So "bounded", "regular" and "uncertain" at a short run length mostly mean "not chaotic yet". Use the chaotic class and λ = 2⟨Y⟩/t, not the other classes.

### Where it is quiet

Fractions in radius windows (`data/window_summary.csv`; λ = 2 × median ⟨Y⟩ / t):

| Window (km) | chaotic at 10 yr | chaotic at 30 yr | λ (1/yr) at 5 / 10 / 30 yr |
|---|---|---|---|
| 139,700 to 139,950 (inner edge) | 98 % | not run | 8.3 / 8.4 / not run |
| 139,950 to 140,150 (inner flank) | 73 % | 91 % | 2.0 / 2.0 / 2.2 |
| **140,197 to 140,247 (core)** | **39 %** | **91 %** | **0.80 / 0.84 / 0.88** |
| 140,150 to 140,700 (quiet zone) | 42 % | 89 % | 0.92 / 0.90 / 0.91 |
| 140,700 to 140,900 (outer flank) | 72 % | 88 % | 2.2 / 2.1 / 2.2 |
| 140,900 to 141,100 (outer edge) | 82 % | 90 % | 5.4 / 5.3 / 4.8 |

- The profile has a minimum of λ across 140,150 to 140,700 km. The core is on the inner side of it.
- Longitude relative to Prometheus does not matter. In the quiet zone the chaotic fraction is 0.396 to 0.435 in the 36 longitude bins. The scatter is 0.011, the same as counting noise (0.011). Use `figures/chaos_profile.png`, not the 2D maps, to read the result.
- There is a comb in radius. The bounded fraction repeats with a period of 8 to 10 km (FFT, `plots.py comb`). The spacing of first-order resonances with Prometheus near m = 110 is (2/3) a / m² = 7.7 km. This is probably the resonance comb, but I did not test it further.

### Close encounters with Prometheus (G1 open issue)

20,000 particles (the first 20,000 of main10), 10 years, CPU f64, one step at a time (`figures/closest_approach.png`, `closest_by_radius.png`):

| Closer than | Particles | Share |
|---|---|---|
| 43 km (moon radius, a hit) | 911 | 4.6 % |
| 1 R_H (63 km) | 1,088 | 5.4 % |
| 3 R_H (190 km) | 2,553 | 12.8 % |
| 7.3 R_H (463 km) | 6,003 | 30.0 % |

- All 2,553 particles within 3 R_H are in the chaotic class. They start below 140,050 km (60 % of the 4,270 particles that start below 140,000 km come within 3 R_H). No particle that starts above 140,000 km comes within 3 R_H, and none above 140,250 km comes within 7.3 R_H. The median closest distance at the core is 533 km (8.4 R_H).
- Meaning for the map: above about 140,250 km (the core and everything outward to Pandora's region), the result does not depend on the missing sub-stepping. Below 140,000 km, many particles have encounters that the 356 s step does not resolve well. The map there is a statement of "very chaotic", not of a value. Prometheus hits are not removed (no collision with moons in the model).
- **Not checked: Pandora.** The outer edge (above 140,900 km) is within about 200 km of the radius of Pandora's periapsis. The outer-edge numbers need the same test.

### Precision check

3,000 particles at 140,150 to 140,700 km, 30 years, CPU f64 and GPU D+ (`configs/val30-*.toml`): the class agrees for 2,985 (99.5 %). The chaotic fraction is 88.3 % (CPU) and 88.4 % (GPU). The median ⟨Y⟩ is 13.2 and 13.3. For single chaotic orbits ⟨Y⟩ differs by a factor 0.65 to 1.56 (5 % to 95 %), which is expected: one compares statistics, not trajectories.

## 2. The moons' share of the apse rate at the core (G0, question 4)

2,000 particles with the core distribution (a = 140,222.43 km, σ_a = 0.5 km, e = 0.00235, apse longitude 353.78°), 50 years, CPU f64. The apse longitude comes from a fit of the radius against longitude over all particles, at 2,955 times (`apse.rs`). The rate is the slope of the unwrapped apse longitude.

| Moons | Rate (deg/day) | Error | Moons' share (deg/day) |
|---|---|---|---|
| none | 2.698412 | 0.000001 | 0 |
| Prometheus | 2.698114 | 0.000060 | −0.00030 |
| Pandora | 2.698869 | 0.000008 | +0.00046 |
| both | 2.699718 | 0.000096 | +0.00131 |
| both, apse phase 120°, other seed | 2.699902 | 0.000085 | +0.00149 |
| both, half time step | 2.699814 | 0.000093 | +0.00140 |
| analytic, Saturn + secular formula | 2.69933 | | +0.00092 |

- **No-moon check:** the fit gives 2.698412 deg/day. PLAN 7.2 (n − κ at 140,222.43 km, J2 to J6) gives 2.698407. They agree to 5 × 10⁻⁶ deg/day. The earlier G0 number for J2 to J6 is 2.6984.
- **Result:** the mean of the three runs with both moons is 2.69981 deg/day, with a spread of 0.00009 and a fit error of 0.0001 (jackknife over 10 blocks), so I quote 2.6998 ± 0.0002. The moons' share is +0.0014 ± 0.0002 deg/day.
- **The fit error is not the whole story.** The apse phase wobbles with the 6.2-year anti-alignment period of the moons. The rate over 10-year windows varies between 2.6983 and 2.7008 deg/day. The observation covers a few years of one phase, so its value may carry the same wobble.
- **Fair comparison.** Our model stops at J6. G0 found that J8 to J12 add +0.0001 deg/day (J2 to J12: 2.69851, ours: 2.69841). The real planet has them, so the fair total is 2.6999 deg/day. That is 0.00035 below the observed value (1.2 sigma on the observed error alone, 1.0 sigma with ours added). The G0 comparison of 6 sigma used Saturn alone.
- **Prometheus alone lowers the rate and the secular formula is wrong in sign.** The formula (Laplace coefficients, Murray and Dermott 7.3, moon radii from their mean motions) gives +0.00072 (Prometheus) and +0.00020 (Pandora), sum +0.00092 deg/day. The simulation gives −0.00030 and +0.00046. The core is at the m = 110 corotation resonance with Prometheus, where the secular average does not apply. The two moons together are not the sum of the two single-moon runs. Treat the analytic number as a rough check of the order of magnitude, not as a prediction.
- **Forced eccentricity (A0 estimate 3 × 10⁻⁵ for Prometheus, 1 × 10⁻⁵ for Pandora).** The rms deviation of the e-vector from the free circle is 1.4 × 10⁻⁴ (Prometheus), 4.7 × 10⁻⁵ (Pandora), 1.5 × 10⁻⁴ (both), and 5 × 10⁻⁶ without moons (frame rounding). It is about 5 times larger than the estimate, which is plausible near a resonance. With e_free = 0.0022 the free part is still about 15 times the forced part, so the fit gives the free precession. With both moons the mean e over the 50 years is 0.00218 (start 0.00235).

## 3. Lyapunov time of the moons (G1)

- **Why not shadow orbits.** I first used shadow pairs with renormalisation (as in ADR 0008). The result depends on the offset d0: 0.95 per year at 1 cm, 0.13 at 1 m, 0.09 at 100 m (100 yr, 40 to 100 pairs; `runs/lyap-study-shadow.log`). It also depends on the renormalisation interval (0.24 per year at 0.05 yr, 0.10 at 1 yr). The likely cause is rounding and integrator-restart noise in the radial direction (the fake growth scales about as 1/d0). The Kepler shear turns it into growth along the track. For a small d0 this fake growth is larger than the real one. I did not measure the noise itself. The chaos map is not affected in the same way: its offset is 1 km (ADR 0008 chose it from a plateau), and the 30-year CPU against GPU check above agrees. The full shadow study is in `data/lyapunov_shadow_study.txt`.
- A second problem: IAS15 restarted with the step of the running integration (1,500 s on a 52,000 s orbit) has a first-step error of 10⁻⁴ (relative). Restarts need a first step of 1 % of the dynamical time.
- **Method used.** The tangent vector is integrated with the orbit as one IAS15 system of dimension 12 (the Jacobian from central differences along the six axes, h = 1 km). It is renormalised every 0.25 yr. Nothing differences two orbits, so there is no noise to amplify. Checks: massless moons give MEGNO → 2 (unit test); 40 pairs at renormalisation 0.05 yr give a chaotic fraction of 11/40 and rate 0.234 ± 0.061, the same as the main run within the error. I tried renormalisation every 1 yr and stopped the run: some tangent vectors then made IAS15 take tiny steps (no result). Do not use intervals longer than 0.25 yr.
- **Shear bias.** Kepler shear gives linear growth, so L = ln(d/d0) grows like ln(t). The plain rate L/t then falls like ln(t)/t: at 100 years it is 0.19 per year (mean of 100 runs), still falling (figure `moon_lyapunov.png`). G1 measured the growth between years 1 and 12: a factor of 12 from shear alone gives ln(12)/11 = 0.23 per year. That is the same size as the G1 rates (0.22 to 0.29), so those rates were not chaos.
- **Result.** 100 independent base trajectories (the nominal state plus a Gaussian position kick of 2 km, Philox stream 8, seed 20261105), one tangent vector each, 100 years. Rate = slope of L over years 30 to 100:

| Group | Runs | Rate (1/yr) |
|---|---|---|
| chaotic (slope above 0.1) | 26 | 0.223 ± 0.061 (spread), ± 0.012 (error of the mean), range 0.13 to 0.37; Lyapunov time 4.5 yr |
| regular | 74 | 0.033 ± 0.010 (shear, about 1/t) |
| all 100 | 100 | 0.083 ± 0.090; median 0.031 |
| nominal start state (kick 0), 8 directions | 8 | 0.175 ± 0.015; Lyapunov time 5.7 yr |

- A MEGNO check gives 2 d⟨Y⟩/dt = 0.067 ± 0.101 (all) and 0.227 ± 0.027 (nominal state).
- Nine of the 100 runs change class between years 30 to 60 and years 60 to 100, so the chaos is intermittent. For the nominal state the slope is 0.31 in years 30 to 60 and 0.13 in years 60 to 100. The mean rate of a chaotic run therefore carries an error from the window.
- **Compared with the literature (0.30 to 0.35 per year).** Our chaotic rate is 0.22 ± 0.06, a factor 1.4 below. One reason could be that our model has only two moons. I did not test this. The result does not depend on d0, the renormalisation interval or the seed.
- **The moons are regular in three quarters of the 2 km neighbourhood.** The real Prometheus state is uncertain by about 10 km (G0: 11.8 km between two ephemerides), so the run-to-run difference matters for any 100-year comparison with observations.

## Limits

- Pandora encounters are not measured (section 1).
- The maps depend on one moon realisation (O7).
- The chaos map says nothing about the width or the persistence of the core. It measures the exponent of a test particle in a fixed moon field.
- Collisions and the ring's own gravity are not in these runs.

## Reproduce

```sh
cargo build --release -p fring-run -p fring-analysis
./target/release/fring-run --config docs/results/configs/chaos-map-main10.toml --quiet   # GPU, 6 min
./target/release/fring-analyse map runs/chaos-map-main10 docs/results/data main10 139700 141100 7 10 2
./target/release/fring-analyse closest docs/results/configs/closest.toml runs/closest.csv runs/chaos-map-main10   # 25 min
./target/release/fring-run --config docs/results/configs/apse-both.toml --quiet           # CPU, 3 min each
./target/release/fring-analyse apse docs/results/data runs/apse-none runs/apse-prom runs/apse-pand runs/apse-both
./target/release/fring-analyse moons runs/apse-both docs/results/data
./target/release/fring-analyse laplace runs/apse-both docs/results/data
./target/release/fring-analyse lyapunov docs/results/data 100 100 8 1.0 0.25 2 20261105 "" tangent   # 11 min
cd crates/fring-analysis/python && uv sync && uv run python plots.py     # figures, needs docs/results/data
```

Figures are in `docs/results/figures/` as PNG (for the note) and `.vl.json` (Vega-Lite spec with data, for the web explainer). Binned map CSV files (`*_map.csv`) are not committed (rebuilt by the `map` command; the specs contain their data).
