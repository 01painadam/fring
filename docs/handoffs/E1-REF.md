# E1-REF: IAS15 reference for encounters with Prometheus
Role: A6  Branch: ws/a6-e1-ref  Status: review
Updated: 2026-11-08

## Goal and acceptance criteria
PLAN E1-REF: REBOUND (IAS15) reference for particles that pass within 1 Hill radius of Prometheus, from the specification only. Done when reference files and a comparison script are in `validation/`. ICR-004 section 6, "E1-REF (A6)".

## Done
- `src/fring_ref/encounter.py`: geometric elements, IAS15 runs with the two moons, closest approach, Hill-frame states.
- Layer 1: `scripts/e1_ref.py` (2,000 encounters, 500 per stratum of measured b, 2-day window, IAS15 epsilon 1e-9 and 1e-11).
- Layer 2: `scripts/e1_l2.py` (300 particles, 1 year, IAS15, closest approach to Prometheus, geometric Delta a, ADR 0008 MEGNO class).
- Comparison: `scripts/e1_compare.py layer1|layer2` (limits and power check of ICR-004).
- Stand-in: `scripts/e1_standin.py` (REBOUND WHFast). **It is not the Rust CPU backend.**
- Tests: `tests/test_encounter.py`, `tests/test_e1_compare.py`, `tests/test_e1_check.py` (16 tests in the suite, about 15 s).

## Next step (one concrete action)
A0: assign a Rust runner (A2 or A5) that reads `reference/e1_l1_states.csv` and `reference/e1_moons_t0.csv`, runs each id for 172,800 s from `t0_s`, and writes the WH result file below; then run `e1_compare.py layer1 --on <on.csv> --off <off.csv>`. Same for layer 2.

## How to verify the current state
```sh
cd validation && uv run pytest -q                       # 16 passed
cd validation && uv run python scripts/e1_ref.py check  # layer 1: sha256 of files, recompute 2 chunks (3 s), "OK"
cd validation && uv run python scripts/e1_l2.py check   # layer 2: sha256, recompute chunk 0 for 20 days (9 s), "OK"
```

## File contract (the WH side)
All files: `#` comment lines, then numeric CSV. Frame: Saturn-centred, Saturn-equator inertial frame of `params/moons.toml`. Units km, km/s, s.
- **Inputs from A6:** `reference/e1_moons_t0.csv` (Prometheus and Pandora at `t0 = j x 10 d`, j = 0 to 35, from IAS15 as V2; the Rust code may use its own ephemeris at `t0`), `reference/e1_l1_states.csv` (`id,stratum,t0_idx,t0_s,x,y,z,vx,vy,vz`: state at `t0_s` seconds after the epoch), `reference/e1_l2_states.csv` (`id,x,y,z,vx,vy,vz` at t = 0, the epoch).
- **Layer 1 WH result:** the state of every id at `t0_s + 172800 s` (2 days): `id,x_km,y_km,z_km,vx_km_s,vy_km_s,vz_km_s`. The script computes the geometric elements before and after with `fring_ref.encounter.geo_elements`, so IAS15 and WH go through the same function.
- **Layer 2 WH result:** `id,min_prometheus_km,x_km,y_km,z_km,vx_km_s,vy_km_s,vz_km_s,megno_mean`: closest distance to Prometheus over the run, final state at the end of the run (`days` in `reference/e1_l2.json`: 365.25), and the mean MEGNO of the run's chaos indicator (ADR 0008).
- Run with sub-stepping on: `--on`. Without: `--off`.

## Results

### Geometric elements (replaces Renner and Sicardy 2006; see Departures)
- Round trip with `epicycle_states` (2,000 states, a 139,500 to 141,000 km, e 0 to 0.0024): max |a - a_in| = 1.62 km (the first-order epicycle map has an O(a e^2) term: a e^2 = 0.81 km at e = 0.0024; the test bound is 2.5 a e^2), max |e - e_in| = 5.8e-6, max phase error 9.3e-4 rad. These are errors of the input map, not of the elements.
- Flatness in IAS15 with the moons off, one particle, 2 days: e = 0.0024: spread (max - min) of the geometric a 1.5e-10 km (standard deviation 2.4e-11), of the osculating a about Saturn's GM 6.2 km (std 2.2 km); e = 0.001: 1.2e-10 km against 2.6 km. Geometric e varies by 3.5e-13.
- The smallest |Delta a| of layer 1 is far above 10 times this floor (1e-9 km used): per stratum 0 encounters have |Delta a_IAS15| < 1e-8 km. Median |Delta a_IAS15| per stratum: 141, 84, 36 and 13.6 km. These are larger than the 0.1 to 1 km estimated before the work: Delta a of the pass is about 2 a Delta v / v with Delta v = 2 GM_m / (b u) = 1.6e-3 km/s at b = 5 R_H, u = 0.04 km/s, i.e. 27 km.

### Layer 1 (`reference/e1_l1_*`, `reports/e1_l1_distribution.*`)
- R_H = 63.300 km (|r| at the epoch times (GM_m/3GM)^(1/3), `hill_radii_km`); constants from `params/` only. Start: Prometheus's position plus x r_hat (|x| 0 to 10 R_H, drawn per stratum) plus 10 R_H along track, upstream; the real separation at the start is 10.0 to 12.8 R_H. e uniform in 0 to 0.0024, random phase; 36 moon epochs t0 = 0, 10, ..., 350 d (inside the first year after the epoch). Seed 20261108.
- Acceptance: measured b (Hermite minimum of 600 s samples) in the stratum and 1e-3 R_H from its edges; the minimum is inside the window; the distance at the end is above 3 R_H. Counts 500, 500, 500, 500 (checked from the reference run).
- Reference error, IAS15 epsilon 1e-9 against 1e-11, `|Delta a(1e-9) - Delta a(1e-11)| / |Delta a(1e-11)|` (median / p95 / max): [0.68,1) 3.0e-13 / 3.1e-12 / 1.5e-10; [1,2) 0 / 2.1e-12 / 1.4e-11; [2,4.1) 0 / 8.2e-12 / 5.1e-10; [4.1,6.3] 0 / 3.6e-11 / 2.1e-10. Delta e: max difference 1.2e-11. Apse: max difference 1.9e-6 rad. The reference error is 9 orders below the limits (5 % and 25 %).
- Distribution against `closest_hist.csv` (KS of measured b inside each stratum against the run's density): p = 0.86, 0.017, 0.50, 0.0054 (D = 0.027, 0.069, 0.037, 0.077). Medians of b agree within 0.03 to 0.10 R_H. What differs: the synthetic set draws b nearly uniformly inside each stratum; the run density rises with b a little more steeply in [1,2) and less in [4.1,6.3]. The shape is close, not equal. **u not checked: no per-encounter data in docs/results/data.** (Synthetic u at the closest sample: 5/50/95 % = 0.020 / 0.042 / 0.076 km/s.) The eccentricity range 0 to 0.0024 is the ICR value; I did not compare it with `a5_closest_ecc.csv`, which holds counts per a bin and no e of single encounters.
- Moons: IAS15 from the epoch, epsilon 1e-9, as V2. Rust against V2 over 20 years: max 5.9e-4 km (`docs/gates/G1.md`, C3 + V2), so at any t0 in the first year the difference is at most that.

### Stand-in compare (layer 1; REBOUND WHFast, not the Rust backend)
| WHFast step | stratum | median | p95 | max |
|---|---|---|---|---|
| h/16 ("on") | [0.68,1) | 3.5e-6 | 2.8e-5 | 7.9e-4 |
| h/16 | [1,2) | 6.4e-7 | 5.0e-6 | 3.7e-4 |
| h/16 | [2,4.1) | 5.0e-8 | 5.3e-7 | 4.3e-5 |
| h/16 | [4.1,6.3] | 3.5e-8 | 4.6e-7 | 3.3e-6 |
| h = 356.2537 s ("off") | [0.68,1) | 9.0e-4 | 7.2e-3 | 0.19 |
| h | [1,2) | 1.6e-4 | 1.3e-3 | 0.094 |
| h | [2,4.1) | 1.3e-5 | 1.4e-4 | 0.011 |
| h | [4.1,6.3] | 8.8e-6 | 1.2e-4 | 8.5e-4 |
| 4 h = 1425 s | [0.68,1) | 2.1e-2 | 0.23 | 3.9 |
| 4 h | [1,2) | 2.7e-3 | 2.8e-2 | 1.5 |
| 4 h | [2,4.1) | 2.1e-4 | 2.2e-3 | 0.18 |
| 4 h | [4.1,6.3] | 1.4e-4 | 1.9e-3 | 0.014 |
| 16 h = 5700 s | [1,2) | 0.18 | 1.03 | 38 |
| 16 h | [2,4.1) | 3.2e-2 | 0.36 | 5.5 |
| 16 h | [4.1,6.3] | 1.8e-2 | 0.41 | 5.7 |

**Finding: with this definition of Delta a the power check has no power at the run's step.** The "on" stand-in passes the limits by 4 orders. The "off" stand-in at h = 356 s has p95 of 0.7 % at most (the limit to exceed is 25 %), so the script prints "NO POWER: sub-stepping is not needed for these strata". The check only gains power at a step 16 times larger. Reason: Delta a of the whole pass is large (14 to 140 km, set by the impulse of the pass) and a second-order scheme gets the impulse right to 1e-3 at h u / d = 0.2; the errors that ICR-004 aims at (energy, Delta C_J at 1e-6 km^2 s^-2) are tiny in comparison with Delta a. This is the result for REBOUND WHFast; the Rust CPU backend may differ, so A0 must run the real one before drawing a conclusion. Delta e and the apse angle show the same ratio (see the columns of `e1_compare.py`).

### Layer 2 (`reference/e1_l2_*`)
- Cost: 20 particles for 30 days with 2 workers took 23.3 s per chunk of 10 (machine load average 20 to 26); this extrapolates to 280 s per chunk for 1 year, so 30 chunks on 5 workers about 28 min. Decision: 300 particles for 1 year (not 100 for 6 months). Measured: 1469 s (24.5 min wall, 5 workers; chunk times 215 to 336 s).
- Particles: guiding radius uniform in 139,600 to 140,050 km (my choice for "below 140,050 km"; 139,600 km is the lower bound of the ADR 0008 sweep), e uniform in 0 to 0.0024, random phases, z = 0; seed 20261109. Sampling every 593.756 s (Hermite) for the closest approach; MEGNO of ADR 0008 (shadow 1 km radial, renormalised every K = 100 steps of 356.2537 s).
- IAS15 result: closest approach to Prometheus below 1, 3, 6.3 R_H: 24, 65, 153 of 300. Chaotic (mean Y >= 5): 33.7 %; regular (Y <= 2.5): 25.7 %; uncertain 40.7 %. Delta a (geometric) 5/50/95 %: -85.8, 0.002, +64.9 km.
- Stand-in (WHFast, h = 356.2537 s, not the Rust backend): counts 23, 63, 154 (limits +-4.9, +-8.1, +-12.4); KS of Delta a D = 0.020, p = 1.0; chaotic fraction 0.337 against 0.337 (3 sigma = 0.082). All pass. Stand-in WHFast at h/16 (23 min wall for 300 particles, 1 year): counts 24, 65, 153 (equal to IAS15), KS D = 0.013, p = 1.0, chaotic fraction 0.337. Layer 2 does not separate "on" from "off" for this stand-in either (no power, as in layer 1). The stand-in files are in `runs/` (git-ignored); regenerate with `e1_standin.py`.
- No epsilon convergence study for layer 2 (chaotic; statistics only). The layer 1 reference error covers the IAS15 integrator.

## Open questions / blockers
- A Rust runner that reads the states and writes the WH result files is not written (A0 assigns it).
- Power check: see the Finding above. If A0 wants a sharper test of the sub-stepping, the metric must change (for example the error of Delta C_J, ICR-004 test D, or a longer start distance); this needs an AJ decision because the limits are his.

## Decisions made (link ADRs)
- No ADR.

## Departures from ICR-004 and the task text
1. **Elements.** Renner and Sicardy 2006 was not available to me. I implemented geometric elements that have the same purpose: a from the z angular momentum (a^2 n(a) = L_z, exactly conserved in the axisymmetric field, so flat to 1.5e-10 km), e from the energy excess over the circular orbit (E - E_circ(a) = 0.5 kappa^2 a^2 e^2), and the apse longitude from the epicycle inverse map. n and kappa are the exact J2, J4, J6 frequencies of PLAN 7.2. If A0 can give the paper's equations, `geo_elements` is the only function to replace.
2. **Apse angle.** `dvarpi_rad` includes the secular apse precession of about 0.09 rad in 2 days, the same for every method. It is reported, not a criterion.
3. **Contract.** The WH side delivers states at the end of the window, not Delta values, so that one function computes the elements.
4. **Window.** The window is exactly 2 days (ICR). For passes at large b the particle is up to 160 R_H from Prometheus at the end; the start is 10.0 to 12.8 R_H.
5. **Layer 2 counts criterion.** "Within sqrt(N) sampling error" is applied as `|N_WH - N_IAS15| <= max(1, sqrt(N_IAS15))`.
6. **Chaotic fraction.** Chaotic means mean Y >= 5 (ADR 0008); the IAS15 side runs its own shadow particle (IAS15, not the WH kernel). The WH side must give `megno_mean` from the run's own indicator.
7. **Closest approach.** IAS15 side: Hermite on 593.756 s samples (layer 2) or 600 s (layer 1). The WH side's definition (per drift segment, ICR-006) is slightly different; differences matter only for the counts near the 1 R_H edge.
