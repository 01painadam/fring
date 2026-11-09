# E1-DIAG: where Rust WH at h differs from h/4 in layer 2
Role: A5  Branch: ws/a5-e1-diag  Status: review
Updated: 2026-11-08

## Goal and acceptance criteria
Find why the Rust CPU WH at the production step h (356.2527 s) has about 5 more of 300 particles inside 3 and 6.3 R_H than at h/4 (70 / 158 against 65 / 153; inside 1 R_H 29 against 25.6 to 26), while IAS15 gives 24 / 65 / 153 and REBOUND WHFast at h gives 23 / 63 / 154. Budget about 1 hour. No change in `fring-cpu` or `fring-model`.

## Done
- `fring-e1` got `--l2-div m` (m times as many steps of h/m, so that h and h/4 share their sample times exactly), `--l2-trace-ids i,j,k` and `--l2-trace-every n` (writes `trace_l2.csv`: state and Prometheus position every n steps of h; default 243 = 1.002 d). Tracing does not change the result (`cmp`).
- Scripts in `apps/fring-run/scripts/`: `e1_diag_counts.py` (step 1), `e1_diag_series.py` (steps 2, 3), `e1_diag_final.py` (end-state statistics). They read `runs/diag/` and use `validation/src/fring_ref` read-only.
- Element definition for all series: the geometric elements of E1-REF (`fring_ref.encounter.geo_elements`): a from L_z, e from the energy excess, mean longitude lam and pericentre longitude varpi from the epicycle inverse map. The same function for both runs.
- WHFast stand-in run: `validation/scripts/e1_standin.py layer2 --div 1 --out runs/standin_l2_h1.csv` (end states only, so no time series for it).

## Results (logs in `runs/logs/e1-diag-*.log`, git-ignored)
### 1. Counts and ids
Counts inside 1 / 3 / 6.3 R_H: IAS15 24 / 65 / 153; WHFast h 23 / 63 / 154; Rust h 29 / 70 / 158; Rust h/4 26 / 65 / 153.
Ids that differ between the four (some method inside, some outside): 25 ids at 1 R_H, 25 at 3 R_H, 13 at 6.3 R_H (50 distinct in total, listed with distances in `runs/logs/e1-diag-table.log`).
- Inside 1 R_H at h but not at h/4 (12): 23, 101, 151, 176, 191, 195, 218, 236, 248, 262, 263, 297. The reverse (9): 38, 54, 103, 108, 115, 179, 193, 206, 234.
- 3 R_H: h only 13 (26, 40, 60, 80, 94, 118, 151, 160, 164, 176, 183, 246, 297); h/4 only 8.
- 6.3 R_H: h only 8 (25, 80, 127, 228, 231, 280, 285, 291); h/4 only 3 (68, 148, 294).
- Many of these have IAS15, WHFast and Rust h/4 agreeing on one closest distance (101: 135.3, 135.2, 135.2 km) and Rust h on another (7.0 km); 176: 232.9, 232.0, 232.6 against 29.8; 262: 188.3, 188.4, 188.6 against 43.4; 297: 221.2, 221.2, 221.1 against 44.5. So h is the odd one out. A smaller number go the other way (54, 234: IAS15 and WHFast 170 and 190 km, h 171 and 187, h/4 40 and 36).
- Controls (10 regular ids, all four agree within 2 % on the closest distance, MEGNO < 2.5, smallest distances): 230, 293, 62, 265, 253, 17, 290, 47, 2, 102.

### 2. The error is large in the whole ensemble, and it scales as h^2
Median |a_final - a_IAS15| (geometric a, km), by IAS15 closest approach (every particle comes within 1500 km of Prometheus in the year):
| IAS15 closest approach | n | WHFast h | Rust h | Rust h/4 |
|---|---|---|---|---|
| 500 to 1500 km | 93 | 0.002 | 0.069 | 0.004 |
| under 500 km | 207 | 0.307 | 3.919 | 0.373 |
All 300: 0.082 / 0.840 / 0.052. Particles with |a_final - a_IAS15| > 5 km: WHFast 49, Rust h 102, Rust h/4 45. Rust h/4 against Rust h: 17 times smaller (h^2 gives 16). Rust h is 13 to 35 times worse than WHFast at the same h.
Heavy tail: id 132 has a closest approach of 33 km in IAS15, WHFast and h/4, but 4.6 km in Rust h (3.7 km with sub-stepping), a final a of 136,627 km (a change of 3080 km) and e = 0.022. It alone doubles the sum of e^2 (1.16e-3 against 0.63e-3) and the final standard deviation of a (254 km against 163 to 171 km).

### 3. When do h and h/4 separate
From daily series of 50 differing ids and the 10 controls (`e1-diag-table.log`):
- Before the first pass the two runs differ only in along-track phase: lam(h) - lam(h/4) = -2 to -5 km at day 100 to 180 (-0.03 km at day 1), growing at 12 km per year or less, with da under 0.001 km and de about 1e-7. Rates before the separation: dvarpi mostly -0.3 to +0.6 deg/yr (it is noise of +-0.02 deg for small e), dlam -0.8 to +1.0 deg/yr for the ids with a pass, and for the 10 controls dvarpi -0.12 to +0.26 deg/yr (230 with a 500 km pass: 0.92) and dlam -0.006 to +0.043 deg/yr (230: -0.34). Controls keep |da| under 0.16 km all year (230: 3.2 km), so no secular drift between passes.
- All 47 ids with a separation (|da| > 1 km) have it at a pass of Prometheus. The pass distances, from a 3-step trace: id 224 closest 78.1 km (h) and 80.7 km (h/4), Delta a of the pass -39.0 and -37.1 km; id 248: 109.4 and 111.8 km, -106.0 and -103.4 km; id 262: 192.4 and 188.6 km, +59.1 and +61.1 km; id 87: 198.2 and 190.7 km, -3.7 and -6.1 km. The pass changes Delta a (h minus h/4) from under 0.001 km to -1.9, -2.7, -2.0 km (about 5 %, 2.5 %, 3 % of the kick; for id 87 from -0.02 to +1.5, then +2.4 km). e jumps by 1e-5, varpi by 0.2 to 0.8 deg.
- So: no jump inside 1 R_H as such; the runs separate at passes of 1.2 to 3 R_H (78 to 200 km), and the cause is already in the few-km phase offset before the pass.
- The matched pairs show no bias in the closest distance itself: median(h - h/4) = -0.08 km, mean +0.04 km, 56 % of the pairs have h < h/4 (106 pairs, both under 400 km, within 20 km). WHFast against h/4: median 0.03 km. Counts under 40 / 63 / 100 / 200 km: IAS15 16 / 24 / 31 / 70, WHFast 17 / 23 / 34 / 68, Rust h 18 / 29 / 37 / 73, h/4 19 / 26 / 35 / 69. So h puts a few more particles on small-distance orbits at all radii; it does not shift every b.

### 4. Hypothesis with numbers
1. At h the Rust WH has a lam drift of up to 12 km per year relative to h/4 (0.005 deg/yr, equal to a mean-motion error of 1.2e-10 of n per year scale; 1 km = 0.0004 deg at a = 139,700 km). It builds up to 2 to 5 km of along-track offset at the first pass.
2. At a pass with b = 80 to 200 km the along-track offset of 2 to 3 km changes the impact geometry by 1 to 4 %, so the kick differs by 2.5 to 5 % (measured: 2 to 3 km of 39 to 106 km). An error in Delta a of 2 to 3 km at one pass changes the phase (n' Delta a: 1 km of a gives 0.22 deg per year, 60 km along track per year) and the geometry of the next pass.
3. Each ring particle has many passes inside 1500 km (every one of the 300 has one in a year), so the changes pile up: 102 particles end more than 5 km from IAS15 at h, against 45 at h/4. In the close ones (|b| < 1.5 R_H) the kick is large (hundreds of km), so a few percent changes whether a later pass is inside 1 R_H. The net count changes by +3.4 +- 0.16 (E1-RUN study), a few particles near the boundary. Particle 132 is an extreme case.
4. Sub-stepping removes only 0.3 because it acts when h v/d is above 0.1, which means only at d under about 100 km, and the error is set by the earlier distant passes and by the phase offset, not by the close pass itself.
Not found: why Rust h is 13 to 35 times worse than WHFast h. WHFast has the same drift-kick structure for the J terms. Differences that could matter (not tested, would need `fring-cpu` or `fring-model` changes or a variant run): the moons are exact in Rust but WHFast integrates them with the same step; the direct and indirect moon kick is evaluated at the mid-step particle position with the moon at the mid-step time (`crates/fring-model/src/physics/wisdom_holman.rs` lines 23 to 35 for the structure, `moon_kick.rs` for the kick); the Kepler drift is about Saturn's point mass and J2 (1.6e-2 of GM) is in the kick (`wh_step`, line 22), where the shadow Hamiltonian error is O(J2 (n h)^2) = 3e-5 of n, which would be a mean-motion error common to all particles (measured lam drift is much smaller, so this is not it unless it cancels in h minus h/4 by the geometric lam definition).

## Next step (one concrete action)
A0: decide. Two experiments (A2, `fring-cpu`) would localise the 13 to 35 times larger error against WHFast: (a) run layer 2 with the moon kick only on Prometheus, and with the indirect term off, at h and h/4; (b) measure the lam drift of one regular particle with no moons at h and h/4 against the exact J2 to J6 frequencies. I did not do these (no `fring-cpu` change allowed here).

## Recommendation
Do not change the E1 limit. For science runs near Prometheus (A3-ANALYSIS, B2) use h/2 or h/4 for the region where passes inside 3 R_H matter (cost 2 to 4 times), or accept a bias of +5 / +5 / +4 particles of 300 in the counts and a heavy-tail risk (one particle in 300 gets a 3000 km kick). Fixing the integrator is possible only after experiment (a) or (b) finds the term. Sub-stepping does not help.

## How to verify the current state
```sh
cargo build --release -p fring-run --bin fring-e1
target/release/fring-e1 layer2 --variants plain --out runs/diag/h -q
target/release/fring-e1 layer2 --variants plain --l2-div 4 --l2-stride 4 --out runs/diag/h4 -q
(cd validation && uv run python scripts/e1_standin.py layer2 --div 1 --out ../runs/standin_l2_h1.csv)
uv run --project validation python apps/fring-run/scripts/e1_diag_counts.py   # step 1 (needs runs/diag/ids.txt for the others)
```
Traces: `--l2-trace-ids $(cat runs/diag/ids.txt)` (ids of step 2: the 50 differing ids, then the 10 controls) for h and for `--l2-div 4 --l2-stride 4`; then `e1_diag_series.py` and `e1_diag_final.py`.

## Open questions / blockers
- The cause of the factor 13 to 35 against WHFast (above).
- The WHFast stand-in gives only end states, so its lam and varpi drift is not measured.

## Decisions made (link ADRs)
- None.
