# WH-EFF-GPU: Wisdom-Holman drift about mu_eff on the GPU (ICR-011 item 5, T4)
Role: A3  Branch: ws/a3-wh-eff-gpu  Status: review
Updated: 2026-11-09

## Goal and acceptance criteria
ICR-011 in `wh.wgsl` (double-single and fast paths) and the Rust twins; test T4; new GPU golden hashes. G1 to G3 tests still pass.

## Done
- `prm` is now 12 vec4 (`PARAM_WORDS` 48): words 0 to 1 mu_eff (DS, drift), 4 to 5 1/mu_eff (DS), 41 to 42 GM (DS, kick), 44 to 45 epsilon (DS), 46 epsilon scaled by h/f32(h) (f32, option D). Epsilon is computed in f64 on the host (`SaturnField::with_drift_radius`).
- `wh.wgsl`: `kepler_c`, `ds_scalars` (so `drift_d`) use mu_eff and 1/mu_eff. `accel_j_c` (C), `kick_dir` (D) and the J2 part of `kick_dir_plus` (D+) fold epsilon into the radial factor: `radial = -eps - a - u b`. J4, J6 of D+ unchanged.
- Twins: `spike::Consts::for_step_drift(p, gms, h, Option<a0>)` (`for_step` and `new` keep epsilon = 0); `spike::ds_scalars`, `wh::kick_dir`, `wh::kick_dir_plus`, `wh::advance` (C), `wh::kepler`, `wh::params`.
- `GpuStepper::new` unchanged (epsilon = 0; the web). `GpuStepper::new_with_drift(gpu, kernels, system, init, StepperOptions { step_s, block_steps, mode, chaos, drift_radius_km })`. `GpuBackend` and its `startup_check` use the config a0.
- Host `field` in `GpuBackend` stays `from_params` (arc geometry, collider, energy): the energy diagnostic uses `specific_energy` (the total force), the start-up check builds `Consts::for_step_drift(.., Some(a0))`.
- Five tests back on `CpuBackend::new`; `with_point_mass_drift` and its golden test removed (`fring-cpu`, one commit).
- `GOLDEN_FULL_RING` (gpu_arc.rs) = `0x4d03_03f6_d3cb_4b5b` (was `0xa999_c5ad_4c72_2112`), M4. `gpu_drop` golden hashes (stepper, epsilon = 0) did not change.
- New `tests/gpu_wh_eff.rs`: GPU T1 and the f32 kick error.

## T4
Twin measure (relative, bound 1e-6), epsilon != 0 after, GM drift before: t_drift_d 2.1e-16 / 3.6e-15; t_kepler_c 8e-16 / 1.7e-15; t_kick_dir_plus 2.3e-12 (same); t_kick_dir 8.6e-11 (was 1.5e-10); t_kick_acc_c / t_accel_j_c 3.7e-11 (was 3e-14: the new radial has two DS terms of 4.6e-3 g; still far below the bound).
Kick error against the f64 kick, in units of h GM/r^2 (200 positions in the band): D 2.5e-9 before, 1.5e-9 after; D+ 4.6e-11 and 4.6e-11; C 2.9e-16 and 1.6e-16. So the f32 kick error does not grow.
GPU T1 (88,582 steps, one year, along-track km at 139,500 / 140,220 / 141,500): D+ -0.0706 / +0.0002 / +0.1034; C -0.0729 / +0.0013 / +0.1096; CPU f64 -0.0717 / +0.0001 / +0.1088. D (reported only): -0.57 / -0.47 / -0.83 (A0 estimated 0.07 per year from rounding of epsilon; the measured D error is larger, about 0.5 to 0.8 km, and was already about 0.5 km off before). Drift about GM: D+ -13.5 / -12.8 / -11.9.
Bounds (M27): D+ and C 0.25 km (largest measured 0.110, margin 2.3; the true O(h^2) split error at the band edge is 0.109); GPU against CPU 0.02 km (largest 0.0054, margin 3.7); GM arm above 10 km.

## Investigation of the two failures (A0 request, 2026-11-09): no bug found
**Split check (a).** Every path goes through the same functions with the same `prm` buffer (main, shadow, `step_d`, `step_dp`, `_r` rule variants, `step_c`, arc kernels in `arc.wgsl` 145 to 156, `chaos_renorm`). Drift: `drift_d` calls `ds_scalars`, which reads `prm_drift_gm()` (words 0, 1) and `prm_inv_drift_gm()` (words 4, 5); `kepler_c` gets `prm_drift_gm()` at wh.wgsl 603, 610 and 655. Kick: `accel_j_c`, `kick_dir` and `kick_dir_plus` read `prm_gm_kick()` (words 41, 42) and epsilon (words 44, 45 DS; 46 f32). No call still reads GM for a drift, and no kick lacks epsilon. `set_moon_radii_km` rewrites the buffer from `prm_words`, which holds the drift words. Twins: `spike::ds_scalars` (drift GM, `inv_gm_d` = 1/drift GM), `wh::kepler`, `wh::advance` C, `wh::kick_*`; `spike.rs` 316 and 563 (options A/B spike twins) keep GM and are not used by `wh.wgsl`.
**Regular orbits (b)** (`tests/gpu_chaos_regular.rs`, no moons, 100 particles, 88,582 steps, chaos on): FTLE per day, CPU median 2.5450e-2, max 2.5505e-2. |GPU - CPU|: D+ median 8.9e-7, max 3.7e-6; C median 5.1e-7, max 2.5e-6; D median 9.6e-6, max 5.1e-5. log growth max 9.307 on CPU, D+ and C (D 9.314). The shadow separation stays at the regular level, so no 12.8 km per year split.
**Large |dL| (c)** (seed 20261102): C max 0.341 is one particle, no. 145 (start r 139,978.9 km, MEGNO -0.37, L = 1.04, a weak regular orbit). The CPU baseline (1e-4 km start offset) moves the same particle by 2.42, so the GPU is 7 times closer to the CPU than the baseline is. The next C values are 0.053 and 0.033. D+ max 0.060 (same particle).
**Spread (d)** three other seeds, FTLE quantile 90 % difference (GPU against CPU), and the baseline (CPU against displaced CPU) which gives the limit:
| seed | tree | baseline q90 | D+ q90 | C q90 |
|---|---|---|---|---|
| 20261102 | GM-GM (3ebacb9) | 0.026 | 0.013 | 0.019 |
| 101 | GM-GM | 0.019 | 0.006 | 0.027 |
| 202 | GM-GM | 0.022 | 0.007 | 0.004 |
| 303 | GM-GM | 0.005 | 0.001 | 0.008 |
| 20261102 | mu-mu | 0.018 | 0.058 | 0.032 |
| 101 | mu-mu | 0.027 | 0.019 | 0.005 |
| 202 | mu-mu | 0.004 | 0.007 | 0.016 |
| 303 | mu-mu | 0.035 | 0.014 | 0.012 |
The rule "q <= 2 x baseline" fails on the old tree too (seed 101: quantile 0, D+ 0.0032 and C 0.0013 against 0.0006) and on the new tree (seed 202: D+ 0.0077 and C 0.0157 against 0.0037). The baseline q90 itself spreads from 0.004 to 0.035. D+ q90 for mu-mu is 0.007 to 0.019 on the three new seeds; 0.058 is the largest of the 8 values (seed 20261102 only). No |dL| rule failed on the six extra runs. So the failure is sampling noise of a rule whose reference (one baseline run) is itself noisy; I changed nothing.
**Arc, mode D, 5 degrees (2).** The arc kernels call the same `drift_d`, `kick_dir`, `kick_dir_plus` with the same `prm`; epsilon (word 46) and mu_eff are used as in `step_d`. Ensembles by `FRING_ARC_SEED` (position difference arc against rotated ring, floor, ratio; bound 4):
| seed | GM-GM | mu-mu |
|---|---|---|
| 0 | 5.47e-3 / 2.20e-3 = 2.5 | 4.25e-3 / 9.17e-4 = 4.6 |
| 1 | 7.47e-3 / 2.29e-3 = 3.3 | 4.22e-3 / 1.18e-3 = 3.6 |
| 2 | 6.66e-3 / 2.33e-3 = 2.9 | 4.42e-3 / 1.48e-3 = 3.0 |
| 3 | 7.13e-3 / 2.06e-3 = 3.5 | 4.17e-3 / 1.04e-3 = 4.0 |
In absolute terms mode D is better with mu_eff (arc against ring 4.2e-3 km, was 5.5 to 7.5e-3). The floor (full ring from a state turned by 0.7 rad) fell by a factor of 1.5 to 2.4, because the f32 radial factor is now small near a0 (the J2 and epsilon terms cancel), so the rotation-dependent rounding is smaller, while the rest of the arc difference did not shrink. The ratio moved from a mean of 3.05 to 3.8; one of four ensembles is above the bound 4 (4.0 is on it). Mode D+ passes everywhere. Mode D is not for science. Decision for A0 (not taken here).

## Next step (one concrete action)
A0 decides on the two failures (see the investigation: no bug found). Then A7: WH-EFF-WEB (`GpuStepper::new_with_drift` with the run a0).

## How to verify the current state
```sh
cargo test -p fring-gpu --release --lib                                  # 24 passed
cargo test -p fring-gpu --release --test gpu_wh --test gpu_wh_eff -- --ignored --nocapture
cargo test -p fring-gpu --release --no-fail-fast -- --ignored --test-threads=1   # all pass except the two above
cargo test -p fring-cpu --test golden
```
Logs: `runs/logs/wheffgpu-*.log` (git-ignored).

## Open questions / blockers
- The two failures (sampling noise of the rules, evidence above).

## Decisions made
- ICR-011 (accepted). A0 approved the 12-vec4 layout and the `fring-cpu` removal in the go-ahead.
