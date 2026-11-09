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

## Open: failing tests (not relaxed, reported to A0)
1. `gpu_arc` `arc_run_is_the_full_ring_run_rotated_by_whole_widths`, mode D (5 degree arc): arc against rotated full ring 4.25e-3 km, bound 4 x floor = 4 x 9.17e-4 = 3.67e-3 km (ratio 4.6; was 2.5 before). Mode D+ passes (9.1e-5 vs 2.2e-4). Mode D is not for science.
2. `chaos_done_when` (G3): D+ FTLE quantile 90 % differs by 0.058, limit 2 x baseline 0.0184; C max |dL| 0.341, limit 0.3. Before ICR-011 (commit 3ebacb9, both arms about GM) the same test gave 0.013 and 0.131. Same class 100 % in all modes, so the chaos classification agrees; these are tail statistics of one 1000-particle sample with a new trajectory set.

## Next step (one concrete action)
A0 decides on the two failures above (tolerance by ADR, or a new baseline sample). Then A7: WH-EFF-WEB (`GpuStepper::new_with_drift` with the run a0).

## How to verify the current state
```sh
cargo test -p fring-gpu --release --lib                                  # 24 passed
cargo test -p fring-gpu --release --test gpu_wh --test gpu_wh_eff -- --ignored --nocapture
cargo test -p fring-gpu --release --no-fail-fast -- --ignored --test-threads=1   # all pass except the two above
cargo test -p fring-cpu --test golden
```
Logs: `runs/logs/wheffgpu-*.log` (git-ignored).

## Open questions / blockers
- The two failures above.

## Decisions made
- ICR-011 (accepted). A0 approved the 12-vec4 layout and the `fring-cpu` removal in the go-ahead.
