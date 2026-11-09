# WH-EFF-CPU: Wisdom-Holman drift about mu_eff (ICR-011), physics and CPU
Role: A2  Branch: ws/a2-wh-eff-cpu  Status: review
Updated: 2026-11-09

## Goal and acceptance criteria
ICR-011 in `fring-model/src/physics/` and `fring-cpu`: every WH drift is a Kepler drift about mu_eff = n(a0)^2 a0^3, the kick adds (mu_eff - GM) r/|r|^3. T1 (one-year along-track error under 0.5 km), T2 (existing tests, bounds unchanged), T3 (E1 layers 1 and 2 at h), new CPU golden hashes. A0 reviews the physics.

## Done
- `SaturnField` has `drift_gm_km3_s2` and `drift_gm_excess` (epsilon). `from_params` sets (GM, 0). `with_drift_radius(a0)` sets both from one f64 computation. `kick_acceleration_saturn(p)` is the J terms plus (drift_gm - GM) r/|r|^3 in one pass (`accel_with(p, point = -epsilon)`). For `from_params` it is bit-identical to `acceleration_without_point_mass` (unit test, `to_bits`).
- `wh_step`, `wh_steps`, `wh_steps_moons`, `kick_acceleration` and the sub-stepping drift (`extended.rs`) use the field's drift GM. The MEGNO shadow goes through the same code.
- `CpuBackend::new` builds the field with `with_drift_radius(frame.reference_semi_major_axis_km)`. `CpuBackend::with_point_mass_drift` (transitional, remove in WH-EFF-GPU) keeps the old field.
- Two struct literals got the new fields: `crates/fring-model/tests/kepler_frames.rs` (now `..from_params`) and `crates/fring-gpu/tests/gpu_collide.rs` (explicit fields).
- Five GPU-against-CPU tests (`gpu_backend`, `done_when`, `chaos_done_when`, `gpu_vs_cpu_loss`, `gpu_backend_collide`) use `with_point_mass_drift`. A3 switches them back in WH-EFF-GPU.
- CPU tests that compare the backend with a serial loop build the loop's field with `.with_drift_radius(140_220.0)` (`backend.rs`, `substep.rs`, `threads.rs`, `v3.rs`). No bound changed.
- Golden hashes (debug build): `0x445bbb7be1df7a8b`, `0x940a556a46c491a9`, `0x8ca83887bf9b1a61`. The old hashes still hold for `with_point_mass_drift` (test `point_mass_drift_keeps_the_old_hashes`).
- T1 `crates/fring-cpu/tests/wh_eff.rs`: mu_eff -0.0717, +0.0001, +0.1088 km at 139,500, 140,220, 141,500 km; GM -13.48, -12.79, -11.94 km. Runs in 0.15 s.
- T3: layer 1 OK; layer 2 OK, 24 / 65 / 153 inside 1 / 3 / 6.3 R_H (IAS15 24 / 65 / 153). Logs in `runs/logs/wheff-e1-*.log` (git-ignored).

## Next step (one concrete action)
A0 reviews the physics and merges. Then WH-EFF-GPU (A3), and WH-EFF-WEB (A7: `fring-web/src/live.rs:211`).

## How to verify the current state
```sh
cargo test -p fring-model --lib gravity            # 4 passed
cargo test -p fring-cpu --test wh_eff -- --nocapture   # mu_eff errors 0.07 / 0.0001 / 0.11 km, GM > 10 km
cargo test -p fring-cpu --test golden              # 2 passed (debug)
cargo build --release -p fring-run --bin fring-e1 && target/release/fring-e1 all --variants plain --out runs/e1 -q
cd validation && uv run python scripts/e1_compare.py layer2 --run ../runs/e1/plain/l2_wh.csv   # layer 2: OK
```

## Open questions / blockers
- The release golden test (board task GOLDEN-REL) was not touched here.

## Decisions made (link ADRs)
- ICR-011 (accepted by AJ 2026-11-09). Two fields instead of one so that an f32 field keeps epsilon accurate (A0 agreed).
