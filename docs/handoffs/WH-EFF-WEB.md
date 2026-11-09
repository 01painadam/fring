# WH-EFF-WEB: the web drifts about mu_eff (ICR-011, A7 item)
Role: A7  Branch: ws/a7-wh-eff-web  Status: review
Updated: 2026-11-09

## Goal and acceptance criteria
`fring-web` builds the Saturn field with `with_drift_radius(a0)` (ICR-011), so the web shows the same dynamics as the CPU and GPU runs; host tests pass; the web dynamics run in `GpuStepper`, so it uses `GpuStepper::new_with_drift`.

## Done
- `live.rs`: `Setup` field is `SaturnField::from_params(..).with_drift_radius(ring.semi_major_axis_km)`. New `StepperSettings` and `Setup::stepper_settings()` (STEP_S, BLOCK_STEPS, `drift_radius_km: Some(a0_km())`).
- `web.rs`: `make_stepper` calls `GpuStepper::new_with_drift` with `StepperOptions` (Mode::Dp, no chaos) built from `stepper_settings()`.
- a0 for the web = `Setup::a0_km()` = 140222.43 km (`params/ring.toml`), as A0 decided. Against the run a0 of 140220: delta eps = -2 eps da/a = -1.59e-7 (absolute), negligible.
- Host test `web_field_and_stepper_use_mu_eff_at_a0`: settings carry a0; excess in (4.5e-3, 4.7e-3) and equal to `with_drift_radius`; `n0_rad_s` and the initial states (Core, two circular bands, 500 particles) are bit for bit those of a `from_params` field.
- Ignored GPU test `web_stepper_with_mu_eff_keeps_the_circular_orbit_at_a0` (`tests/live_gpu.rs`).

## Measurements (M4, Metal, release; one circular orbit at a0, no moons, 87,660 steps of 360 s = 1 Julian year)
- Along-track error against n(r) t: mu_eff +0.0101 km; GM drift -13.0565 km (ICR-011 T1: about 12.8 km and 0.0001 km).
- Bounds (M27): mu_eff arm below 0.1 km (10 times the measurement); GM arm above 10 km (T1 limit, margin 1.3).
- wasm size 942,859 bytes (W2-DROP 916,957; +25,902, +2.8 %; includes other merged changes to fring-gpu and fring-model since W2-DROP).

## Deviation from the plan
`fring-gpu` is a wasm32-only dependency of the library (`Cargo.toml`), so `StepperOptions` is not visible on the host. I did not change `Cargo.toml`. `Setup::stepper_settings()` returns a plain struct with the numeric fields; `web.rs` adds Mode::Dp and `chaos: None`. The host test cannot check those two.

## Next step (one concrete action)
A0: review; run `?bench` in Chrome (`http://localhost:8080/?bench&counts=1e5&modes=off,side,overlay`, earlier value mean 60.0 fps; stall check `counts=1e4,3e5&modes=off,side`).

## How to verify the current state
```sh
cargo test -p fring-web                          # 35 lib tests pass
cargo test -p fring-web --release -- --ignored --nocapture   # GPU tests; prints the two drift numbers
cargo check -p fring-web --target wasm32-unknown-unknown
apps/fring-web/scripts/build.sh                  # wasm size: 942859 bytes
```

## Open questions / blockers
- Browser `?bench` check is not done (A0 runs it).

## Decisions made
- ICR-011 (accepted). a0 = `Setup::a0_km()` (A0, 2026-11-09).
