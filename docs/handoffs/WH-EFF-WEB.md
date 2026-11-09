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

## Finding 2026-11-09: the live ring was not drawn because Chrome rejects arc.wgsl (not this change)
- Console (headless Chrome, CDP), same on the public demo (main) and on this branch: `Error while parsing WGSL: :1081:31 error: value nan cannot be represented as 'f32'` at `store(i, V3(vec2<f32>(bitcast<f32>(0x7fc00000u), 0.0), p.y, p.z), v);` (`crates/fring-gpu/shaders/arc.wgsl:216`, `t_arc_wrap`, included into module "wh"). Pipelines `step_dp` and `step_dp_r` are invalid, so every submit with a step pass is dropped. Paused frames have no step pass, so they draw. A3 fixes it on `ws/a3-wgsl-nan`.
- Why no native test can reproduce it: naga (native wgpu) accepts a constant NaN; only Tint (Chrome) rejects it. So the regression guard is at run time: `App::new` installs `device.on_uncaptured_error`; `frame` turns the first message into `Phase::Failed(gpu_error_text(msg))`; the page shows "The GPU rejected its program" with the WGSL text; a bench cell then reads "not run: <error>". Host test `first_gpu_error_becomes_the_failure_text` covers the mapping (first error kept, text, cut length). Browser check (CDP screenshot of localhost:8080 with the old arc.wgsl) shows the message.
- #86 needs main with the A3 fix merged in before AJ benches again.

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
- Finding for A3 (GPU behaviour, not web; A0 request 2026-11-09). Through `GpuStepper::new_with_drift` (Mode::Dp, no moons, mu_eff at a0 = 140222.43 km) the along-track error after one year at a0 is about +0.009 km, not the +0.0002 km of `gpu_wh_eff` (D+, 356.25 s, one `kernels.apply`). Scratch runs, T1-style start (position (a0,0,0), velocity `circular_speed(a0)`, n at a0), M4/Metal, release; not committed:

  | step h (s) | block_steps | steps | mu_eff (km) | GM (km) |
  |---|---|---|---|---|
  | 360 | 128 | 87,660 | +0.0097 | -13.0565 |
  | 360 | 4096 | 87,660 | +0.0094 | -13.0558 |
  | 356.25 | 128 | 88,582 | +0.0088 | -12.7850 |
  | 356.25 | 4096 | 88,582 | +0.0087 | -12.7847 |

  `MAX_BLOCK_STEPS` is 4096, so one block of 88,582 steps is not possible; 4096 is the largest. Conclusions: (1) the block size changes the result by 0.0003 km at most, and the step by 0.001 km, so neither explains the factor 40 to 50; (2) the GM arm agrees with T1 after h^2 scaling (-12.7847 against T1 -12.8), so the harness and reference are right; (3) the `Band::Circular` start (r0 - a0 = -6.3e-4 km, circular speed at r0, n at r0) gives +0.0101 km, so the start is not the cause either. Not tested, for A3: the T1 test uses mode D+ and the web uses Dp; T1 calls `kernels.apply` with a zero-filled moon table, the stepper fills its own rows; a0 = 140222.43 here against 140220 there (this should change the error by about 2e-4 km). The size, +0.009 km, is below the O(h^2) split error at the band edge (0.07 to 0.11 km, ICR-011), so the web test bound (0.1 km) is unchanged.

## Decisions made
- ICR-011 (accepted). a0 = `Setup::a0_km()` (A0, 2026-11-09).
