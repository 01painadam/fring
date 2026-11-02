# W2-DROP: the web run goes on when particles are dropped
Role: A7  Branch: ws/a7-w2-drop  Status: review (code done; browser check by AJ not done)
Updated: 2026-11-02

## Goal and acceptance criteria
(PLAN.md section 9, W2-DROP; ICR-006; AJ decisions of 2026-11-02.) The web run goes on when particles are dropped: the radius rule is on (radii from `params/moons.toml`); dropped particles are hidden in the live picture and in a "Set to now" reference (none at t = 0); the page shows the counts. At the end of the moon ephemeris the run pauses and does not restart; Restart starts again at t = 0.
Done when: in Chrome on the Mac mini, a full run at 10^5 particles (default band, and a band close to Prometheus) goes to the end with no stop, shows the counts, and pauses at the end; `?bench` at 10^5 still gives mean 30 fps or more in each compare mode; the host and native tests pass.

## Done
Commits (all on `ws/a7-w2-drop`):
- 22b3f65 `COMPUTE_BUDGET_PARTICLE_STEPS_PER_S` = 5.5e8 (was 6.6e8). Calculation below.
- abd2a5e `moon_radii_for(system, moons_toml)` and `builtin_moon_radii_for(system)` in `live.rs`: radii from `fring_model::params::moon_radii_from_toml_str`, matched by track NAME, an error if a name has no radius. Host test `moon_radii_follow_the_tracks_by_name`: [43.09, 40.63] with Pandora, [43.09] without.
- 65c08c8 `shaders/project.wgsl` binding 3 (`status: array<u32>`); word `8 + i` not 0 gives disc diameter 0. Native test `tests/drop_gpu.rs`: two particles are made unbound, then counts (word 0 = 5, word 1 = 2, words 2 to 7 = 0, reasons), live sizes (0 for the two, 2.5 for the rest), the t = 0 copy (all visible) and a reference copy taken after the drops (same sizes as live). It asserts that some low words are not zero (M24). `project_gpu.rs` and `compare_gpu.rs` only add the binding 3 buffer; tolerances are unchanged.
- 8ac6bfa `web.rs`: `set_moon_radii_km` after `GpuStepper::new` in `make_stepper`; live bind group binding 3 = `status_buffer()`; new `ref_status` buffer (`status_bytes(cap)` = (8 + cap) * 4 bytes), copied in `make_stepper` (t = 0) and in the "Set to now" block, in the same encoder and before `encode_steps`.
- 68cd910 Counts: 32-byte staging buffer (words 0 to 7), 1 s period, an epoch so that a read from an earlier run is ignored, counts zero in `make_stepper`. `failed_index` and its stop are removed. `dropped_text()` gives `Dropped: 12 unbound · 45 inside Prometheus · 3 inside Pandora` (Pandora part left out when Pandora is off); the page shows it in a line under the time readout. Host test `drop_text_shows_the_counts_and_leaves_out_pandora_when_off`.
- 3d28636 End pause: field `ended`, set when at most one step is left (`limit_to_end`, host test `the_run_stops_at_the_end_of_the_moon_data`); only stepping stops. `ended()` for JS (Play is disabled). The message `End of the moon data. Press Restart to start again from t = 0.` is built from the state in `message()`. `make_stepper` (Restart, count, band, preset, Pandora) clears `ended`. A kernel error still gives `Phase::Failed`.

Speed limit (budget 5.5e8 = 0.70 x the rule-on Dp rate 7.86e8, W2-DROP-GPU): h = 5.5e8 x 360 s / 3600 / N = 5.5e8 x 0.1 / N.
- N = 10^5: 550 h/s, shown as 500 (the slider maximum).
- N = 3x10^5: 183.3 h/s.
- N = 10^6: 55.0 h/s.
`speed_limit_follows_the_compute_budget` checks 500, 550/3 and 55.

Definition of done on the last code commit (exit codes read directly, zsh `eval` loop, M25):
- `cargo fmt --all -- --check`: 0
- `cargo clippy --workspace --all-targets -- -D warnings`: 0
- `cargo test --workspace`: 0
- `cargo test -p fring-web --release -- --ignored` (the native GPU tests, including `drop_gpu`, `project_gpu`, `compare_gpu`, `live_gpu`): 0
- `cargo check -p fring-model --target wasm32-unknown-unknown`: 0
- `cargo check -p fring-gpu --target wasm32-unknown-unknown`: 0
- `cargo check -p fring-web --target wasm32-unknown-unknown`: 0
- `cargo clippy -p fring-web --target wasm32-unknown-unknown -- -D warnings`: 0
- `apps/fring-web/scripts/build.sh`: 0, wasm size 916,957 bytes (it needs `~/.cargo/bin` in PATH for `wasm-bindgen`; without it, exit 1 with "wasm-bindgen not found")

## Next step (one concrete action)
AJ: run the browser steps B1 to B5 and both `?bench` commands below in Chrome (not headless) on the Mac mini and write the numbers here. Then A0 reviews and opens the pull request.

## How to verify the current state
Host and native: `cargo test -p fring-web --lib` (31 passed) and `cargo test -p fring-web --release -- --ignored` (needs a GPU).

Browser (AJ):
`apps/fring-web/scripts/build.sh && python3 -m http.server 8080 -d apps/fring-web/dist`, then open `http://localhost:8080` in Chrome.
- B1, full run at 10^5. Default band, speed 500 h/s. Expected: the line "Dropped: ..." shows zeros at t = 0, then the numbers rise. The run reaches the end with no stop. At the end the time stops, Play is disabled and the message "End of the moon data. Press Restart to start again from t = 0." shows.
- B2, band close to Prometheus (for example a min of 139,700 km and a max of 140,000 km), 10^5. Expected: the "inside Prometheus" count rises, the dots vanish, and the run does not stop.
- B3, "Set to now" after drops, then Compare = Side by side or Overlay. Expected: the reference does not show particles that were dropped before the copy. Press Restart: the counts go to zero, the reference goes back to t = 0 (message "The reference was reset to t = 0.").
- B4, at the end (paused). Expected: camera drag and zoom, Compare and "Set to now" still work; Pause or Play do nothing (the button is disabled). Restart, a new particle count, a new band, a new preset and the Pandora switch each start at t = 0 and enable Play; the end message disappears.
- B5, Pandora off. Expected: the line has no Pandora part. Zoom to the maximum on a dot: no jitter at a few km (M24).
- Bench, mean fps: `http://localhost:8080/?bench&counts=1e5&modes=off,side,overlay`. Expected: mean 30 fps or more in each mode (W2-COMPARE gave 60.0).
- Stall re-check from W2-COMPARE: `http://localhost:8080/?bench&counts=1e4,3e5&modes=off,side`. In W2-COMPARE the 3x10^5 side cell had one frame of 6.4 s. Expected: no frame above about 100 ms. If it comes back, the cause is probably not the drop code; report the numbers.

## Open questions / blockers
- None for the code. Not yet measured in a browser: the fps with the rule on, and the stall re-check.
- `AGENTS.md` does not list `cargo clippy -p fring-web --target wasm32-unknown-unknown -- -D warnings` (host clippy does not compile `web.rs`). A0 may add it.
- `build.sh` needs `wasm-bindgen` in PATH (`~/.cargo/bin`); this is not in the Homebrew rustup PATH line of `CLAUDE.md`.

## Decisions made (link ADRs)
- No ADR. Decisions of A0 and AJ of 2026-11-02: moon radii by name from `params/moons.toml` through `moon_radii_from_toml_str`; counts line under the time readout; at the end only stepping stops (message text above); the button keeps the name "Restart"; budget 5.5e8.
