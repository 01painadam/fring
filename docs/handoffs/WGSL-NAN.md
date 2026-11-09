# WGSL-NAN: constant NaN in WGSL breaks the browser
Role: A3  Branch: ws/a3-wgsl-nan  Status: review
Updated: 2026-11-09

## Goal and acceptance criteria
Chrome rejected `bitcast<f32>(0x7fc00000u)` in `arc.wgsl` (test kernel `t_arc_wrap`) as a constant NaN. The "wh" module was invalid, so the demo dropped every frame with a step pass. Fix it, add a guard, prove it in Chrome.

## Done
- `t_arc_wrap` marks a failure with x = -1.0e30 km (finite). `gpu_arc.rs` checks that sentinel.
- Test `gpu_arc` `wrap_kernel_matches_its_twin_and_the_f64_wrap` (needs a GPU, `--ignored`): passes (16 failures, 0 disagreements). With the sentinel in the test changed on purpose to -2.0e30 it fails ("twin and GPU disagree"). Restored.
- Guard `crates/fring-gpu/tests/wgsl_guard.rs` (no GPU): scans all `shaders/**/*.wgsl` for `bitcast<f32>(` of an integer literal and for division by a literal zero. It failed on the old line, passes now. Nothing found in `shaders/collide/`.
- Chrome (headless, A7 script copy in `/private/tmp/claude-501/a3-cdp/`): no WGSL or "Invalid" errors; only a favicon 404. Time ran 1.1 d to 6.0 d in 25 s at 60 fps; screenshot shows the ring, Saturn and both moons.

## Next step (one concrete action)
A0 opens the PR and merges; the demo redeploys from main.

## How to verify the current state
- `cargo test -p fring-gpu --test wgsl_guard` : 2 passed.
- `cargo test -p fring-gpu --test gpu_arc wrap_kernel -- --ignored` : 1 passed (GPU).
- `apps/fring-web/scripts/build.sh`, serve `dist/`, open in Chrome: ring moves, no console errors.

## Open questions / blockers
- Native tests cannot see Tint-only rules. The guard covers only these two forms. Other const-eval faults (overflow of an f32 literal, for example) are not scanned.

## Decisions made (link ADRs)
- Finite sentinel -1.0e30 km, not a run-time NaN. No ADR.
