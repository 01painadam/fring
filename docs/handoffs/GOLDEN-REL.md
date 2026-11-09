# GOLDEN-REL: release golden test fails on a clean main
Role: A2  Branch: ws/a2-golden-rel  Status: review
Updated: 2026-11-09

## Goal and acceptance criteria
`cargo test -p fring-cpu --release --test golden` must pass with the unchanged hashes, like the debug build.

## Done
- Cause: `kepler_drift` called `x.sin()` and `x.cos()` on the same value. In release, LLVM fuses them into macOS `___sincos_stret`, which differs from libm `sin` by up to 2 ulp (first mismatch: x = 0.3858, 0x3fd8154e812d3eea vs ...eeb). Debug does not fuse. First differing step of the golden run: 172, particle 275, vx 0xbfffeb3d9ed1e909 (debug) vs ...e90b (release). No FMA, no rayon order effect, no debug-only path.
- Fix: `black_box(x).sin()` and `black_box(x).cos()` as two separate calls in `kepler_drift` (both sites), with a comment. Golden hashes are unchanged.
- Guard: unit test `kepler_drift_bits_do_not_depend_on_the_profile` in `physics/kepler.rs` (hash recorded in debug, runs in both; fails in release without the fix).
- `___sincos_stret` call sites (otool): release golden test binary 20 -> 18; release `fring-run` 35 -> 31. The golden path (with moons and chaos) no longer reaches a fused site.

## Not changed (A0 decides)
Other sin/cos pairs on the same value remain fused in release: `physics/frames.rs` (58, 76, 107), `moon_radius.rs` (160, 356), `arc.rs` (180-302), `initial.rs` (80-91, sin_cos at 191, 213), `rng.rs` 212, and the `Real` impl in `real.rs`. They are not reached by the golden test. A cleaner fix is one helper in `real.rs`.

## Next step
A0 review and merge.

## How to verify the current state
cargo test -p fring-cpu --release --test golden   # 2 passed
cargo test -p fring-cpu --test golden             # 2 passed
cargo test -p fring-model --release kepler_drift_bits   # 1 passed

## Open questions / blockers
- Whether to fix the other fused sites (list above).

## Decisions made
- None needing an ADR; no hash or tolerance changed.
