# DET-SINCOS: one deterministic sin_cos helper
Role: A2  Branch: ws/a2-det-sincos  Status: review
Updated: 2026-11-09

## Goal and acceptance criteria
One `sin_cos` helper in `crates/fring-model/src/real.rs`, used at every same-argument sin/cos pair in non-test code of `fring-model` that release can fuse into `___sincos_stret` (GOLDEN-REL). A release test proves profile independence. Debug bits and tolerances do not change.

## Done
- `real::sin_cos<T: Real>(x) -> (sin, cos)`: one `core::hint::black_box` per call (works on wasm32; no `std` needed).
- Sites moved to the helper: `kepler_drift` (2), `frames.rs` (3 functions), `rng::box_muller`, `arc.rs` (`centre_cos_sin`, `half_width_cos_sin`, `width_cos_sin`, `fold`), `initial.rs` (`epicycle_state` 3 pairs, clump, moonlet), `moon_radius.rs` `particle_level` arc width. Pairs in `#[cfg(test)]` code are unchanged.
- Test `physics::profile_bits::sin_cos_users_bits_do_not_depend_on_the_profile`: one FNV hash over all these functions, recorded in debug (0x06c3bdb7b1b1d749). Passes in debug and release. Checked on base commit 9f134b1 with the test copied in: debug gives the same hash (debug unchanged), release gives 0x6104bc0d8b69bd16 (the test detects the fusion).
- CI step `cargo test -p fring-model --release --lib` after "Test".
- `___sincos_stret` call sites (otool -tv): release `fring-run` 31 -> 3; release golden test binary 18 -> 2. The rest are outside `fring-model` (see below).

## Next step (one concrete action)
A0 reviews the pull request.

## How to verify the current state
cargo test -p fring-model --lib profile_bits            # 1 passed
cargo test -p fring-model --release --lib               # 67 passed, about 1.3 s
cargo test -p fring-cpu --test golden                   # 2 passed (debug)
cargo test -p fring-cpu --release --test golden         # 2 passed

## Open questions / blockers
Follow-up (outside fring-model, not in this task): same-argument pairs that release can still fuse:
- `crates/fring-collide/src/pair.rs:103`, `crates/fring-collide/src/hard_sphere.rs:352`
- `crates/fring-analysis/src/apse.rs:39`, `crates/fring-analysis/src/lyapunov.rs:405-406`
- `crates/fring-gpu/src/bin.rs:328-329, 1123, 1416, 1434, 1463`
- `crates/fring-collide/examples/k2_sort_bench.rs:66`
- `crates/fring-gpu/src/ds.rs:264-275` (f64 fallback `sin`/`cos` on separate trait methods)

## Decisions made (link ADRs)
- Free function, not a `Real` method: the trait stays small and no other implementor changes.
