# DET-SINCOS-2: real::sin_cos outside fring-model
Role: A3 + A4 + A8 (A0 allowed the edits in the three crates for this task only)  Branch: ws/a3-det-sincos-2  Status: review
Updated: 2026-11-10

## Goal and acceptance criteria
Use `fring_model::real::sin_cos` at the fused same-argument sin/cos pairs in non-test code of `fring-collide`, `fring-analysis` and `fring-gpu/src/bin.rs`. Release `fring-run` had 3 `___sincos_stret` call sites; they go to 0 or each remaining one is explained. Debug bits unchanged. A release test proves profile independence.

## Done
- Sites changed: `pair.rs` `impact_normal`, `hard_sphere.rs` `resolve_pairs` (arc width, not hoisted), `apse.rs` `fit_ring` (was std `sin_cos()`, fuses the same way), `bin.rs` `params` (sector table), `examples/k2_sort_bench.rs`.
- `otool -tv target/release/fring-run | grep -c ___sincos_stret`: 3 -> 0 (the 3 were `bin::params`, `HardSphereModel::resolve_pairs`, `pair::impact_normal`). `___sincosf_stret`: 0 before and after.
- Tests `tests/profile_bits.rs` in the three crates (Philox seed 20261110 or fixed grid). Recorded debug hashes: collide 0xc816c70891997513, analysis 0x40c9956358a61c6e, gpu 0xf6617f1330142bf6.
- Base be05695 with only the tests copied in: debug hash equals the recorded one in all three. Release: collide 0x0cf121c9489ccd4a (fails), analysis 0x405d6db2f8ef8044 (fails), gpu 0xf6617f1330142bf6 (passes).
- The gpu test cannot detect the fusion: the sector table uses 64 fixed angles and fused and separate results are equal for all of them. The test is a guard only. The otool count is the proof for that site.
- Cost of `impact_normal` (release, one thread, 1e7 calls, ns per call): before 11.68, 11.48, 11.61; after 12.48, 13.20, 12.75 (about +1.1 ns, +10 percent of this function alone).
- A0 check on head 5b8e3a2: DoD green (431 passed, 0 failed, 110 ignored); otool count 3 (base sources) and 0 (head) reproduced; the base release and debug hashes above reproduced (`runs/logs/a0-verify-det-sincos-2-5b8e3a2.log` in the main checkout). The timing was not repeated.
- The 3 extra tests in release are the K3 tests in `fring-collide/tests/k3_hard_sphere.rs` (ignored in debug, slow); they pass with the new `resolve_pairs`.

## Left unchanged (reason)
- Test-only pairs: `lyapunov.rs:405-406` (`#[cfg(test)]`), `bin.rs` 1123, 1416, 1434, 1463 (after `#[cfg(test)]` at line 1096), `fring-gpu/src/arc.rs:219-220`, all of `tests/`. They do not reach the simulation.
- `apps/fring-web/src/view.rs:37`: wasm32 has no `___sincos_stret`; display only.

## Next step (one concrete action)
A0 merges the pull request after green CI. The CI line is in `.github/workflows/ci.yml` (A0 commit 5b8e3a2).

## How to verify the current state
cargo test -p fring-collide -p fring-analysis -p fring-gpu --release --test profile_bits   # 3 x "1 passed"
cargo test -p fring-collide -p fring-analysis -p fring-gpu --test profile_bits             # 3 x "1 passed"
cargo build --release -p fring-run && otool -tv target/release/fring-run | grep -c ___sincos_stret   # 0
cargo test -p fring-cpu --test golden; cargo test -p fring-model --lib profile_bits                   # pass

CI line (added by A0 after the `fring-model` release step):
`cargo test -p fring-collide -p fring-analysis -p fring-gpu --release --test profile_bits`

## Open questions / blockers
- None.

## Decisions made (link ADRs)
- No ADR. No hash or tolerance changed.
