# W2-DROP-GPU: dropped particles in GpuStepper (ICR-006)
Role: A3  Branch: ws/a3-w2-drop-gpu  Status: review
Updated: 2026-11-02 21:10

## Goal and acceptance criteria
ICR-006 in `crates/fring-gpu`: a drop reason for each particle in the status buffer (`8 + cap` words), an early return for a dropped particle in the D and Dp step kernels, counts by reason, and an opt-in moon-radius rule (minimum distance over each drift between two kicks, linear interpolation of the particle-minus-moon vector, mean radius from `params/moons.toml`). Done when: the tests of ICR-006 item 4 pass natively; all existing tests pass and `GpuBackend` results are bit for bit unchanged with the rule off; `cargo check -p fring-gpu --target wasm32-unknown-unknown` passes; `fring-bench` shows no slow-down beyond its noise.

## Done
- Plan accepted by A0 (decisions 1 to 8 in the A0 go-ahead). Commits on `ws/a3-w2-drop-gpu`:
  - 87b66e9 and 319b128: this file; `tests/gpu_drop.rs` golden hashes of the rule-off stepper, measured on the unchanged code (M4, Metal). They stay valid in every later commit.
  - a08c681: status buffer of `8 + cap` words (`STATUS_HEADER_WORDS`, `REASON_*`, `reason_inside_moon`, `status_words`, `status_init_bytes` in `wh_gpu.rs`).
  - f87142b: `step_d` and `step_dp` return early for a dropped particle, set reason, count and word 0 for a failed one (`drop_particle`), and stop the loop at the first failure.
  - 6027258: `moon_rel` and `seg_min_d2` (WGSL) with Rust twins `wh::moon_rel` and `wh::segment_min_d2`; prm grows to 44 words (`wh::params_with_radii`, `wh::with_radii`); `t_moon_rel` and `t_segment_min` test entries.
  - b1dd710: `wh::sync_rows` (moon positions at the two sync times of a block; sentinel `SYNC_NONE_KM` outside the ephemeris); two spare rows in each table buffer.
  - 8d93ecf: Rust twin of a whole block with the rule, `wh::advance_dropping`.
  - 30d9411: kernels `step_d_r` and `step_dp_r`, `GpuStepper::set_moon_radii_km`, `moon_radii_km`, `Mode::entry_rule`.
  - bc66a86: timing test (information).
- Tests (all `#[ignore]`, `cargo test -p fring-gpu --release -- --ignored`):
  - `tests/gpu_drop.rs` (8): `rule_off_state_bits_are_the_golden_bits`; `status_buffer_has_8_plus_cap_words_with_word_0_no_failure`; `an_unbound_particle_gets_reason_1_and_is_frozen_and_others_are_unchanged` (two unbound particles, modes D and Dp: reason 1, counts, word 0, others bit for bit equal to a run without the failure, frozen state over 3 blocks); `a_crossing_between_two_kicks_is_dropped_in_the_right_block_and_r_plus_1_km_is_not` (modes D and Dp, Prometheus and Pandora: both ends of the drift outside the radius, dropped with reason 2 + k in block 1 or later, state frozen at the start of that block, counts and word 0, the others bit for bit the same as in the run with radius closest approach - 1 km in which nothing drops); `rule_with_tiny_radii_gives_the_bits_of_the_rule_off_stepper`; `gpu_and_twin_agree_on_who_is_dropped_near_the_moons` (366 crafted particles, D and Dp: 173 dropped each, 0 on the edge); `set_moon_radii_checks_its_arguments`; `rate_with_the_rule_off_and_on` (timing only).
  - `tests/gpu_wh.rs`: `moon_radius_functions_match_their_rust_twins` (M24: low words of the particle and moon positions are not zero; worst relative error vs twin 1.2e-7 for `moon_rel`, 3.7e-7 for `seg_min_d2`).
  - Unit tests in `wh.rs` (`radius_tests`, 4): segment cases, sync rows and sentinel, params, `advance_dropping` equals `advance` bit for bit when nothing drops and freezes otherwise.
  - Existing tests unchanged and passing: `gpu_stepper` (bytes of state and status equal), `gpu_backend` (5), `gpu_twins`, `done_when`, `gpu_chaos`, `chaos_done_when`, `gpu_selftest`.
- Accuracy of the rule, checked: the closest approaches chosen by the tests are 25.1 to 25.2 km; the test margin is 1 km (radius = closest approach +- 1 km) and the GPU and the Rust twin agree on all 366 particles, none within 0.05 km of a radius. The size of the A0 estimate of the segment error (0.1 km) is not measured against an IAS15 run; ICR-006 accepts the linear rule.
- Behaviour changes: with the rule off, results of live particles are bit for bit unchanged (golden hashes, `gpu_stepper`). A failed particle is now frozen and costs one load per block. After a failure `GpuBackend` still returns the error as before. Mode C is unchanged (word 0 only). Chaos: rule on is a `Config` error; with the rule off chaos keeps its behaviour (the shadow dispatch shares the status buffer, so a shadow failure also freezes the main particle; `GpuBackend` stops the run at once).
- The ephemeris end: a drift whose sync row lies outside the ephemeris is not checked (only the last half drift of a run that ends within half a step of the ephemeris end).
- Different from the plan: the Rust twin of the whole block (`advance_dropping`) is its own commit before the kernels (as planned), the timing test is C9, and `wh::with_radii` was added as the single place that patches the prm words.
- Definition of done on the pushed head bc66a86 (exit codes read directly, loop with `eval`): `cargo fmt --all -- --check` 0; `cargo clippy --workspace --all-targets -- -D warnings` 0; `cargo test --workspace` 0 (223 passed, 0 failed, 36 ignored); `cargo check -p fring-model --target wasm32-unknown-unknown` 0; `cargo check -p fring-gpu --target wasm32-unknown-unknown` 0; `cargo test -p fring-gpu --release -- --ignored` 0 (21 passed, 0 failed).
- Bench, `fring-bench --sizes 100000,1000000`, A = baseline binary at 938344d, B = head 30d9411 (the later commit only adds a test), run interleaved A1 B1 A2 B2 A3 B3 on the same quiet M4, rule off through `GpuBackend`. CSVs are in the session scratchpad (not in the repository); the raw `rate_best` values (particle-steps/s) of the GPU rows are here:

| backend | N | chaos | A1 | A2 | A3 | B1 | B2 | B3 |
|---|---|---|---|---|---|---|---|---|
| gpu_c | 100000 | false | 4.5584e8 | 4.5886e8 | 4.5454e8 | 4.5449e8 | 4.5482e8 | 4.5878e8 |
| gpu_c | 100000 | true | 2.2253e8 | 2.2277e8 | 2.2197e8 | 2.2261e8 | 2.2204e8 | 2.2229e8 |
| gpu_c | 1000000 | false | 4.7015e8 | 4.6987e8 | 4.6750e8 | 4.7118e8 | 4.6706e8 | 4.6760e8 |
| gpu_c | 1000000 | true | 2.3341e8 | 2.3103e8 | 2.3286e8 | 2.3264e8 | 2.3314e8 | 2.3285e8 |
| gpu_d | 100000 | false | 1.0814e9 | 1.0852e9 | 1.0791e9 | 1.0871e9 | 1.0816e9 | 1.0865e9 |
| gpu_d | 100000 | true | 5.2667e8 | 5.2503e8 | 5.2620e8 | 5.2744e8 | 5.2660e8 | 5.2712e8 |
| gpu_d | 1000000 | false | 1.0940e9 | 1.0942e9 | 1.0923e9 | 1.0956e9 | 1.0977e9 | 1.0962e9 |
| gpu_d | 1000000 | true | 5.3969e8 | 5.4001e8 | 5.3915e8 | 5.4148e8 | 5.3936e8 | 5.4010e8 |
| gpu_dp | 100000 | false | 9.1929e8 | 9.2601e8 | 9.2403e8 | 9.2084e8 | 9.2380e8 | 9.1835e8 |
| gpu_dp | 100000 | true | 4.5303e8 | 4.5315e8 | 4.5252e8 | 4.5346e8 | 4.5382e8 | 4.5360e8 |
| gpu_dp | 1000000 | false | 9.4614e8 | 9.4341e8 | 9.4763e8 | 9.4579e8 | 9.5040e8 | 9.4768e8 |
| gpu_dp | 1000000 | true | 4.6680e8 | 4.6721e8 | 4.6628e8 | 4.6848e8 | 4.6747e8 | 4.6959e8 |

  Best of three per binary (spread = (max - min) / max over the three runs): B vs A best is between -0.2 % and +0.5 % in all 12 rows; the spread of A is 0.1 to 1.0 %, of B 0.1 to 0.9 %; B's best lies inside A's spread range in every row, so there is no slow-down beyond the noise. Three repeats were enough (the rule was not to go to 5 unless B's best fell outside A's spread). The two rows with the largest differences: gpu_dp 1000000 chaos true +0.5 % (faster) and gpu_dp 100000 false -0.2 %.
- Rule on (information, `rate_with_the_rule_off_and_on`, best of 5, 300 steps in one block): D 100000: 1.048e9 off, 9.135e8 on (-12.9 %); Dp 100000: 9.072e8, 7.862e8 (-13.3 %); D 1000000: 1.080e9, 9.510e8 (-12.0 %); Dp 1000000: 9.494e8, 8.313e8 (-12.4 %). The web budget of 6.6e8 particle-steps/s for compute is still met at 8.3e8.

## Next step (one concrete action)
A0 review: run the Definition of done on the pushed head, check the bench table against the raw rows, then start W2-DROP (A7): `set_moon_radii_km(&[43.09, 40.63])` before the first step, read words 0 to 7 of `status_buffer()`, hide particles with a nonzero word `8 + i`.

## How to verify the current state
- `cargo test -p fring-gpu --release -- --ignored` gives 21 passed (needs a GPU).
- `cargo test --workspace` gives 223 passed, 36 ignored.
- `cargo check -p fring-gpu --target wasm32-unknown-unknown` passes.

## Open questions / blockers
- None. For A7: the radii are `radius_km` of `params/moons.toml` in the order of `System.moons.tracks` (Prometheus, Pandora); call `set_moon_radii_km` before the first `encode_steps`; the buffer has `8 + capacity` words (about 4 MB at 10^6 particles).

## Decisions made (link ADRs)
- Separate entry points `step_d_r` and `step_dp_r` for the radius rule; 2 sync-time moon rows per block; prm grows to 44 words; setter `GpuStepper::set_moon_radii_km`; rule with chaos or mode C is a `Config` error.
- With the rule off, a failed particle is now frozen (ICR-006 item 2). Results before the first failure are bit for bit unchanged.
- `MAX_MOONS` stays 4. Status words 6 and 7 (moons 4 and 5) stay reserved: ICR-006 allows 6 moons, the kernels take 4.
- Accuracy of the rule (A0 estimate, to be checked): the moon path between two kicks has a sagitta L^2 / (8 a) = 5,940^2 / (8 x 139,380) = 31.6 km, so only the particle-minus-moon vector is interpolated. That vector is almost straight: tidal acceleration 3 n^2 d = 3 x (1.17e-4)^2 x 50 = 2e-6 km/s^2 gives 0.03 km at mid-drift; the moon's pull at 43 km (0.01071 / 43^2 = 5.8e-6 km/s^2) gives about 0.09 km. The error of the segment test is about 0.1 km against a 43 km radius.
