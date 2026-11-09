# ICR-012-IMPL: inclination_deg for the core distribution
Role: A2 (+ A0 files)  Branch: ws/a2-icr-012-impl  Status: review
Updated: 2026-11-09

## Goal and acceptance criteria
`docs/interfaces/ICR-012.md`: optional `inclination_deg` (default 0, finite, in [0, 90)) on `Distribution::Core`; `i_rad = inclination_deg` in radians and `z_phase_rad = 2 pi u_z` (second word of `counter(id, 1)`); exactly `i_rad = 0`, `z_phase = 0` at 0. Acceptance 1 to 5 of the ICR.

## Done
- `config.rs`: field with `#[serde(default)]`, doc, validation. `to_toml_string` keeps the field (used only in round-trip tests in `crates/fring-model/tests/config.rs`; no test or file compares its text).
- `initial.rs`: Core arm and module doc.
- Tests: `core_inclination_absent_equals_zero_bitwise` (6 components, N = 1000, and z = vz = +0.0), `core_inclination_gives_a_uniform_vertical_epicycle` (N = 10^4, sigma = 0: in-plane bits equal to i = 0, max |z| <= a i (1 + 1e-12), (z/(a i))^2 + (vz/(a i nu))^2 = 1 within 1e-12, |mean| < 0.0212, |rms - 1/sqrt 2| < 0.0075), `core_inclination_defaults_to_zero_and_is_checked` (config; rejects -1, -0.001, 90, 120, nan, inf).
- `inclination_deg: 0.0` in the 6 literals of other roles (own commit); `docs/PLAN.md` 7.4 one phrase.
- The `profile_bits` hash 0x06c3bdb7b1b1d749 is unchanged: core states at i = 0 are bit-identical to main.

## Next step (one concrete action)
A0 reviews the pull request.

## How to verify the current state
cargo test -p fring-model                              # all pass
cargo test -p fring-model --release --lib              # 69 passed
cargo test -p fring-cpu --test golden                  # 2 passed
cargo test -p fring-cpu --release --test golden        # 2 passed
cargo check -p fring-web --target wasm32-unknown-unknown

## Open questions / blockers
- None. `band` and `eccentric_band` have no inclination field (ICR-012 point 4).

## Decisions made (link ADRs)
- None needing an ADR.
