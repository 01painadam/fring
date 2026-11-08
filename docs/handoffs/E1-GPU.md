# E1-GPU: flag and freeze and the radius rule on the GPU (no sub-stepping)
Role: A3  Branch: ws/a3-e1-gpu  Status: doing
Updated: 2026-11-08

## Goal and acceptance criteria
ICR-004 amendment 2, part 1, items 3 and 4. GPU: drop reasons, exact `drop_step` words (`8 + cap + i`), frozen state = block start state, the radius rule through the generic functions of `fring-model`, `GpuBackend::dropped`, `set_moon_radii_km`; `[substep]` is a `Config` error. Tests G1 to G6 (see the plan in the A0 thread).

## Done
- G1 twins (`tests/twins_loss.rs`, normal tests): `segment_min_d2` twin against the generic function (f32 and f64), `drift_hits_moon`, `advance_dropping` (field `hit_rel_km` added to `DropOutcome`).

## Next step (one concrete action)
Kernel work: widen `ctl` to 32 bytes with `first_step`, `drop_step` words, `status_words = 8 + 2*cap`, arc log base moved (see the plan).

## How to verify the current state
```sh
cargo test -p fring-gpu --test twins_loss       # 4 tests pass
```

## Open questions / blockers
- G5 (GPU against E1-REF) needs E1-RUN (A5).

## Decisions made (link ADRs)
- A0 2026-11-08: `ctl` 32 bytes and arc log move accepted; the end half drift of a block gets `first + n - 1` on the GPU (the CPU can give one step later where it merges drifts); `freeze` is allowed with chaos and with the arc.
