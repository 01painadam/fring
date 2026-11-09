# D1-CPU: arc wrap and arc collisions on the CPU backend
Role: A2  Branch: ws/a2-d1-cpu  Status: review
Updated: 2026-11-08

## Goal and acceptance criteria
PLAN section 9: the CPU backend calls the D1 wrap (`physics::arc`) after each step; `fring-run` accepts `domain.kind = "arc"` (A5). Done when a CPU energy test in the arc passes (no moons: E and Lz unchanged by the wrap to 1e-13 relative) and a `fring-run` arc config runs. A0 scope: also lift the arc refusal for stochastic and hard-sphere collisions (K4 needs it).

## Done
- `crates/fring-cpu/src/lib.rs`: `CpuBackend` has `arc: Option<ArcGeometry>`. `new` folds the start state into the arc (as `GpuBackend`), refuses `moon_radius = "params"` with an arc (as the GPU). In arc mode `step` uses blocks of one step, and `wrap_states` runs after each step (plain and extended path), before the chaos renormalisation and the collision step. Order: integrate, wrap, renorm, collide.
- Shadow: gets the same rotation as its main particle (k turns), never a wrap of its own.
- Frozen particles (`reason != 0`) are skipped.
- Wrap failure (more than `MAX_WRAPS` widths, or not finite) is handled as the GPU does: `crates/fring-gpu/shaders/arc.wgsl` `arc_run` line 168 `drop_particle(i, 1u, seg)` (`wh.wgsl:441`): reason 1 (unbound or invalid), `drop_step = first_step + seg - 1`, the step that failed. CPU: with `freeze`, reason `DROP_UNBOUND` and `drop_step = step - 1` (block of one step); without freeze, `Error::Backend("particle <id> (index <i>) has an unbound or invalid orbit")`, the existing text, lowest index.
- `collide.rs`: arc domains for both models, `Some(width)` and the rotated separation for seam pairs; refusal removed.
- No change to `StepStats`, `Backend`, config, or `fring-model`. Wraps are not counted.
- Full-ring path untouched: `cargo test -p fring-cpu --test golden` passed before and after commit 1 (1 passed, 0.57 s and 1.12 s).

## Numbers (logs: `runs/logs/d1cpu_arc_tests.log`, `runs/logs/d1cpu_cost.log`; release)
- PLAN test (`wrap_keeps_e_and_lz`, 300 particles, 0.5 deg arc, 3000 steps, E and Lz just before and after each wrap call, no integrator error): 9031 wrap events, max |dE/E| 1.27e-15, max |dLz/Lz| 6.1e-16 (bound 1e-13).
- Why f64 meets 1e-13: a wrap is a rotation by cos/sin of the width, about 3 roundings of 1.1e-16 per component, so about 6e-16 per wrap.
- Arc against the full ring turned by whole widths (`arc_against_rotated_ring`, 3000 steps, 298 of 300 rotated): max position difference 8.96e-7 km, |dE/E| 6.0e-14, |dLz/Lz| 3.0e-14. This is not the PLAN bound: the two runs round differently in the integrator. Test bounds are loose (1e-6 km, 1e-12).
- 360 deg arc against ring (400 steps): max relative difference 9.8e-12 (blocks of one step change the rounding).
- Chaos: log growth arc against ring, max difference 2.2e-4 (200 particles, 2000 steps, d0 = 1e-3 km); the noise comes from rounding at d0 = 1e-3 km; a shadow wrapped alone would differ by about 10. Test bound 1e-3.
- Hard-sphere seam pair against the same pair mid-arc (10 deg arc, gap 0.08 km, contact 0.1 km, 1 m/s): relative dv difference A 4.0e-11, B 4.0e-11. The ARC-CELLS-CPU expectation of 3e-9 is rounding of the inertial coordinates (3e-11 km at 1.4e5 km) divided by the gap (13 mm there: 3e-11 / 1.3e-2 = 2e-9). Here the gap is 80 mm, so the expected size is about 4e-10; we measured 4e-11. The test bound is 2e-8 (the ARC-CELLS bound).
- Stochastic collisions in a 3 deg arc (2000 particles, 300 steps): 11596 collisions; bits equal for splits 300 and 100+200.
- Cost (`arc_cost`, 10^5 particles, no moons, 500 steps after 20): full ring 2.20 ms per step (22 ns per particle-step), arc 5 deg 4.95 ms per step (49.5 ns). Ratio 2.25. Cause: blocks of one step (no merged half drifts, one rayon fork per step) plus the wrap.

## Next step (one concrete action)
A0 reviews the PR. A5 then lets `fring-run` accept `domain.kind = "arc"` (`check_supported`). After ICR-011 (WH-EFF-CPU), put the wrap inside the WH loops so that arc runs can use long blocks again.

## How to verify the current state
```
cargo test -p fring-cpu --test golden                    # 1 passed
cargo test -p fring-cpu --test arc -- --nocapture        # 9 passed (1 ignored: arc_cost)
cargo test -p fring-cpu --release --test arc -- --ignored --nocapture   # cost
cargo test -p fring-cpu
```

## Open questions / blockers
- None. The `fring-run` part is A5's.

## Decisions made (link ADRs)
- A0 2026-11-08: blocks of one step in arc mode for now; no `StepStats` wrap counter.
