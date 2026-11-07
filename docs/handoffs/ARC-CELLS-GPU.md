# ARC-CELLS-GPU: K5-BIN cell of a particle in an arc domain
Role: A3  Branch: ws/a3-arc-cells  Status: review
Updated: 2026-11-07

## Goal and acceptance criteria
PLAN ARC-CELLS (A3 part): after the D1 wrap a particle may be up to EDGE_TOL (cos h + sin h) rad (at most 1.4e-6 rad, 0.2 km) outside either edge. `cell_f64`, `cell_ds` and `cell_of` must put it in a valid cell. Done when: edge tests on both sides of both edges, an arc across 180 degrees with 7 degrees, the GPU sort equal to its twin, the full ring unchanged.

## Done
- `BinGeometry.arc: bool` (new `pub` field; `new` sets false, `from_config` sets true for `Domain::Arc`). `alpha_rad` is the longitude of the arc centre for an arc, so inertial longitude minus alpha is the signed angle s, folded to (-pi, pi]; y = a0 s + D/2. Full ring: no change.
- Edge rule (A0 decision: the cell of the periodic image, the same as the CPU twin of A4; first version was a clamp): y in (-slack, 0) is taken at y + D (the last cell n_y - 1), y in [D, D + slack) at y - D (cell 0). D = 2 x (D/2) from `prm[4]` is exact in DS. Beyond that: `OutOfRange`. `ARC_SLACK_RAD = 1.01 sqrt(2) EDGE_TOL` = 1.43e-6 rad (the largest slack the wrap leaves, plus 1 %); in km `slack_km = a0 ARC_SLACK_RAD` = 0.20 km.
- `prm` words (used only for an arc): [0].y = flag, [4].xy = D/2 (DS), [4].z = slack km. For the ring they are zero.
- Files: `crates/fring-gpu/src/bin.rs` (`cell_f64`, `cell_ds`, `params`, geometry), `shaders/bin.wgsl` (`cell_of`), new `tests/gpu_bin_arc.rs`.

## Numbers
- Edge tests (f64 and DS twin; arcs 7/180, 7/0, 5/30, 10/-90, 30/175 degrees; t = 0, 12345.6, 3e7 s): inside, exactly at the edge, 0.5 x and 1 x the wrap slack outside, valid; exactly on an edge: the first or the last cell; outside by 0.5 x and 1 x the slack: the image cell (outside the start edge the last cell, outside the end edge cell 0); 2 x slack and 1e-3 rad outside: `OutOfRange`; both edges.
- GPU sort against twin, exact (keys, sorted ids, bucket starts, counts); twin against f64 only within 1.01e-7 km of an edge: 7/180, 5/30, 30/175 degrees, 2480 particles each: 2320 binned, 160 out of range, as built.
- End to end (`GpuBackend`, 7 degree arc at 180 degrees, 100000 particles, 300 steps, binned after each 100): out of range with the arc rule 0, with the old plain-domain rule 1; all 100000 binned.
- Full ring: the existing `gpu_bin` tests pass unchanged; `params` for a ring is word for word the same (test `full_ring_parameters_are_unchanged`).

## Next step (one concrete action)
A0 reviews. A4 part of ARC-CELLS (K2 `cell_key`, K3 seam pairs) is separate. `fring-gpu` users who build a `BinGeometry` by struct literal must add `arc: false` (none in the repository).

## How to verify the current state
```
cargo test -p fring-gpu --lib bin
cargo test -p fring-gpu --release --test gpu_bin_arc -- --include-ignored --nocapture   # 6 tests
cargo test -p fring-gpu --release --test gpu_bin -- --ignored
```

## Open questions / blockers
- None. The slack of the wrap is angular; the binning uses a0 for the km conversion, as for y, so both use the same angle.

## Decisions made (link ADRs)
- Periodic image (A0, 2026-11-07), not a clamp: the arc is periodic and the CPU twin uses the image. No ADR. No tolerance changed.
