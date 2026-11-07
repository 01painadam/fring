# ARC-CELLS-CPU: cell index of the arc and hard-sphere pairs across the seam
Role: A4  Branch: ws/a4-arc-cells (from origin/main 0c635b6)  Status: review
Updated: 2026-11-07

## Goal and acceptance criteria
From `docs/PLAN.md`, ARC-CELLS (A3 + A4), CPU part: the cell index of an arc uses the signed angle s from the arc centre in (-pi, pi], y = a0 (s + dtheta/2); a particle within the D1 wrap slack (EDGE_TOL (cos h + sin h) <= 1.4e-6 rad) goes into a valid cell; K3 `find_pairs` and `resolve_pairs` see pairs across the seam (partner rotated by +-dtheta, position and velocity). *Done when:* edge tests on both sides of both edges, an arc across 180 degrees with dtheta = 7 degrees, a hard-sphere pair across the seam. The GPU part (K5-BIN) is A3's.

## Done
- `Domain` has a new field `arc: Option<ArcGeometry>` and a constructor `Domain::arc_domain` (D = a0 dtheta, phi0 = centre - dtheta/2). `cell_key` for an arc: s from the inertial longitude and `ArcGeometry::centre_at_rad(t)`, y = a0 (s + dtheta/2). The full ring takes the old expression (bit-identical; K2 tests unchanged).
- Choice for the slack: a particle outside an edge goes to the cell of its **periodic image** (`CellGrid::key` takes y modulo D), not the edge cell. The image is where its partners are after the wrap. Clamping to the edge cell would lose pairs: with 10 m cells, a particle 0.15 km outside is 15 cells from its partners. Exactly at an edge: cell 0 or the last cell (same place).
- K3: `find_pairs` returns `Vec<Pair { i, j, shift }>`. For an arc, a candidate that is not close directly is tried with the partner rotated by +-dtheta (in the same and in neighbouring cells too, because a slack particle sits in the cell of its image). `resolve_pairs(pairs, h_s, width_rad, separation(i, j, shift), vel)` rotates the partner velocity for g and the impulse back for the partner. `HardSphereModel::arc_domain` added. Shift 0 uses the old operations (full ring unchanged).
- Tests: `tests/arc_cells.rs` (6 tests): 7 degree arc at 180 degrees (task example, several t), monotonic cell index through 180 degrees, edges (inside, exactly at, inside the slack, beyond; both edges; 5, 7, 10, 30 degree arcs; centres 0, 180, 90, -120; two times), full ring unchanged, pairs against brute force with images (2000 particles wrapped by `ArcGeometry::wrap`, 7 and 10 degrees, t = 0, 1e5, 3e8), seam pair against the same pair in the middle (both id orders, receding pair untouched).

## Next step (one concrete action)
A0: review and open the PR. A3 does the GPU part (K5-BIN `cell_f64`, `cell_of`) with the same rule (periodic image for the slack).

## How to verify the current state
```sh
cargo test -p fring-collide --test arc_cells -- --nocapture   # 6 tests, a few seconds
cargo test -p fring-collide --release                         # all quick tests incl. K1 hard sphere
cargo clippy --workspace --all-targets -- -D warnings
```

## Open questions / blockers
- A seam pair matches the same pair in the middle of the arc to about 3e-9 relative (not 1e-15), because the inertial coordinates are 1.4e5 km with a rounding of 3e-11 km and the pair is 13 mm apart (3e-11 / 1.3e-2 = 2e-9 in the direction of d). The test bound is 2e-8. The algebra is exact; the limit is the input rounding.
- SPEC.md 8 wording is proposed in the report, not committed.

## Decisions made (link ADRs)
- A4: periodic image (not clamp) for the wrap slack, see Done.
