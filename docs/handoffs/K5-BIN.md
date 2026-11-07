# K5-BIN: GPU binning by cell (counting sort), with a Rust twin
Role: A3  Branch: ws/a3-k5-bin  Status: review (A0 re-runs the checks and opens the pull request)
Updated: 2026-11-07

## Goal and acceptance criteria
PLAN section 9, phase 4: "GPU binning: counting sort by cell at each step, with a Rust twin. Done when: the GPU sort agrees with a CPU sort (random cases and edge cases); `fring-bench` shows its cost." A0 go-ahead (2026-11-07): hashed buckets (no dense grid: about 4e10 cells for the full ring), sort inside each bucket by (key, id), cost measured by `examples/k5_bin_bench.rs` (A0 adds the `fring-bench` variant later).

## Done
- `crates/fring-gpu/src/bin.rs`: `BinGeometry` (cell sizes, L_y = D / round(D / L_y,config), domain, phi0, n0; `from_config`), `Cell`/`Binned`/key packing (`pack_key`, `unpack_key`, `hash_key`), the Rust twin `cell_ds` (same operations as the shader), the f64 reference `cell_f64`, `bin_twin` (counting sort in Rust), `cells_reference` (std sort), `runs`, and `GpuBinner` (the GPU side).
- `crates/fring-gpu/shaders/bin.wgsl`: 8 passes in one compute pass: `bin_clear`, `bin_keys` (key of each particle and count per bucket, atomics), `scan_local`, `scan_sums`, `scan_add` (multi-workgroup exclusive prefix sum, 1024 counts per workgroup), `bin_scatter` (atomic), `bin_rank` (see decision 2), `bin_emit`.
- `crates/fring-gpu/tests/gpu_bin.rs` (5 tests, `#[ignore]`) and 9 unit tests in `bin.rs`.
- `crates/fring-gpu/examples/k5_bin_bench.rs` (cost). Log: `runs/logs/k5-bin-bench.log` (git-ignored).

## The interface for K5 (A4) and for the `fring-bench` variant (A5/A0)
```rust
let g = BinGeometry::from_config(&config, n0)?;            // n0 = physics::frames::reference_mean_motion(field, a0)
let binner = GpuBinner::for_stepper(&gpu, &stepper, g)?;   // or GpuBinner::new(gpu, state, status, n, cap, g, buckets)
stepper.encode_steps(&mut enc, n)?;
binner.encode(&mut enc, stepper.time_s());                 // records the passes; the caller submits
```
- `time_s` is the time of the state (`stepper.time_s()` after `encode_steps`). The host forms alpha = (n0 t + phi0) mod 2 pi in f64 and passes it as DS; the shader never forms n0 t. Each `encode` call makes its own 16-byte uniform buffer, so several calls can share one submission.
- Buffers (all `STORAGE | COPY_SRC`): `sorted_buffer()` (n ids), `sorted_keys_buffer()` (n keys of 2 u32: hi, lo), `bucket_start_buffer()` (M + 1 words first), `particle_keys_buffer()` (key of particle i), `info_buffer()` (word `info_offset_words()`: number binned, number not binned, number out of range). Layout and the key format are in the header of `bin.wgsl` and the module docs.
- K5 walks runs of equal key in `sorted_keys`; the members of a cell are in id order (SPEC 5.5, Q4). Two different keys in one bucket are separate runs (tested with M = 1, 2, 4, 64).
- A particle with a drop reason (status word 8 + i not 0), a bad state, or a cell outside the range of the key (|ix| > 2^21 - 2, |iz| > 2^20 - 2, n_y at most 2^21, or a longitude outside the arc) has key `NO_KEY` and takes no part. Their numbers are in the info words.
- Needs 7 storage buffers per shader stage (the WebGPU default is 8; checked in `new`). Native helpers (`read_words`, `read_output`, `read_particle_keys`, `run_timed`) are `cfg(not(wasm32))`; the rest builds for wasm32. The caller must have run the DS self-test, as for the step kernels.
- Memory at N = 1e7 (M = 2^24): counts 67 MB, start 67 MB, keys 80 MB, sorted 40 MB, sorted keys 80 MB (about 334 MB, besides the state).

## The cell and the precision
Cell = (floor(x/Lx), floor(y_c/Ly), floor(z/Lz)), x = rho - a0, y_c = a0 ((phi - phi0) mod 2 pi) (SPEC 5.1). All from the DS state: rho = DS sqrt of X^2 + Y^2; the longitude is k pi/32 + atan(y'/x') with k from an f32 atan2 (any k within the half sector is valid), (x', y') the position turned by -k pi/32 with DS cos and sin from a table of 65 entries (computed in f64 on the host), so that |y'/x'| < 0.0493 and the series of atan (DS terms up to u/3, f32 tail) is exact to 1e-18. Key: 22 + 21 + 21 bits in two u32 words, ordered like (ix, iy, iz).

Edge rule (A0 decision, a new test tolerance; stated in the doc comments of the tests):
- GPU against the DS twin: the same key for each particle, and the same sorted arrays, exactly. A particle within 1e-6 km of an edge may get the neighbouring cell on one side only (the twin and the GPU round the DS operations differently, because the division and square-root seeds differ); the edge test checks the exact agreement for offsets of 1e-6 km and more.
- Twin or GPU against f64: a different cell only if the f64 coordinate is within delta of a cell edge on an axis where the cells differ (the y edges include 0 and D). delta = 10 x the largest |dx|, |dy|, |dz| between the twin and f64 over the random cases, at most 1e-6 km.
- Measured (unit test `twin_agrees_with_f64_up_to_the_edge_rule`, 3 x 200,000 random particles: ring at t = 0 and t = 1e8 s, arc 12,237 km): largest error 1.001e-8 km (the y of the full ring, y up to 8.8e5 km), delta = 1.001e-7 km (tests use 1.01e-7), cell mismatches 0 of 600,000, none outside delta. Edge test (`particles_next_to_an_edge_obey_the_rule`, particles at edge + offset from -1e-5 to +1e-5 km on each axis, 19,800 cases): 892 mismatches with f64 (twin), all within delta, all with |offset| below 1e-6 km. On the GPU: 884 keys differ from f64, 36 from the twin (all |offset| below 1e-6 km, all within delta).

## Results (M4, Metal, release; GPU shared with other runs)
Tests on the head (see the commit list for the head): `cargo test -p fring-gpu --release --test gpu_bin -- --ignored` passes (5 tests, about 1 s). Random cases: ring t = 0 and t = 3.1e8 s, arc 5 degrees, N = 1, 63, 60,001 and 100,000, M = default and 2^20, 1,981 dropped particles in the 100,000 case: keys, `bucket_start`, `sorted`, `sorted_keys` and the three counts equal the twin exactly; the runs equal the std sort of the twin cells; 0 particles differ from f64.

Cost (`k5_bin_bench`, band of g3_bench, cells scaled to about 5 particles per occupied cell, mode D+, median of 7 repeats of several passes in one command buffer, WH block = the backend's default block):

| N | M | cells (x, y, z) | occupancy | bin pass min / median ms | WH step ms | bin / WH step | bin / WH block |
|---|---|---|---|---|---|---|---|
| 1e5 | 2^17 | 16.6 km, 3325 km, 13.3 km | 4.98 | 0.38 / 0.39 | 0.133 | 2.9 | 0.0014 (2000 steps) |
| 1e6 | 2^20 | 4.7 km, 947 km, 3.8 km | 4.96 | 6.31 / 6.73 | 1.307 | 4.8 | 0.024 (200 steps) |
| 1e7 | 2^24 | 1.4 km, 282 km, 1.1 km | 4.55 | 105.7 / 106.9 | 15.73 | 6.7 | 0.336 (20 steps) |

Stages at N = 1e7 (ms): clear 2.1, keys 20.0, scan 3.3, scatter 58.9, rank 13.4, emit 9.3. The scatter dominates: random atomic writes into a 64 MiB array (particles are in random order relative to the cells). The cells are large because a band of 1e5 to 1e7 particles in the full ring is dilute; the occupancy, not the cell size, sets the cost. With `every_steps` = 1 the bin pass costs 3 to 7 WH steps at these sizes (so about 75 % to 87 % of the time of a step plus a bin). Binning every k steps divides this by k. The full table is in `runs/logs/k5-bin-bench.log`.

Stress case, all particles in one cell (best of 3, including the submit): 4,096 members 0.86 ms; 16,384 members 7.9 ms; 32,767 members (the SPEC limit of candidates) 28 ms. The cost is quadratic in the size of one cell (len^2 comparisons over len threads); at 1e5 members in one cell it would be about 0.26 s (extrapolated, not run).

## Decisions made
1. Hashed buckets (A0). Hash: a 32-bit mix (`hash_key`, "lowbias32" type) of the two key words; M = next power of two at least N.
2. Deviation from the go-ahead (A0 asked for an insertion sort of each bucket by one thread): my first version did exactly that and measured 877 ms for 4,096 members in one cell and 17.5 s for 16,384 (a single GPU thread with dependent reads; 17 s is also near the macOS watchdog). It is replaced by `bin_rank`: one thread per particle counts the members of its bucket that come before it in (key, id) order and writes itself to `start[bucket] + rank`. The order is total, so the result is deterministic and the same as the sort by (key, id). Cost above. The scatter writes its ids into the `.x` of `sorted_keys` (scratch) so no 8th storage buffer is needed.
3. Out-of-range cells are not binned (key `NO_KEY`), not clamped, except y within 1e-6 km above D, which is the last cell (rounding at the end of the domain).

## Next step (one concrete action)
A0: re-run the Definition of done on the pushed head, open the pull request, and decide who adds the `fring-bench` variant (it can call `GpuBinner::encode` after `GpuStepper::encode_steps`; `examples/k5_bin_bench.rs` shows the setup and the cell scaling). Then D1.

## How to verify the current state
```sh
cargo test -p fring-gpu --lib bin::                                   # 9 tests, about 2 s
cargo test -p fring-gpu --release --test gpu_bin -- --ignored         # 5 tests, about 1 s, needs a GPU
cargo run -p fring-gpu --release --example k5_bin_bench -- --n 100000,1000000,10000000   # about 20 s
cargo check -p fring-gpu --target wasm32-unknown-unknown
```

## Open questions / blockers
- None. For A4 (K5): the SPEC says the cell velocity statistics use ubar etc.; binning gives only the cells. SPEC 5.1 still says "the config today has one `cell_size_km`" (A4 updates it in K2).
- The bin pass cost is 3 to 7 WH steps at every_steps = 1; the scatter is the part to optimise if it matters (for example by sorting the particles by cell in memory every few hundred steps so the scatter is local). Not done: not asked.
