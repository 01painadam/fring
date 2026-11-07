# K5-BIN: GPU binning by cell (counting sort), with a Rust twin
Role: A3  Branch: ws/a3-k5-bin  Status: doing
Updated: 2026-11-07

## Goal and acceptance criteria
PLAN section 9, phase 4: "GPU binning: counting sort by cell at each step, with a Rust twin. Done when: the GPU sort agrees with a CPU sort (random cases and edge cases); `fring-bench` shows its cost." A0 go-ahead: hashed buckets (no dense grid), sort each bucket by (key, id), cost measured by `examples/k5_bin_bench.rs` (A0 adds the `fring-bench` variant later).

## Done
- Rust twin and f64 reference (`crates/fring-gpu/src/bin.rs`), shader (`shaders/bin.wgsl`), unit tests of the twin.

## Next step (one concrete action)
Write the GPU side (`GpuBinner`) in `bin.rs`, then `tests/gpu_bin.rs`.

## How to verify the current state
```sh
cargo test -p fring-gpu --lib bin::    # 9 tests pass
```

## Open questions / blockers
- None.

## Decisions made (link ADRs)
- A0 (K5-BIN go-ahead): hashed buckets; edge rule delta = 10 x largest measured twin-vs-f64 error, at most 1e-6 km; bench example in `fring-gpu`.
