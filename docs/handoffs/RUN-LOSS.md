# RUN-LOSS: fring-run part of ICR-004 (loss, radii, drops.csv)
Role: A5  Branch: ws/a5-run-loss  Status: review
Updated: 2026-11-09 16:00

## Goal and acceptance criteria
`fring-run` part of ICR-004 (section 5): reads `[loss]`, gives the moon radii of `params/moons.toml` to the CPU and GPU backends when `moon_radius = "params"`, writes `drops.csv` (header of ICR-004, also when the run stops with an error); `drops.csv` writer and reader in `fring-io`; `docs/snapshot-format.md` 8b and `chaos.csv` flag bit 1 (A0 reviews); with `[loss]` absent, results bit for bit unchanged.

## Done
- `fring-io`: `drops.rs` (`DropRow`, `write_drops_csv`, `read_drops_csv`, `read_drops_csv_report` with crash rule R1, `DROPS_COLUMNS`, `DROPS_VERSION`), `RunDir::{drops_path, write_drops_file, read_drops_file}`. Tests: `crates/fring-io/tests/drops.rs` (5).
- `fring-run` (`apps/fring-run/src/lib.rs`):
  - Radii: `moon_radii_for(moons_text, &System)` gives the radii in the track order of the `System` (= `[moons] include` order, tested). Set on the CPU and the GPU backend when `moon_radius = "params"`, and on the CPU also for `[substep]` alone. Nothing is read or called when neither is on.
  - `drops.csv`: `collect_drops` calls `Backend::dropped()` and `particles()` once, at the end. With freeze on, the file always exists (header only if nobody was lost). If the stepping loop fails, `drops.csv` is written best effort and the first error is returned (no `meta.json`).
  - `drop_rows` (pub): rows ordered by id, `time_s = start + drop_step x step_s`, `start` = `init.time_s`.
  - `chaos.csv` flags: bit 1 (value 2) is set from the drop report (see Open questions, finding for A3).
- `docs/snapshot-format.md`: section 3 (file list), 8a (flag bit 1), new 8b. A0 reviews.
- Tests: `apps/fring-run/tests/loss.rs` (7) and unit tests in `lib.rs` (4 new: `drop_rows`, radii order, failed step with a mock backend, finished run with the same mock).

## Measurements (rule M27)
- Bit for bit: a small run (200 particles, 0.02 years, 2 moons, from `acceptance-1y-1e4.toml`) on main (f03c9c9) and after the change, SHA-256 of every file except `config.toml` and `meta.json` (wall time): CPU 6 files identical, GPU 6 files identical. Logs: `runs/logs/{before,after}-{cpu,gpu}.sha` (not committed, `runs/` is ignored).
- Clump of 300 particles (sigma 40 km) at Prometheus, 0.0025 years (219 steps), radius rule on: CPU and GPU both lose 24 particles, all reason 2, with the same ids and the same `drop_step` (144 to 175). The test checks this without a tolerance. The frozen positions agree to 1e-8 relative in the one case that I printed.

## Next step (one concrete action)
A0 reviews the PR (snapshot-format 8a, 8b), then A8 (A3-ANALYSIS) can read `drops.csv` with `RunDir::read_drops_file` / `fring_io::read_drops_csv`.

## How to verify the current state
```
cargo test -p fring-io --test drops           # 5 passed
cargo test -p fring-run --lib                 # 12 passed
cargo test -p fring-run --test loss           # 7 passed (GPU tests skip without a GPU)
```

## Open questions / blockers
- Finding for A3 (GPU chaos): `chaos_renorm` in `crates/fring-gpu/shaders/chaos.wgsl` sets bit 1 only at the next renormalisation (`f3.y = f32(u32(f3.y) | 2u)`). A particle lost in the last `renorm_steps` steps of a run has no bit 1 in `GpuBackend::chaos().flags`. `fring-run` ORs the bit in from the drop report, so `chaos.csv` is right. A direct user of `GpuBackend::chaos()` is not covered. (My first plan said the GPU never sets the bit. This was wrong: my grep looked for `| 2` and the shader has `| 2u`.)
- `drops.csv` written on an error path has the state after the last completed `step` call; on the GPU after a failed step the buffers may be inconsistent. Best effort, stated in 8b.

## Decisions made (link ADRs)
- A0 answers of 2026-11-09: header-only file with freeze; `time_s` from `init.time_s`; radii on the CPU for `[substep]` alone; no `meta.json` field.
