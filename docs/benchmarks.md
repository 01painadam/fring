# Benchmarks

Task B1. Particle-steps per second of the CPU and GPU backends, measured by `fring-bench` through the `Backend` trait, as `fring-run` uses it.

```sh
cargo run -p fring-bench --release          # about 2.5 minutes on the M4
cargo run -p fring-bench --release -- --help
```

## Machine

- Chip: Apple M4 (10 physical cores, 10 threads)
- Memory: 16 GiB
- OS: macOS 26.5.1
- GPU adapter: Apple M4 (Metal)
- Rust: rustc 1.99.1 (58cf8f9c3 2026-10-13)
- wgpu: 30.0.1
- Git commit: 47f0e9ed0715fc6b3538ba486cdcb2dceebaccb0-dirty
- Date: 2026-11-02
- GPU buffer limits: max_storage_buffer_binding_size 4294967292 bytes, max_buffer_size 4294967295 bytes
- Run time of the matrix: 152 s

The git commit above is the code that was measured (`-dirty` because the benchmark file itself was not yet committed). The CSV with every number is in `runs/bench/bench-2026-11-02.csv` (not in git).

## Method

- **Workload:** core particles (a = 140,220 km, sigma = 5 km, e = 0.0024, seed 1), both moons (Prometheus and Pandora), Saturn J2 to J6. Step h = 356.2537 s, which is 150 steps per orbit at the core. Test particles only: no collisions yet.
- **Backends:** `CpuBackend` (f64, rayon, all 10 threads); `GpuBackend` in mode D+ (production), C (reference) and D (fast, does not meet PLAN section 10, so it is not for science runs). Modes are the ADR 0006 modes.
- **Chaos indicator:** off, and on with ICR-003 `[diagnostics.chaos]` (shadow particle, d0 = 1 km, K = 100). The GPU modes run it at every N. The CPU runs it at N = 10^4 and 10^5 only.
- **What is timed:** one call `step(steps_per_call)`. For the GPU this includes the host work (moon table, command submission) and the wait for the GPU to finish. It does not include building the backend, the start-up self-test or the warm-up.
- **Warm-up and steps per call:** a probe of 10 steps, grown until it takes at least 0.375 s, warms up the caches and pipelines. Then the step count is chosen so that one call takes about 1.5 s (at least 10, at most 30,000 steps, in multiples of 10). Where the limits bite, a call is shorter or longer: the GPU at N = 10^4 hits the 30,000-step cap (about 0.3 to 1.5 s per call); the CPU at N = 10^7 and GPU C with chaos at N = 10^7 use the minimum of 10 steps (about 1.5 s and 0.45 s).
- **Statistics:** 3 timed calls; the table gives the best (highest rate) and the spread (slowest to fastest).
- **With read-back:** the extra column is the rate if a frame is read back every 1460 steps (`frame_every_steps` of the example config). The read-back time is measured separately (best of 2 `particles()` calls on a stale host copy, plus `chaos()` when the indicator is on), because a real 1460-step interval at N = 10^7 would take 15 s per sample. The rate is `N * 1460 / (N * 1460 / rate + read-back time)`. The read-back is small: at most 0.13 s against 1460 steps of work.
- A point that does not fit is reported as "not run" with the reason. No point failed on this machine.

## Results

| Backend | N | Chaos | Steps per call | Particle-steps/s (best of 3) | Min to max | With read-back every 1460 steps | Read-back time (s) |
|---|---:|---|---:|---:|---|---:|---:|
| CPU f64 | 10000 | off | 8700 | 6.28e7 | 6.18e7 to 6.28e7 | 6.28e7 | 0.000 |
| CPU f64 | 10000 | on | 4680 | 3.22e7 | 3.09e7 to 3.22e7 | 3.22e7 | 0.000 |
| GPU D (not for science) | 10000 | off | 30000 | 1.00e9 | 1.00e9 to 1.00e9 | 9.70e8 | 0.001 |
| GPU D (not for science) | 10000 | on | 30000 | 4.87e8 | 4.86e8 to 4.87e8 | 4.76e8 | 0.001 |
| GPU D+ (production) | 10000 | off | 30000 | 8.67e8 | 8.66e8 to 8.67e8 | 8.43e8 | 0.000 |
| GPU D+ (production) | 10000 | on | 30000 | 4.21e8 | 4.21e8 to 4.21e8 | 4.13e8 | 0.001 |
| GPU C (reference) | 10000 | off | 30000 | 4.09e8 | 4.09e8 to 4.09e8 | 4.04e8 | 0.000 |
| GPU C (reference) | 10000 | on | 30000 | 2.02e8 | 2.02e8 to 2.02e8 | 2.00e8 | 0.001 |
| CPU f64 | 100000 | off | 990 | 6.57e7 | 6.40e7 to 6.57e7 | 6.57e7 | 0.000 |
| CPU f64 | 100000 | on | 490 | 3.38e7 | 3.15e7 to 3.38e7 | 3.38e7 | 0.000 |
| GPU D (not for science) | 100000 | off | 16200 | 1.10e9 | 1.10e9 to 1.10e9 | 1.09e9 | 0.001 |
| GPU D (not for science) | 100000 | on | 8090 | 5.39e8 | 5.39e8 to 5.39e8 | 5.36e8 | 0.002 |
| GPU D+ (production) | 100000 | off | 14220 | 9.49e8 | 9.47e8 to 9.49e8 | 9.42e8 | 0.001 |
| GPU D+ (production) | 100000 | on | 6980 | 4.66e8 | 4.66e8 to 4.66e8 | 4.64e8 | 0.002 |
| GPU C (reference) | 100000 | off | 7040 | 4.70e8 | 4.67e8 to 4.70e8 | 4.68e8 | 0.001 |
| GPU C (reference) | 100000 | on | 3470 | 2.32e8 | 2.32e8 to 2.32e8 | 2.31e8 | 0.002 |
| CPU f64 | 1000000 | off | 90 | 6.78e7 | 6.42e7 to 6.78e7 | 6.78e7 | 0.002 |
| GPU D (not for science) | 1000000 | off | 1650 | 1.11e9 | 1.11e9 to 1.11e9 | 1.10e9 | 0.004 |
| GPU D (not for science) | 1000000 | on | 810 | 5.46e8 | 5.45e8 to 5.46e8 | 5.43e8 | 0.012 |
| GPU D+ (production) | 1000000 | off | 1420 | 9.57e8 | 9.55e8 to 9.57e8 | 9.54e8 | 0.004 |
| GPU D+ (production) | 1000000 | on | 700 | 4.72e8 | 4.70e8 to 4.72e8 | 4.70e8 | 0.011 |
| GPU C (reference) | 1000000 | off | 700 | 4.76e8 | 4.74e8 to 4.76e8 | 4.74e8 | 0.012 |
| GPU C (reference) | 1000000 | on | 350 | 2.36e8 | 2.36e8 to 2.36e8 | 2.35e8 | 0.011 |
| CPU f64 | 10000000 | off | 10 | 6.33e7 | 5.96e7 to 6.33e7 | 6.33e7 | 0.019 |
| GPU D (not for science) | 10000000 | off | 150 | 1.07e9 | 1.07e9 to 1.07e9 | 1.06e9 | 0.039 |
| GPU D (not for science) | 10000000 | on | 30 | 5.27e8 | 5.08e8 to 5.27e8 | 5.25e8 | 0.110 |
| GPU D+ (production) | 10000000 | off | 130 | 9.22e8 | 9.07e8 to 9.22e8 | 9.20e8 | 0.039 |
| GPU D+ (production) | 10000000 | on | 30 | 4.56e8 | 4.43e8 to 4.56e8 | 4.55e8 | 0.110 |
| GPU C (reference) | 10000000 | off | 60 | 4.61e8 | 4.56e8 to 4.61e8 | 4.60e8 | 0.038 |
| GPU C (reference) | 10000000 | on | 10 | 2.24e8 | 2.22e8 to 2.24e8 | 2.23e8 | 0.126 |

Reading the table: the best rate is the figure to quote. The spread is below 4 % for the GPU and below 7 % for the CPU, so the machine was quiet. CPU rates do not depend on N (about 6.3 to 6.8 x 10^7). GPU D+ is flat from N = 10^5 (9.5 x 10^8); N = 10^4 is 9 % lower because the dispatch is too small to fill the GPU.

Ratios (N = 10^6 to 10^7): D+ is 2.0 times faster than C and 0.86 times as fast as D. The chaos indicator costs a factor 2.0 on the GPU (a second integration of the shadow particle) and 1.9 on the CPU.

## Comparison with the PLAN section 11 targets

Test particles, no collisions yet. PLAN section 11 sets the targets for an 8 h (28,800 s) run of 30 years (about 18,000 orbits) at 100 steps per orbit: 1.8 x 10^6 steps. The production step is 150 steps per orbit (gate G1), so a 30-year run has 2.7 x 10^6 steps and needs 1.5 times the PLAN rate. The table uses the production step.

| Target (150 steps per orbit) | Needed (particle-steps/s) | GPU D+, chaos off | GPU D+, chaos on | GPU C, chaos off | CPU f64 |
|---|---:|---:|---:|---:|---:|
| Must: 10^6 superparticles | 9.4 x 10^7 | 9.57 x 10^8 (10 times) | 4.72 x 10^8 (5.0 times) | 4.76 x 10^8 (5.1 times) | 6.78 x 10^7 (0.72 times) |
| Stretch: 10^7 superparticles | 9.4 x 10^8 | 9.22 x 10^8 (0.98 times) | 4.56 x 10^8 (0.49 times) | 4.61 x 10^8 (0.49 times) | 6.33 x 10^7 (0.07 times) |

Run time of 30 years at 150 steps per orbit (2.7 x 10^12 particle-steps for 10^6 particles, 2.7 x 10^13 for 10^7):

| N | GPU D+, chaos off | GPU D+, chaos on |
|---|---:|---:|
| 10^6 | 2,820 s (47 min) | 5,720 s (1.6 h) |
| 10^7 | 29,300 s (8.1 h) | 59,200 s (16.4 h) |

- **Must:** met by a wide margin for test particles (10 times with D+, 5 times with the chaos indicator or mode C). Collisions will use part of the rest.
- **Stretch:** just missed with D+ and the indicator off (8.1 h against 8 h). At the PLAN's 100 steps per orbit it would be met (5.4 h). With the indicator on, or with mode C, it takes about 16 h. The step review at S1 (150 against 300 steps per orbit, gate G1) also affects this target.
- **CPU:** below the must target at the production step (0.72 times; it would be 1.1 times at 100 steps per orbit). It is a reference and a test tool, not a production path.
- Collisions are not in these numbers. Re-run `fring-bench` when the collision kernels (A4) are in.

## Memory

| | Bytes per particle | N = 10^6 | N = 10^7 |
|---|---:|---:|---:|
| GPU state (mode D, D+ or C) | 48 | 48 MB | 480 MB |
| GPU with chaos indicator (state, shadow state, accumulators) | 128 | 128 MB | 1.28 GB |

The largest single buffer is the state buffer. This GPU (Apple M4, Metal) allows `max_storage_buffer_binding_size` 4,294,967,292 bytes and `max_buffer_size` 4,294,967,295 bytes, so the state of up to 89,478,485 particles fits one buffer (4 GiB / 48 B). This is a limit of the buffer API, not a tested size. The largest N that was run is 10^7. The memory is shared with the CPU (16 GiB in total), so with the indicator on (128 B) the practical limit is lower than the buffer limit; a run at 10^8 would need 12.8 GB and was not tried.
