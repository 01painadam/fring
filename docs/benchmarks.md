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
- Collisions are not in these numbers. The B2 section below has them.

## Memory

| | Bytes per particle | N = 10^6 | N = 10^7 |
|---|---:|---:|---:|
| GPU state (mode D, D+ or C) | 48 | 48 MB | 480 MB |
| GPU with chaos indicator (state, shadow state, accumulators) | 128 | 128 MB | 1.28 GB |

The largest single buffer is the state buffer. This GPU (Apple M4, Metal) allows `max_storage_buffer_binding_size` 4,294,967,292 bytes and `max_buffer_size` 4,294,967,295 bytes, so the state of up to 89,478,485 particles fits one buffer (4 GiB / 48 B). This is a limit of the buffer API, not a tested size. The largest N that was run is 10^7. The memory is shared with the CPU (16 GiB in total), so with the indicator on (128 B) the practical limit is lower than the buffer limit; a run at 10^8 would need 12.8 GB and was not tried.

## B2: collisions, freeze and the radius rule (task B2)

Date 2026-11-09, commit 229cf54, same machine as above. `fring-bench` with feature rows; the run took 1,423 s (159 points). Every number below is computed from the raw CSV (`runs/bench/b2-2026-11-09.csv`, one row per repeat, not in git) by the script that is in the log `runs/logs/b2-summary.txt`.

```sh
cargo build -p fring-bench --release
target/release/fring-bench --features base,freeze,loss,coll,coll+loss,substep --states core,band,dense --every 1,10 --repeats 5 --sensitivity --out runs/bench/b2-2026-11-09.csv
```

**Scope.** On the GPU, "E1-GPU" means flag and freeze plus the radius rule. Sub-stepping is CPU only (ICR-004 amendment 2: `[substep]` is a config error on the GPU), so the `substep` rows are CPU rows.

### Method

- **Features.** `freeze`: `[loss] on_unbound = "freeze"`. `loss`: freeze and `moon_radius = "params"` (radii of `params/moons.toml`). `coll`: `model = "stochastic"`, restitution constant 0.5 (a cost figure, not a physical choice). `coll+loss`: both. `substep`: `[substep]` (accuracy 0.1, max_level 4) with freeze and the rule, CPU only.
- **Collision parameters** (`crates/fring-collide/SPEC.md` section 10): particle radius s = 1e-5 km (1 cm); weight W = 8e8 x (10^6 / N), so tau stays 0.3 at every N; cells L_x = 0.05 km, L_y = 10 km, L_z = 0.04 km (H/2 for c = 1 cm/s); `every_steps` 1 and 10. The sensitivity point has L_x = 0.2 km. Domain: full ring (the GPU refuses the arc with collisions).
- **States.** `core`: the B1 cold core (sigma 5 km, seed 1), comparable with B1. `band`: the A3-ANALYSIS band, 139,700 to 141,100 km, circular orbits, seed 20261102, reference orbit 140,222.43 km (`docs/results/configs/chaos-map-full30.toml`), generated directly; it has the close passes of Prometheus and Pandora, so it shows the cost of the rule. `dense`: the core with sigma 0.3 km (the resolved-core width of SPEC section 10), the only state with a noticeable number of candidate pairs.
- **Backends.** GPU D+ and the CPU (CPU up to N = 10^6; modes C and D are not measured: C refuses collisions and `[loss]`). Chaos indicator off in all rows.
- **Repeats.** 5 timed calls per point (B1: 3), each row of the CSV is one call. The table gives the median, min to max and the spread (max - min) / median of the 5 raw rates. For the collision rows every repeat runs on a new backend in the initial state (plus one backend for the calibration and the read-back), because collisions cool the ring and repeats on one backend would measure different states. The candidates per particle are the same in all 5 repeats of a row (for example 1.2650e-5 for GPU, `coll+loss`, band, 10^6, `every_steps` 10), which shows that the state is the same.
- **Ratio to base** is the median rate divided by the median rate of the `base` row of the same backend, state and N (so the `coll` and `coll+loss` ratios include the cost of the rule).

### Noise (from the raw rows)

GPU: the spread is 1.1 % or less at N = 10^7, at most 4.6 % at 10^6 (the `coll` rows with `every_steps` 1) and at most 5.4 % at 10^4. Of 81 GPU points, 5 are above 3 %. CPU: at most 10.7 %, 27 of 70 points are above 3 % (the CPU figures are good to about 10 %). The base rows agree with B1 (GPU D+ core: 9.52e8 at 10^6 against 9.57e8 in B1, 9.20e8 at 10^7 against 9.22e8; -0.5 % and -0.3 %).

### GPU D+ results, N = 10^6 and 10^7

| Backend | Features | State | N | every_steps | L_x (km) | Steps per call | Median particle-steps/s | Min to max | Spread | Ratio to base | Candidates per particle per collision step | Pairs per particle per collision step | Weighted occupancy |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| GPU D+ (production) | base | core | 1000000 | - | - | 1420 | 9.52e8 | 9.51e8 to 9.52e8 | 0.1 % | 1.000 | - | - | - |
| GPU D+ (production) | freeze | core | 1000000 | - | - | 1420 | 9.52e8 | 9.47e8 to 9.52e8 | 0.5 % | 1.000 | - | - | - |
| GPU D+ (production) | loss | core | 1000000 | - | - | 1260 | 8.46e8 | 8.46e8 to 8.46e8 | 0.0 % | 0.888 | - | - | - |
| GPU D+ (production) | coll | core | 1000000 | 1 | 0.05 | 180 | 1.22e8 | 1.17e8 to 1.23e8 | 4.6 % | 0.128 | 1.065e-4 | 1.045e-4 | 1.02 |
| GPU D+ (production) | coll | core | 1000000 | 10 | 0.05 | 850 | 5.68e8 | 5.64e8 to 5.70e8 | 1.1 % | 0.596 | 1.005e-3 | 7.914e-4 | 1.02 |
| GPU D+ (production) | coll+loss | core | 1000000 | 1 | 0.05 | 170 | 1.17e8 | 1.17e8 to 1.18e8 | 0.8 % | 0.123 | 1.078e-4 | 1.057e-4 | 1.03 |
| GPU D+ (production) | coll+loss | core | 1000000 | 10 | 0.05 | 780 | 5.23e8 | 5.17e8 to 5.25e8 | 1.6 % | 0.549 | 9.780e-4 | 7.907e-4 | 1.02 |
| GPU D+ (production) | coll | core | 1000000 | 10 | 0.2 | 820 | 5.53e8 | 5.50e8 to 5.54e8 | 0.8 % | 0.581 | 1.336e-3 | 1.174e-3 | 1.09 |
| GPU D+ (production) | base | core | 10000000 | - | - | 130 | 9.20e8 | 9.19e8 to 9.20e8 | 0.1 % | 1.000 | - | - | - |
| GPU D+ (production) | freeze | core | 10000000 | - | - | 130 | 9.20e8 | 9.19e8 to 9.20e8 | 0.1 % | 1.001 | - | - | - |
| GPU D+ (production) | loss | core | 10000000 | - | - | 120 | 8.14e8 | 8.13e8 to 8.14e8 | 0.1 % | 0.885 | - | - | - |
| GPU D+ (production) | coll | core | 10000000 | 1 | 0.05 | 10 | 6.90e7 | 6.87e7 to 6.91e7 | 0.5 % | 0.075 | 1.652e-4 | 1.321e-4 | 1.32 |
| GPU D+ (production) | coll | core | 10000000 | 10 | 0.05 | 60 | 4.16e8 | 4.16e8 to 4.18e8 | 0.4 % | 0.453 | 1.657e-3 | 1.329e-3 | 1.31 |
| GPU D+ (production) | coll+loss | core | 10000000 | 1 | 0.05 | 10 | 6.74e7 | 6.71e7 to 6.76e7 | 0.7 % | 0.073 | 1.652e-4 | 1.321e-4 | 1.32 |
| GPU D+ (production) | coll+loss | core | 10000000 | 10 | 0.05 | 50 | 3.92e8 | 3.92e8 to 3.93e8 | 0.3 % | 0.426 | 1.662e-3 | 1.333e-3 | 1.31 |
| GPU D+ (production) | base | band | 1000000 | - | - | 1430 | 9.54e8 | 9.54e8 to 9.55e8 | 0.1 % | 1.000 | - | - | - |
| GPU D+ (production) | freeze | band | 1000000 | - | - | 1430 | 9.55e8 | 9.54e8 to 9.55e8 | 0.1 % | 1.000 | - | - | - |
| GPU D+ (production) | loss | band | 1000000 | - | - | 1270 | 8.48e8 | 8.48e8 to 8.48e8 | 0.1 % | 0.888 | - | - | - |
| GPU D+ (production) | coll | band | 1000000 | 1 | 0.05 | 190 | 1.30e8 | 1.27e8 to 1.30e8 | 2.5 % | 0.136 | 3.684e-8 | 3.684e-8 | 1.00 |
| GPU D+ (production) | coll | band | 1000000 | 10 | 0.05 | 870 | 5.84e8 | 5.83e8 to 5.87e8 | 0.6 % | 0.612 | 1.207e-5 | 1.655e-6 | 1.00 |
| GPU D+ (production) | coll+loss | band | 1000000 | 1 | 0.05 | 180 | 1.24e8 | 1.24e8 to 1.26e8 | 1.5 % | 0.130 | 3.889e-8 | 3.889e-8 | 1.00 |
| GPU D+ (production) | coll+loss | band | 1000000 | 10 | 0.05 | 800 | 5.35e8 | 5.30e8 to 5.36e8 | 1.2 % | 0.560 | 1.265e-5 | 1.450e-6 | 1.00 |
| GPU D+ (production) | coll | band | 1000000 | 10 | 0.2 | 870 | 5.84e8 | 5.81e8 to 5.86e8 | 0.8 % | 0.611 | 4.943e-6 | 2.069e-6 | 1.00 |
| GPU D+ (production) | base | band | 10000000 | - | - | 130 | 9.21e8 | 9.20e8 to 9.21e8 | 0.1 % | 1.000 | - | - | - |
| GPU D+ (production) | freeze | band | 10000000 | - | - | 130 | 9.21e8 | 9.20e8 to 9.21e8 | 0.1 % | 1.000 | - | - | - |
| GPU D+ (production) | loss | band | 10000000 | - | - | 120 | 8.15e8 | 8.14e8 to 8.15e8 | 0.1 % | 0.885 | - | - | - |
| GPU D+ (production) | coll | band | 10000000 | 1 | 0.05 | 10 | 8.07e7 | 7.98e7 to 8.07e7 | 1.1 % | 0.088 | 1.000e-8 | 1.000e-8 | 1.00 |
| GPU D+ (production) | coll | band | 10000000 | 10 | 0.05 | 60 | 4.56e8 | 4.56e8 to 4.57e8 | 0.3 % | 0.496 | 1.333e-7 | 1.333e-7 | 1.00 |
| GPU D+ (production) | coll+loss | band | 10000000 | 1 | 0.05 | 10 | 7.81e7 | 7.77e7 to 7.83e7 | 0.8 % | 0.085 | 1.000e-8 | 1.000e-8 | 1.00 |
| GPU D+ (production) | coll+loss | band | 10000000 | 10 | 0.05 | 60 | 4.26e8 | 4.26e8 to 4.27e8 | 0.1 % | 0.463 | 1.333e-7 | 1.333e-7 | 1.00 |
| GPU D+ (production) | base | dense | 1000000 | - | - | 1420 | 9.55e8 | 9.55e8 to 9.55e8 | 0.0 % | 1.000 | - | - | - |
| GPU D+ (production) | freeze | dense | 1000000 | - | - | 1430 | 9.55e8 | 9.55e8 to 9.55e8 | 0.1 % | 1.000 | - | - | - |
| GPU D+ (production) | loss | dense | 1000000 | - | - | 1270 | 8.48e8 | 8.47e8 to 8.48e8 | 0.1 % | 0.887 | - | - | - |
| GPU D+ (production) | coll | dense | 1000000 | 1 | 0.05 | 160 | 1.10e8 | 1.10e8 to 1.11e8 | 1.0 % | 0.115 | 2.139e-3 | 1.720e-3 | 1.35 |
| GPU D+ (production) | coll | dense | 1000000 | 10 | 0.05 | 810 | 5.43e8 | 5.32e8 to 5.43e8 | 2.1 % | 0.568 | 2.251e-2 | 1.853e-2 | 1.25 |
| GPU D+ (production) | coll+loss | dense | 1000000 | 1 | 0.05 | 150 | 1.05e8 | 1.05e8 to 1.06e8 | 0.9 % | 0.110 | 2.160e-3 | 1.731e-3 | 1.36 |
| GPU D+ (production) | coll+loss | dense | 1000000 | 10 | 0.05 | 750 | 5.00e8 | 4.98e8 to 5.02e8 | 0.7 % | 0.523 | 2.207e-2 | 1.823e-2 | 1.26 |
| GPU D+ (production) | coll | dense | 1000000 | 10 | 0.2 | 730 | 4.97e8 | 4.96e8 to 5.00e8 | 0.9 % | 0.521 | 3.515e-2 | 2.167e-2 | 2.17 |
| GPU D+ (production) | base | dense | 10000000 | - | - | 30 | 9.12e8 | 9.07e8 to 9.13e8 | 0.6 % | 1.000 | - | - | - |
| GPU D+ (production) | freeze | dense | 10000000 | - | - | 130 | 9.21e8 | 9.21e8 to 9.22e8 | 0.1 % | 1.010 | - | - | - |
| GPU D+ (production) | loss | dense | 10000000 | - | - | 120 | 8.15e8 | 8.14e8 to 8.15e8 | 0.1 % | 0.894 | - | - | - |
| GPU D+ (production) | coll | dense | 10000000 | 1 | 0.05 | 10 | 4.03e7 | 4.03e7 to 4.04e7 | 0.3 % | 0.044 | 5.447e-3 | 2.139e-3 | 5.95 |
| GPU D+ (production) | coll | dense | 10000000 | 10 | 0.05 | 30 | 2.93e8 | 2.92e8 to 2.93e8 | 0.1 % | 0.321 | 5.365e-2 | 2.106e-2 | 5.81 |
| GPU D+ (production) | coll+loss | dense | 10000000 | 1 | 0.05 | 10 | 3.97e7 | 3.96e7 to 3.98e7 | 0.4 % | 0.044 | 5.447e-3 | 2.139e-3 | 5.95 |
| GPU D+ (production) | coll+loss | dense | 10000000 | 10 | 0.05 | 30 | 2.80e8 | 2.79e8 to 2.81e8 | 0.5 % | 0.307 | 5.365e-2 | 2.106e-2 | 5.81 |

### CPU results, N = 10^6

| Backend | Features | State | N | every_steps | L_x (km) | Steps per call | Median particle-steps/s | Min to max | Spread | Ratio to base | Candidates per particle per collision step | Pairs per particle per collision step | Weighted occupancy |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| CPU f64 | base | core | 1000000 | - | - | 90 | 6.06e7 | 5.89e7 to 6.17e7 | 4.6 % | 1.000 | - | - | - |
| CPU f64 | freeze | core | 1000000 | - | - | 90 | 6.17e7 | 6.13e7 to 6.21e7 | 1.2 % | 1.018 | - | - | - |
| CPU f64 | loss | core | 1000000 | - | - | 90 | 5.96e7 | 5.91e7 to 6.02e7 | 2.0 % | 0.982 | - | - | - |
| CPU f64 | coll | core | 1000000 | 1 | 0.05 | 10 | 1.10e7 | 1.10e7 to 1.11e7 | 1.5 % | 0.182 | 1.267e-4 | 1.238e-4 | 1.03 |
| CPU f64 | coll | core | 1000000 | 10 | 0.05 | 60 | 4.27e7 | 4.23e7 to 4.27e7 | 0.9 % | 0.704 | 1.244e-3 | 1.216e-3 | 1.03 |
| CPU f64 | coll+loss | core | 1000000 | 1 | 0.05 | 10 | 1.01e7 | 1.01e7 to 1.02e7 | 0.7 % | 0.167 | 1.267e-4 | 1.238e-4 | 1.03 |
| CPU f64 | coll+loss | core | 1000000 | 10 | 0.05 | 60 | 4.08e7 | 4.04e7 to 4.08e7 | 1.0 % | 0.672 | 1.244e-3 | 1.216e-3 | 1.03 |
| CPU f64 | substep | core | 1000000 | - | - | 80 | 5.92e7 | 5.80e7 to 5.97e7 | 2.9 % | 0.977 | - | - | - |
| CPU f64 | base | band | 1000000 | - | - | 110 | 7.41e7 | 7.38e7 to 7.49e7 | 1.5 % | 1.000 | - | - | - |
| CPU f64 | freeze | band | 1000000 | - | - | 110 | 7.51e7 | 7.25e7 to 7.52e7 | 3.5 % | 1.013 | - | - | - |
| CPU f64 | loss | band | 1000000 | - | - | 100 | 7.17e7 | 7.16e7 to 7.19e7 | 0.5 % | 0.967 | - | - | - |
| CPU f64 | coll | band | 1000000 | 1 | 0.05 | 10 | 1.18e7 | 1.14e7 to 1.19e7 | 4.3 % | 0.159 | 0.000e0 | 0.000e0 | 1.00 |
| CPU f64 | coll | band | 1000000 | 10 | 0.05 | 70 | 4.83e7 | 4.67e7 to 4.87e7 | 4.1 % | 0.652 | 0.000e0 | 0.000e0 | 1.00 |
| CPU f64 | coll+loss | band | 1000000 | 1 | 0.05 | 10 | 1.09e7 | 1.08e7 to 1.09e7 | 1.1 % | 0.147 | 0.000e0 | 0.000e0 | 1.00 |
| CPU f64 | coll+loss | band | 1000000 | 10 | 0.05 | 60 | 4.61e7 | 4.59e7 to 4.63e7 | 0.8 % | 0.622 | 0.000e0 | 0.000e0 | 1.00 |
| CPU f64 | substep | band | 1000000 | - | - | 100 | 6.94e7 | 6.59e7 to 6.98e7 | 5.6 % | 0.937 | - | - | - |
| CPU f64 | base | dense | 1000000 | - | - | 90 | 6.30e7 | 5.99e7 to 6.35e7 | 5.8 % | 1.000 | - | - | - |
| CPU f64 | freeze | dense | 1000000 | - | - | 90 | 6.36e7 | 6.15e7 to 6.40e7 | 3.9 % | 1.011 | - | - | - |
| CPU f64 | loss | dense | 1000000 | - | - | 90 | 6.09e7 | 6.08e7 to 6.15e7 | 1.3 % | 0.968 | - | - | - |
| CPU f64 | coll | dense | 1000000 | 1 | 0.05 | 10 | 1.10e7 | 1.10e7 to 1.11e7 | 0.7 % | 0.175 | 2.754e-3 | 2.045e-3 | 1.48 |
| CPU f64 | coll | dense | 1000000 | 10 | 0.05 | 60 | 4.26e7 | 4.25e7 to 4.28e7 | 0.7 % | 0.676 | 2.477e-2 | 1.898e-2 | 1.42 |
| CPU f64 | coll+loss | dense | 1000000 | 1 | 0.05 | 10 | 1.01e7 | 9.65e6 to 1.02e7 | 5.2 % | 0.161 | 2.754e-3 | 2.045e-3 | 1.48 |
| CPU f64 | coll+loss | dense | 1000000 | 10 | 0.05 | 50 | 4.01e7 | 3.98e7 to 4.04e7 | 1.5 % | 0.637 | 2.532e-2 | 1.929e-2 | 1.44 |
| CPU f64 | substep | dense | 1000000 | - | - | 80 | 5.52e7 | 5.28e7 to 5.85e7 | 10.3 % | 0.876 | - | - | - |

### Collision rows at N = 10^4 and 10^5 (cost figures only)

W = 8e8 x (10^6 / N) is 80 times (10^4) and 8 times (10^5) the value at 10^6. At these N the weight breaks the step rule of SPEC section 5.6 (W sigma g_max dt / V_c > 1 in many cells; see the overfull cells in the CSV), and eight points stop with the "more than 32,767 candidates" error. These rows are cost figures only. The 30-year times and the S1 line use only 10^6 and 10^7.

| Backend | Features | State | N | every_steps | L_x (km) | Steps per call | Median particle-steps/s | Min to max | Spread | Ratio to base | Candidates per particle per collision step | Pairs per particle per collision step | Weighted occupancy |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| GPU D+ (production) | coll | core | 10000 | 1 | 0.05 | 4090 | 2.58e7 | 2.55e7 to 2.69e7 | 5.4 % | 0.030 | 4.955e-4 | 1.612e-4 | 1.00 |
| GPU D+ (production) | coll | core | 10000 | 10 | 0.05 | not run: a cell needs more than 32,767 candidates in one step (the run stops) | | | | | | | |
| GPU D+ (production) | coll+loss | core | 10000 | 1 | 0.05 | 3880 | 2.59e7 | 2.57e7 to 2.64e7 | 2.8 % | 0.030 | 4.656e-4 | 1.388e-4 | 1.00 |
| GPU D+ (production) | coll+loss | core | 10000 | 10 | 0.05 | not run: a cell needs more than 32,767 candidates in one step (the run stops) | | | | | | | |
| GPU D+ (production) | coll | core | 100000 | 1 | 0.05 | 1910 | 1.27e8 | 1.27e8 to 1.28e8 | 0.7 % | 0.136 | 1.799e-4 | 9.804e-5 | 1.00 |
| GPU D+ (production) | coll | core | 100000 | 10 | 0.05 | 6590 | 4.31e8 | 4.30e8 to 4.31e8 | 0.2 % | 0.461 | 2.205e-2 | 3.410e-3 | 1.01 |
| GPU D+ (production) | coll+loss | core | 100000 | 1 | 0.05 | 1820 | 1.21e8 | 1.20e8 to 1.23e8 | 2.4 % | 0.130 | 1.533e-4 | 8.785e-5 | 1.00 |
| GPU D+ (production) | coll+loss | core | 100000 | 10 | 0.05 | 6310 | 4.06e8 | 4.06e8 to 4.07e8 | 0.1 % | 0.435 | 2.010e-2 | 3.112e-3 | 1.00 |
| GPU D+ (production) | coll | band | 10000 | 1 | 0.05 | 4020 | 2.72e7 | 2.70e7 to 2.75e7 | 1.6 % | 0.032 | 0.000e0 | 0.000e0 | 1.00 |
| GPU D+ (production) | coll | band | 10000 | 10 | 0.05 | 30000 | 2.11e8 | 2.10e8 to 2.12e8 | 0.9 % | 0.246 | 2.532e-3 | 5.940e-5 | 1.00 |
| GPU D+ (production) | coll+loss | band | 10000 | 1 | 0.05 | 4180 | 2.74e7 | 2.67e7 to 2.75e7 | 3.1 % | 0.032 | 0.000e0 | 0.000e0 | 1.00 |
| GPU D+ (production) | coll+loss | band | 10000 | 10 | 0.05 | 30000 | 1.93e8 | 1.91e8 to 1.96e8 | 2.5 % | 0.224 | 2.532e-3 | 5.940e-5 | 1.00 |
| GPU D+ (production) | coll | band | 100000 | 1 | 0.05 | 1960 | 1.29e8 | 1.28e8 to 1.31e8 | 2.3 % | 0.137 | 9.184e-7 | 3.469e-7 | 1.00 |
| GPU D+ (production) | coll | band | 100000 | 10 | 0.05 | 8620 | 5.73e8 | 5.72e8 to 5.74e8 | 0.3 % | 0.612 | 2.211e-4 | 2.542e-5 | 1.00 |
| GPU D+ (production) | coll+loss | band | 100000 | 1 | 0.05 | 1880 | 1.24e8 | 1.23e8 to 1.26e8 | 2.4 % | 0.133 | 9.096e-7 | 3.138e-7 | 1.00 |
| GPU D+ (production) | coll+loss | band | 100000 | 10 | 0.05 | 7850 | 5.24e8 | 5.22e8 to 5.31e8 | 1.8 % | 0.559 | 1.942e-4 | 2.064e-5 | 1.00 |
| GPU D+ (production) | coll | dense | 10000 | 1 | 0.05 | 3590 | 2.37e7 | 2.37e7 to 2.38e7 | 0.6 % | 0.027 | 4.547e-3 | 1.342e-3 | 1.00 |
| GPU D+ (production) | coll | dense | 10000 | 10 | 0.05 | not run: a cell needs more than 32,767 candidates in one step (the run stops) | | | | | | | |
| GPU D+ (production) | coll+loss | dense | 10000 | 1 | 0.05 | 3580 | 2.36e7 | 2.36e7 to 2.45e7 | 4.1 % | 0.027 | 4.522e-3 | 1.331e-3 | 1.00 |
| GPU D+ (production) | coll+loss | dense | 10000 | 10 | 0.05 | not run: a cell needs more than 32,767 candidates in one step (the run stops) | | | | | | | |
| GPU D+ (production) | coll | dense | 100000 | 1 | 0.05 | 1820 | 1.20e8 | 1.19e8 to 1.21e8 | 2.0 % | 0.126 | 2.240e-3 | 1.258e-3 | 1.01 |
| GPU D+ (production) | coll | dense | 100000 | 10 | 0.05 | not run: a cell needs more than 32,767 candidates in one step (the run stops) | | | | | | | |
| GPU D+ (production) | coll+loss | dense | 100000 | 1 | 0.05 | 1760 | 1.15e8 | 1.14e8 to 1.15e8 | 0.2 % | 0.121 | 2.101e-3 | 1.204e-3 | 1.01 |
| GPU D+ (production) | coll+loss | dense | 100000 | 10 | 0.05 | not run: a cell needs more than 32,767 candidates in one step (the run stops) | | | | | | | |

### Cost of one collision pass

Cost of one collision pass = (1 / rate(coll) - 1 / rate(base)) x N x every_steps; for `coll+loss` the base is the `loss` row. The cost does not depend on `every_steps` (the GPU rows with 1 and 10 agree within 3 %; the CPU rows within 7 %). Range over the collision rows (states, `coll` and `coll+loss`, `every_steps`), L_x = 0.05 km:

| N | M = next power of two of N | M / N | Pass, GPU (ms) | ns per particle | Pass, CPU (ms) |
|---:|---:|---:|---:|---:|---:|
| 10^4 | 2^14 | 1.64 | 0.37 | 37 | 0.75 |
| 10^5 | 2^17 | 1.31 | 0.68 to 0.75 | 7 | 6.0 |
| 10^6 | 2^20 | 1.05 | 6.6 to 8.3 | 7 to 8 | 71 to 85 |
| 10^7 | 2^24 | 1.68 | 111 (band), 132 (core), 232 to 239 (dense) | 11 to 24 | not run |

The cost scales with N, not with the hash size M: from 10^5 to 10^6, N rises 10 times and M 8 times, while the cost per particle does not change (7 ns); M / N is 1.31 and 1.05 there. Below 10^5 a floor of about 0.3 ms remains (command submission, one wait for the tally). At 10^7 the cost per particle is 1.6 to 3.4 times larger, and it rises with the occupancy of the state (band 11 ns, core 13 ns, dense 23 ns at weighted occupancy 1.0, 1.3 and 5.9), so part of it is the cell kernel and not the hash. A smaller hash at small N would save at most the floor of about 0.3 ms, which is below the time of 40 integrator steps at 10^4, so it is not worth a task (for A4). The sensitivity point (L_x = 0.2 km, 10^6, `every_steps` 10) costs 7.57 ms (core), 6.66 ms (band) and 9.63 ms (dense) against 7.12, 6.64 and 7.96 ms with L_x = 0.05 km: the cell size changes the cost by up to 21 % (dense), where the candidates per particle rise from 2.3e-2 to 3.5e-2.

### Freeze and the radius rule

Freeze alone costs nothing (ratio 1.000 to 1.010 at 10^6 and 10^7). Freeze and the rule cost 11 to 12 % on the GPU (ratio 0.885 to 0.894 at 10^6 and 10^7, all three states), as G6 of E1-GPU found (-11 to -16 %). The state does not change it, so the cost is the test itself, not the number of close passes. On the CPU the rule costs 1 to 4 % at 10^6, within the noise.

### 30-year times

Steps in 30 years: 30 x 365.25 x 86,400 s / 356.2537 s = 2,657,455. Time = N x steps / rate. GPU D+, medians, state `dense` (collision rows; the other states are in the log):

| N | test particles (`base`) | freeze + rule | + collisions, `every_steps` 10 | + collisions, `every_steps` 1 |
|---:|---:|---:|---:|---:|
| 10^6 | 0.77 h | 0.87 h | 1.48 h | 7.01 h |
| 10^7 | 8.1 h | 9.06 h | 26.3 h | 185.8 h |

(`band` and `core`: 10^6 with collisions 1.4 h (`every_steps` 10) and 5.9 to 6.3 h (`every_steps` 1); 10^7: 17.3 to 18.8 h and 94.5 to 109.6 h.) The time with `every_steps` k can be predicted from the `loss` and the `every_steps` 1 rows: time = N x steps x (1 / rate_loss + (1 / rate_coll1 - 1 / rate_loss) / k). This reproduces the measured `every_steps` 10 time to 1.5 % at 10^6 and 10^7 (for dense: 1.49 h against 1.48 h, 26.7 h against 26.3 h). For k = 3 it gives 2.92 h at 10^6 and 68.0 h at 10^7 (dense). This k = 3 time is a prediction, not a measurement.

**Stretch target (10^7 particles, 30 years).** 8.1 h at G2 (test particles). With the rule it is 9.1 h, and with collisions it is 26 h at best (`every_steps` 10) and 186 h for `every_steps` 1. The stretch target is not met with collisions. Remember that SPEC section 10 gives about 16 tau = 4.8 collisions per orbit at tau = 0.3, which is 0.032 per step for `every_steps` 1, and 0.32 per collision step for `every_steps` 10, above the limit of 0.1 of SPEC section 5.6. So `every_steps` 10 is a lower bound of the cost, and a physical run uses `every_steps` 1 to 3.

### End-to-end check with `fring-run`

`fring-run` (commit 229cf54, GPU D+), config in the handoff: 10^6 band particles, 1 year, `every_steps` 10, freeze and the radius rule, `frame_every_steps` and `stats_every_steps` 10^8. Wall time 170.5 s (`time`; `meta.json` says 170.0 s) for 88,582 steps: 5.20e8 particle-steps/s. The bench row (GPU, `coll+loss`, band, 10^6, `every_steps` 10) is 5.35e8 (median of 5 repeats: 5.303e8 to 5.364e8). The run is 2.8 % slower than the bench (2.6 % with the `meta.json` time). The run includes the start-up, a year of evolution (5,389 particles were lost and frozen; in a year the ring state changes) and the writing of the final files. This is within the spread of the bench rows.

### S1 budget (PLAN section 11: 8 h per overnight run)

**Verdict.** A single 30-year run with all features (freeze, rule, collisions) at 10^6 particles fits one night: 1.5 h with `every_steps` 10, 2.9 h with `every_steps` 3 (prediction) and 7.0 h with `every_steps` 1 (dense state, the slowest). At 10^7 it does not fit: 26.3 h (`every_steps` 10) is 3.3 nights, and `every_steps` 1 is 186 h. Matrix cost, M runs at 10^6, all with collisions (an upper bound: the runs without collisions cost 0.87 h each), formula M x T / 8 h with T the time of one run:

| `every_steps` | T (h) | M = 8 | M = 16 | M = 32 |
|---:|---:|---:|---:|---:|
| 10 | 1.48 | 11.8 h, 1.5 nights | 23.7 h, 3.0 nights | 47.4 h, 5.9 nights |
| 3 (prediction) | 2.92 | 23.4 h, 2.9 nights | 46.7 h, 5.8 nights | 93.4 h, 11.7 nights |
| 1 | 7.01 | 56.1 h, 7.0 nights | 112.2 h, 14.0 nights | 224.4 h, 28.1 nights |

So S1 fits the budget of section 11 at 10^6 per run (one run per night), and the matrix fits a few nights if `every_steps` is 3 or more; with `every_steps` 1 the matrix of 32 runs takes 28 nights. The number of nights that AJ has is for the S1 decision. Runs at 10^7 do not fit in a night with collisions.

**Does the CPU collision cost matter for S1?** No, if S1 runs on the GPU. The CPU pass costs 71 to 85 ms at 10^6 (about 10 times the GPU pass; `CellList::group` is 60 to 70 % of it, K2-CPU), and the CPU `base` run alone takes 30 years x 10^6 particles / 6.1e7 = 12.2 h, longer than a night. A parallel pair search or a faster `CellList::group` (deferred by K2-SORT) matters only if S1 or A8 use the CPU backend at 10^6 particles or more; it would not change the verdict for the GPU.

### Memory at N = 10^7 with collisions

GPU D+, chaos off (from `crates/fring-gpu/src/stepper.rs`, `bin.rs`, `collide.rs`, `backend.rs`; the run at 10^7 completed): state 480 MB, status and drop steps 80 MB, bin counts and starts 67 MB each, bin keys and sorted keys 80 MB each, sorted index 40 MB, collide buffers 160 MB and 80 MB, read-back stage 480 MB. Total about 1.6 GB (1.1 GB without the stage); the largest buffer is 480 MB. Host peak at start-up about 2.5 to 3 GB. Chaos is refused with collisions.
