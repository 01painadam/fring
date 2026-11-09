# Snapshot format (version 1)

Task F3. Written by A5 (exception to ownership; A0 reviews). Code: `crates/fring-io`. Plan reference: `docs/PLAN.md` section 4.6.

## 1. Purpose

A run writes one directory. Native tools (`fring-run`, `fring-analysis`) and the browser replay viewer read it. Rules for the format:

- Every binary file starts with a magic string, a version and a header length.
- Everything is little-endian. Floating-point numbers are IEEE 754.
- A reader is strict: it rejects damaged files and never guesses.
- The core read and write functions work on `std::io::Read` and `std::io::Write`, so they compile for `wasm32-unknown-unknown`.

## 2. Conventions

- Units: km, s, km/s, rad/s. The unit is in the field name.
- Frame files hold positions (and optional velocities) in the **rotating frame** as `f32`. `fring-io` does no frame conversion. The caller gives it arrays that are already in the rotating frame.
- The rotating frame turns about +z (prograde) at `n0_rad_s`. Its axes coincide with the inertial axes at `t_ref_s`. By convention `t_ref_s = 0`, the epoch (`time_s = 0`), so frames of different runs share one rotating frame. The header stores the value anyway, so a frame file describes itself.
- Precision: `f32` has 24 bits. At a radius of 1.4 x 10^5 km one unit in the last place is about 0.016 km. A position that is converted back to the inertial frame has this error, by design. Do not use frames for energy checks. Use `stats.csv` for that.

## 3. Run directory

```
<run>/
  config.toml          byte-for-byte copy of the input file
  meta.json            code version, git hash, machine, wall time, format versions
  moons.bin            inertial moon states at each output time
  stats.csv            one row per statistics output
  chaos.csv            only with [diagnostics.chaos]: chaos indicator per particle (section 8a)
  drops.csv            only with [loss] on_unbound = "freeze": the lost particles (section 8b)
  frames/000000.bin    one file per frame output
  frames/000001.bin
  ...
```

Frame file names are the frame index with at least 6 digits (`000042.bin`, and `1000000.bin` after 999 999). A reader sorts by the number, not by the text. The index counts output frames (0, 1, 2, ...). The step number is in the header.

## 4. Writer rules

- Write each file to `<name>.tmp` in the same directory, then rename it. A reader then never sees a half-written file. A run that stops in the middle leaves at most one `.tmp` file, which readers ignore.
- Write `config.toml` first (copy the bytes of the input file; do not serialise the parsed config again).
- `moons.bin` can be written once, at the start: the moon ephemeris is computed before the run (`docs/PLAN.md` section 6.3), so the moon states at all output times are known.
- `stats.csv` grows during a run. It is the one exception to the `.tmp` rule: the writer appends rows (`write_stats_header`, then `write_stats_row`). The runner formats each row into one buffer and writes it with a single `write_all`, so a crash can leave at most one unfinished last line. How a reader treats it: section 8.
- Check array lengths before writing. All arrays of one frame have the same length as `n_particles`.

## 5. Frame file `frames/NNNNNN.bin`

Header, at offset 0. `header_len` is 72 in version 1.

| Offset | Type | Field | Meaning |
|---|---|---|---|
| 0 | 8 bytes | magic | ASCII `FRINGFRM` |
| 8 | u32 | version | 1 |
| 12 | u32 | header_len | bytes from file start to the body; multiple of 8; at least 72 |
| 16 | u32 | frame_type | 1 = particle positions in the rotating frame, f32 |
| 20 | u32 | flags | bit 0: velocities present. Other bits must be 0 |
| 24 | u64 | step | step number |
| 32 | u64 | n_particles | N |
| 40 | f64 | time_s | simulation time |
| 48 | f64 | a0_km | reference semi-major axis a0 |
| 56 | f64 | n0_rad_s | rotation rate of the frame |
| 64 | f64 | t_ref_s | epoch at which the frames coincide (0 by convention) |

Body, from offset `header_len`, structure of arrays:

1. `id`: N x u32
2. `x_km`: N x f32 (rotating frame)
3. `y_km`: N x f32
4. `z_km`: N x f32
5. if bit 0 of flags: `vx_km_s`, `vy_km_s`, `vz_km_s`: N x f32 each (rotating frame, velocity relative to the rotating frame)

Body size: 16 N bytes without velocities, 28 N bytes with. N = 10^6 gives 16 MB (28 MB). N = 10^7 gives 160 MB (280 MB). Every array starts at a multiple of 4 bytes, so a browser can view it as a typed array.

File size must be exactly `header_len + body size`.

## 6. Conversion to the inertial frame

With `theta = n0_rad_s * (time_s - t_ref_s)`, and rotating-frame values `(x, y, z)`, `(vx, vy, vz)`:

```
X  = x cos(theta) - y sin(theta)        Y  = x sin(theta) + y cos(theta)        Z = z
VX = vx cos(theta) - vy sin(theta) - n0 Y      VY = vx sin(theta) + vy cos(theta) + n0 X      VZ = vz
```

The last terms are the velocity of the rotating frame, `n0 z_hat x r`.

## 7. `moons.bin`

Header, `header_len` = 32 + size of the name table, rounded up to a multiple of 8.

| Offset | Type | Field |
|---|---|---|
| 0 | 8 bytes | magic `FRINGMNS` |
| 8 | u32 | version = 1 |
| 12 | u32 | header_len |
| 16 | u32 | n_moons (0, 1, 2, ...) |
| 20 | u32 | flags (must be 0) |
| 24 | u64 | n_records |
| 32 | names | for each moon: u32 byte length, then UTF-8 bytes |

Padding bytes after the names are zero. Records follow from `header_len`. One record is `1 + 6 n_moons` f64 values: `time_s`, then for each moon `x, y, z` (km) and `vx, vy, vz` (km/s) in the **inertial** frame. A record is `8 (1 + 6 n_moons)` bytes, so every f64 is 8-byte aligned. With 0 moons, a record is only `time_s`.

## 8. `stats.csv`

Plain CSV: a header row with column names, then one row per output. No comment lines, so pandas and other tools read it with defaults. Columns in version 1:

`time_s, steps_total, collisions_since_last, max_rel_energy_error`

- A reader finds columns by name. It ignores unknown columns and accepts any order. Only `time_s` is required; a missing column gives the default.
- An empty field means "no value" (`None`).
- Floating-point values use the shortest text that parses back to the same number (Rust `{:?}`). Not-a-number and infinity are written `NaN`, `inf`, `-inf`.
- **Crash rule (R1).** A row is complete only when its newline has been written. A reader drops a last line that has no final newline, even if it parses, because a truncated number (`0.0001` cut from `0.000123`) can still parse and be wrong. `read_stats_csv_report` says whether it dropped a line. Any other bad row, including a finished ragged row, is an error. A header without a newline counts as an empty file.
- New `StepStats` fields add columns at the end. The format version is in `meta.json`.

**Collision columns (ICR-009; additive to version 1).** The collision diagnostics follow the columns above, in this order. Each column has the name of its field in `CollisionStats` (`crates/fring-model/src/backend.rs`, `crates/fring-collide/SPEC.md` section 9). The runner writes them from task ICR-009-IO.

`energy_loss_km2_s2, candidates, occupied_cells, occupancy_sum, occupancy_sq_sum, lonely_particles, overfull_cells, gmax_exceeded, deep_collisions`

- `energy_loss_km2_s2` is a float in km² s⁻² (per unit superparticle mass). All the others are integers.
- Each value counts since the previous row, as `collisions_since_last` does. A value is zero when it does not apply to the collision model (for example `deep_collisions` with `model = "stochastic"`), and all are zero without collisions.
- Derived values: mean occupancy of an occupied cell = `occupancy_sum / occupied_cells`; particle-weighted occupancy (the mean number of particles in the cell of a particle) = `occupancy_sq_sum / occupancy_sum`.
- A file written before ICR-009-IO has none of these columns. It reads as zero (the default of a missing column).

## 8a. `chaos.csv` (added by ICR-003; additive to version 1)

Written by `fring-run` at the end of a run that has a `[diagnostics.chaos]` table. Without the table the file does not exist. Plain CSV with the same rules as `stats.csv` (header row, shortest round-trip floats, `NaN`), one row per particle in particle order. Columns, all required:

`id, a0_km, lambda_rel_prometheus_deg, ftle_per_day, log_growth, megno_mean, flags`

- `id`: particle id.
- `a0_km`: Keplerian semi-major axis of the initial state (point mass GM, vis-viva), km. It is not the orbital radius: for a circular orbit near the core in J2 to J6 it is about 640 km larger (A1-ANALYSIS). A map against radius must use the initial radius from the config, not this column.
- `lambda_rel_prometheus_deg`: initial inertial longitude of the particle minus that of Prometheus, degrees in [-180, 180). `NaN` if the run has no Prometheus.
- `ftle_per_day`: finite-time Lyapunov exponent L / t, day^-1, with t the time of the last renormalisation.
- `log_growth`: L, the sum of ln(d / d0) over the renormalisations (dimensionless).
- `megno_mean`: the mean MEGNO. About 2 for a regular orbit, about lambda t / 2 for a chaotic one. Classes used in G3: regular <= 2.5, chaotic >= 5 (ADR 0008).
- `flags`: bit 0 means that an interval had a zero or non-finite separation and was skipped. Bit 1 (value 2) means that the particle was lost (ICR-004, `on_unbound = "freeze"`; the reason is in `drops.csv`, section 8b). The CPU backend sets this bit when it makes the report; the GPU backend sets it only at its next renormalisation. So `fring-run` also sets it from the drop report. The file then marks every lost particle, also one that was lost after the last renormalisation, and the CPU and GPU files agree. Older readers ignore the bit.

The definition of the indicator (shadow offset, renormalisation, MEGNO formula) is in `docs/adr/0008-chaos-indicator.md`. `meta.json` gets an optional `chaos` object (section 9).

## 8b. `drops.csv` (added by ICR-004; additive to version 1)

Written by `fring-run` only when the config has `[loss]` with `on_unbound = "freeze"`; without it the file does not exist. With `freeze`, the file always exists: if no particle was lost it has the header row only. Plain CSV with the same rules as `stats.csv` (header row, shortest round-trip floats). Columns, all required, in this order:

`id, reason, drop_step, time_s, x_km, y_km, z_km, vx_km_s, vy_km_s, vz_km_s`

- One row per lost particle, ordered by `id`.
- `reason`: 1 = unbound or invalid orbit (the Kepler drift fails); `2 + k` = inside moon `k` (`moon_radius = "params"`), with `k` the position in `[moons] include` (0 to 5).
- `drop_step`: the 0-based global step, counted from the start of the run, in which the loss was found. It is exact and has the same meaning on the CPU and the GPU.
- `time_s`: `start + drop_step x step_s`, in seconds since the epoch (`start` is the start time of the run).
- `x_km` to `vz_km_s`: the frozen state in the inertial frame (Saturn-centred, km and km/s). **Within one block the frozen state is unspecified**: the backends keep the state at the start of the block of the loss (ICR-004 section 2). Use `reason` and `drop_step` for physics, not the state.
- Frames still contain frozen particles at their frozen position. A reader that needs the live set at frame step `s` takes the particles with `drop_step >= s` or not in `drops.csv`.
- The runner writes the file at the end of the run, through a `.tmp` file and a rename, so a reader never sees a half file. It also writes the file when a step fails (the state is then the one after the last completed step call, and `meta.json` does not exist). If that write fails, the first error is the one reported. A reader still applies the crash rule R1 as for `stats.csv`: a last line without a newline is dropped (`read_drops_csv_report` says so).
- `fring-io`: `write_drops_csv`, `read_drops_csv`, `read_drops_csv_report`, `RunDir::write_drops_file`, `RunDir::read_drops_file`. `meta.json` has no field for `drops.csv`.

## 9. `meta.json`

A JSON object (Rust struct `RunMeta`): `code_version`, `git_hash`, `machine`, `wall_time_s`, `frame_format_version`, `moons_format_version`, `stats_format_version`, and an optional `chaos` object (only with `[diagnostics.chaos]`; ICR-003): `method`, `offset_km`, `renorm_steps`, `n_renorm`, `chaos_format_version`. `fring-run` fills the values. `fring-io` does not call git and does not read a clock. Readers of frames do not need `meta.json`: a0 and n0 are in every frame header.

## 10. Versions and compatibility

- A reader checks the magic (wrong magic is an error), then the version. It rejects a version that it does not know.
- Within one version, a writer may append header fields after the known ones; `header_len` grows. A reader of that version skips the bytes between the fields that it knows and `header_len`. So added header fields do not break older readers.
- A change of the body, or of the meaning or position of an existing field, needs a new version number. Older readers then reject the file instead of misreading it.
- Unknown flag bits and unknown frame types are errors in version 1.
- There is no checksum and no compression in version 1. W3 may add compression with a new version.

## 11. Damaged files

The reader returns `Error::Io` with a clear message in each case:

- Bad magic, unknown version, `header_len` not a multiple of 8 or too small or larger than 4096, unknown flags or frame type.
- Truncated file (less data than header and `n_particles` say). The reader allocates memory in chunks while data arrives, so a corrupt header with a huge count fails as "truncated" without a large allocation. `n_particles` above 2^32 is rejected at once.
- Trailing data after the expected end.
- A velocity flag with missing arrays is a truncated file.
- Invalid UTF-8 in moon names, a ragged CSV row, or a number that does not parse.

## 12. Open items

- Compression for the web replay (task W3).
- `stats.csv` after a crash: decided in R1, see the crash rule in section 8.
- Optional per-particle columns (for example size) need a new frame type.
