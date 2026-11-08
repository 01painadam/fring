# E1-CPU: ICR-004 and per-particle sub-stepping in the CPU backend
Role: A2  Branch: ws/a2-e1-cpu  Status: review
Updated: 2026-11-08

## Goal and acceptance criteria
ICR-004 (Accepted 2026-11-08), sections 1 to 6, A2 line of "Roles affected", migration item 4. PLAN E1: `[loss]` and `[substep]` in the config, `Backend::dropped`, `physics/moon_radius.rs`, `CpuBackend` with drop reasons, `drop_step`, levels, Hermite sub-kicks and a shadow that follows its main particle, tests C1 to C8 and the energy test D. The PLAN "done when" also asks for agreement with E1-REF; that comparison follows when A6 has E1-REF (not started).

## Done
- C1: golden hashes, recorded on the unchanged code and committed first (`crates/fring-cpu/tests/golden.rs`, commit 0556811). They still pass.
- `crates/fring-model/src/physics/moon_radius.rs`: `segment_min_d2`, `drift_hits_moon`, `hermite_moon`, `substep_level`, and `particle_level` with the arc images (turn by -dtheta and +dtheta, largest level of the three; `None` = particle only). T1 to T4 and an arc test are unit tests there. wasm32 check builds.
- `config.rs`: `[loss]` (`on_unbound`, `moon_radius`), `[substep]` (`accuracy` default 0.1, `max_level` default 4), `validate` rules of ICR section 1 (not the GPU mode C rule), helpers `freeze_unbound`, `moon_radius_rule`, `loss_or_substep_active`.
- `backend.rs`: `DROP_LIVE`, `DROP_UNBOUND`, `drop_inside_moon`, `DropReport`, provided `Backend::dropped`.
- `KickTable::fill_ext`: moon velocities at the kick rows, the two sync rows, `moons_at_subkick` (rows for n = 1, cubic Hermite else). `fill` is unchanged.
- `crates/fring-cpu/src/extended.rs`: the extended particle loop `advance_ext` (freeze, radius rule on every drift, levels, merged drift, shadow stepped with its main particle). It runs only when `[substep]`, `moon_radius = "params"` or `on_unbound = "freeze"` is present. Otherwise `advance_chunk` runs as before (the old chaos path too). A `[loss]` with only defaults gives the old path and the old error text.
- `CpuBackend::set_moon_radii_km(&[f64])`, `dropped()`, `ChaosReport.flags` bit 1 (value 2).
- Tests: C2, C3, C7 (flags) in `tests/drops.rs`; C4, C5, C6, C7 (levels) in `tests/substep.rs`; test D in `tests/energy_encounter.rs`; C8 in `tests/bench_substep.rs`.
- Level-0 shortcut with squares only (`level_zero_certain`, unit test: 2 million random cases, never says 0 where the rule gives more). C8 went from +9.3 % to +0.6 %.

## Conventions chosen (for A3 and A5)
- `drop_step` of a drift: the first half drift of a block belongs to the first step of the block. A drift after kick (or sub-kick) of step k belongs to step k, also the last drift to the block end and the merged drift to the next step. For the GPU (segment s = 0 for the first half drift, s = k + 1 for the drift after kick k): `drop_step = first_step + max(s - 1, 0)`.
- Frozen state: state at the start of the block (ICR section 2). The shadow of a lost particle is frozen with it. A Kepler failure of the shadow alone counts as a loss of the pair with reason 1 (not in the ICR; rare).
- Reasons: 1 unbound, 2 + k inside moon k, k = 0 to 5. The extended path returns `Error::Config` for more than 6 moons.
- With `on_unbound = "error"` and `[substep]` the first unbound particle gives the old error text.
- The next level is a `Vec<u8>` (255 unknown) in `CpuBackend`. The first step of a run takes the state at the block start, the moon at sync row 0 and `dt = h`. A sync row outside the ephemeris is not an error (the radius test skips that drift; the Hermite sub-kicks use the two nearest kick rows).
- The ICR example "63 km, u = 0.03 km/s gives L = 1" ignores the margin `d_eff = max(d - u dt, r)` of section 3. With the margin it is d_eff = 47 km and L = 2. T4 follows the formula of section 3.

## Measured results (raw lines in `runs/logs/`, git-ignored; commands below)
- C4, C6, C7 (levels and flags), C2, C3: pass (normal tests).
- C5 levels: pass. 60 particles with encounters (7 with L > 0), 100 days, blocks of 4096, 1000 and 37 steps: levels agree in 1.000000 of 2,880,000 steps. Positions: median 3.7e-6 km, 90th percentile 1.1e-5 km, **maximum 1.24 km**. **The ICR limit "positions agree to 1e-3 km" is not met** (`c5_positions_icr_limit`, ignored, fails). Cause: rounding grows through the encounters. A change of 1e-12 km in one start coordinate gives 4.3e-5 km after 100 days (1e-9 km gives 1.5e-3 km), and blocks of 37 steps have 650 block ends. Blocks of 4095, 2048 and 1000 give a maximum of 9e-6, 1.2e-5 and 2e-3 km. Needs a decision by A0 (statistics instead of positions, or a shorter window).
- T3 velocity limit: **not met**. `hermite_moon` position error 1.16e-3 km (limit 5e-3 passes), velocity error 1.0e-5 km/s (ICR limit 1e-6). A cubic Hermite with nodes h apart cannot give 1e-6 km/s (the derivative of the interpolation error is about h^3 x'''' / 125). `t3_velocity_limit_of_the_icr` keeps the ICR limit (ignored, fails); the normal test uses 3e-5. The velocity only enters the level rule through u = |w - mdot|, a relative error of 3e-4 at u = 0.03 km/s.
- C8: **pass**, ratio 1.0064. Three runs each, 12,800 particles of 139,700 to 141,100 km, 88,600 steps of 356.25 s, real moons. Off: 24.939, 24.996, 25.003 s. On: 25.098, 25.148, 25.181 s. Best off 4.547e7, best on 4.519e7 particle-steps/s. The radius rule costs about 4 % (information).
- Test D, with the quantity the backend conserves. The backend adds Saturn's reflex term (ADR 0004), which is a static uniform force in the rotating frame. The conserved quantity is `C_J + g (r . m_hat)` with `g = GM_m/a_m^2` (7.7e-8 km² s⁻² amplitude). The ICR formula without the term has a floor of 2.7e-9 to 7e-8 (see `d_ii_quick_literal_cj.log`). With the term, WH at h/64 converges as h² to 1e-13.
  - (i), strata by the measured closest distance (reference run), b chosen by a pilot so that the distance is uniform in each stratum; 500 particles per stratum except [0.68, 1) R_H with 368. p95 of |dCJ| in km² s⁻²:
    | stratum (R_H) | sub-stepped | today | ratio | reference h/64 | reference / sub-stepped |
    |---|---|---|---|---|---|
    | [0.68, 1) | 2.99e-9 | 1.89e-10 | 15.8 | 1.7e-13 | 5.7e-5 |
    | [1, 2) | 1.57e-9 | 1.71e-10 | 9.2 | 1.7e-13 | 1.1e-4 |
    | [2, 4.1) | 1.28e-10 | 1.28e-10 | 1.00 (L = 0) | 1.1e-13 | 8.9e-4 |
    | [4.1, 6.3) | 7.37e-11 | 7.37e-11 | 1.00 (L = 0) | 1.1e-13 | 1.5e-3 |
    The absolute limit 1e-6 passes by a factor of 600. **The limit "<= 0.1 x today" fails in all strata with b >= 1 R_H**: the sub-stepped net error is 9 times larger than today's for [1, 2), and equal for [2, 6.3] where no particle is sub-stepped. Today's WH is already at 1.7e-10 net because a symplectic step returns to its own shadow Hamiltonian after the pass. The reference check passes in 3 of 4 strata ([4.1, 6.3) gives 1.5e-3 against 1e-3; the reference is at the rounding floor 1.1e-13).
  - During the pass (information, p95 of the largest |dCJ| at the step ends): [0.68, 1) sub-stepped 4.2e-8, today 8.6e-8; [1, 2) 1.7e-8 and 3.1e-8; [2, 4.1) 4.2e-9 for both; [4.1, 6.3) 3.7e-10 for both. The gain from sub-stepping is only a factor of 2, not 16.
  - Check of the cause: with the exact moon position at each sub-kick (a temporary change, not committed) six particles with b of 1.1 to 1.64 R_H gave the same net error (1e-10 to 1.3e-8) as with Hermite. So the cubic Hermite is **not** the limit. The state-dependent step (no time symmetry, ICR section 3 "no hysteresis") seems to be. This is a finding for A0; I did not change `accuracy` or `max_level`.
  - (ii), 100 particles, a = a_m + 300 to 600 km (b = 4.7 to 9.5 R_H, so L = 0 almost always: sub-stepped and today are bit for bit equal), samples every 30 days to 51 years: rms |dCJ| at N = 100 encounters 6.6e-12 km² s⁻² (limit 2.7e-6, passes); slope of ln rms against ln N 0.15 (limit 0.7, passes); the rms is flat (6.6e-12 to 2.6e-10 at N = 12 to 150). Reference (20 years, 243 samples): ratio to the sub-stepped rms 0.02 to 0.42 (limit 1e-3, **not met**: the reference is at the rounding floor 1e-12). (ii) does not exercise sub-stepping. It took 23 minutes (100 particles are one rayon chunk, so one thread).

## Next step (one concrete action)
A0 reviews and decides: (1) the C5 position limit, (2) the T3 velocity limit, (3) the test D relative limit and the reference check, (4) merge. Then E1-GPU (A3) can start from the conventions above.

## How to verify the current state
```sh
cd .claude/worktrees/a2-e1-cpu            # branch ws/a2-e1-cpu
cargo test -p fring-cpu -p fring-model     # normal tests, all pass
cargo test -p fring-cpu --release -- --ignored c5_block_length c8 # pass
# expected failures (limits not met, see above):
cargo test -p fring-cpu --release --test substep c5_positions_icr_limit -- --ignored
cargo test -p fring-cpu --release --test energy_encounter d_i_ -- --ignored --nocapture   # 20 s
cargo test -p fring-cpu --release --test energy_encounter d_ii -- --ignored --nocapture   # 23 min; E1_D_QUICK=1 for a short run
cargo test -p fring-cpu --release --test bench_substep -- --ignored --nocapture           # 2.5 min
```

## Open questions / blockers
- Out of scope here: GPU (E1-GPU, A3); `drops.csv`, the `fring-run` and `fring-io` keys and `docs/snapshot-format.md` 8b (A5); the line in `crates/fring-collide/SPEC.md` (A4); the E1-REF comparison (A6).
- For A5 (`fring-run`): read `<params_dir>/moons.toml` with `moon_radii_from_toml_str`, match the names to the moon tracks of the `System` (not by position), and call `CpuBackend::set_moon_radii_km` before the first `step`. The backend reads no files. If `[substep]` or `moon_radius = "params"` is on and the radii were never set, the first `step` returns `Error::Config`.
- For D1-CPU (the arc): `advance_ext` passes `arc = None` to `particle_level`. D1-CPU must pass the `ArcGeometry` so that the images are used. The radius rule with the arc stays a `Config` error.
- For A3: T1 f32 part (the `Wf32` twin of `segment_min_d2` against the generic function with `T = f32` and `T = f64`, relative < 1e-6) belongs to E1-GPU. The generic function has the same order of operations. `fill_ext` is the host table that the GPU needs (velocities, sync rows).
- For A0 (out of scope, decide later): how A6 (E1-REF) runs the CPU WH on its synthetic encounter states without reading the implementation. `CpuBackend` with `System` and `InitialState` is the public interface.
- The C5 and T3 limits and the D relative limit are not changed. They need an A0 decision (ADR for any change).
- `particle_level` and `substep_level` use the formula of section 3 exactly; the trigger shortcut `level_zero_certain` is CPU only.

## Decisions made (link ADRs)
- A0 2026-11-08: radii by `CpuBackend::set_moon_radii_km`, arc images in the shared function, extended path only when a key changes the behaviour. No ADR.
