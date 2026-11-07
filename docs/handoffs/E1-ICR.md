# E1-ICR: ICR-004, lost particles and per-particle sub-stepping
Role: A2  Branch: ws/a2-e1-icr  Status: review
Updated: 2026-11-07

## Goal and acceptance criteria
Write `docs/interfaces/ICR-004.md` (PLAN.md section 8, E1-ICR): flag and freeze on the CPU with the drop reasons of ICR-006; the Rust twin of the radius rule moves to `fring-model/src/physics/`; trigger and scheme of the sub-stepping for both backends; the REBOUND comparison set and statistics; whether `fring-run` uses the radius rule. Done when A0 writes "Accepted" after AJ's decision. AJ decided on 2026-11-07 ("go with recommendations"); one number (the energy-test limit) is open.

## Done
- `docs/interfaces/ICR-004.md` written (Status: open). No code changed. The board is not edited (A0 keeps it).
- The counting run (A0 approved, CPU release, under 5 minutes): `runs/logs/e1-icr-f.log` (git-ignored; the numbers are in the ICR). Three `fring-run` runs of 2 years: 12,800 particles of the full band (41 s), 6,400 eccentric particles at 140,150 to 140,225 km and at 140,625 to 140,700 km (25 s each), frames every 100 steps with velocities. Configs and the analysis script `f3.py` are in the session scratchpad (not in the repository); to repeat: the `[particles.distribution]` blocks are in the ICR table caption, the rule is in ICR section 3.

## Working that did not fit in the ICR
- **Why the first plan numbers were wrong.** The plan round estimated f = 1.0e-3 for a trigger of 6.9 R_H: `2 b / (2 pi a) = 876 / 875,000 = 1.0e-3` for a particle that is always in radial reach. The measured share inside 6.3 R_H is 1.1e-4 to 1.8e-4, because only about 25 % of the band is in radial reach of Prometheus (the band starts at 139,700 km, Prometheus reaches 139,689 km at apoapsis) and the eccentric bins are in reach only near the right apse phase. The first plan also used a fixed `v_rel` of 0.109 km/s. The measured `u` at d = 100 to 400 km is 0.002 to 0.055 km/s (the shear 1.5 n x d is 0.035 km/s at 200 km), so `h u / d` is below 0.1 for most passes down to about 100 km.
- **Level rule check at one point.** d = 63.4 km, u = 0.03 km/s, GM = 0.01071: `v_eff² = 0.03² + 2 x 0.01071 / 63.4 = 9.0e-4 + 3.4e-4`, `v_eff` = 0.035 km/s, `h v_eff / d = 356.25 x 0.035 / 63.4 = 0.197`, so L = 1 (0.098). At d = 43.09, u = 0.15: `v_eff² = 0.0225 + 4.97e-4`, v_eff = 0.152, `h v_eff/d = 1.26`, L = 4 gives 0.079 (< 0.1), so the cap of 16 resolves a pass at the radius.
- **Shear limit of `accuracy`.** `h x 1.5 n0 = 356.25 x 1.5 x 1.1756e-4 = 0.0628`. With `accuracy = 0.02` the rule gave level 2 for every sample (log). So `accuracy >= 0.07`.
- **Hermite error.** `(n0 h)^4 a / 384 = (0.04188)^4 x 139,380 / 384 = 3.08e-6 x 139,380 / 384 = 1.1e-3 km` for a bracket of one step. A bracket of a half step (at the block ends) is 16 times smaller. Linear interpolation: sagitta `L² / (8 a)` = 31.6 km (A3, ICR-006). Quadratic through three kicks: about 1 km.
- **Why the decision comes from the last kick.** `wh_steps_moons` merges the half drifts (`moon_kick.rs`), so the position at `t_k` does not exist in the middle of a block. The kick at the end of step k has the position and already computes the distance. Carrying a one-byte "next level" across blocks makes the rule independent of the block structure; the only dependence left is rounding. Without it, step 0 of a block would use a different state (`t_0` with `dt = h`) than the steps inside the block (`dt = 1.5 h`).
- **Energy-test estimate.** Per pass the energy error of a second-order step is about `(GM_m/d)(h u/d)²`: at d = 200 km, `0.01071/200 = 5.4e-5 km² s⁻²`; `(0.05..0.1)² = 2.5e-3..1e-2` gives 1.4e-7 to 5.4e-7 per encounter; sqrt(100) of the smaller gives 1.4e-6 km² s⁻² = 5e-9 v_c² (v_c² = GM/r = 3.793e7/140,000 = 271 km² s⁻²). Today's step (`h u/d` = 0.5 to 1 at 63 km) is about 100 times worse. This is an estimate; the limit is open for AJ.
- **E1-REF power check.** At b = 2 R_H = 127 km and u = 0.04 km/s, `h u / b = 0.11`, so the unsub-stepped error may already be small in the outer strata. The ICR asks for a failure of the unsub-stepped run in at least one stratum with b >= 1 R_H, and says what A6 reports if there is none.
- **Dropped decisions that I did not take.** I did not add fields to `StepStats` or `ChaosReport` (the two `fring-io` test literals without `..` would fail, and `stats.csv` has fixed columns). Drop counts go through `Backend::dropped()` and `drops.csv`.

## Next step (one concrete action)
A0 reviews `docs/interfaces/ICR-004.md`, asks AJ for the energy-test limit (options a, b, c in the ICR), and writes "Accepted". Then E1-CPU (A2) and E1-REF (A6) can start; E1-GPU (A3) follows E1-CPU.

## How to verify the current state
- `git log --oneline origin/main..HEAD` shows the commits of this task (documents only).
- `git diff --stat origin/main..HEAD` lists only `docs/interfaces/ICR-004.md` and `docs/handoffs/E1-ICR.md`.
- The measured numbers: `runs/logs/e1-icr-f.log` (main checkout of the worktree; git-ignored).

## Open questions / blockers
- Energy-test limit (ICR "Decisions for AJ", options a to c). Recommendation (a).
- A3 confirms the status-word layout and the table layout in E1-GPU.

## Decisions made (link ADRs)
- AJ 2026-11-07: decisions 1 to 4 of the ICR (default `error`, rule opt-in, `accuracy` 0.1 and `max_level` 4 with the E1-REF limits, `drops.csv` sidecar). No ADR; they are in ICR-004.
