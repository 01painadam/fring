# A3-ANALYSIS: 30-year chaos map over the full band, freeze and radius rule on
Role: A8  Branch: ws/a8-a3-analysis  Status: review
Updated: 2026-11-09

## Goal and acceptance criteria
PLAN A3-ANALYSIS: 30-year chaos map over 139,700 to 141,100 km with flag and freeze and the radius rule on (E1-GPU; no sub-stepping, ICR-004 amendment 2). Done when the run completes and the inner-edge numbers of A1-ANALYSIS are confirmed or replaced.

## Done
- Configs `docs/results/configs/chaos-map-full10-freeze.toml`, `-full10-rule.toml`, `-full30.toml` (seed 20261102, N 200,000, band as main10, so the same ids as main10).
- `crates/fring-analysis/src/loss.rs` and `fring-analyse windows <run> <out.csv> <years>`: per-window table with survivors, lost by reason, chaotic fraction (survivors, and lost counted as chaotic), lambda of the survivors.
- Pilot run `full10-freeze` (GPU, freeze only, WH-EFF-GPU in, K4 running on the CPU): wall time 404.6 s (main10 in A1: 384 s, +5 %). 2 of 200,000 particles lost (unbound), both at 139,700 to 139,950 km. Chaotic fraction per window against A1 main10 (same ids): inner edge 0.9786 / 0.9789, flank 0.7311 / 0.7321, core 0.3932 / 0.3938, quiet zone 0.4177 / 0.4187, outer flank 0.7146 / 0.7150, outer edge 0.8228 / 0.8220; all differences below 0.003, lambda within 1.0 % (largest: outer flank 2.103 to 2.124; the others 0.25 % or less). The mu_eff effect (with DET-SINCOS and freeze) is small.
- Byte identity: `fring-analyse map <raw main10> ... main10 139700 141100 7 10 2` gives `main10_hist.csv` and `main10_profile.csv` equal (`cmp`) to the committed files.

## Next step (one concrete action)
A0: review the note `docs/results/a3-analysis.md` and the numbers in `docs/results/data/a3_*`, open the PR. Final runs are in `runs/final-*` (git-ignored); times in `runs/logs/a3-final-times.log`.

## How to verify the current state
`cargo test -p fring-analysis --release` (25 tests); byte identity: `target/release/fring-analyse map /Users/adampain/Projects/fring/runs/chaos-map-main10 <dir> main10 139700 141100 7 10 2` then `cmp` with `docs/results/data/main10_{hist,profile}.csv`; `target/release/fring-analyse windows runs/pilot-full10-freeze /tmp/w.csv 10`.

## Open questions / blockers
- Raw main10 run (read only): `/Users/adampain/Projects/fring/runs/chaos-map-main10/`. No raw main30; only `docs/results/data/main30_{hist,profile}.csv`.

- Final runs done (RUN-LOSS merged): full10-freeze 395 s (chaos.csv identical to the pilot), full10-rule 416 s, full30 1237 s. Note written with verdicts: chaotic fractions all confirmed; lambda verdicts as pre-registered (per-particle median): inner edge, inner flank, quiet zone and outer edge 'replaced'. A0 review: with the A1 definition (median of 2 km bin medians, `python/window_a1.py`, reproduces main10) inner flank and quiet zone agree to 0.3 %; inner edge is selection by the rule; outer edge 30 yr is -4.2 % and not shown to be selection (a 30-year freeze-only run would settle it).

## Decisions made
- A0 2026-11-09: three-run design; N = 200,000; no new ICR; use the `fring-io` drops reader, no own parser.
