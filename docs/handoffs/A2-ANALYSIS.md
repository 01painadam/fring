# A2-ANALYSIS: Pandora close approaches; chaos map of eccentric orbits
Role: A8  Branch: ws/a8-a2-analysis  Status: done (A0 review 2026-11-02; AJ authorised the merge)
Updated: 2026-11-02 23:30

## Goal and acceptance criteria
PLAN section 9 (A2-ANALYSIS): (a) closest approach to Pandora for the particles of the 10-year chaos map; (b) chaos map of core-like eccentric orbits (ICR-007 `eccentric_band`, e = 0.00235 and apse 353.78 deg from `params/ring.toml`) across 140,150 to 140,700 km, 10 and 30 years, against the e = 0 control. Done when `docs/results/a2-analysis.md` says whether the outer edge needs sub-stepping (h/4 test) and whether eccentric core orbits are more or less chaotic.

## Done
- Code: `closest` (several moons, segment minimum, window, id join), `Point.a_km`, stats (z, KS, Holm, bootstrap), `ecc-configs`.
- Configs in `docs/results/configs/` (`ecc/`, `edge-h*.toml`, `closest-pandora.toml`).

## A0 review (2026-11-02)
- Definition of done again on the pushed head 1961dbd (log `runs/logs/a0-review-a2-analysis.log`, exit codes read directly).
- Each key number of the note traced to its CSV row: `edge_step.csv`, `ecc_windows_e10.csv`, `ecc_windows_e30.csv`, `ecc_repeats_e10.csv`, `ecc_bins_e10.csv`, `ecc_bins_e30.csv`, `closest_pandora_classes.csv`, `closest_pandora_rbins.csv`. Counting sd of the step test recomputed: sqrt(0.8134 x 0.1866 x 2 / 30,000) = 0.00318.
- Corrections in the note (A0 commit): the Holm p at 30 years is at most 2.1e-30, not below 1e-30 (smallest per-bin z 11.5); the apse-phase lambda values compared with the base runs are 4.02, 2.92 and 3.46 (the note had the seed-repeat values 4.09, 2.90 and 3.42); the control lambda per bin is 0.75 to 1.34 (not 0.73 to 1.37); the orbit ends are 2.1 R_H from Prometheus and 1.3 R_H from Pandora (not "2 R_H" for both). Added: the test is unpaired; the class columns of `closest_pandora_pandora_cum.csv` are empty (written before the join).
- No change to a conclusion.

## Next step (one concrete action)
None for A2-ANALYSIS. Follow-ups are in "Open questions"; A0 decides at gate G4 planning whether they become tasks.

## How to verify the current state
`cargo test -p fring-analysis --release` (12+ tests). Runs: `runs/logs/ecc-status-e10.csv` (name, exit, seconds).

## Open questions / blockers
- The h/4 test covers the circular outer window only. Eccentric orbits have b_crit 6.3 R_H against 4.1 R_H; a step test for `eccentric_band` near the zone ends would close this (about 5 min of GPU).
- The cause of the higher chaos of eccentric orbits (close encounters at the orbit ends, or eccentricity resonances) is not separated. `closest` on `eccentric_band` would separate it.
- The excess depends on the apse phase (-0.09 to -0.19 in the fraction at +120 deg). The 30-year runs cover more than one cycle of the relative apse motion (Prometheus and the core: about 19 years), but only one starting phase was run at 30 years.
- Candidates for A3-ANALYSIS or E1 (phase 4). Not started.

## Decisions made
- A0 and AJ (2026-11-02): see the go-ahead message; paired e = 0 control, 25 km bins, 5,000 particles, h/4 test decides sub-stepping.
