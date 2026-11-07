# L3-PARAMS: GR03b moon masses as a variant file
Role: A1  Branch: ws/a1-l3-params  Status: review
Updated: 2026-11-07

## Goal and acceptance criteria
The GR03b moon masses as a variant file in `params/`, cited in `params/sources.md`. Done when the file and its citation are merged. The nominal values do not change.

## Done
- `params/variants/gr03b-moons.toml`: copy of `moons.toml`, only the two GM values changed.
- `params/sources.md`: section "Variant: GR03b moon masses".
- Products (G = 37931206.234): 5.80e-10 x G = 0.0220001, so 0.02200 (4 s.f.). 3.43e-10 x G = 0.0130104, so 0.01301 (4 s.f.). Ratios to nominal: 2.054 and 1.405 (agree with L3 note).
- Python `tomllib` parses the file; `diff` against `moons.toml` shows only the two GM lines (plus the header).

## Next step (one concrete action)
A0 reviews and opens the PR. A4-ANALYSIS (A8) then needs an optional argument to load the file.

## How to verify the current state
`diff params/moons.toml params/variants/gr03b-moons.toml` shows only the header and two `gm_km3_s2` lines.

## Open questions / blockers
- The GR03b m/M values come from the L3 task (A1 read, A0 re-checked). Not re-opened here.
- Loader: `moons_from_toml_str` and `moon_radii_from_toml_str` take text and ignore unknown keys, so they read the variant as is. No loader or test needs to know about the file. A8's `--moons <file>` option (A4-ANALYSIS) must be added in `fring-analysis`/`fring-run`; I did not edit crates. The Rust loader was not run on the variant file.
- The variant duplicates the states of `moons.toml`. If `moons.toml` changes, update it too (the header says so).

## Decisions made (link ADRs)
- None.
