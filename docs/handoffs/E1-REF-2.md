# E1-REF-2: e1_compare.py for ICR-004 amendment 2, part 3
Role: A6  Branch: ws/a6-e1-ref-2  Status: review
Updated: 2026-11-08

## Goal and acceptance criteria
`validation/scripts/e1_compare.py` applies the E1-REF limits to the WH run without sub-stepping (production). A run with sub-stepping is optional and reported with no pass or fail. No power check. Exit code 0 only if every limit of the run without sub-stepping is met.
- Layer 1: median < 5 % and p95 < 25 % of |da_WH - da_IAS15| / |da_IAS15| for b >= 1 R_H. Stratum [0.68, 1) reported only.
- Layer 2: counts inside 1, 3 and 6.3 R_H within sqrt(N) of IAS15, KS of Delta a p > 0.01, chaotic fraction within 3 sigma.

## Done
- New command line: `e1_compare.py layer1|layer2 --run <file> [--substep <file>] [--json <file>]`. `--run` is required. `--on` and `--off` are removed (no aliases).
- `power_off` removed; `verdict_on` renamed `verdict`. Layer 2 now grades `--run` and reports `--substep`.
- Exit codes: 0 all limits of `--run` met; 1 a limit failed (an empty stratum is NaN and fails for strata 1 to 3); 2 bad input (missing or empty file, fewer columns than the contract, ids differ) or bad arguments.
- JSON keys: `run` and `substep`. The layer 2 `limits` key is filled.
- Result file formats are unchanged. A layer 2 file may have extra columns after the 9 of the contract (the reference file has `min_pandora_km` as column 10).
- Tests in `validation/tests/test_e1_compare.py` (limits, empty stratum, no power check, CLI exit codes for both layers).
- Stand-in check on real data: `e1_standin.py layer1 --div 1` took about 2 s. Compare with `--run`: exit 0. Median / p95: [0.68,1) 8.95e-4 / 7.16e-3; [1,2) 1.64e-4 / 1.27e-3; [2,4.1) 1.26e-5 / 1.35e-4; [4.1,6.3] 8.81e-6 / 1.17e-4. Same as the E1-REF handoff.

## Superseded text
In `docs/handoffs/E1-REF.md`, the "Finding: power check has no power" paragraph and the old command lines (`--on`, `--off`, "limits and power check") are superseded by ICR-004 amendment 2 and this task. That file is not edited.

## Next step (one concrete action)
A0: review, open the PR, update the `AGENTS.md` command list and the E1-RUN board row (command lines below). Then E1-RUN (A5) runs the Rust backend and calls the script; E1-GPU (G5) uses it later.

## How to verify the current state
```sh
cd validation && uv run pytest -q     # 20 passed (about 15 s)
cd validation && uv run python scripts/e1_standin.py layer1 --div 1 --out ../runs/standin_l1_h1.csv   # about 2 s
cd validation && uv run python scripts/e1_compare.py layer1 --run ../runs/standin_l1_h1.csv           # strata 1 to 3 pass, "layer 1: OK", exit 0
cd validation && uv run python scripts/e1_compare.py layer2 --run reference/e1_l2_ref.csv             # IAS15 against itself: "layer 2: OK"
cd validation && uv run python scripts/e1_compare.py layer1 --on x                                    # argparse error, exit 2
```
Command lines for `AGENTS.md`:
```sh
cd validation && uv run python scripts/e1_compare.py layer1 --run <wh.csv> [--substep <wh_substep.csv>]   # E1-REF limits on the run without sub-stepping; the sub-stepped run is reported only (also: layer2)
cd validation && uv run python scripts/e1_standin.py layer1|layer2 --div 1 --out <file>   # REBOUND WHFast stand-in at the run step h, NOT the Rust backend
```

## Open questions / blockers
- None. The layer 2 stand-in was not run (about 25 min); layer 2 is covered by synthetic tests.

## Decisions made (link ADRs)
- ICR-004 amendment 2, part 3 (AJ, 2026-11-08). No new ADR.
