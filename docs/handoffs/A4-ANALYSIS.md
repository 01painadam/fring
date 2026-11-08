# A4-ANALYSIS: the L3 deciding run
Role: A8  Branch: ws/a8-a4-analysis  Status: doing
Updated: 2026-11-08 08:40

## Goal and acceptance criteria
PLAN.md, item A4-ANALYSIS. A note in `docs/results/` gives, for the GR03b masses and our masses, the chaotic-group rate (with the error of the mean), the chaotic fraction and the mean over all runs, and applies the L3 rule (0.30 to 0.35 per year: the masses explain the factor 1.4; near 0.22: they do not).

## Done
- `setup::load_moons_file`, `lyapunov::gr_fit`, `lyapunov::gr_anchor` with tests.
- `fring-analyse lyapunov`: argument 10 (moons file) and 11 (`L`). Default output checked identical to the old binary (6 runs, 8 yr, `diff -r`).
- 10-run check with our masses equals rows 0 to 9 of `docs/results/data/lyapunov_pairs.csv`.

## Next step (one concrete action)
Run (2) (our masses) is running in `runs/a4/ours`. When it ends, compare `runs/a4/ours/lyapunov_pairs.csv` with `docs/results/data/lyapunov_pairs.csv` (`diff`). If they differ, stop and tell A0. If equal, run (1) (GR03b), then run (3), then write the note.

## How to verify the current state
```sh
export PATH="/opt/homebrew/opt/rustup/bin:$HOME/.cargo/bin:$PATH"
cargo test -p fring-analysis        # 22 tests pass
```

## Open questions / blockers
- None.

## Decisions made (link ADRs)
- Raw output goes to `runs/a4/<name>/` (git-ignored). Only small files are committed to `docs/results/data/` (AJ via A0, 2026-11-08).
