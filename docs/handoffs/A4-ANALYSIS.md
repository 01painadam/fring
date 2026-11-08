# A4-ANALYSIS: the L3 deciding run
Role: A8  Branch: ws/a8-a4-analysis  Status: review
Updated: 2026-11-08 09:30

## Goal and acceptance criteria
PLAN.md, item A4-ANALYSIS. A note in `docs/results/` gives, for the GR03b masses and our masses, the chaotic-group rate (with the error of the mean), the chaotic fraction and the mean over all runs, and applies the L3 rule (0.30 to 0.35 per year: the masses explain the factor 1.4; near 0.22: they do not).

## Done
- Code: `setup::load_moons_file`, `lyapunov::gr_fit`, `gr_anchor`, arguments 10 and 11 of `fring-analyse lyapunov`. Default output identical to the old binary.
- Trust check passed: the re-run with our masses equals the committed A1 files (byte for byte).
- Runs: ours (537 s), GR03b (644 s), 3000 yr with 2 km kick (1065 s), 3000 yr without kick (805 s). Raw output is in `runs/a4/` (git-ignored).
- Result: GR03b chaotic-group rate 0.2145 +- 0.0109 per year (34 % chaotic): near 0.22, so the masses do not explain the factor 1.4.
- Note: `docs/results/a4-analysis.md`. Small data files are in `docs/results/data/`.

## Next step (one concrete action)
A0 reviews the note and opens the pull request.

## How to verify the current state
```sh
export PATH="/opt/homebrew/opt/rustup/bin:$HOME/.cargo/bin:$PATH"
cargo test -p fring-analysis        # 22 tests pass
```

## Open questions / blockers
- None.

## Decisions made (link ADRs)
- Raw output goes to `runs/a4/<name>/` (git-ignored). Only small files are committed to `docs/results/data/` (AJ via A0, 2026-11-08).
