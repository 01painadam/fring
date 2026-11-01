# CLAUDE.md

@AGENTS.md

The rules in `AGENTS.md` apply. This file adds instructions for Claude Code only.

## Operating mode: lean

AJ uses a Pro subscription. Usage limits are the main constraint, so:

- **One main session at a time.** The main session takes one role per task (it reads the role from the task row on the board). Do not run many roles in parallel.
- **At most one subagent at a time,** and only for these cases:
  - A6 validation work, because it must be independent of the implementation.
  - A read-only search of the repository that would otherwise fill the main context.
- **Do not spawn an agent for a task that the main session can do in a few steps.**
- **Pick the next task** from `docs/status/BOARD.md`: the first `todo` task in the current phase whose blockers are `done`. If none, stop and tell AJ.

## Model use

Default model: Sonnet. Switch to Opus only for these tasks, then switch back.

| Use Opus for | Reason |
|---|---|
| A0 interface design (F2) and gate reviews | Mistakes here affect every role |
| G1 GPU precision spike | Numerical analysis with f32 |
| K1 collision specification | Statistical physics design |
| Debugging a validation failure that is not obvious after one attempt | Expensive to get wrong |

Use Haiku (or do it without a model call) for formatting, changelogs and small edits to docs.

## Save context and usage

- Start from the handoff file. Read only the files and `PLAN.md` sections that the task needs.
- Use `cargo test -p <crate>` and filtered tests (`cargo test <name>`) instead of the full workspace while you work. Run the full Definition of done once, at the end.
- Pipe long output through `tail` or `grep`. Do not paste large logs into the context; save them to `runs/logs/` and quote the important lines.
- Long simulations run in the background (`run_in_background`). Do not wait in a loop for them; write the run ID in the handoff file and continue with other work, or end the session.
- Do not re-read a file that you just wrote.

## Before the session ends

If you are near a usage limit, or the task is at a natural stop:

1. Commit (`wip:` prefix if not finished).
2. Update the handoff file, especially "Next step" and "How to verify".
3. Update your row on the board.
4. Tell AJ in two or three sentences: task, state, next step.

## Talking to AJ

- AJ is new to Rust and was a physicist. Explain Rust choices briefly; do not explain physics basics.
- Show calculations and numbers, not only conclusions.
- Use ASD-STE-100 simplified English and British spelling.
- Ask before a decision that is expensive to reverse (interfaces, file formats, tolerances, dependencies).
