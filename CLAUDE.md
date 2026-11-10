# CLAUDE.md

@AGENTS.md

The rules in `AGENTS.md` apply. This file adds instructions for Claude Code only.

## Operating mode: lean

AJ uses a Pro subscription. Usage limits are the main constraint, so:

- **One main session at a time.** The main session is A0. It runs board tasks with subagents (one agent per task, each in its own worktree) and keeps the board.
- **Run tasks in parallel wherever possible (AJ, 2026-11-08).** Start every ready task that can run beside the others, and run their plan rounds at the same time. Limits:
  - One agent per role at a time. Tasks of the same role edit the same files, so they run one after the other.
  - Only tasks whose files do not overlap.
  - A6 validation work stays independent of the implementation (see `AGENTS.md`).
  - If usage gets near the limit, let the running agents finish and start no new ones.
- **Do not spawn an agent for a task that the main session can do in a few steps.**
- **Pick the next tasks** from `docs/status/BOARD.md`: the `todo` tasks in the current phase whose blockers are `done`, in board order. If none, stop and tell AJ.

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

## Toolchain on AJ's Mac

Rust comes from Homebrew `rustup`, which is keg-only. If `cargo` is not found, start the command with `export PATH="/opt/homebrew/opt/rustup/bin:$HOME/.cargo/bin:$PATH" &&`. `~/.cargo/bin` holds `wasm-bindgen`, which `apps/fring-web/scripts/build.sh` needs.

## Worktrees: remove the stale ones as you go (AJ, 2026-11-10)

AJ gave standing permission: "from now on lets make a point of deleting stale work trees - you have permission to do the tidyup on that as we go". Do not ask again for each one.

- Stale means: its PR is merged or closed, or AJ dropped the task. Remove it soon after the merge, in the same session.
- Check first:
  - `git -C <wt> status --porcelain` is empty, and `git -C <wt> log @{u}..` shows no unpushed commits.
  - Nothing under its git-ignored folders (`runs/`, logs) is still needed: a report cites it, or an open task or gate uses it. `git worktree remove` deletes those files too. Move the needed data to the main checkout's `runs/` first, or keep the worktree and say why.
- Then: `git worktree remove .claude/worktrees/<name>` and `git branch -d ws/<branch>`. Never use `--force` or `-D` without asking AJ. The permission covers local worktrees and local branches only; leave remote branches alone (the auto-mode check refuses `git push origin --delete`, and AJ has not asked for it).
- Move git-ignored run data that may still be needed to `runs/_worktrees/<name>/` in the main checkout before the removal (same disk, so the move is instant).
- Why: each worktree has its own `target/` of 3 to 5 GB. On 2026-11-09 the disk filled up (12 worktrees, about 48 GB of `target/`) and stopped the K4 runs overnight. Check `df -h /System/Volumes/Data` before long runs.

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
