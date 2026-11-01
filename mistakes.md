# Mistakes and lessons

Read this file at the start of a session. When a mistake costs time, add an entry so that no agent makes it twice. Newest first. Each entry: what happened, why, and the rule now.

## 2026-11-01 (phase 0)

### M1. Merging the base of stacked PRs closed the other PRs

- **What:** PRs #4 to #7 had the base `ws/a0-f2`, the branch of PR #3. `gh pr merge 3 --merge --delete-branch` deleted that branch. GitHub then **closed** #4 to #7; it did not move them to `main`. `gh` also deleted the local branch.
- **Repair used:** push the old head commit back to the branch name (`git push origin "${SHA}:refs/heads/ws/a0-f2"`), `gh pr reopen N`, `gh pr edit N --base main`, merge, then delete the branch.
- **Rule:** before you merge the base PR of a stack, move every dependent PR to `main` with `gh pr edit N --base main`. Or merge the base PR without `--delete-branch`, and delete the branch only after the other PRs are merged.

### M2. zsh changed a git refspec

- **What:** `git push origin $SHA:refs/heads/x` failed. zsh read `:r` as a modifier of `$SHA` ("remove extension").
- **Rule:** in zsh, put braces and quotes around variables before a colon: `"${SHA}:refs/heads/x"`.

### M3. A command chain continued after a failure

- **What:** `step1 && step2; gh pr create ...`. Step 1 failed (cargo not found), so no commit was made, but `gh pr create` still ran and failed with "No commits between main and ...".
- **Rule:** join dependent steps with `&&` only. Never put `;` before a step that needs the earlier steps.

### M4. `cargo` not found

- **What:** Rust comes from Homebrew rustup, which is keg-only. The tool shell does not always load `~/.zshrc`.
- **Rule:** start cargo commands with `export PATH="/opt/homebrew/opt/rustup/bin:$PATH" &&` (also in `CLAUDE.md`).

### M5. A subagent's worktree disappeared

- **What:** the Agent tool with `isolation: "worktree"` deletes the worktree when the agent stops without changes. A plan-first round (no edits) lost it, and the agent could not continue.
- **Rule:** create the worktree yourself before you start the agent: `git worktree add .claude/worktrees/<name> -b ws/<role>-<task> origin/<base>` (`.claude/worktrees/` is in `.git/info/exclude`). Remove it after the merge.

### M6. The Write tool refused a file for a subagent

- **What:** the Write tool refused `docs/literature/summary.md` for a subagent ("Subagents should return findings as text"). It allowed other `.md` files.
- **Rule:** tell each subagent: on a refusal, do not try another route; return the full text of the file. The main session writes it and says so in the PR.

### M7. Neighbouring rows of the board conflicted

- **What:** two branches changed adjacent rows of `docs/status/BOARD.md`. Git reports changes on adjacent lines as a conflict.
- **Rule:** when several task branches run at the same time, only A0 edits the board. Otherwise, merge the other branch into yours first and resolve the conflict.

### M8. Actions outside the task were blocked

- **What:** the permission check blocked an edit of `~/.zshrc`, `gh pr edit` and `gh pr ready`. AJ allowed them later.
- **Rule:** for an action outside the repository or on GitHub that AJ has not allowed, ask AJ first, not after the block.

### M9. A count in a handoff was wrong

- **What:** the F1 handoff said "30 tasks". The board has 32, because "V1 to V3" in the plan is three tasks. A grep found the error.
- **Rule:** count with a command before you write a number.

### M10. A text replacement in a test did not match

- **What:** a config test replaced a block that starts with `kind = "core"`, but the example file had a comment after that text. The guard `assert!(EXAMPLE.contains(from))` found it at once.
- **Rule:** keep a "does the text exist" guard in every test that edits text.

### M11. `timeout` does not exist on macOS

- **Rule:** use the `timeout` parameter of the tool, not the shell command.
