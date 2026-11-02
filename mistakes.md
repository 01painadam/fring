# Mistakes and lessons

Read this file at the start of a session. When a mistake costs time, add an entry so that no agent makes it twice. Newest first. Each entry: what happened, why, and the rule now.

## 2026-11-02 (phase 2, A1-ANALYSIS)

### M19. A0 read Kepler shear as chaos

- **What:** at gate G1, A0 took the growth of a moon offset between years 1 and 12 (a factor of 11 to 25) as a Lyapunov rate of 0.22 to 0.29 per year, "as in the literature". A1-ANALYSIS showed that linear (shear) growth alone gives a factor of up to 12, which is ln(12)/11 = 0.23 per year. The plain rate L/t of a regular orbit falls only like ln(t)/t, and it was still 0.19 per year at 100 years. The converged rate (tangent vector, slope over years 30 to 100) is 0.22 per year in the 26 % of runs that are chaotic, and about 0.03 per year (not exponential) in the others.
- **Rule:** a finite-time growth rate of an orbit separation is not a Lyapunov exponent. Compare it with the shear growth of a regular orbit (or use MEGNO, which gives 2 for linear growth), and fit the late-time slope of ln(d) only after the shear part is small.

### M20. A shadow-orbit Lyapunov rate measured rounding noise

- **What:** the first moon Lyapunov runs used shadow orbits with renormalisation. The rate depended on the start offset (0.95 per year at 1 cm, 0.09 at 100 m) and on the renormalisation interval. Rounding and restart noise in the radial direction grew by Kepler shear, and for a small offset it was larger than the real growth.
- **Rule:** a Lyapunov estimate must not change with the offset d0 or the renormalisation interval. Show that it does not, or use the tangent (variational) vector, which has no difference of two orbits. The particle chaos indicator (ADR 0008) is on the d0 plateau, so it is not affected.

### M21. IAS15 restarted with the running step was wrong by 10⁻⁴

- **What:** `Ias15::new` restarted with the step of the running integration (1,500 s on a 52,000 s orbit) had a relative error of 10⁻⁴ in the first step.
- **Rule:** start or restart IAS15 with a first step of about 1 % of the dynamical time, and let it adapt.

### M22. The `a0_km` of `chaos.csv` is not the orbital radius

- **What:** `a0_km` is the point-mass Keplerian semi-major axis (vis-viva with GM only). For a circular orbit near the core in J2 to J6 it is about 640 km larger than the orbital radius. A map against `a0_km` puts every particle in the wrong place.
- **Rule:** for maps against radius, take the initial radius from the config (or recompute the initial states), not `a0_km`. `docs/snapshot-format.md` section 8a says this now.

## 2026-11-01 (phase 2)

### M17. A small sample hid the tail of an error distribution

- **What:** the G1 spike compared 47 regular particles and found that option D met PLAN section 10 (max Δλ 6.1 × 10⁻⁶ rad). The G2 test with 4,890 regular particles found a max Δλ of 1.15 × 10⁻⁴ rad, 19 times larger. The median was the same; only the tail was different. G2 needed one more mode (D+) and one more round of work.
- **Rule:** a precision or tolerance claim needs the full distribution (max, 99th percentile, median) over thousands of cases, not tens. Size the sample before a gate decision depends on it. A test that filters cases (for example by a shadow rule) must also assert a minimum number of cases.

### M18. A0 wrote a decision for a subagent but did not send it

- **What:** the G2 subagent stopped and asked how to continue. A0 wrote the decision in the conversation but did not call SendMessage. The subagent waited until AJ asked for the status.
- **Rule:** when a subagent waits for a decision, send the decision in the same turn in which you write it, and confirm that the agent has resumed.
- **It happened again (G3, 2026-11-02):** A0 wrote "I am sending the decision now" and ended the turn without the call. AJ had to say "continue".
- **Stricter rule:** make the SendMessage call first. Write "sent" only after the tool result says that the agent has resumed. If the send tool is deferred, load it before you write the message.

### M16. WGSL on Metal is not IEEE: fast math removes double-single error terms

- **What:** wgpu-hal compiles Metal shaders with fast math on, and naga has no switch to turn it off. The compiler rewrote `(a + b) - a` as `b`, so plain double-single arithmetic had only f32 accuracy. Knuth's TwoSum with a guard passed alone, but failed again once the compiler inlined it into the double-single add (G1, ADR 0006).
- **Rule:** never assume IEEE evaluation order in WGSL. Guard every error-free transformation (multiply key sums by a 1.0 read from a uniform). Test the real kernels against their Rust twins on the GPU, not only the primitives. Any GPU path that uses double-single runs the self-test at start-up and fails closed.

## 2026-11-01 (phase 1)

### M12. The plan's baseline method was not accurate enough for the science

- **What:** PLAN 7.3 chose the leapfrog at 100 to 200 steps per orbit. In C2 it slowed the apse precession at the core by 9.7 % and gave 52,000 km of along-track error after 100 orbits (ADR 0003).
- **Rule:** before you build on a planned method, measure its error on the quantity that the science needs (here ϖ̇ = n − κ, a small difference of large numbers), not only on a generic test.

### M13. A long background script ran twice

- **What:** the V2 script was started twice (the first start survived a shell exit). Both copies used 4 cores and wrote the same files.
- **Rule:** before you start a long run, check `ps` for a copy; after the start, check that only one copy runs.

### M14. A pipe hid an exit code

- **What:** `cmd | tail -1; echo $?` printed 0, the status of `tail`, although `cmd` had exited with 2.
- **Rule:** check exit codes without a pipe, or use `${pipestatus[1]}` in zsh.

### M15. Evidence was left in a temporary folder

- **What:** a comparison plot for the gate was only in the session scratchpad, which is deleted later.
- **Rule:** copy gate evidence into the repository (for example `docs/gates/G<n>/`) before the session ends.

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
