# W2-COMPARE: compare the live ring with a stored reference
Role: A7  Branch: ws/a7-w2-compare  Status: doing
Updated: 2026-11-02

## Goal and acceptance criteria
(PLAN.md section 9, W2-COMPARE; AJ decisions of 2026-11-02.) A "Compare" control: Off, Side by side (reference left, live right), Overlay (reference faint under the live dots). The reference is t = 0; "Set to now" stores the live state as the new reference (GPU copy). Same view mode, linked camera; the reference uses its own time for the frame angle and the moon dots. A "Particles" control (10^4, 10^5, 3x10^5, 10^6) rebuilds the stepper at t = 0; a count that does not fit gives a message; the speed is limited at large counts.
Done when: the three modes and "Set to now" work in Chrome on the Mac mini; the count control works; a `?bench` table (each count x each mode, 500 h/s requested) is below, with mean >= 30 fps at 10^5 in each mode; the default count is chosen from that table (A0 proposes, AJ confirms).

## Done
- Commit 1: pure helpers in `live.rs` (compare mode, count list, speed limit from one budget constant, buffer sizes, `count_fits`, faint head instances), with host tests.

## Next step (one concrete action)
Commit 2: make the particle count a run-time setting in `web.rs` (instances buffer sized by count, rebuild path).

## How to verify the current state
`cargo test -p fring-web` (host tests pass); `cargo check -p fring-web --target wasm32-unknown-unknown`.

## Open questions / blockers
- Browser checks B1 to B5 need AJ or A0 in Chrome (headless Chrome shows a black canvas).

## Decisions made (A0, 2026-11-02)
- Every rebuild (band, Pandora, Restart, ephemeris wrap, count) resets the reference to the new t = 0; a message only if a "Set to now" reference was lost.
- Half widths are floor(w/2); a 1-pixel gap is fine.
- The reference copy holds planes 0 and 1 (x, y) only: 16*cap bytes. A view mode that reads z needs plane 2 in the copy as well.
- Speed limit: hard clamp min(500, budget/N) h/s shown on the page; the governor stays as a backstop. Budget: `COMPUTE_BUDGET_PARTICLE_STEPS_PER_S` = 6.6e8 (W2-GPU bench, Dp, 30 % of the frame kept for rendering).
- Out-of-memory handling is best effort and not tested.
- No cache of the reference projection for now; decide from the bench table.
