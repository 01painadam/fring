# W2-COMPARE: compare the live ring with a stored reference
Role: A7  Branch: ws/a7-w2-compare  Status: doing
Updated: 2026-11-02

## Goal and acceptance criteria
(PLAN.md section 9, W2-COMPARE; AJ decisions of 2026-11-02.) A "Compare" control: Off, Side by side (reference left, live right), Overlay (reference faint under the live dots). The reference is t = 0; "Set to now" stores the live state as the new reference (GPU copy). Same view mode, linked camera; the reference uses its own time for the frame angle and the moon dots. A "Particles" control (10^4, 10^5, 3x10^5, 10^6) rebuilds the stepper at t = 0; a count that does not fit gives a message; the speed is limited at large counts.
Done when: the three modes and "Set to now" work in Chrome on the Mac mini; the count control works; a `?bench` table (each count x each mode, 500 h/s requested) is below, with mean >= 30 fps at 10^5 in each mode; the default count is chosen from that table (A0 proposes, AJ confirms).

## Done
- Commit 1: pure helpers in `live.rs` (compare mode, count list, speed limit from one budget constant, buffer sizes, `count_fits`, faint head instances), with host tests.
- Commit 2: `App::set_particle_count` (pre-check `count_fits`, instances buffer sized by count, old buffers freed first, speed clamp by `max_hours_for`); native test `one_million_particles_fit_and_run`. Note: the workgroup size is 64, so 10^6 particles give cap 1,000,000 (state 48 MB, reference 16 MB, instances 32 MB).
- Commit 3: `index.html`: Particles select (the old band select is now labelled Band), message before the rebuild, speed readout shows the limit.
- Commit 4: `App::set_reference_now` (copy of planes 0 and 1, 16*cap bytes, before the steps of the next frame), reference buffer made and filled at every `make_stepper` (t = 0), `ref_time_s`; message when a "Set to now" reference is lost; native test `reference_copy_is_the_state_at_the_time_of_the_copy`.
- Commit 5: second projection pass for the reference (own uniform and bind group, own time for the frame angle and moons, written into the second half of the instance buffer), skipped when Compare is Off; `App::set_compare`; native test `reference_projection_matches_the_twin_in_the_second_half_of_the_buffer` (26 particles within 500 km, worst error 0.151 of the tolerance, non-zero low words).
- Commit 6: Overlay. `points.wgsl` draws colour ids >= 10 as a warm faint reference; `draw` draws the reference range first, then the live range; Compare select (Off, Overlay) on the page; native test `render_shader_makes_a_pipeline`.

## Next step (one concrete action)
Commit 7: Side by side. `view::Layout` (half rectangles, odd widths, pointer mapping) with host tests; two viewports in `draw`; per-half frame uniform; input mapping in `pan`/`zoom`; add the option to the Compare select.

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
