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
- Commit 7: Side by side. `view::split_halves` (floor(w/2) per half, odd widths) and `half_local_x` with host tests (`halves_of_even_and_odd_widths`, `pointer_maps_to_the_same_place_in_each_half`); one render pass, two viewports and scissor rects, one shared frame uniform (both halves have the same size); pan, zoom and the view width use the size of one picture.
- Commit 8: "Set to now" button, labels over the two halves in Side by side (reference time and live time), reference time in the readout in Overlay; `App::reference_is_custom`.
- Commit 9: `?bench` runs every count x compare mode (`counts=`, `modes=`), 3 s warm-up and 10 s per cell at 500 h/s requested; table in the page, one CSV row per cell and the user agent in the console.

## Next step (one concrete action)
AJ or A0 runs browser steps B1 to B5 (below) in Chrome; then commit 10: paste the `?bench` table here, propose the default count (A0), AJ confirms.

## How to verify the current state
Definition of done on the head of commit 9 (exit codes read directly): fmt 0, clippy 0, `cargo test --workspace` 0 (219 passed, 0 failed), wasm32 checks of fring-model, fring-gpu, fring-web all 0, `cargo test -p fring-web --release -- --ignored` 0 (6 native GPU tests). wasm size 869,527 bytes.
`cargo test -p fring-web` (host tests pass); `cargo check -p fring-web --target wasm32-unknown-unknown`.

## Browser steps for AJ or A0 (Chrome on the Mac mini, not headless)
Build: `apps/fring-web/scripts/build.sh`, serve: `python3 -m http.server 8080 -d apps/fring-web/dist`, open http://localhost:8080.

- B1, Particles. Choose 10 000, 300 000, 1 000 000, then 100 000. Expected: "Building N particles..." shows, then the ring reappears at t = 0; the speed readout shows "(limit)" when the slider is above 220 h/s (300 000) or 66 h/s (1 000 000) and nothing at 100 000 and below; the footer count agrees; no error text. Expected rebuild time at 10^6: a second or two.
- B2, Overlay. Compare = Overlay, speed 100 h/s for a few seconds. Expected: warm-coloured faint dots and faint moon dots at their t = 0 places under the blue live dots; the readout ends with "reference t = 0.0 d". Switch View to Rotating: the faint ring stays still while the live outer material falls behind.
- B3, Side by side. Compare = Side by side. Expected: the canvas splits in two, labels "reference start, t = 0.0 d" (left) and "live t = ... d" (right). Drag and scroll in either half: both halves move and zoom together. Switch the four views: the left half shows the reference at its own time (moon dots at their t = 0 places; the rotating and moon views use the reference time). Resize the window to an odd width: no error.
- B4, Set to now. Press "Set to now" with the speed at 100 h/s. Expected: the left half becomes the picture of that moment ("reference set at ...") and stays still while the right half goes on. Press Restart: the reference goes back to t = 0 with the message "The reference was reset to t = 0."
- B5, Bench. Open `http://localhost:8080/?bench` (full matrix, about 3 minutes; or `?bench&counts=1e5&modes=off,side,overlay`). Expected: the table fills one row per cell; the console shows the user agent once, then one CSV row per cell (count, mode, frames, mean fps, p5 fps, worst ms, mean steps/frame, effective h/s, rebuild ms). Copy the table and the CSV rows into this file. Pass: mean fps >= 30 at 100 000 in each mode.
- Zoom check (M24): at the maximum zoom in each mode, the dots move smoothly, with no jitter at a few km.

## Limits of this work
- Out-of-memory on a weak device is handled only by the pre-check `count_fits` against the device limits (default WebGPU limits, 128 MiB binding). A real out-of-memory error is not caught and is not tested.
- No Chrome here: B1 to B5 are not run. Host tests, native GPU tests and the wasm build pass (see the numbers in the final report).
- The reference copy holds planes 0 and 1 (x, y) only (16*cap bytes). A view mode that reads z needs plane 2 in the copy as well.
- The reference projection is redone every frame (no cache); decide from the bench table.

## Open questions / blockers
- Browser checks B1 to B5 need AJ or A0 in Chrome (headless Chrome shows a black canvas).

## Decisions made (A0, 2026-11-02)
- Every rebuild (band, Pandora, Restart, ephemeris wrap, count) resets the reference to the new t = 0; a message only if a "Set to now" reference was lost.
- Half widths are floor(w/2); a 1-pixel gap is fine.
- The reference copy holds planes 0 and 1 (x, y) only: 16*cap bytes. A view mode that reads z needs plane 2 in the copy as well.
- Speed limit: hard clamp min(500, budget/N) h/s shown on the page; the governor stays as a backstop. Budget: `COMPUTE_BUDGET_PARTICLE_STEPS_PER_S` = 6.6e8 (W2-GPU bench, Dp, 30 % of the frame kept for rendering).
- Out-of-memory handling is best effort and not tested.
- No cache of the reference projection for now; decide from the bench table.
