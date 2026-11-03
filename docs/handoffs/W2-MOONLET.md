# W2-MOONLET: moonlet scenario and moonlet view in the web page
Role: A7 (done by the A0 session)  Branch: ws/a7-w2-moonlet  Status: review
Updated: 2026-11-03 09:30

## Goal and acceptance criteria
From `docs/PLAN.md` section 9 (added by AJ, 2026-11-03): a "Moonlet" scenario in the web page: all particles start in a loosely packed moonlet (`moonlet` distribution, ICR-008) at a selected orbital radius, uniform in a selected packing radius, 180° from Prometheus at t = 0. A "Velocity" control selects the start velocities: synchronous, centre or circular. A view "On the moonlet" is centred on the moonlet centre's circular orbit from t = 0 and turns with it. *Done when:* host tests pass (the view keeps the centre at the origin; a synchronous particle has zero view velocity at t = 0); in Chrome on the Mac mini the scenario runs in each velocity mode and the view works with Compare; `?bench` at 10⁵ still gives mean 30 fps or more.

## Done
- ICR-008 (PR #36): `Distribution::Moonlet` and `MoonletVelocity` in `fring-model`, with tests.
- `live.rs`: `Band::Moonlet { a_km, radius_km, velocity }`; `Setup::moonlet_longitude_deg` (Prometheus at t = 0 plus 180°) and `Setup::moonlet_guide`; `Guide` (radius, longitude at t = 0, exact n); `ViewFrame::Moonlet(Guide)` (`ViewFrame` lost `Eq`, because `Guide` holds f64); `view_transform` turns at n of the guide, oriented as the moon views (Saturn down, motion to the left); `moonlet_velocity(index)`; `MOONLET_RADIUS_KM = 20`.
- `view.rs`: `Camera::moonlet_view(radius)`, half-height 10 packing radii (200 km at 20 km).
- `web.rs`: `set_moonlet(a_km, radius_km, velocity)`, `moonlet_defaults_km()`, view 4 in `set_view` (an error unless the band is a moonlet). A new orbit or radius resets the camera of the moonlet view; a new velocity mode keeps it, to compare the modes at one zoom.
- `index.html`: Band option "Moonlet (loosely packed)" with the rows Orbit, packing radius and Velocity (the "a from / to" row hides); View option "On the moonlet (turns with its centre)", greyed unless the band is the moonlet. Choosing the moonlet band switches to the moonlet view; leaving it while in that view goes back to Inertial. A note paragraph explains the scenario.
- Checks: `cargo test -p fring-web` 34 passed (2 new, and 1 new camera test in `view.rs`); `cargo test -p fring-web -- --ignored` 7 GPU tests passed; `cargo clippy -p fring-web --target wasm32-unknown-unknown -- -D warnings` clean; `build.sh`: wasm 925,605 bytes (main: 916,934).
- Not done: the browser check. The Chrome extension was not connected in this session.

## Next step (one concrete action)
AJ: the Chrome check M1 to M6 below. Then A0 merges the PR (the merge deploys the public page).

## How to verify the current state
Build and serve: `apps/fring-web/scripts/build.sh` then `python3 -m http.server 8080 -d apps/fring-web/dist`, open `http://localhost:8080` in Chrome.

Expected numbers (ICR-008, R = 20 km at 140,222 km, orbit 14.9 h): the guiding centres spread over ±4R, ±2R and ±R (synchronous, centre, circular), so the strand grows by 3π Δa per orbit: about 1,508, 754 and 377 km per orbit, and wraps the ring in about 1.0, 2.0 and 4.0 years (17, 35 and 70 s at 500 h/s).

- M1. Band "Moonlet (loosely packed)". Expected: the view changes to "On the moonlet"; a ball of dots at the centre of the window (radius 20 km, window half-height 200 km); Saturn is straight down.
- M2. Velocity "Synchronous", Play at a low speed (about 1 h/s). Expected: the ball starts at rest, then stretches along the orbit: the inner (lower) part moves to the left (ahead), the outer part to the right. Zoom out to follow it: about 1,500 km of strand after one orbit (14.9 h).
- M3. Velocity "Centre's velocity" and "Own circular orbit". Expected: each restarts at t = 0; the strand grows about half and a quarter as fast as in M2. The camera stays where it was.
- M4. Orbit 139600 km (close to Prometheus) and packing radius 50 km. Expected: a larger ball; Prometheus passes and pulls the strand; the "Dropped" counts can rise. A bad entry (radius 0, or radius at least the orbit) shows a message and changes nothing.
- M5. Compare "Side by side" and "Overlay" in the moonlet view, then "Set to now". Expected: the reference uses the same moonlet view at its own time.
- M6. Band "Core (narrow)". Expected: the view goes back to Inertial and "On the moonlet" is greyed. Then `http://localhost:8080/?bench&counts=100000`: mean 30 fps or more in each compare mode (the bench uses the default band; the kernels did not change).

## Open questions / blockers
- None for the code. `fring-analysis::map::guiding_radius_km` uses the initial |r| for a moonlet (ICR-008, "Roles affected"); a chaos map of a moonlet run would need it changed.

## Decisions made (link ADRs)
- AJ (2026-11-03): a Velocity control with all three modes; the start is fixed 180° from Prometheus; a view that turns with the moonlet centre at t = 0 (ICR-008).
- A0: the moonlet view starts at 10 packing radii; a velocity change keeps the camera.
