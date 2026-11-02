# W2-DROP-GPU: dropped particles in GpuStepper (ICR-006)
Role: A3  Branch: ws/a3-w2-drop-gpu  Status: doing
Updated: 2026-11-02 20:15

## Goal and acceptance criteria
ICR-006 in `crates/fring-gpu`: a drop reason for each particle in the status buffer (`8 + cap` words), an early return for a dropped particle in the D and Dp step kernels, counts by reason, and an opt-in moon-radius rule (minimum distance over each drift between two kicks, linear interpolation of the particle-minus-moon vector, mean radius from `params/moons.toml`). Done when: the tests of ICR-006 item 4 pass natively; all existing tests pass and `GpuBackend` results are bit for bit unchanged with the rule off; `cargo check -p fring-gpu --target wasm32-unknown-unknown` passes; `fring-bench` shows no slow-down beyond its noise.

## Done
- Plan accepted by A0 (decisions 1 to 8 in the A0 go-ahead). Baseline bench binary built at 938344d.

## Next step (one concrete action)
C2: commit the golden-bits test (`tests/gpu_drop.rs`) on the unchanged code.

## How to verify the current state
`cargo test -p fring-gpu --release --test gpu_drop -- --ignored --nocapture`

## Open questions / blockers
- None.

## Decisions made (link ADRs)
- Separate entry points `step_d_r` and `step_dp_r` for the radius rule; 2 sync-time moon rows per block; prm grows to 44 words; setter `GpuStepper::set_moon_radii_km`; rule with chaos or mode C is a `Config` error.
- With the rule off, a failed particle is now frozen (ICR-006 item 2). Results before the first failure are bit for bit unchanged.
- `MAX_MOONS` stays 4. Status words 6 and 7 (moons 4 and 5) stay reserved: ICR-006 allows 6 moons, the kernels take 4.
- Accuracy of the rule (A0 estimate, to be checked): the moon path between two kicks has a sagitta L^2 / (8 a) = 5,940^2 / (8 x 139,380) = 31.6 km, so only the particle-minus-moon vector is interpolated. That vector is almost straight: tidal acceleration 3 n^2 d = 3 x (1.17e-4)^2 x 50 = 2e-6 km/s^2 gives 0.03 km at mid-drift; the moon's pull at 43 km (0.01071 / 43^2 = 5.8e-6 km/s^2) gives about 0.09 km. The error of the segment test is about 0.1 km against a 43 km radius.
