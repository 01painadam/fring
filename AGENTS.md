# AGENTS.md

Rules for every agent (human or AI) that works in this repository. Read this file first, then your handoff file, then only the sections of `docs/PLAN.md` that your task needs.

## Project

`fring` simulates Saturn's F ring with its shepherd moons Prometheus and Pandora over decades, with inelastic collisions, on a GPU (wgpu/Metal) with a CPU f64 reference. Results go to a web app and a written explainer. Full plan: `docs/PLAN.md`.

## Ownership

Edit only the directories your role owns. To change anything else, write an interface change request (see Coordination).

| Role | Owns |
|---|---|
| A0 Lead | `Cargo.toml`, `AGENTS.md`, `CLAUDE.md`, `docs/` (except `docs/literature/`), `crates/fring-model/` (except `src/physics/`), CI |
| A1 Literature | `docs/literature/`, `params/` |
| A2 CPU | `crates/fring-cpu/`, `crates/fring-model/src/physics/` (A0 reviews each change) |
| A3 GPU | `crates/fring-gpu/` (except `shaders/collide/`) |
| A4 Collisions | `crates/fring-collide/`, `crates/fring-gpu/shaders/collide/` |
| A5 I/O and runner | `crates/fring-io/`, `apps/fring-run/`, `apps/fring-bench/` |
| A6 Validation | `validation/` |
| A7 Web | `apps/fring-web/` |
| A8 Analysis | `crates/fring-analysis/`, `docs/results/` |

Every role can edit its own handoff files and its own row on the board.

`crates/fring-model/src/physics/` holds the physics that the CPU, GPU and web code share: force functions, analytic formulas, moon interpolation, frame conversions and initial conditions. It is generic over `Real` and must compile for wasm32 (AJ decision, 2026-11-01).

A6 writes oracle tests from the physics specification only. A6 does not read the implementation code of A2, A3 or A4.

## Coordination

Sessions stop without warning (usage limits). All state that the next session needs must be in the repository, not in a conversation.

### Files

| File | Purpose | Who writes |
|---|---|---|
| `docs/status/BOARD.md` | One row per task: ID, role, status, branch, blocked by | Each role edits only its own rows; A0 adds tasks |
| `docs/handoffs/<task-id>.md` | Current state of one task | The role that owns the task |
| `docs/interfaces/ICR-NNN.md` | Request to change a shared interface | Any role; A0 decides |
| `docs/gates/G<n>.md` | Evidence for a gate and AJ's decision | A0 writes; AJ approves |
| `docs/adr/NNNN-<title>.md` | Decisions that are not obvious from the code | Any role; A0 approves |

Task status values: `todo`, `doing`, `blocked`, `review`, `done`.

### Handoff file template

```markdown
# <task-id>: <title>
Role: <A-n>  Branch: ws/<a-n>-<task>  Status: <status>
Updated: <YYYY-MM-DD HH:MM>

## Goal and acceptance criteria
(copy from PLAN.md)

## Done
- ...

## Next step (one concrete action)
...

## How to verify the current state
<exact commands and expected output>

## Open questions / blockers
- ...

## Decisions made (link ADRs)
- ...
```

### Session routine

At the start:

1. `git pull` and check out your task branch.
2. Read `docs/status/BOARD.md`, your handoff file and `mistakes.md`. Do not read the whole repository.
3. Run the "How to verify" commands. If the result does not agree with the handoff, fix the handoff first.

During the work:

- Commit and update the handoff file at each checkpoint: after each passing test, before each large change, and at least every 30 minutes of work. A work-in-progress commit is acceptable on a task branch (prefix the message with `wip:`).
- Keep the "Next step" line correct at all times. If the session stops, this line is the only instruction that the next session gets.

At the end of a task:

1. All checks pass (Definition of done).
2. Status `review` on the board; open a pull request with the evidence.
3. A0 reviews and merges, then sets `done`.

### Interfaces

Shared interfaces are the `Backend` trait and types in `fring-model`, the snapshot format, the config schema and the collision specification. To change one:

1. Write `docs/interfaces/ICR-NNN.md`: what, why, which roles it affects, migration.
2. Set your task to `blocked` if you cannot continue without it.
3. Do not change the interface until A0 writes "Accepted" in the ICR.

## Definition of done

- `cargo fmt --all -- --check` passes.
- `cargo clippy --workspace --all-targets -- -D warnings` passes.
- `cargo test --workspace` passes (long tests may be marked `#[ignore]` and run in CI or at gates).
- The acceptance criteria of the task are met, and the pull request shows the evidence (numbers, plots or logs).
- The handoff file is complete and the board is updated.
- The pull request description has a short plain-language summary for a Rust beginner.

## Gates

Phases end at gates G0 to G6 (`docs/PLAN.md`, section 9). No role starts a task of the next phase before AJ writes "Approved" in `docs/gates/G<n>.md`.

## Conventions

- **Units:** km, s, kg. GM in km³ s⁻². Radians in code, degrees only in configs and plots. Put the unit in the name or doc comment of every physical quantity.
- **Frames:** inertial (Saturn-centred, equatorial), rotating at n₀, and local guiding-centre coordinates (x radial, y azimuthal, z vertical). See `docs/PLAN.md` section 4.2.
- **Precision:** CPU is f64 and is the reference. GPU is f32 (with double-single where ADR says so). Every WGSL function has a Rust twin and a comparison test.
- **Constants:** only from `params/`. Every value has a citation in `params/sources.md`.
- **Randomness:** only the seeded Philox RNG in `fring-model`.
- **Tolerances:** do not relax a test tolerance without an ADR approved by A0.
- **Chaotic orbits:** compare statistics, never trajectories.
- **Writing:** British English. No em dashes in docs. Prefer short, clear text.

## Commands

A0 keeps this list correct. `rust-toolchain.toml` pins the compiler (Rust 1.99, with rustfmt, clippy and the wasm32 target). rustup installs it on first use.

```sh
cargo build --workspace
cargo test --workspace
cargo test -p <crate>                       # test one crate (cheaper)
cargo test --workspace -- --ignored         # long tests
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all
cargo check -p fring-model --target wasm32-unknown-unknown   # fring-model must build for the web
```

CI (`.github/workflows/ci.yml`) runs `cargo fmt --check`, clippy, the tests and the wasm32 check on macOS for each pull request and each push to `main`.

These commands work after their tasks are done. Until then, the apps only print "not implemented yet".

```sh
cargo run -p fring-run --release -- --config runs/<name>.toml   # after R1
cargo run -p fring-bench --release                              # after B1
```

Validation commands (A6; a `uv` project in `validation/`, Python 3.12, needs `uv`):

```sh
cd validation && uv sync                                  # install rebound, reboundx, numpy, scipy, matplotlib, pytest
cd validation && uv run pytest -q                         # reference force vs finite difference, PLAN 7.2 and REBOUNDx
cd validation && uv run python scripts/v1_precession.py   # V1, about 5 minutes; writes reference/v1_* and reports/v1_precession.png
cd validation && uv run python scripts/v2_moons.py --check   # V2, recompute the first day, compare with the committed file
cd validation && uv run python scripts/v2_moons.py           # V2, about 15 minutes (5 parallel runs); writes reference/v2_*, reports/v2_*.png
```

Web commands: A7 adds them here when task W1 is done.
