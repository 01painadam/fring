//! Snapshot read and write, run statistics.
//!
//! Owner: A5 (see `AGENTS.md`). The format is specified in
//! `docs/snapshot-format.md`.
//!
//! The core functions work on `std::io::Read` and `std::io::Write`, so they
//! build for `wasm32-unknown-unknown`. Only the `fs` module touches the file system.

mod bytes;
pub mod frame;

pub use frame::{
    FRAME_TYPE_ROTATING_F32, FRAME_VERSION, Frame, FrameHeader, RotatingVelocities,
    read_frame, read_frame_header, write_frame,
};
