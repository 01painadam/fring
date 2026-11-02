//! Snapshot read and write, run statistics.
//!
//! Owner: A5 (see `AGENTS.md`). The format is specified in
//! `docs/snapshot-format.md`.
//!
//! The core functions work on `std::io::Read` and `std::io::Write`, so they
//! build for `wasm32-unknown-unknown`. Only the `fs` module touches the file system.

mod bytes;
pub mod chaos;
pub mod frame;
pub mod fs;
pub mod layout;
pub mod meta;
pub mod moons;
pub mod stats;

pub use chaos::{CHAOS_COLUMNS, CHAOS_VERSION, ChaosRow, read_chaos_csv, write_chaos_csv};

pub use stats::{
    STATS_COLUMNS, STATS_VERSION, StatsRead, read_stats_csv, read_stats_csv_report,
    write_stats_csv, write_stats_header, write_stats_row,
};

pub use moons::{MOONS_VERSION, MoonRecord, MoonsFile, read_moons, write_moons};

pub use frame::{
    FRAME_TYPE_ROTATING_F32, FRAME_VERSION, Frame, FrameHeader, RotatingVelocities, read_frame,
    read_frame_header, write_frame,
};
pub use layout::{RunDir, frame_file_name, parse_frame_file_name};
pub use meta::{ChaosMeta, RunMeta, read_meta, write_meta};
