//! `meta.json`: facts about the run. `fring-run` fills the values; this crate
//! does not call git and does not read a clock.

use crate::frame::FRAME_VERSION;
use crate::moons::MOONS_VERSION;
use crate::stats::STATS_VERSION;
use fring_model::{Error, Result};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};

/// Contents of `meta.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunMeta {
    /// Version of the code (for example the crate version).
    pub code_version: String,
    /// Git commit hash of the code.
    pub git_hash: String,
    /// Machine description (host name, CPU or GPU).
    pub machine: String,
    /// Wall-clock time of the run in seconds.
    pub wall_time_s: f64,
    /// Version of the frame files.
    pub frame_format_version: u32,
    /// Version of `moons.bin`.
    pub moons_format_version: u32,
    /// Version of `stats.csv`.
    pub stats_format_version: u32,
}

impl RunMeta {
    /// A meta record with the format versions of this crate and the other
    /// fields given by the caller.
    pub fn new(code_version: &str, git_hash: &str, machine: &str, wall_time_s: f64) -> Self {
        Self {
            code_version: code_version.to_string(),
            git_hash: git_hash.to_string(),
            machine: machine.to_string(),
            wall_time_s,
            frame_format_version: FRAME_VERSION,
            moons_format_version: MOONS_VERSION,
            stats_format_version: STATS_VERSION,
        }
    }
}

/// Writes `meta.json` (pretty-printed).
pub fn write_meta<W: Write>(w: &mut W, meta: &RunMeta) -> Result<()> {
    serde_json::to_writer_pretty(&mut *w, meta)
        .map_err(|e| Error::Io(format!("meta.json: {e}")))?;
    w.write_all(b"\n")?;
    Ok(())
}

/// Reads `meta.json`.
pub fn read_meta<R: Read>(r: &mut R) -> Result<RunMeta> {
    serde_json::from_reader(r).map_err(|e| Error::Io(format!("meta.json: {e}")))
}
