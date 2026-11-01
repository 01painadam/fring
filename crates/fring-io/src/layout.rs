//! Paths inside a run directory. Pure string work, no file system access.

use std::path::{Path, PathBuf};

/// A run directory (`docs/snapshot-format.md` section 3).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunDir {
    root: PathBuf,
}

impl RunDir {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn config_path(&self) -> PathBuf {
        self.root.join("config.toml")
    }

    pub fn meta_path(&self) -> PathBuf {
        self.root.join("meta.json")
    }

    pub fn moons_path(&self) -> PathBuf {
        self.root.join("moons.bin")
    }

    pub fn stats_path(&self) -> PathBuf {
        self.root.join("stats.csv")
    }

    pub fn frames_dir(&self) -> PathBuf {
        self.root.join("frames")
    }

    /// `frames/NNNNNN.bin`: at least 6 digits.
    pub fn frame_path(&self, index: u64) -> PathBuf {
        self.frames_dir().join(frame_file_name(index))
    }
}

/// File name of frame `index`: `000042.bin`.
pub fn frame_file_name(index: u64) -> String {
    format!("{index:06}.bin")
}

/// Parses a frame file name (6 or more digits, then `.bin`) to its index.
pub fn parse_frame_file_name(name: &str) -> Option<u64> {
    let digits = name.strip_suffix(".bin")?;
    if digits.len() < 6 || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}
