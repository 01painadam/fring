//! File system wrappers around the `Read`/`Write` functions.
//!
//! Every write goes to `<name>.tmp` first and is then renamed, so a reader
//! never sees a half-written file. Do not use this module in the browser:
//! it compiles for wasm32 but file access fails there.

use crate::chaos::{ChaosRow, read_chaos_csv, write_chaos_csv};
use crate::drops::{DropRow, read_drops_csv, write_drops_csv};
use crate::frame::{Frame, read_frame, write_frame};
use crate::layout::{RunDir, parse_frame_file_name};
use crate::meta::{RunMeta, read_meta, write_meta};
use crate::moons::{MoonsFile, read_moons, write_moons};
use crate::stats::{read_stats_csv, write_stats_csv};
use fring_model::{Result, StepStats};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

/// `path` with `.tmp` added to the file name.
fn tmp_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".tmp");
    path.with_file_name(name)
}

/// Writes a file atomically: `f` writes to `<path>.tmp`, then it is renamed.
/// On error the temporary file is removed.
pub fn write_atomic<F>(path: &Path, f: F) -> Result<()>
where
    F: FnOnce(&mut BufWriter<File>) -> Result<()>,
{
    let tmp = tmp_path(path);
    let result = (|| {
        let mut w = BufWriter::new(File::create(&tmp)?);
        f(&mut w)?;
        w.flush()?;
        w.into_inner().map_err(|e| e.into_error())?.sync_all()?;
        fs::rename(&tmp, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result
}

fn open(path: &Path) -> Result<BufReader<File>> {
    let f =
        File::open(path).map_err(|e| fring_model::Error::Io(format!("{}: {e}", path.display())))?;
    Ok(BufReader::new(f))
}

impl RunDir {
    /// Creates the run directory and its `frames` directory.
    pub fn create(&self) -> Result<()> {
        fs::create_dir_all(self.frames_dir())?;
        Ok(())
    }

    /// Stores a byte-for-byte copy of the input config file.
    pub fn write_config_copy(&self, config_bytes: &[u8]) -> Result<()> {
        write_atomic(&self.config_path(), |w| Ok(w.write_all(config_bytes)?))
    }

    /// Reads the stored config copy.
    pub fn read_config_copy(&self) -> Result<Vec<u8>> {
        Ok(fs::read(self.config_path())?)
    }

    pub fn write_frame_file(&self, index: u64, frame: &Frame) -> Result<()> {
        write_atomic(&self.frame_path(index), |w| write_frame(w, frame))
    }

    pub fn read_frame_file(&self, index: u64) -> Result<Frame> {
        read_frame(&mut open(&self.frame_path(index))?)
    }

    /// Indices of the frame files that exist, sorted. Ignores `.tmp` files.
    pub fn list_frames(&self) -> Result<Vec<u64>> {
        let mut out = Vec::new();
        for entry in fs::read_dir(self.frames_dir())? {
            let name = entry?.file_name();
            if let Some(i) = name.to_str().and_then(parse_frame_file_name) {
                out.push(i);
            }
        }
        out.sort_unstable();
        Ok(out)
    }

    pub fn write_moons_file(&self, file: &MoonsFile) -> Result<()> {
        write_atomic(&self.moons_path(), |w| write_moons(w, file))
    }

    pub fn read_moons_file(&self) -> Result<MoonsFile> {
        read_moons(&mut open(&self.moons_path())?)
    }

    pub fn write_stats_file(&self, rows: &[StepStats]) -> Result<()> {
        write_atomic(&self.stats_path(), |w| write_stats_csv(w, rows))
    }

    pub fn read_stats_file(&self) -> Result<Vec<StepStats>> {
        read_stats_csv(&mut open(&self.stats_path())?)
    }

    pub fn write_chaos_file(&self, rows: &[ChaosRow]) -> Result<()> {
        write_atomic(&self.chaos_path(), |w| write_chaos_csv(w, rows))
    }

    pub fn read_chaos_file(&self) -> Result<Vec<ChaosRow>> {
        read_chaos_csv(&mut open(&self.chaos_path())?)
    }

    pub fn write_drops_file(&self, rows: &[DropRow]) -> Result<()> {
        write_atomic(&self.drops_path(), |w| write_drops_csv(w, rows))
    }

    pub fn read_drops_file(&self) -> Result<Vec<DropRow>> {
        read_drops_csv(&mut open(&self.drops_path())?)
    }

    pub fn write_meta_file(&self, meta: &RunMeta) -> Result<()> {
        write_atomic(&self.meta_path(), |w| write_meta(w, meta))
    }

    pub fn read_meta_file(&self) -> Result<RunMeta> {
        read_meta(&mut open(&self.meta_path())?)
    }
}
