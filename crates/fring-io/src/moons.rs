//! `moons.bin`: inertial moon states at each output time.
//!
//! Layout: `docs/snapshot-format.md` section 7.

use crate::bytes::{
    expect_end, read_err, read_f64s, read_u32, read_u64, write_f64s,
};
use fring_model::{BodyState, Error, Result, Vec3};
use std::io::{Read, Write};

/// ASCII magic at the start of `moons.bin`.
pub const MOONS_MAGIC: [u8; 8] = *b"FRINGMNS";
/// Version of `moons.bin` written by this crate.
pub const MOONS_VERSION: u32 = 1;
/// Fixed part of the header, in bytes.
const FIXED_LEN: u32 = 32;
/// Largest header that a reader accepts, in bytes.
const HEADER_LEN_MAX: u32 = 1 << 20;

/// Moon states at one output time (inertial frame, km and km/s).
#[derive(Clone, Debug, PartialEq)]
pub struct MoonRecord {
    pub time_s: f64,
    /// One entry per moon, in the order of [`MoonsFile::names`].
    pub bodies: Vec<BodyState>,
}

/// Contents of `moons.bin`. The number of moons may be 0, 1 or more.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MoonsFile {
    pub names: Vec<String>,
    pub records: Vec<MoonRecord>,
}

fn pad8(n: u64) -> u64 {
    n.div_ceil(8) * 8
}

/// Writes `moons.bin` (version 1).
pub fn write_moons<W: Write>(w: &mut W, file: &MoonsFile) -> Result<()> {
    let n_moons = file.names.len();
    for (i, rec) in file.records.iter().enumerate() {
        if rec.bodies.len() != n_moons {
            return Err(Error::Input(format!(
                "moons: record {i} has {} bodies, expected {n_moons}",
                rec.bodies.len()
            )));
        }
    }
    let mut names = Vec::new();
    for name in &file.names {
        names.extend_from_slice(&(name.len() as u32).to_le_bytes());
        names.extend_from_slice(name.as_bytes());
    }
    let header_len = pad8(u64::from(FIXED_LEN) + names.len() as u64);
    if header_len > u64::from(HEADER_LEN_MAX) {
        return Err(Error::Input("moons: names are too long".into()));
    }
    names.resize((header_len - u64::from(FIXED_LEN)) as usize, 0);
    w.write_all(&MOONS_MAGIC)?;
    w.write_all(&MOONS_VERSION.to_le_bytes())?;
    w.write_all(&(header_len as u32).to_le_bytes())?;
    w.write_all(&(n_moons as u32).to_le_bytes())?;
    w.write_all(&0u32.to_le_bytes())?;
    w.write_all(&(file.records.len() as u64).to_le_bytes())?;
    w.write_all(&names)?;
    let mut row = Vec::with_capacity(1 + 6 * n_moons);
    for rec in &file.records {
        row.clear();
        row.push(rec.time_s);
        for b in &rec.bodies {
            row.extend_from_slice(&[
                b.position_km.x,
                b.position_km.y,
                b.position_km.z,
                b.velocity_km_s.x,
                b.velocity_km_s.y,
                b.velocity_km_s.z,
            ]);
        }
        write_f64s(w, &row)?;
    }
    Ok(())
}

/// Reads `moons.bin`. The reader must end exactly at the end of the file.
pub fn read_moons<R: Read>(r: &mut R) -> Result<MoonsFile> {
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)
        .map_err(|e| read_err(e, "moons magic"))?;
    if magic != MOONS_MAGIC {
        return Err(Error::Io(format!(
            "not a moons file: bad magic {magic:02x?}"
        )));
    }
    let version = read_u32(r, "moons version")?;
    if version != MOONS_VERSION {
        return Err(Error::Io(format!(
            "unsupported moons version {version} (this reader knows version {MOONS_VERSION})"
        )));
    }
    let header_len = read_u32(r, "moons header_len")?;
    if !(FIXED_LEN..=HEADER_LEN_MAX).contains(&header_len) || header_len % 8 != 0 {
        return Err(Error::Io(format!("bad moons header_len {header_len}")));
    }
    let n_moons = read_u32(r, "moons count")? as usize;
    let flags = read_u32(r, "moons flags")?;
    if flags != 0 {
        return Err(Error::Io(format!("unknown moons flag bits {flags:#x}")));
    }
    let n_records = read_u64(r, "moons record count")?;
    let mut rest = vec![0u8; (header_len - FIXED_LEN) as usize];
    r.read_exact(&mut rest)
        .map_err(|e| read_err(e, "moons name table"))?;
    let mut names = Vec::new();
    let mut at = 0usize;
    for _ in 0..n_moons {
        let len_bytes = rest
            .get(at..at + 4)
            .ok_or_else(|| Error::Io("moons name table is too short".into()))?;
        let len = u32::from_le_bytes(len_bytes.try_into().expect("4 bytes")) as usize;
        at += 4;
        let text = rest
            .get(at..at.saturating_add(len))
            .ok_or_else(|| Error::Io("moons name table is too short".into()))?;
        names.push(
            String::from_utf8(text.to_vec())
                .map_err(|_| Error::Io("moon name is not valid UTF-8".into()))?,
        );
        at += len;
    }
    let width = (1 + 6 * n_moons) as u64;
    let mut records = Vec::with_capacity(n_records.min(1024) as usize);
    for _ in 0..n_records {
        let row = read_f64s(r, width, "moons record")?;
        let bodies = row[1..]
            .chunks_exact(6)
            .map(|c| BodyState {
                position_km: Vec3::new(c[0], c[1], c[2]),
                velocity_km_s: Vec3::new(c[3], c[4], c[5]),
            })
            .collect();
        records.push(MoonRecord {
            time_s: row[0],
            bodies,
        });
    }
    expect_end(r, "moons file")?;
    Ok(MoonsFile { names, records })
}
