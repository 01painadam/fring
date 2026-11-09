//! `drops.csv`: the particles that were lost during a run with
//! `on_unbound = "freeze"` (ICR-004, `docs/snapshot-format.md` section 8b).

use fring_model::{Error, Result};
use std::io::{Read, Write};

/// Version of `drops.csv` written by this crate.
pub const DROPS_VERSION: u32 = 1;

/// Column names, in order.
pub const DROPS_COLUMNS: [&str; 10] = [
    "id",
    "reason",
    "drop_step",
    "time_s",
    "x_km",
    "y_km",
    "z_km",
    "vx_km_s",
    "vy_km_s",
    "vz_km_s",
];

/// One row of `drops.csv`: one lost particle.
#[derive(Clone, Debug, PartialEq)]
pub struct DropRow {
    /// Particle id.
    pub id: u32,
    /// 1 unbound or invalid orbit; `2 + k` inside moon `k` (moon order of the run).
    pub reason: u8,
    /// 0-based global step in which the loss was found.
    pub drop_step: u64,
    /// `start + drop_step x step_s`, seconds.
    pub time_s: f64,
    /// Frozen inertial position, km (unspecified within one block).
    pub position_km: [f64; 3],
    /// Frozen inertial velocity, km/s.
    pub velocity_km_s: [f64; 3],
}

/// Result of [`read_drops_csv_report`].
#[derive(Clone, Debug, PartialEq)]
pub struct DropsRead {
    /// The complete rows, in file order.
    pub rows: Vec<DropRow>,
    /// True if an unfinished last line (no final newline) was dropped (crash rule R1).
    pub dropped_partial: bool,
}

fn bad(line: usize, msg: impl std::fmt::Display) -> Error {
    Error::Io(format!("drops.csv line {line}: {msg}"))
}

/// Writes the header and all rows (the header also when there are no rows).
/// Floats use the shortest text that parses back exactly.
pub fn write_drops_csv<W: Write>(w: &mut W, rows: &[DropRow]) -> Result<()> {
    let mut buf = String::new();
    buf.push_str(&DROPS_COLUMNS.join(","));
    buf.push('\n');
    for r in rows {
        let [x, y, z] = r.position_km;
        let [vx, vy, vz] = r.velocity_km_s;
        buf.push_str(&format!(
            "{},{},{},{:?},{x:?},{y:?},{z:?},{vx:?},{vy:?},{vz:?}\n",
            r.id, r.reason, r.drop_step, r.time_s
        ));
    }
    w.write_all(buf.as_bytes())?;
    Ok(())
}

/// Reads `drops.csv`. Columns are found by name (all required, any order,
/// unknown columns ignored). Crash rule R1: a last line without a newline is
/// dropped, even if it parses. Every other bad row is an error.
pub fn read_drops_csv_report<R: Read>(r: &mut R) -> Result<DropsRead> {
    let mut text = String::new();
    r.read_to_string(&mut text)
        .map_err(|e| Error::Io(format!("drops.csv: {e}")))?;
    let complete_len = text.rfind('\n').map_or(0, |i| i + 1);
    let dropped_partial = complete_len < text.len();
    text.truncate(complete_len);
    let mut lines = text.lines().enumerate();
    let (_, head) = lines
        .next()
        .ok_or_else(|| Error::Io("drops.csv is empty (no header row)".into()))?;
    let names: Vec<&str> = head.split(',').map(str::trim).collect();
    let c: Vec<usize> = DROPS_COLUMNS
        .iter()
        .map(|n| {
            names
                .iter()
                .position(|m| m == n)
                .ok_or_else(|| bad(1, format!("missing column {n}")))
        })
        .collect::<Result<_>>()?;
    let mut rows = Vec::new();
    for (i, line) in lines {
        let n = i + 1;
        if line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').map(str::trim).collect();
        if f.len() != names.len() {
            return Err(bad(
                n,
                format!("{} fields, header has {}", f.len(), names.len()),
            ));
        }
        let num = |k: usize| {
            f[c[k]].parse::<f64>().map_err(|_| {
                bad(
                    n,
                    format!("{} {:?} is not a number", DROPS_COLUMNS[k], f[c[k]]),
                )
            })
        };
        let int = |k: usize| {
            f[c[k]].parse::<u64>().map_err(|_| {
                bad(
                    n,
                    format!("{} {:?} is not an integer", DROPS_COLUMNS[k], f[c[k]]),
                )
            })
        };
        let small = |k: usize, max: u64| {
            let v = int(k)?;
            if v > max {
                return Err(bad(n, format!("{} {v} is too large", DROPS_COLUMNS[k])));
            }
            Ok(v)
        };
        rows.push(DropRow {
            id: small(0, u64::from(u32::MAX))? as u32,
            reason: small(1, u64::from(u8::MAX))? as u8,
            drop_step: int(2)?,
            time_s: num(3)?,
            position_km: [num(4)?, num(5)?, num(6)?],
            velocity_km_s: [num(7)?, num(8)?, num(9)?],
        });
    }
    Ok(DropsRead {
        rows,
        dropped_partial,
    })
}

/// As [`read_drops_csv_report`], without the partial-line flag.
pub fn read_drops_csv<R: Read>(r: &mut R) -> Result<Vec<DropRow>> {
    Ok(read_drops_csv_report(r)?.rows)
}
