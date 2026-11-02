//! `chaos.csv`: the chaos indicator of every particle, written at the end of
//! a run that has `[diagnostics.chaos]` (ICR-003, `docs/snapshot-format.md`
//! section 9).

use fring_model::{Error, Result};
use std::io::{Read, Write};

/// Version of `chaos.csv` written by this crate (recorded in `meta.json`).
pub const CHAOS_VERSION: u32 = 1;

/// Column names, in order.
pub const CHAOS_COLUMNS: [&str; 7] = [
    "id",
    "a0_km",
    "lambda_rel_prometheus_deg",
    "ftle_per_day",
    "log_growth",
    "megno_mean",
    "flags",
];

/// One row of `chaos.csv`.
#[derive(Clone, Debug, PartialEq)]
pub struct ChaosRow {
    /// Particle id.
    pub id: u32,
    /// Keplerian semi-major axis of the initial state (point mass GM), km.
    pub a0_km: f64,
    /// Initial inertial longitude of the particle minus that of Prometheus, in
    /// degrees in [-180, 180). NaN without Prometheus.
    pub lambda_rel_prometheus_deg: f64,
    /// Finite-time Lyapunov exponent in day⁻¹.
    pub ftle_per_day: f64,
    /// L, the sum of ln(d / d0) (dimensionless).
    pub log_growth: f64,
    /// Mean MEGNO.
    pub megno_mean: f64,
    /// Bit 0: an interval was skipped (zero or non-finite separation).
    pub flags: u32,
}

fn bad(line: usize, msg: impl std::fmt::Display) -> Error {
    Error::Io(format!("chaos.csv line {line}: {msg}"))
}

/// Writes the header and all rows. Floats use the shortest text that parses
/// back exactly.
pub fn write_chaos_csv<W: Write>(w: &mut W, rows: &[ChaosRow]) -> Result<()> {
    writeln!(w, "{}", CHAOS_COLUMNS.join(","))?;
    for r in rows {
        writeln!(
            w,
            "{},{:?},{:?},{:?},{:?},{:?},{}",
            r.id,
            r.a0_km,
            r.lambda_rel_prometheus_deg,
            r.ftle_per_day,
            r.log_growth,
            r.megno_mean,
            r.flags
        )?;
    }
    Ok(())
}

/// Reads `chaos.csv` (columns by name, all required).
pub fn read_chaos_csv<R: Read>(r: &mut R) -> Result<Vec<ChaosRow>> {
    let mut text = String::new();
    r.read_to_string(&mut text)
        .map_err(|e| Error::Io(format!("chaos.csv: {e}")))?;
    let mut lines = text.lines().enumerate();
    let (_, head) = lines
        .next()
        .ok_or_else(|| Error::Io("chaos.csv is empty (no header row)".into()))?;
    let names: Vec<&str> = head.split(',').map(str::trim).collect();
    let col = |name: &str| {
        names
            .iter()
            .position(|n| *n == name)
            .ok_or_else(|| bad(1, format!("missing column {name}")))
    };
    let c: Vec<usize> = CHAOS_COLUMNS
        .iter()
        .map(|n| col(n))
        .collect::<Result<_>>()?;
    let mut out = Vec::new();
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
                    format!("{} {:?} is not a number", CHAOS_COLUMNS[k], f[c[k]]),
                )
            })
        };
        let int = |k: usize| {
            f[c[k]].parse::<u32>().map_err(|_| {
                bad(
                    n,
                    format!("{} {:?} is not an integer", CHAOS_COLUMNS[k], f[c[k]]),
                )
            })
        };
        out.push(ChaosRow {
            id: int(0)?,
            a0_km: num(1)?,
            lambda_rel_prometheus_deg: num(2)?,
            ftle_per_day: num(3)?,
            log_growth: num(4)?,
            megno_mean: num(5)?,
            flags: int(6)?,
        });
    }
    Ok(out)
}
