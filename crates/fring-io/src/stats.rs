//! `stats.csv`: plain CSV with a header row, one row per statistics output.
//!
//! Layout: `docs/snapshot-format.md` section 8.

use fring_model::{Error, Result, StepStats};
use std::io::{Read, Write};

/// Column names written by this version, in order.
pub const STATS_COLUMNS: [&str; 4] = [
    "time_s",
    "steps_total",
    "collisions_since_last",
    "max_rel_energy_error",
];

fn bad(line: usize, msg: impl std::fmt::Display) -> Error {
    Error::Io(format!("stats.csv line {line}: {msg}"))
}

/// Writes the header row only. Use with [`write_stats_row`] to append during a run.
pub fn write_stats_header<W: Write>(w: &mut W) -> Result<()> {
    writeln!(w, "{}", STATS_COLUMNS.join(","))?;
    Ok(())
}

/// Writes one row. Floats use the shortest text that parses back exactly.
pub fn write_stats_row<W: Write>(w: &mut W, s: &StepStats) -> Result<()> {
    let energy = s
        .max_rel_energy_error
        .map_or(String::new(), |v| format!("{v:?}"));
    writeln!(
        w,
        "{:?},{},{},{}",
        s.time_s, s.steps_total, s.collisions_since_last, energy
    )?;
    Ok(())
}

/// Writes the header and all rows.
pub fn write_stats_csv<W: Write>(w: &mut W, rows: &[StepStats]) -> Result<()> {
    write_stats_header(w)?;
    for s in rows {
        write_stats_row(w, s)?;
    }
    Ok(())
}

/// Reads `stats.csv`. Columns are found by name; unknown columns are ignored;
/// an empty field is `None` (or the default for non-optional fields that are
/// missing as a whole column).
pub fn read_stats_csv<R: Read>(r: &mut R) -> Result<Vec<StepStats>> {
    let mut text = String::new();
    r.read_to_string(&mut text)
        .map_err(|e| Error::Io(format!("stats.csv: {e}")))?;
    let mut lines = text.lines().enumerate();
    let (_, head) = lines
        .next()
        .ok_or_else(|| Error::Io("stats.csv is empty (no header row)".into()))?;
    let names: Vec<&str> = head.split(',').map(str::trim).collect();
    let col = |name: &str| names.iter().position(|n| *n == name);
    let c_time = col("time_s").ok_or_else(|| bad(1, "missing required column time_s"))?;
    let c_steps = col("steps_total");
    let c_coll = col("collisions_since_last");
    let c_energy = col("max_rel_energy_error");

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
        let parse_f64 = |s: &str, what: &str| {
            s.parse::<f64>()
                .map_err(|_| bad(n, format!("{what} {s:?} is not a number")))
        };
        let parse_u64 = |s: &str, what: &str| {
            s.parse::<u64>()
                .map_err(|_| bad(n, format!("{what} {s:?} is not an integer")))
        };
        let mut s = StepStats {
            time_s: parse_f64(f[c_time], "time_s")?,
            ..StepStats::default()
        };
        if let Some(c) = c_steps.filter(|&c| !f[c].is_empty()) {
            s.steps_total = parse_u64(f[c], "steps_total")?;
        }
        if let Some(c) = c_coll.filter(|&c| !f[c].is_empty()) {
            s.collisions_since_last = parse_u64(f[c], "collisions_since_last")?;
        }
        if let Some(c) = c_energy.filter(|&c| !f[c].is_empty()) {
            s.max_rel_energy_error = Some(parse_f64(f[c], "max_rel_energy_error")?);
        }
        out.push(s);
    }
    Ok(out)
}
