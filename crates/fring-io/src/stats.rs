//! `stats.csv`: plain CSV with a header row, one row per statistics output.
//!
//! Layout: `docs/snapshot-format.md` section 8.

use fring_model::{CollisionStats, Error, Result, StepStats};
use std::io::{Read, Write};

/// Version of `stats.csv` written by this crate (recorded in `meta.json`).
pub const STATS_VERSION: u32 = 1;

/// Column names written by this version, in order. The last nine are the
/// collision columns (ICR-009): the fields of [`CollisionStats`] in struct
/// order, with column name = field name.
pub const STATS_COLUMNS: [&str; 13] = [
    "time_s",
    "steps_total",
    "collisions_since_last",
    "max_rel_energy_error",
    "energy_loss_km2_s2",
    "candidates",
    "occupied_cells",
    "occupancy_sum",
    "occupancy_sq_sum",
    "lonely_particles",
    "overfull_cells",
    "gmax_exceeded",
    "deep_collisions",
];

/// Index in [`STATS_COLUMNS`] of the first integer collision counter
/// (`candidates`). The counters fill the columns from here to the end.
const FIRST_COUNTER: usize = 5;

/// The integer collision counters, in column order.
fn counters(c: &CollisionStats) -> [u64; STATS_COLUMNS.len() - FIRST_COUNTER] {
    // No `..` in the pattern: a new field in `CollisionStats` stops the build
    // here, so the columns cannot drift away from the struct.
    let CollisionStats {
        energy_loss_km2_s2: _,
        candidates,
        occupied_cells,
        occupancy_sum,
        occupancy_sq_sum,
        lonely_particles,
        overfull_cells,
        gmax_exceeded,
        deep_collisions,
    } = *c;
    [
        candidates,
        occupied_cells,
        occupancy_sum,
        occupancy_sq_sum,
        lonely_particles,
        overfull_cells,
        gmax_exceeded,
        deep_collisions,
    ]
}

/// As [`counters`], but references to fill in when a row is read.
fn counters_mut(c: &mut CollisionStats) -> [&mut u64; STATS_COLUMNS.len() - FIRST_COUNTER] {
    let CollisionStats {
        energy_loss_km2_s2: _,
        candidates,
        occupied_cells,
        occupancy_sum,
        occupancy_sq_sum,
        lonely_particles,
        overfull_cells,
        gmax_exceeded,
        deep_collisions,
    } = c;
    [
        candidates,
        occupied_cells,
        occupancy_sum,
        occupancy_sq_sum,
        lonely_particles,
        overfull_cells,
        gmax_exceeded,
        deep_collisions,
    ]
}

fn bad(line: usize, msg: impl std::fmt::Display) -> Error {
    Error::Io(format!("stats.csv line {line}: {msg}"))
}

/// Writes the header row only. Use with [`write_stats_row`] to append during a run.
pub fn write_stats_header<W: Write>(w: &mut W) -> Result<()> {
    writeln!(w, "{}", STATS_COLUMNS.join(","))?;
    Ok(())
}

/// Writes one row. Floats use the shortest text that parses back exactly.
///
/// The row is built in memory first and then written with one call, so the
/// newline that completes it (crash rule R1) is always its last byte.
pub fn write_stats_row<W: Write>(w: &mut W, s: &StepStats) -> Result<()> {
    use std::fmt::Write as _;
    let energy = s
        .max_rel_energy_error
        .map_or(String::new(), |v| format!("{v:?}"));
    let mut line = format!(
        "{:?},{},{},{},{:?}",
        s.time_s, s.steps_total, s.collisions_since_last, energy, s.collision.energy_loss_km2_s2
    );
    for v in counters(&s.collision) {
        // Writing to a String cannot fail.
        let _ = write!(line, ",{v}");
    }
    line.push('\n');
    w.write_all(line.as_bytes())?;
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

/// Result of [`read_stats_csv_report`].
#[derive(Clone, Debug, PartialEq)]
pub struct StatsRead {
    /// The complete rows, in file order.
    pub rows: Vec<StepStats>,
    /// True if the file ended in an unfinished line (no final newline), which
    /// was dropped. That is what a crash during a write leaves behind.
    pub dropped_partial: bool,
}

/// Reads `stats.csv`. Columns are found by name; unknown columns are ignored;
/// an empty field is `None` (or the default for non-optional fields that are
/// missing as a whole column). A file from before ICR-009-IO has no collision
/// columns, so its `collision` fields read as zero.
///
/// Crash rule: a row is complete only when its newline has been written, so
/// a last line without a newline is dropped, even if it would parse (a
/// truncated number can still parse and be wrong). Every other bad row is an error.
pub fn read_stats_csv<R: Read>(r: &mut R) -> Result<Vec<StepStats>> {
    Ok(read_stats_csv_report(r)?.rows)
}

/// As [`read_stats_csv`], and also says whether a partial last line was dropped.
pub fn read_stats_csv_report<R: Read>(r: &mut R) -> Result<StatsRead> {
    let mut text = String::new();
    r.read_to_string(&mut text)
        .map_err(|e| Error::Io(format!("stats.csv: {e}")))?;
    let complete_len = text.rfind('\n').map_or(0, |i| i + 1);
    let dropped_partial = complete_len < text.len();
    text.truncate(complete_len);
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
    let c_loss = col("energy_loss_km2_s2");
    let c_counters: Vec<Option<usize>> = STATS_COLUMNS[FIRST_COUNTER..]
        .iter()
        .map(|name| col(name))
        .collect();

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
        if let Some(c) = c_loss.filter(|&c| !f[c].is_empty()) {
            s.collision.energy_loss_km2_s2 = parse_f64(f[c], "energy_loss_km2_s2")?;
        }
        let names_counters = &STATS_COLUMNS[FIRST_COUNTER..];
        for ((slot, name), c) in counters_mut(&mut s.collision)
            .into_iter()
            .zip(names_counters)
            .zip(&c_counters)
        {
            if let Some(c) = c.filter(|&c| !f[c].is_empty()) {
                *slot = parse_u64(f[c], name)?;
            }
        }
        out.push(s);
    }
    Ok(StatsRead {
        rows: out,
        dropped_partial,
    })
}
