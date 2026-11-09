//! CSV and Markdown output.

use crate::matrix::Point;
use crate::measure::{FRAME_STEPS, Timing};
use fring_io::sysinfo::MachineFacts;

/// A point and what happened to it.
#[derive(Clone, Debug, PartialEq)]
pub struct Outcome {
    /// The point.
    pub point: Point,
    /// Timing, or the reason why the point was not run.
    pub result: Result<Timing, String>,
}

/// Facts about the run itself, for the header of the table.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RunEnv {
    /// GPU adapter and graphics backend, for example "Apple M4 (Metal)".
    pub adapter: String,
    /// `max_storage_buffer_binding_size` of the device, bytes.
    pub max_binding_bytes: u64,
    /// `max_buffer_size` of the device, bytes.
    pub max_buffer_bytes: u64,
    /// wgpu version from `Cargo.lock`.
    pub wgpu_version: String,
    /// Git commit (with `-dirty` if there are uncommitted changes).
    pub git: String,
    /// Wall time of the whole matrix, seconds.
    pub wall_s: f64,
}

/// Bytes of the GPU state buffer per particle (6 f64-sized words, ADR 0006).
pub const GPU_STATE_BYTES: u64 = 48;
/// Bytes of GPU memory per particle with the chaos indicator (state, shadow
/// state and accumulators).
pub const GPU_CHAOS_BYTES: u64 = 128;

/// Largest N whose state buffer (the biggest single buffer) fits the limits.
pub fn max_particles(max_binding_bytes: u64, max_buffer_bytes: u64) -> u64 {
    (max_binding_bytes.min(max_buffer_bytes) / GPU_STATE_BYTES).min(u64::from(u32::MAX) / 2)
}

/// CSV header.
pub const CSV_HEADER: &str = "backend,n_particles,chaos,status,steps_per_call,rate_best,rate_min,rate_max,readback_s,rate_with_readback,note";

/// CSV text (header and one line per outcome).
pub fn csv(outcomes: &[Outcome]) -> String {
    let mut s = String::from(CSV_HEADER);
    s.push('\n');
    for o in outcomes {
        let p = &o.point;
        let head = format!("{},{},{}", p.variant.name(), p.n, p.chaos);
        match &o.result {
            Ok(t) => s.push_str(&format!(
                "{head},ok,{},{:.4e},{:.4e},{:.4e},{:.4e},{:.4e},\n",
                t.steps_per_call,
                t.rate_best,
                t.rate_min,
                t.rate_max,
                t.readback_s,
                t.rate_with_readback
            )),
            Err(why) => {
                // Keep the CSV simple: no commas or line breaks in the note.
                let note = why.replace([',', '\n'], ";");
                s.push_str(&format!("{head},not run,,,,,,,{note}\n"));
            }
        }
    }
    s
}

/// CSV header of a B2 run: one row per timed repeat.
pub const CSV_HEADER_B2: &str = "backend,feature,state,n_particles,every_steps,radial_cell_km,repeat,status,steps_per_call,rate,candidates_per_particle,pairs_per_particle,weighted_occupancy,occupied_cells_per_collision_step,overfull_cells,note";

fn cell_km(p: &Point) -> f64 {
    if !p.feature.collisions() {
        0.0
    } else if p.wide_cells {
        crate::WIDE_CELL_KM
    } else {
        crate::CELL_RADIAL_KM
    }
}

/// CSV text of a B2 run (header, then one line per repeat of each outcome).
pub fn csv_b2(outcomes: &[Outcome]) -> String {
    let mut s = String::from(CSV_HEADER_B2);
    s.push('\n');
    for o in outcomes {
        let p = &o.point;
        let head = format!(
            "{},{},{},{},{},{}",
            p.variant.name(),
            p.feature.name(),
            p.state.name(),
            p.n,
            p.every_steps,
            cell_km(p)
        );
        match &o.result {
            Ok(t) => {
                for (k, r) in t.raw_rates.iter().enumerate() {
                    // The statistics of this repeat if they were kept per
                    // repeat, else those of all repeats together.
                    let coll = if t.repeat_coll.len() == t.raw_rates.len() {
                        t.repeat_coll.get(k)
                    } else {
                        t.coll.as_ref()
                    };
                    let (cand, pairs, occ, cells, over) = match coll {
                        Some(c) => (
                            format!("{:.4e}", c.candidates_per_particle(p.n)),
                            format!("{:.4e}", c.collisions_per_particle(p.n)),
                            format!("{:.4}", c.weighted_occupancy()),
                            format!(
                                "{:.1}",
                                c.occupied_cells as f64 / c.coll_steps.max(1) as f64
                            ),
                            c.overfull_cells.to_string(),
                        ),
                        None => Default::default(),
                    };
                    s.push_str(&format!(
                        "{head},{},ok,{},{r:.4e},{cand},{pairs},{occ},{cells},{over},\n",
                        k + 1,
                        t.steps_per_call
                    ));
                }
            }
            Err(why) => {
                let note = why.replace([',', '\n'], ";");
                s.push_str(&format!("{head},,not run,,,,,,,,{note}\n"));
            }
        }
    }
    s
}

/// Markdown table of a B2 run (one line per point; the ratio is to the
/// `base` row of the same backend, state and N).
pub fn markdown_table_b2(outcomes: &[Outcome]) -> String {
    let mut s = String::from(
        "| Backend | Features | State | N | every_steps | L_x (km) | Steps per call | Median particle-steps/s | Min to max | Spread | Ratio to base | Candidates per particle per collision step | Pairs per particle per collision step | Weighted occupancy |\n",
    );
    s.push_str("|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|\n");
    for o in outcomes {
        let p = &o.point;
        let head = format!(
            "| {} | {} | {} | {} | {} | {} |",
            p.variant.label(),
            p.feature.name(),
            p.state.name(),
            p.n,
            if p.feature.collisions() {
                p.every_steps.to_string()
            } else {
                "-".to_string()
            },
            if p.feature.collisions() {
                cell_km(p).to_string()
            } else {
                "-".to_string()
            }
        );
        match &o.result {
            Ok(t) => {
                let base = outcomes.iter().find_map(|b| {
                    let q = &b.point;
                    (q.feature == crate::matrix::Feature::Base
                        && q.variant == p.variant
                        && q.state == p.state
                        && q.n == p.n)
                        .then(|| b.result.as_ref().ok().map(|t| t.rate_median))
                        .flatten()
                });
                let ratio = base.map_or("-".to_string(), |b| format!("{:.3}", t.rate_median / b));
                let min = t.raw_rates.iter().copied().fold(f64::MAX, f64::min);
                let max = t.raw_rates.iter().copied().fold(f64::MIN, f64::max);
                let (cand, pairs, occ) = match &t.coll {
                    Some(c) => (
                        format!("{:.3e}", c.candidates_per_particle(p.n)),
                        format!("{:.3e}", c.collisions_per_particle(p.n)),
                        format!("{:.2}", c.weighted_occupancy()),
                    ),
                    None => ("-".into(), "-".into(), "-".into()),
                };
                s.push_str(&format!(
                    "{head} {} | {} | {} to {} | {:.1} % | {ratio} | {cand} | {pairs} | {occ} |\n",
                    t.steps_per_call,
                    sci(t.rate_median),
                    sci(min),
                    sci(max),
                    100.0 * crate::measure::spread(&t.raw_rates)
                ));
            }
            Err(why) => s.push_str(&format!("{head} not run: {why} | | | | | | | |\n")),
        }
    }
    s
}

/// Three significant figures in scientific notation, for example `9.52e8`.
pub fn sci(x: f64) -> String {
    format!("{x:.2e}").replace("e+", "e")
}

/// Markdown table of the outcomes.
pub fn markdown_table(outcomes: &[Outcome]) -> String {
    let mut s = format!(
        "| Backend | N | Chaos | Steps per call | Particle-steps/s (best of 3) | Min to max | With read-back every {FRAME_STEPS} steps | Read-back time (s) |\n"
    );
    s.push_str("|---|---:|---|---:|---:|---|---:|---:|\n");
    for o in outcomes {
        let p = &o.point;
        let chaos = if p.chaos { "on" } else { "off" };
        let head = format!("| {} | {} | {chaos} |", p.variant.label(), p.n);
        match &o.result {
            Ok(t) => s.push_str(&format!(
                "{head} {} | {} | {} to {} | {} | {:.3} |\n",
                t.steps_per_call,
                sci(t.rate_best),
                sci(t.rate_min),
                sci(t.rate_max),
                sci(t.rate_with_readback),
                t.readback_s
            )),
            Err(why) => s.push_str(&format!("{head} not run: {why} | | | | |\n")),
        }
    }
    s
}

/// Markdown list of machine details.
pub fn markdown_machine(f: &MachineFacts, env: &RunEnv) -> String {
    format!(
        "- Chip: {} ({} physical cores, {} threads)\n\
         - Memory: {}\n\
         - OS: {}\n\
         - GPU adapter: {}\n\
         - Rust: {}\n\
         - wgpu: {}\n\
         - Git commit: {}\n\
         - Date: {}\n\
         - GPU buffer limits: max_storage_buffer_binding_size {} bytes, max_buffer_size {} bytes\n\
         - Run time of the matrix: {:.0} s\n",
        f.chip,
        f.physical_cores,
        f.logical_cores,
        f.memory,
        f.os,
        env.adapter,
        f.rustc,
        env.wgpu_version,
        env.git,
        f.date,
        env.max_binding_bytes,
        env.max_buffer_bytes,
        env.wall_s
    )
}

/// Version of `wgpu` from the text of `Cargo.lock`, or `"unknown"`.
pub fn wgpu_version(cargo_lock: &str) -> String {
    let mut lines = cargo_lock.lines();
    while let Some(l) = lines.next() {
        if l.trim() == "name = \"wgpu\""
            && let Some(v) = lines.next()
            && let Some(v) = v.trim().strip_prefix("version = \"")
        {
            return v.trim_end_matches('"').to_string();
        }
    }
    "unknown".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::Variant;

    fn ok() -> Outcome {
        Outcome {
            point: Point::b1(Variant::GpuDp, 1_000_000, false),
            result: Ok(Timing {
                steps_per_call: 300,
                rate_best: 9.52e8,
                rate_min: 9.4e8,
                rate_max: 9.52e8,
                readback_s: 0.05,
                rate_with_readback: 9.0e8,
                raw_rates: vec![9.4e8, 9.52e8, 9.5e8],
                rate_median: 9.5e8,
                coll: None,
                repeat_coll: Vec::new(),
            }),
        }
    }

    fn failed() -> Outcome {
        Outcome {
            point: Point::b1(Variant::Cpu, 10, true),
            result: Err("too big, sorry".to_string()),
        }
    }

    #[test]
    fn csv_lines() {
        let s = csv(&[ok(), failed()]);
        let lines: Vec<&str> = s.lines().collect();
        assert_eq!(lines[0], CSV_HEADER);
        assert!(lines[1].starts_with("gpu_dp,1000000,false,ok,300,9.5200e8,"));
        assert_eq!(lines[2], "cpu_f64,10,true,not run,,,,,,,too big; sorry");
        // Same number of fields in every line.
        let cols = |l: &str| l.split(',').count();
        assert!(lines.iter().all(|l| cols(l) == cols(lines[0])));
    }

    #[test]
    fn b2_csv_and_table() {
        use crate::matrix::Feature;
        use crate::measure::CollTotals;
        let mut with = ok();
        with.point.feature = Feature::Coll;
        with.point.every_steps = 10;
        if let Ok(t) = &mut with.result {
            t.raw_rates = vec![4.0e8, 4.2e8];
            t.rate_median = 4.1e8;
            t.coll = Some(CollTotals {
                coll_steps: 2,
                candidates: 100_000,
                collisions: 20_000,
                occupied_cells: 10,
                occupancy_sum: 10,
                occupancy_sq_sum: 30,
                ..CollTotals::default()
            });
        }
        let outs = [ok(), with, failed()];
        let s = csv_b2(&outs);
        let lines: Vec<&str> = s.lines().collect();
        assert_eq!(lines[0], CSV_HEADER_B2);
        // 3 + 2 repeats and one "not run" row, all with the same field count.
        assert_eq!(lines.len(), 1 + 3 + 2 + 1);
        let cols = |l: &str| l.split(',').count();
        assert!(lines.iter().all(|l| cols(l) == cols(lines[0])), "{s}");
        assert!(lines[4].starts_with(
            "gpu_dp,coll,core,1000000,10,0.05,1,ok,300,4.0000e8,5.0000e-2,1.0000e-2,3.0000,"
        ));
        assert!(lines[6].contains("not run"));
        let t = markdown_table_b2(&outs);
        // Ratio to the base row: 4.1e8 / 9.5e8 = 0.432.
        assert!(t.contains("| 0.432 |"), "{t}");
        assert!(t.contains("5.000e-2"), "{t}");
    }

    #[test]
    fn markdown_lines() {
        let s = markdown_table(&[ok(), failed()]);
        assert!(s.contains("| GPU D+ (production) | 1000000 | off | 300 | 9.52e8 | 9.40e8 to 9.52e8 | 9.00e8 | 0.050 |"));
        assert!(s.contains("not run: too big, sorry"));
    }

    #[test]
    fn limits_and_lock() {
        // 4 GiB binding limit: 89,478,485 particles at 48 bytes.
        assert_eq!(max_particles(1 << 32, 1 << 33), 89_478_485);
        assert_eq!(sci(951_234_567.0), "9.51e8");
        let lock = "[[package]]\nname = \"wgpu\"\nversion = \"30.0.1\"\n";
        assert_eq!(wgpu_version(lock), "30.0.1");
        assert_eq!(wgpu_version(""), "unknown");
    }
}
