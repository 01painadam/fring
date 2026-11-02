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
            point: Point {
                variant: Variant::GpuDp,
                n: 1_000_000,
                chaos: false,
            },
            result: Ok(Timing {
                steps_per_call: 300,
                rate_best: 9.52e8,
                rate_min: 9.4e8,
                rate_max: 9.52e8,
                readback_s: 0.05,
                rate_with_readback: 9.0e8,
            }),
        }
    }

    fn failed() -> Outcome {
        Outcome {
            point: Point {
                variant: Variant::Cpu,
                n: 10,
                chaos: true,
            },
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
