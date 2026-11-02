//! Facts for `meta.json` and `docs/benchmarks.md`, found without extra
//! crates. Nothing here fails: a missing tool gives `"unknown"`.
//!
//! These functions start child processes (`git`, `sysctl`, ...), so they only
//! work on native targets. Moved here from `fring-run` for `fring-bench` (B1).

use std::path::Path;
use std::process::Command;

/// Output of a command, trimmed, or `None` if it fails or prints nothing.
fn output(cmd: &str, args: &[&str], dir: Option<&Path>) -> Option<String> {
    let mut c = Command::new(cmd);
    c.args(args);
    if let Some(d) = dir {
        c.current_dir(d);
    }
    let out = c.output().ok().filter(|o| o.status.success())?;
    let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (!text.is_empty()).then_some(text)
}

/// Git commit hash of `dir` (`-dirty` appended if there are uncommitted
/// changes), or `"unknown"` outside a git repository or without git.
pub fn git_hash(dir: &Path) -> String {
    match output("git", &["rev-parse", "HEAD"], Some(dir)) {
        None => "unknown".to_string(),
        Some(hash) => {
            if output("git", &["status", "--porcelain"], Some(dir)).is_some() {
                format!("{hash}-dirty")
            } else {
                hash
            }
        }
    }
}

/// `host | os arch | cpu | n threads | backend`.
pub fn machine(backend: &str) -> String {
    let host = output("hostname", &[], None).unwrap_or_else(|| "unknown".to_string());
    let cpu = output("sysctl", &["-n", "machdep.cpu.brand_string"], None)
        .unwrap_or_else(|| "cpu unknown".to_string());
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    format!(
        "{host} | {} {} | {cpu} | {threads} threads | {backend}",
        std::env::consts::OS,
        std::env::consts::ARCH
    )
}

/// Machine facts for a benchmark table. Every field is `"unknown"` if the
/// tool that gives it is missing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MachineFacts {
    /// CPU brand string, for example "Apple M4".
    pub chip: String,
    /// Physical cores.
    pub physical_cores: String,
    /// Logical cores (threads that the OS offers).
    pub logical_cores: String,
    /// Installed memory, for example "16 GiB".
    pub memory: String,
    /// Operating system and version, for example "macOS 26.1".
    pub os: String,
    /// Output of `rustc --version`.
    pub rustc: String,
    /// Date as `YYYY-MM-DD` (local time).
    pub date: String,
}

fn sysctl(key: &str) -> String {
    output("sysctl", &["-n", key], None).unwrap_or_else(|| "unknown".to_string())
}

/// Formats a byte count as GiB with no decimals when it is whole.
pub fn format_gib(bytes: u64) -> String {
    let gib = bytes as f64 / (1u64 << 30) as f64;
    if (gib - gib.round()).abs() < 0.05 {
        format!("{} GiB", gib.round() as u64)
    } else {
        format!("{gib:.1} GiB")
    }
}

/// Reads the machine facts (macOS `sysctl` and `sw_vers`; other systems give
/// `"unknown"` for most fields).
pub fn machine_facts() -> MachineFacts {
    let memory = sysctl("hw.memsize")
        .parse::<u64>()
        .map_or_else(|_| "unknown".to_string(), format_gib);
    let os = match output("sw_vers", &["-productVersion"], None) {
        Some(v) => format!("macOS {v}"),
        None => std::env::consts::OS.to_string(),
    };
    MachineFacts {
        chip: sysctl("machdep.cpu.brand_string"),
        physical_cores: sysctl("hw.physicalcpu"),
        logical_cores: std::thread::available_parallelism()
            .map_or_else(|_| sysctl("hw.logicalcpu"), |n| n.get().to_string()),
        memory,
        os,
        rustc: output("rustc", &["--version"], None).unwrap_or_else(|| "unknown".to_string()),
        date: output("date", &["+%Y-%m-%d"], None).unwrap_or_else(|| "unknown".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gib_format() {
        assert_eq!(format_gib(16 << 30), "16 GiB");
        assert_eq!(format_gib(3 << 29), "1.5 GiB");
    }

    #[test]
    fn facts_do_not_panic() {
        let f = machine_facts();
        assert!(!f.chip.is_empty() && !f.date.is_empty());
    }
}
