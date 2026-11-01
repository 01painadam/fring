//! Facts for `meta.json`, found without extra crates. Nothing here fails:
//! a missing tool gives `"unknown"`.

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
