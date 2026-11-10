//! Command-line arguments, parsed by hand (no external crate).

use crate::matrix::{Feature, State};
use std::path::PathBuf;

/// What the user asked for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Command {
    /// Run the benchmark.
    Run(Args),
    /// Print the usage text.
    Help,
    /// Print the version.
    Version,
}

/// Arguments of a benchmark run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Args {
    /// Particle counts (`--sizes`); default 10^4 to 10^7.
    pub sizes: Vec<usize>,
    /// Short run: smaller sizes, shorter measurements (`--quick`).
    pub quick: bool,
    /// Skip GPU mode D, which does not meet PLAN section 10 (`--no-mode-d`).
    pub no_mode_d: bool,
    /// CSV path (`--out`); default `runs/bench/bench-<date>.csv`.
    pub out: Option<PathBuf>,
    /// B2 feature sets (`--features`). Empty means the B1 matrix.
    pub features: Vec<Feature>,
    /// Particle states of the B2 rows (`--states`); default core and band.
    pub states: Vec<State>,
    /// Integrator steps per collision step (`--every`); default 1 and 10.
    pub every: Vec<u64>,
    /// Timed repeats per point (`--repeats`); default 3 (B1) or 5 (B2).
    pub repeats: Option<usize>,
    /// Add the cell-size sensitivity point (`--sensitivity`).
    pub sensitivity: bool,
}

impl Args {
    /// True if this is a B2 run (feature rows).
    pub fn is_b2(&self) -> bool {
        !self.features.is_empty()
    }

    /// Timed repeats per point.
    pub fn repeats(&self) -> usize {
        self.repeats.unwrap_or(if self.is_b2() { 5 } else { 3 })
    }
}

/// The default sizes.
pub const DEFAULT_SIZES: [usize; 4] = [10_000, 100_000, 1_000_000, 10_000_000];
/// The sizes of `--quick`.
pub const QUICK_SIZES: [usize; 2] = [10_000, 100_000];

/// Usage text.
pub const USAGE: &str = "\
fring-bench: particle-steps per second of the CPU and GPU backends

USAGE:
    fring-bench [--quick] [--sizes <n,n,...>] [--no-mode-d] [--out <file.csv>]
                [--features <list>] [--states <list>] [--every <list>]
                [--repeats <n>] [--sensitivity]

Runs core particles with both moons at 150 steps per orbit (h = 356.2537 s)
through the Backend trait. Backends: CPU (f64, all threads), GPU D+ (production),
GPU C (reference) and GPU D (fast, does not meet PLAN section 10). The GPU
modes run with the chaos indicator off and on; the CPU runs it on only for
N <= 100000. Each point: warm-up, then best of 3. The default matrix takes
about 5 minutes on an Apple M4.

OPTIONS:
        --quick           N = 10^4 and 10^5 only, 0.3 s per measurement
        --sizes <list>    comma-separated particle counts (default
                          10000,100000,1000000,10000000)
        --no-mode-d       skip GPU mode D
    -o, --out <file>      CSV file (default: runs/bench/bench-<date>.csv)
        --features <list> B2: feature rows instead of the B1 matrix. Names:
                          base, freeze, loss (freeze + radius rule), coll
                          (stochastic collisions), coll+loss, substep (CPU
                          only). Backends: GPU D+ and CPU (CPU up to 10^6)
        --states <list>   B2: core, band, dense (default all three)
        --every <list>    B2: integrator steps per collision step
                          (default 1,10)
        --repeats <n>     timed repeats per point (default 3; 5 with
                          --features); every repeat is a row in the CSV
        --sensitivity     B2: add GPU D+ coll at N = 10^6 with L_x = 0.2 km
    -h, --help            print this text
    -V, --version         print the version

A Markdown table is printed to stdout. Progress goes to stderr.
";

/// Parses the arguments after the program name. The error is a message for
/// the user (exit code 2).
pub fn parse_args<I>(args: I) -> Result<Command, String>
where
    I: IntoIterator<Item = String>,
{
    let mut it = args.into_iter();
    let (mut quick, mut no_mode_d) = (false, false);
    let (mut sizes, mut out) = (None, None);
    let (mut features, mut states, mut every) = (Vec::new(), None, None);
    let (mut repeats, mut sensitivity) = (None, false);
    while let Some(arg) = it.next() {
        let (flag, inline) = match arg.split_once('=') {
            Some((f, v)) if f.starts_with("--") => (f.to_string(), Some(v.to_string())),
            _ => (arg.clone(), None),
        };
        let mut value = |name: &str| -> Result<String, String> {
            inline
                .clone()
                .or_else(|| it.next())
                .ok_or_else(|| format!("{name} needs a value"))
        };
        match flag.as_str() {
            "-h" | "--help" => return Ok(Command::Help),
            "-V" | "--version" => return Ok(Command::Version),
            "--quick" => quick = true,
            "--no-mode-d" => no_mode_d = true,
            "--sizes" => sizes = Some(parse_sizes(&value("--sizes")?)?),
            "--features" => features = parse_features(&value("--features")?)?,
            "--states" => states = Some(parse_states(&value("--states")?)?),
            "--every" => every = Some(parse_every(&value("--every")?)?),
            "--repeats" => repeats = Some(parse_repeats(&value("--repeats")?)?),
            "--sensitivity" => sensitivity = true,
            "-o" | "--out" => out = Some(PathBuf::from(value("--out")?)),
            _ => return Err(format!("unknown argument {arg:?}")),
        }
    }
    let sizes = sizes.unwrap_or_else(|| {
        if quick {
            QUICK_SIZES.to_vec()
        } else {
            DEFAULT_SIZES.to_vec()
        }
    });
    Ok(Command::Run(Args {
        sizes,
        quick,
        no_mode_d,
        out,
        features,
        states: states.unwrap_or_else(|| vec![State::Core, State::Band, State::Dense]),
        every: every.unwrap_or_else(|| vec![1, 10]),
        repeats,
        sensitivity,
    }))
}

fn parse_features(text: &str) -> Result<Vec<Feature>, String> {
    text.split(',')
        .map(|p| Feature::parse(p.trim()).ok_or_else(|| format!("unknown feature {p:?}")))
        .collect()
}

fn parse_states(text: &str) -> Result<Vec<State>, String> {
    text.split(',')
        .map(|p| State::parse(p.trim()).ok_or_else(|| format!("unknown state {p:?}")))
        .collect()
}

fn parse_every(text: &str) -> Result<Vec<u64>, String> {
    text.split(',')
        .map(|p| match p.trim().parse::<u64>() {
            Ok(k) if (1..=1000).contains(&k) => Ok(k),
            _ => Err(format!("bad --every value {p:?} (1 to 1000)")),
        })
        .collect()
}

fn parse_repeats(text: &str) -> Result<usize, String> {
    match text.trim().parse::<usize>() {
        Ok(k) if (1..=50).contains(&k) => Ok(k),
        _ => Err(format!("bad --repeats value {text:?} (1 to 50)")),
    }
}

fn parse_sizes(text: &str) -> Result<Vec<usize>, String> {
    let mut v = Vec::new();
    for part in text.split(',') {
        let n: usize = part
            .trim()
            .replace('_', "")
            .parse()
            .map_err(|_| format!("bad particle count {part:?} in --sizes"))?;
        if n == 0 || n > 100_000_000 {
            return Err(format!("particle count {n} must be in 1 to 10^8"));
        }
        v.push(n);
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(args: &[&str]) -> Result<Command, String> {
        parse_args(args.iter().map(|s| s.to_string()))
    }

    #[test]
    fn defaults() {
        let Command::Run(a) = p(&[]).unwrap() else {
            panic!()
        };
        assert_eq!(a.sizes, DEFAULT_SIZES);
        assert!(!a.quick && !a.no_mode_d && a.out.is_none());
        // No new flags: the B1 run (3 repeats, no feature rows).
        assert!(!a.is_b2() && a.repeats() == 3 && !a.sensitivity);
    }

    #[test]
    fn b2_flags() {
        let Command::Run(a) = p(&[
            "--features",
            "base,coll+loss,substep",
            "--states=band",
            "--every",
            "1,10",
            "--sensitivity",
        ])
        .unwrap() else {
            panic!()
        };
        assert!(a.is_b2() && a.repeats() == 5);
        assert_eq!(a.features.len(), 3);
        assert_eq!(a.states, [State::Band]);
        assert_eq!(a.every, [1, 10]);
        assert!(a.sensitivity);
        let Command::Run(a) = p(&["--features", "base", "--repeats", "1"]).unwrap() else {
            panic!()
        };
        assert_eq!(a.repeats(), 1);
        assert_eq!(a.states, [State::Core, State::Band, State::Dense]);
        assert!(p(&["--features", "x"]).unwrap_err().contains("unknown"));
        assert!(p(&["--every", "0"]).unwrap_err().contains("bad"));
        assert!(p(&["--repeats", "0"]).unwrap_err().contains("bad"));
    }

    #[test]
    fn quick_and_sizes() {
        let Command::Run(a) = p(&["--quick"]).unwrap() else {
            panic!()
        };
        assert_eq!(a.sizes, QUICK_SIZES);
        let Command::Run(a) = p(&["--sizes=1000,2_000", "-o", "x.csv", "--no-mode-d"]).unwrap()
        else {
            panic!()
        };
        assert_eq!(a.sizes, [1000, 2000]);
        assert_eq!(a.out.unwrap().to_str(), Some("x.csv"));
        assert!(a.no_mode_d);
    }

    #[test]
    fn help_version_and_errors() {
        assert_eq!(p(&["-h"]).unwrap(), Command::Help);
        assert_eq!(p(&["-V"]).unwrap(), Command::Version);
        assert!(p(&["--sizes"]).unwrap_err().contains("needs a value"));
        assert!(p(&["--sizes", "a"]).unwrap_err().contains("bad"));
        assert!(p(&["--sizes", "0"]).unwrap_err().contains("must be"));
        assert!(p(&["--bogus"]).unwrap_err().contains("unknown"));
    }
}
