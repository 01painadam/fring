//! Command-line arguments, parsed by hand (no external crate).

use std::path::PathBuf;

/// What the user asked for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Command {
    /// Run a simulation.
    Run(Args),
    /// Print the usage text.
    Help,
    /// Print the version.
    Version,
}

/// Arguments of a run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Args {
    /// Path of the config file (`--config`).
    pub config: PathBuf,
    /// Output directory (`--out`); default `runs/<run.name>`.
    pub out: Option<PathBuf>,
    /// No progress lines (`--quiet`).
    pub quiet: bool,
}

/// Usage text.
pub const USAGE: &str = "\
fring-run: run an F ring simulation from one config file

USAGE:
    fring-run --config <file.toml> [--out <dir>] [--quiet]

OPTIONS:
    -c, --config <file>   run config (required)
    -o, --out <dir>       output directory (default: runs/<run.name>);
                          must not exist or must be empty
    -q, --quiet           no progress lines (errors are still printed)
    -h, --help            print this text
    -V, --version         print the version

Exit codes: 0 done; 1 failure during the run; 2 bad usage, config or
parameters, or a feature that is not implemented yet.
";

/// Parses the arguments after the program name. The error is a message for
/// the user (exit code 2).
pub fn parse_args<I>(args: I) -> Result<Command, String>
where
    I: IntoIterator<Item = String>,
{
    let mut it = args.into_iter();
    let mut config = None;
    let mut out = None;
    let mut quiet = false;
    while let Some(arg) = it.next() {
        // Accept `--flag value` and `--flag=value`.
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
            "-q" | "--quiet" => quiet = true,
            "-c" | "--config" => config = Some(PathBuf::from(value("--config")?)),
            "-o" | "--out" => out = Some(PathBuf::from(value("--out")?)),
            _ => return Err(format!("unknown argument {arg:?}")),
        }
    }
    let config = config.ok_or("--config <file> is required")?;
    Ok(Command::Run(Args { config, out, quiet }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(args: &[&str]) -> Result<Command, String> {
        parse_args(args.iter().map(|s| s.to_string()))
    }

    #[test]
    fn parses_flags() {
        let c = p(&["--config", "a.toml", "-o", "x", "-q"]).unwrap();
        assert_eq!(
            c,
            Command::Run(Args {
                config: "a.toml".into(),
                out: Some("x".into()),
                quiet: true
            })
        );
        let c = p(&["--config=a.toml"]).unwrap();
        assert!(matches!(c, Command::Run(a) if a.out.is_none() && !a.quiet));
    }

    #[test]
    fn help_and_version() {
        assert_eq!(p(&["--help"]).unwrap(), Command::Help);
        assert_eq!(p(&["-V"]).unwrap(), Command::Version);
    }

    #[test]
    fn rejects_bad_usage() {
        assert!(p(&[]).unwrap_err().contains("--config"));
        assert!(p(&["--config"]).unwrap_err().contains("needs a value"));
        assert!(p(&["--bogus"]).unwrap_err().contains("unknown"));
        assert!(
            p(&["--config", "a", "extra"])
                .unwrap_err()
                .contains("unknown")
        );
    }
}
