//! Native CLI for overnight runs.
//!
//! Owner: A5 (see `AGENTS.md`). The logic is in the library (`lib.rs`).

use fring_run::cli::{Command, USAGE, parse_args};
use std::process::ExitCode;

fn main() -> ExitCode {
    match parse_args(std::env::args().skip(1)) {
        Ok(Command::Help) => {
            print!("{USAGE}");
            ExitCode::SUCCESS
        }
        Ok(Command::Version) => {
            println!("fring-run {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        Ok(Command::Run(args)) => match fring_run::run(&args) {
            Ok(_) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("fring-run: {e}");
                ExitCode::from(e.exit_code())
            }
        },
        Err(msg) => {
            eprintln!("fring-run: {msg}\n\n{USAGE}");
            ExitCode::from(2)
        }
    }
}
