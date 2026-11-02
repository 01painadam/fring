//! Benchmarks: particle-steps per second for CPU and GPU.
//!
//! Owner: A5 (see `AGENTS.md`). The logic is in the library (`lib.rs`).

use fring_bench::cli::{Command, USAGE, parse_args};
use std::process::ExitCode;

fn main() -> ExitCode {
    match parse_args(std::env::args().skip(1)) {
        Ok(Command::Help) => {
            print!("{USAGE}");
            ExitCode::SUCCESS
        }
        Ok(Command::Version) => {
            println!("fring-bench {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        Ok(Command::Run(args)) => match fring_bench::run(&args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("fring-bench: {e}");
                ExitCode::from(1)
            }
        },
        Err(msg) => {
            eprintln!("fring-bench: {msg}\n\n{USAGE}");
            ExitCode::from(2)
        }
    }
}
