//! The error type for all `fring` crates that use the shared interfaces.

use std::fmt;

/// An error from config parsing, input checks or a backend.
///
/// Each variant has a message for a human. Code should not parse the message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// The run config is not valid TOML, has an unknown field, or a value is out of range.
    Config(String),
    /// Input data is not consistent (for example, particle arrays of different lengths).
    Input(String),
    /// A backend failed (for example, the GPU device was lost).
    Backend(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Config(msg) => write!(f, "config error: {msg}"),
            Error::Input(msg) => write!(f, "input error: {msg}"),
            Error::Backend(msg) => write!(f, "backend error: {msg}"),
        }
    }
}

impl std::error::Error for Error {}

/// Short form of `std::result::Result<T, fring_model::Error>`.
pub type Result<T> = std::result::Result<T, Error>;
