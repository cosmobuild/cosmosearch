use std::error::Error as StdError;
use std::fmt;

use thiserror::Error;

/// The cause of a failure in the encoding library.
#[derive(Debug)]
pub struct EncodingError(Box<dyn StdError + Send + Sync>);

impl EncodingError {
    pub(crate) fn new(source: impl Into<Box<dyn StdError + Send + Sync>>) -> Self {
        EncodingError(source.into())
    }
}

impl fmt::Display for EncodingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl StdError for EncodingError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.0.source()
    }
}

/// Errors from reading and writing on-disk formats.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// An object was written in a format this version does not understand.
    #[error("unsupported format {format}")]
    UnsupportedFormat {
        /// The format the object declares.
        format: u32,
    },

    /// The stored bytes could not be encoded or decoded.
    #[error("the stored table could not be encoded or decoded")]
    Encoding(#[source] EncodingError),

    /// The stored table decodes but is inconsistent.
    #[error("the stored table is damaged: {reason}")]
    Malformed {
        /// What is wrong with it.
        reason: &'static str,
    },

    /// A value read from a table is invalid.
    #[error(transparent)]
    Core(#[from] cosmosearch_core::Error),
}

/// Result type of this crate.
pub type Result<T, E = Error> = std::result::Result<T, E>;

pub(crate) fn malformed(reason: &'static str) -> Error {
    Error::Malformed { reason }
}
