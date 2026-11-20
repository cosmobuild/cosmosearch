use thiserror::Error;

/// Errors from reading and writing vector packs and IVF files.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// An object was written in a format this version does not understand.
    #[error("unsupported format {format}")]
    UnsupportedFormat {
        /// The format the object declares.
        format: u32,
    },

    /// The stored bytes are damaged.
    #[error("the stored vectors are damaged: {reason}")]
    Malformed {
        /// What is wrong with them.
        reason: &'static str,
    },

    /// A vector cannot be stored as given.
    #[error("invalid vector: {reason}")]
    InvalidVector {
        /// What is wrong with it.
        reason: &'static str,
    },

    /// An id or range read from the bytes is invalid.
    #[error(transparent)]
    Core(#[from] cosmosearch_core::Error),
}

/// Result type of the vector formats.
pub type Result<T, E = Error> = std::result::Result<T, E>;

pub(crate) fn malformed(reason: &'static str) -> Error {
    Error::Malformed { reason }
}
