use cosmosearch_core::ExtractFailure;
use thiserror::Error;

/// Errors from extracting one file.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// This file could not be extracted. Other files are not affected.
    #[error(transparent)]
    File(#[from] ExtractFailure),

    /// The file's location or a core type is invalid.
    #[error(transparent)]
    Core(#[from] cosmosearch_core::Error),
}

/// Result type of this crate.
pub type Result<T, E = Error> = std::result::Result<T, E>;
