use thiserror::Error;

/// Errors from recording, writing, merging and reading heat.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// The store failed, or the outcome of a write is unknown.
    #[error(transparent)]
    Store(#[from] cosmosearch_store::Error),

    /// A setting is out of range.
    #[error("invalid heat settings: {reason}")]
    InvalidSettings {
        /// What is wrong.
        reason: &'static str,
    },

    /// An amount of heat is not finite or is negative.
    #[error("an amount of heat must be finite and not negative")]
    InvalidAmount,

    /// A stored heat object does not decode.
    #[error("heat object `{key}` is malformed: {reason}")]
    Malformed {
        /// The object's key within the heat scope.
        key: String,
        /// What is wrong with it.
        reason: &'static str,
    },

    /// A stored heat object has a format this version cannot read.
    #[error("heat object `{key}` has format {format}, which this version cannot read")]
    UnsupportedFormat {
        /// The object's key within the heat scope.
        key: String,
        /// The format it has.
        format: u32,
    },
}

/// Result type of this crate.
pub type Result<T, E = Error> = std::result::Result<T, E>;
