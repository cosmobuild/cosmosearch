use thiserror::Error;

/// Errors from parsing and validating core types.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// A source ref is malformed, for example it has no locator or ETag.
    #[error("invalid source ref `{input}`: {reason}")]
    InvalidRef {
        /// The rejected input.
        input: String,
        /// What is wrong with it.
        reason: &'static str,
    },

    /// A location is not of the form `scheme://authority/key`.
    #[error("invalid location `{input}`: {reason}")]
    InvalidLocation {
        /// The rejected input.
        input: String,
        /// What is wrong with it.
        reason: &'static str,
    },

    /// A byte range does not hold at least one byte.
    #[error("invalid byte range {start}-{end}: start must be below end")]
    InvalidRange {
        /// The requested start.
        start: u64,
        /// The requested end.
        end: u64,
    },

    /// A run of rows ends before it starts.
    #[error("invalid rows {first}-{last}: the first must not come after the last")]
    InvalidRows {
        /// The requested first row.
        first: u64,
        /// The requested last row.
        last: u64,
    },

    /// An ETag is empty or contains a character that is not allowed.
    #[error("invalid ETag `{0}`")]
    InvalidETag(String),

    /// A media type is not a type and a subtype joined by `/`.
    #[error("invalid media type `{0}`")]
    InvalidMediaType(String),

    /// An id is not a valid path segment.
    #[error("invalid {kind} `{value}`: {reason}")]
    InvalidId {
        /// Which kind of id, such as "shard id".
        kind: &'static str,
        /// The rejected value.
        value: String,
        /// What is wrong with it.
        reason: &'static str,
    },

    /// A level name is not one of `L0` to `L2`.
    #[error("invalid level `{0}`: expected L0, L1 or L2")]
    InvalidLevel(String),

    /// A manifest version is zero, too large, or not a valid 12-digit
    /// `v000000000123.json` file name.
    #[error("invalid manifest version `{0}`")]
    InvalidVersion(String),

    /// An extraction limit is zero, which no file could meet.
    #[error("the limit on {limit} must not be zero")]
    InvalidLimit {
        /// The limit that was set to zero.
        limit: crate::LimitKind,
    },

    /// An object was written in a format this version does not understand.
    #[error("unsupported format {format}")]
    UnsupportedFormat {
        /// The format the object declares.
        format: u32,
    },

    /// A manifest, stub, pointer or index record is not valid JSON of the expected shape.
    #[error("manifest data is not valid JSON of the expected shape")]
    ManifestJson(#[source] serde_json::Error),

    /// An embedding or reranker identity is incomplete or asks for an impossible size.
    #[error("invalid identity: {reason}")]
    InvalidIdentity {
        /// What is wrong with it.
        reason: &'static str,
    },

    /// A manifest decodes but is inconsistent.
    #[error("invalid manifest: {reason}")]
    InvalidManifest {
        /// What is wrong with it.
        reason: String,
    },
}

impl Error {
    /// Whether stored metadata is corrupt, as opposed to merely unreadable by this version.
    pub fn is_damaged_data(&self) -> bool {
        matches!(self, Error::ManifestJson(_) | Error::InvalidManifest { .. })
    }
}

/// Result type of this crate.
pub type Result<T, E = Error> = std::result::Result<T, E>;
