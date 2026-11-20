use std::error::Error as StdError;
use std::fmt;

use cosmosearch_core::{ETag, FileSource, IndexId, Location, ManifestVersion};
use thiserror::Error;

use crate::ObjectKey;

/// The cause of a failure inside a storage backend.
#[derive(Debug)]
pub struct BackendError(Box<dyn StdError + Send + Sync>);

impl BackendError {
    /// Wraps a backend's own error.
    pub fn new(source: impl Into<Box<dyn StdError + Send + Sync>>) -> Self {
        BackendError(source.into())
    }
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl StdError for BackendError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.0.source()
    }
}

/// Errors from storage and from the index protocols built on it.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// The object does not exist or is not visible yet.
    #[error("object `{key}` not found")]
    NotFound {
        /// The key that was read.
        key: ObjectKey,
    },

    /// A create-only write found the key occupied, or the store reported a conflict.
    #[error("object `{key}` already exists")]
    AlreadyExists {
        /// The key that was written.
        key: ObjectKey,
    },

    /// The object no longer has the expected ETag.
    #[error("object `{key}` changed: it no longer has ETag `{expected}`")]
    Stale {
        /// The key that was read.
        key: ObjectKey,
        /// The ETag the read was pinned to.
        expected: ETag,
    },

    /// A range read returned fewer bytes than asked for.
    #[error("object `{key}` is shorter than the range that was read")]
    ShortRead {
        /// The key that was read.
        key: ObjectKey,
    },

    /// A write may or may not have been applied, and may still be applied later.
    #[error("the outcome of writing `{key}` is unknown")]
    OutcomeUnknown {
        /// The key that was written.
        key: ObjectKey,
        /// Why the outcome is unknown.
        #[source]
        source: BackendError,
    },

    /// The backend definitely refused the operation.
    #[error("the store refused the operation on `{key}`")]
    Refused {
        /// The key involved.
        key: String,
        /// Why it was refused.
        #[source]
        source: BackendError,
    },

    /// A read or list failed, possibly only for now.
    #[error("storage failed for `{key}`")]
    Backend {
        /// The key or prefix involved.
        key: String,
        /// The cause.
        #[source]
        source: BackendError,
    },

    /// The store accepted a second create-only write of the same key.
    #[error("the store at `{location}` ignores create-only writes")]
    CreateOnlyIgnored {
        /// The metadata location.
        location: Location,
    },

    /// The index was created for a different data location.
    #[error("this index covers `{recorded}`, not `{given}`")]
    WrongDataLocation {
        /// The location recorded in the manifest.
        recorded: Location,
        /// The location given when opening.
        given: Location,
    },

    /// The index was created with a different way of receiving files.
    #[error("this index uses file source {recorded:?}, not {given:?}")]
    WrongFileSource {
        /// The mode recorded in the manifest.
        recorded: FileSource,
        /// The mode given when opening.
        given: FileSource,
    },

    /// A manifest from a different index was offered for commit.
    #[error("a manifest for index `{found}` cannot be committed to index `{expected}`")]
    WrongIndex {
        /// The index being written.
        expected: IndexId,
        /// The index the manifest belongs to.
        found: IndexId,
    },

    /// The data location is the metadata location or inside it, so `destroy` would remove
    /// the host's files.
    #[error("the data location `{data}` is inside the metadata location `{metadata}`")]
    DataInsideMetadata {
        /// The data location.
        data: Location,
        /// The metadata location.
        metadata: Location,
    },

    /// A manifest was offered for commit before the version it follows exists.
    #[error("version {version} does not exist yet, so nothing can follow it")]
    ParentMissing {
        /// The missing version.
        version: ManifestVersion,
    },

    /// A retained manifest is missing or does not decode, so what it references is unknown.
    #[error("the manifest of version {version} cannot be read")]
    ManifestUnreadable {
        /// Its version.
        version: ManifestVersion,
    },

    /// A key cannot be used.
    #[error("invalid key `{key}`: {reason}")]
    InvalidKey {
        /// The rejected key.
        key: String,
        /// What is wrong with it.
        reason: &'static str,
    },

    /// A core type is invalid, for example a stored manifest that does not decode.
    #[error(transparent)]
    Core(#[from] cosmosearch_core::Error),
}

/// Result type of this crate.
pub type Result<T, E = Error> = std::result::Result<T, E>;
