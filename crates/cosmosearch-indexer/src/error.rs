use cosmosearch_core::{FileSource, IndexId, IndexKind, Level, ShardId};
use thiserror::Error;

/// Errors from recording file changes.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// The call does not match how files reach this index.
    #[error("this call needs file source {needed:?}, but the index uses {actual:?}")]
    WrongFileSource {
        /// The file source this call works with.
        needed: FileSource,
        /// The file source of the index.
        actual: FileSource,
    },

    /// A job belongs to another index, for example one destroyed and created again.
    #[error("this job is for index `{found}`, not `{expected}`")]
    WrongIndex {
        /// The index being worked on.
        expected: IndexId,
        /// The index the job was planned for.
        found: IndexId,
    },

    /// One segment was asked to hold files planned for different shards.
    #[error("a segment cannot hold files of both shard `{first}` and shard `{second}`")]
    MixedShards {
        /// The first shard seen.
        first: ShardId,
        /// The other shard seen.
        second: ShardId,
    },

    /// The work needs another index plug-in: the one of this kind, with the same embedding
    /// identity, that built the segments it replaces.
    #[error("this work needs the index plug-in `{kind}`")]
    WrongPlugin {
        /// The kind of plug-in needed.
        kind: IndexKind,
    },

    /// The work cannot replace or build a segment of this level.
    #[error("this work cannot be done at level {level}")]
    WrongLevel {
        /// The level.
        level: Level,
    },

    /// A worker thread stopped before finishing.
    #[error("a worker thread stopped before finishing")]
    Worker(#[source] tokio::task::JoinError),

    /// An index plug-in failed.
    #[error(transparent)]
    Index(#[from] cosmosearch_core::IndexError),

    /// Extraction failed for a reason other than the file itself.
    #[error(transparent)]
    Extract(#[from] cosmosearch_chunk::Error),

    /// The index has no manifest that resolves.
    #[error("the index has no readable manifest")]
    NoManifest,

    /// The index has no shard with this id, for example because it was split since.
    #[error("the index has no shard `{shard}`")]
    UnknownShard {
        /// The shard asked for.
        shard: ShardId,
    },

    /// A key falls in no shard, which a valid manifest rules out.
    #[error("no shard covers the key `{key}`")]
    NoShard {
        /// The key.
        key: String,
    },

    /// The commit may or may not have landed. Sending the same changes again is safe.
    #[error("the outcome of the commit is unknown")]
    CommitUnknown,

    /// Other writers kept winning, and the changes were not recorded.
    #[error("the commit lost to other writers {attempts} times")]
    TooManyConflicts {
        /// How many commits were tried.
        attempts: usize,
    },

    /// Storage failed or refused.
    #[error(transparent)]
    Store(#[from] cosmosearch_store::Error),

    /// A stored file table could not be read.
    #[error(transparent)]
    Format(#[from] cosmosearch_format::Error),

    /// A core type is invalid.
    #[error(transparent)]
    Core(#[from] cosmosearch_core::Error),
}

impl Error {
    /// Whether the same call can succeed if made again: storage failed for now, or a commit
    /// did not land. The file involved is not at fault and nothing was recorded against it.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Error::CommitUnknown
                | Error::TooManyConflicts { .. }
                | Error::Store(
                    cosmosearch_store::Error::Backend { .. }
                        | cosmosearch_store::Error::OutcomeUnknown { .. }
                )
        )
    }
}

/// Result type of this crate.
pub type Result<T, E = Error> = std::result::Result<T, E>;
