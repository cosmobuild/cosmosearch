use std::collections::BTreeSet;
use std::fmt;

use cosmosearch_core::{ExtractorProfile, IndexError, IndexId, IndexKind, ShardId, SourceRef};
use thiserror::Error;

/// A plug-in, named as it is recorded in the manifest.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[non_exhaustive]
pub enum Plugin {
    /// An extractor, by profile.
    Extractor(ExtractorProfile),
    /// An index, by kind.
    Index(IndexKind),
}

impl fmt::Display for Plugin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Plugin::Extractor(profile) => write!(f, "extractor `{profile}`"),
            Plugin::Index(kind) => write!(f, "index `{kind}`"),
        }
    }
}

fn fusion_setting(kind: Option<&IndexKind>) -> String {
    match kind {
        Some(kind) => format!("the fusion weight of index `{kind}`"),
        None => "the fusion rank constant".to_owned(),
    }
}

fn names(plugins: &BTreeSet<Plugin>) -> String {
    plugins
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

/// Errors from opening, searching and reading an index.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// The metadata location is empty.
    #[error("no index exists at this metadata location")]
    NotIndexed,

    /// The metadata location holds a partial index, which `destroy` clears.
    #[error("the index at this metadata location is incomplete")]
    Incomplete,

    /// The index holds segments built with plug-ins the host did not add.
    #[error("this index needs plug-ins that were not added: {}", names(.0))]
    MissingPlugins(BTreeSet<Plugin>),

    /// The same plug-in was added twice.
    #[error("{0} was added twice")]
    DuplicatePlugin(Plugin),

    /// A fusion setting is not finite and above zero: the rank constant, or the weight of the
    /// index kind named.
    #[error("{} must be finite and above zero, not {value}", fusion_setting(.kind.as_ref()))]
    InvalidFusion {
        /// The index kind whose weight it is, or `None` for the rank constant.
        kind: Option<IndexKind>,
        /// The value given.
        value: f64,
    },

    /// A boost's weight is not finite and at least zero, or a recency boost's half-life is
    /// zero.
    #[error(
        "a boost's weight must be finite and at least zero, and its half-life above zero; the weight is {weight}"
    )]
    InvalidBoost {
        /// The weight given.
        weight: f64,
    },

    /// The request asks for more hits than [`SearchRequest::MAX_LIMIT`].
    ///
    /// [`SearchRequest::MAX_LIMIT`]: crate::SearchRequest::MAX_LIMIT
    #[error(
        "a search may ask for at most {} hits, not {limit}",
        crate::SearchRequest::MAX_LIMIT
    )]
    LimitTooLarge {
        /// The limit asked for.
        limit: usize,
    },

    /// The index this reader opened was removed from its metadata location.
    #[error("the index was removed from its metadata location")]
    IndexRemoved,

    /// The index this reader opened was removed and another created in its place.
    #[error("index `{opened}` was replaced by index `{found}`")]
    IndexReplaced {
        /// The index this reader opened.
        opened: IndexId,
        /// The index the location holds now.
        found: IndexId,
    },

    /// A ref points outside the data location.
    #[error("`{reference}` is outside the data location")]
    OutsideData {
        /// The ref.
        reference: SourceRef,
    },

    /// A search's scope names a prefix that no file of the data location can have, such as
    /// a path relative to the data location instead of a whole object key.
    #[error("the scope prefix `{prefix}` is outside the data location")]
    ScopeOutsideData {
        /// The prefix.
        prefix: String,
    },

    /// A key names no whole object key inside the data location.
    #[error("the key `{key}` is outside the data location")]
    FileOutsideData {
        /// The key.
        key: String,
    },

    /// A key names a file the index does not hold.
    #[error("the file `{key}` is not in the index")]
    FileNotInIndex {
        /// The key.
        key: String,
    },

    /// A ref points at a file the index does not hold.
    #[error("`{reference}` names a file that is not in the index")]
    NotInIndex {
        /// The ref.
        reference: SourceRef,
    },

    /// The file a ref points at has changed since the ref was made.
    #[error("`{reference}` is stale: the file has changed")]
    StaleRef {
        /// The ref.
        reference: SourceRef,
    },

    /// The file is in the index, but no indexed passage of it is at the ref's locator.
    #[error("`{reference}` names no indexed passage")]
    NoPassage {
        /// The ref.
        reference: SourceRef,
    },

    /// A segment the shard needs stayed missing after a refresh, so the shard needs a rebuild.
    #[error("shard `{shard}` cannot be read until it is rebuilt")]
    ShardUnread {
        /// The shard.
        shard: ShardId,
    },

    /// Extracting a file again at read time failed.
    #[error(transparent)]
    Extract(#[from] cosmosearch_chunk::Error),

    /// A worker thread stopped before finishing.
    #[error("a worker thread stopped before finishing")]
    Worker(#[source] tokio::task::JoinError),

    /// An index plug-in failed.
    #[error(transparent)]
    Index(#[from] IndexError),

    /// Storage failed or refused.
    #[error(transparent)]
    Store(#[from] cosmosearch_store::Error),

    /// A stored table could not be read.
    #[error(transparent)]
    Format(#[from] cosmosearch_format::Error),

    /// A core type is invalid.
    #[error(transparent)]
    Core(#[from] cosmosearch_core::Error),
}

/// Result type of this crate.
pub type Result<T, E = Error> = std::result::Result<T, E>;
