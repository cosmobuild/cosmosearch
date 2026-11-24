//! Core types shared by every cosmosearch crate.
//!
//! ```
//! use cosmosearch_core::SourceRef;
//!
//! let source_ref: SourceRef = "s3://example-bucket/a.txt#bytes=0-4@etag=9f3a".parse()?;
//! assert_eq!(source_ref.location().key(), "a.txt");
//! # Ok::<(), cosmosearch_core::Error>(())
//! ```

#![warn(clippy::arithmetic_side_effects)]

mod attributes;
mod budget;
mod cache;
mod chunk;
#[cfg(feature = "test-util")]
mod damage;
mod element;
mod embed;
mod error;
mod etag;
mod extract;
#[cfg(feature = "test-util")]
mod fake;
mod filter;
mod heat;
mod ids;
mod level;
mod location;
mod manifest;
mod media;
mod passages;
mod plugin;
mod refs;
mod rerank;
mod results;
mod rows;
mod scope;
mod search;
mod split;
mod status;
mod time;
mod usage;
mod version;

pub use attributes::{AttributeValue, Attributes, Number};
pub use budget::Budget;
pub use cache::{Admission, CachedFile, Placement, SegmentCache, SharedCache};
pub use chunk::{Chunk, ChunkHash, indexed_text};
#[cfg(feature = "test-util")]
pub use damage::extract_damaged;
pub use element::ElementPath;
pub use embed::{
    EmbedCause, EmbedError, Embedder, EmbeddingIdentity, EmbeddingIdentityBuilder, Embeddings,
    Metric, SharedEmbedder,
};
pub use error::{Error, Result};
pub use etag::ETag;
pub use extract::{
    ExtractFailure, ExtractLimits, ExtractLimitsBuilder, Extraction, LimitKind, MAX_CHUNK_BYTES,
    Passage,
};
#[cfg(feature = "test-util")]
pub use fake::{FakeEmbedder, FakeReranker};
pub use filter::Filter;
pub use heat::{Action, HeatPolicy, HeatSink, ShardCost, SharedHeat, Signal};
pub use ids::{
    CommitId, ExtractorProfile, IndexId, IndexKind, ModelId, SegmentFileName, SegmentId, ShardId,
};
pub use level::Level;
pub use location::{Location, Scheme};
pub use manifest::{
    BuiltBy, FileSource, IndexRecord, KeyRange, Manifest, ManifestObject, ManifestPointer,
    ManifestStub, SegmentEntry, SegmentFile, ShardEntry,
};
pub use media::MediaType;
pub use passages::{PassageCache, SharedPassages};
pub use plugin::{Extractor, Index, SharedExtractor, SharedIndex};
pub use refs::{ByteRange, Locator, RowRange, SourceRef};
pub use rerank::{
    RerankCause, RerankError, RerankScores, Reranker, RerankerIdentity, SharedReranker,
};
pub use results::{CachedResults, ResultCache, SearchIdentity, SearchKey, SharedResults};
pub use rows::{AllowedRows, RowSet};
pub use scope::Scope;
pub use search::{
    Built, BuiltFile, Hit, Hits, IndexCause, IndexError, Query, SegmentFiles, SegmentSource,
};
pub use split::text_ranges;
pub use status::Status;
pub use time::Timestamp;
pub use usage::Usage;
pub use version::ManifestVersion;
