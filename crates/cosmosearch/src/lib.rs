//! Hybrid keyword and semantic search over large collections of files in S3-compatible
//! storage, built for RAG, with no database dependency.
//!
//! Layered indexes and usage-based promotion keep search fast without the cost of indexing
//! everything up front. cosmosearch works with S3, any S3-compatible store, or a local folder.
//! It keeps a small index next to your data and reads it straight from the store when you
//! search, and every result points back to the exact place in the original file it came from.
//!
//! An [`Engine`] searches one index. You build it from the plug-ins you want (an index kind,
//! extractors, a reranker, caches); text and Markdown extraction are built in, and anything you
//! leave out costs you no dependencies. The `index` module (feature `indexer`) creates the
//! index and keeps it up to date.
//!
//! The types a search needs are at the top level. Each area has its own module: [`store`]
//! for storage and the index lifecycle, [`core`] for shared types and the plug-in contracts,
//! and, behind their features, `index`, `lexical`, `vector`, `extract`, `embed_local`,
//! `rerank_local`, `http`, `heat` and `cache`.
//!
//! The [README](https://github.com/cosmobuild/cosmosearch#readme) has a complete example
//! against S3 and a table of features. This smaller one uses the in-memory store from the
//! `test-util` feature and searches an index that has no files yet:
//!
//! ```
//! # #[cfg(feature = "test-util")]
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use cosmosearch::store::MemoryStore;
//! use cosmosearch::{DataStore, Engine, FileSource, MetadataStore, Query, Scope, SearchRequest};
//!
//! let store = MemoryStore::new();
//! let data = DataStore::new(store.clone(), "s3://example-bucket/docs/".parse()?)?;
//! let metadata = MetadataStore::new(store, "s3://example-bucket/index".parse()?)?;
//! metadata.create(data.location(), FileSource::Listing).await?;
//!
//! let engine = Engine::builder(data, metadata).open().await?;
//! assert_eq!(engine.manifest().version().get(), 1);
//! let request = SearchRequest::new(Query::new("refund")).scope(Scope::prefix("docs/contracts/"));
//! let results = engine.search(&request).await?;
//! assert!(results.hits().is_empty());
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "test-util"))]
//! # fn main() {}
//! ```

mod engine;
#[cfg(feature = "indexer")]
mod indexing;

pub use cosmosearch_core::{
    Budget, ETag, FileSource, Filter, IndexId, IndexOptions, Location, Locator, ManifestVersion,
    Query, Scope, ShardId, SourceRef, Status,
};
pub use cosmosearch_format::{FileEntry, FileReason, FileStatus};
pub use cosmosearch_query::{
    Boost, Coverage, DEFAULT_TABLE_BUDGET, Error, Plugin, Reading, RerankFailure, Reranking,
    Result, SearchHit, SearchRequest, SearchResults, Snippet, TableCache,
};
pub use cosmosearch_store::{CreateOutcome, DataStore, MetadataStore};
pub use engine::{Engine, EngineBuilder};

/// Every type of the plug-in contracts and the index's records, from `cosmosearch-core`:
/// what a plug-in implements and what the other modules' calls take and return.
pub mod core {
    pub use cosmosearch_core::*;
}

/// Data and metadata locations, the storage backends, index lifecycle and garbage
/// collection, from `cosmosearch-store`.
pub mod store {
    pub use cosmosearch_store::*;
}

/// Indexing jobs and the file table, from `cosmosearch-indexer` and `cosmosearch-format`.
#[cfg(feature = "indexer")]
pub mod index {
    pub use cosmosearch_chunk::Extractors;
    pub use cosmosearch_format::{FileEntry, FileReason, FileStatus};
    pub use cosmosearch_indexer::*;

    pub use crate::indexing::{Indexed, index_pending};
}

/// BM25 over tantivy, from `cosmosearch-lexical`.
#[cfg(feature = "lexical")]
pub mod lexical {
    pub use cosmosearch_lexical::*;
}

/// The vector index and the vector store, from `cosmosearch-vector`.
#[cfg(feature = "vector")]
pub mod vector {
    pub use cosmosearch_vector::*;
}

/// Extractors for the formats beyond text and Markdown, one feature each.
pub mod extract {
    #[cfg(feature = "csv")]
    pub use cosmosearch_extract_csv::Csv;
    #[cfg(feature = "docx")]
    pub use cosmosearch_extract_docx::Docx;
    #[cfg(feature = "html")]
    pub use cosmosearch_extract_html::Html;
    #[cfg(feature = "json")]
    pub use cosmosearch_extract_json::Json;
    #[cfg(feature = "pdf")]
    pub use cosmosearch_extract_pdf::Pdf;
}

/// The local embedder, from `cosmosearch-embed-local`.
#[cfg(feature = "embed-local")]
pub mod embed_local {
    pub use cosmosearch_embed_local::*;
}

/// The local reranker, from `cosmosearch-rerank-local`.
#[cfg(feature = "rerank-local")]
pub mod rerank_local {
    pub use cosmosearch_rerank_local::*;
}

/// The embedder and reranker that call HTTP endpoints, from `cosmosearch-http`.
#[cfg(any(feature = "embed-http", feature = "rerank-http"))]
pub mod http {
    pub use cosmosearch_http::*;
}

/// Heat: counters, writers, merging, the promotion policy and the spend plan, from
/// `cosmosearch-heat`.
#[cfg(feature = "heat")]
pub mod heat {
    pub use cosmosearch_heat::*;
}

/// The cache of segment files in memory and on disk, the result cache and the passage
/// cache, from `cosmosearch-cache`.
#[cfg(feature = "cache")]
pub mod cache {
    pub use cosmosearch_cache::*;
}
