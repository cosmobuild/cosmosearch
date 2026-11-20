//! Searches and reads a cosmosearch index through the manifest snapshot a reader holds.
//!
//! ```
//! # #[cfg(feature = "test-util")]
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use cosmosearch_chunk::Extractors;
//! use cosmosearch_core::{FileSource, Query, Scope};
//! use cosmosearch_query::{Plugins, Reader, ReaderOptions, SearchRequest};
//! use cosmosearch_store::{DataStore, MemoryStore, MetadataStore};
//!
//! let store = MemoryStore::new();
//! let data = DataStore::new(store.clone(), "s3://example-bucket/docs/".parse()?)?;
//! let metadata = MetadataStore::new(store, "s3://example-bucket/index".parse()?)?;
//! metadata.create(data.location(), FileSource::HostNamed).await?;
//!
//! let plugins = Plugins::new(Extractors::builtin()?);
//! let reader =
//!     Reader::open(data, metadata, FileSource::HostNamed, plugins, ReaderOptions::new()).await?;
//! let request = SearchRequest::new(Query::new("refund")).scope(Scope::prefix("docs/contracts/"));
//! let results = reader.search(&request).await?;
//! assert!(results.hits().is_empty());
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "test-util"))]
//! # fn main() {}
//! ```

#![warn(clippy::arithmetic_side_effects)]

mod cached;
mod coverage;
mod deadline;
mod error;
mod fuse;
mod passages;
mod plugins;
mod read;
mod reader;
mod request;
mod rerank;
mod search;
mod snippet;
mod tables;

pub use coverage::Coverage;
pub use error::{Error, Plugin, Result};
pub use plugins::Plugins;
pub use read::Reading;
pub use reader::{Reader, ReaderOptions};
pub use request::SearchRequest;
pub use rerank::{RerankFailure, Reranking};
pub use search::{SearchHit, SearchResults};
pub use snippet::Snippet;
