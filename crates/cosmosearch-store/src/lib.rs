//! Object storage access for cosmosearch.
//!
//! ```
//! # #[cfg(feature = "test-util")]
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use cosmosearch_core::FileSource;
//! use cosmosearch_store::{CreateOutcome, MemoryStore, MetadataStore};
//!
//! let metadata = MetadataStore::new(MemoryStore::new(), "s3://example-bucket/index".parse()?)?;
//! let data = "s3://example-bucket/docs/".parse()?;
//! let outcome = metadata.create(&data, FileSource::Listing).await?;
//! assert!(matches!(outcome, CreateOutcome::Created(_)));
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "test-util"))]
//! # fn main() {}
//! ```

#[cfg(any(feature = "s3", feature = "fs"))]
mod adapter;
mod backend;
mod collection;
mod commit;
mod error;
#[cfg(feature = "test-util")]
mod fault;
mod key;
mod lifecycle;
mod locations;
#[cfg(feature = "test-util")]
mod memory;
mod retention;
mod scope;
mod swap;

#[cfg(any(feature = "s3", feature = "fs"))]
pub use adapter::ObjectStoreBackend;
#[cfg(feature = "s3")]
pub use adapter::{S3Options, S3OptionsBuilder};
pub use backend::{ObjectInfo, ReadStore, WriteStore};
pub use collection::{CollectionReport, CollectionSettings};
pub use error::{BackendError, Error, Result};
#[cfg(feature = "test-util")]
pub use fault::{Fault, FaultStore, Op};
pub use key::ObjectKey;
pub use lifecycle::{CreateOutcome, OpenState, ReadyIndex};
pub use locations::{DataStore, MetadataStore};
#[cfg(feature = "test-util")]
pub use memory::MemoryStore;
pub use retention::Retained;
pub use scope::ScopedStore;
pub use swap::{CommitOutcome, IndexHandle, new_commit_id};
