//! Vectors for cosmosearch: a reuse cache of embeddings, keyed by embedding identity and
//! chunk hash, so a chunk seen before is not embedded again; the spherical k-means that
//! partitions a segment's vectors into clusters; and [`Ivf`], the L2 index plug-in that
//! stores one IVF file per segment and searches it in two rounds of parallel range reads.
//!
//! ```
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use std::time::SystemTime;
//!
//! use cosmosearch_core::{ChunkHash, EmbeddingIdentity, FileSource};
//! use cosmosearch_store::{CreateOutcome, MemoryStore, MetadataStore};
//! use cosmosearch_vector::VectorStore;
//!
//! let metadata = MetadataStore::new(MemoryStore::new(), "s3://example-bucket/index".parse()?)?;
//! let CreateOutcome::Created(index) =
//!     metadata.create(&"s3://example-bucket/docs/".parse()?, FileSource::Listing).await?
//! else {
//!     return Ok(());
//! };
//! let store = VectorStore::new(&index)?;
//! let identity = EmbeddingIdentity::builder("example-model", "r1", 3, 512).build()?;
//! let hash = ChunkHash::of("refund policy applies");
//!
//! store.put(&identity, &[(hash, &[0.5, -0.25, 1.0])], SystemTime::now()).await?;
//! let lookup = store.get(&identity, &[hash, ChunkHash::of("new text")]).await;
//! assert_eq!(lookup.found().len(), 1);
//! assert_eq!(lookup.missing(), [ChunkHash::of("new text")]);
//! # Ok(())
//! # }
//! ```

#![warn(clippy::arithmetic_side_effects)]

mod centroids;
mod error;
/// The file formats of the vector index: binary and int8 vectors, IVF files and vector packs.
pub mod format;
mod ivf;
mod kmeans;
mod lookup;
mod name;
mod probe;
mod random;
#[cfg(test)]
mod recall;
mod store;
mod sweep;
#[cfg(test)]
mod testdata;

pub use centroids::Centroids;
pub use error::{Error, Result};
pub use ivf::Ivf;
pub use kmeans::KMeans;
pub use lookup::Lookup;
pub use store::VectorStore;
pub use sweep::SweepReport;
