//! A RAM and disk cache of cosmosearch segment files, shared by every engine in a process.
//!
//! A [`Cache`] keeps aligned blocks of segment files, 64 KiB unless set, under a byte budget
//! for memory and another for disk. Memory evicts its least recently used blocks to disk, and
//! disk drops its least recently used blocks. An engine reads its segment files through the
//! cache, which fetches only the blocks it lacks, adjacent ones in one store read, and fetches
//! a block that several searches miss at once only once. [`Cache::purge`] removes an index's
//! blocks from both tiers.
//!
//! The disk tier outlives the process. [`Cache::save`] writes the blocks in memory to it, and
//! a cache built on a directory an earlier process left keeps the blocks found there, each
//! checked against a hash when it is read, so a restart reads from disk and not the store.
//!
//! A [`SearchCache`] keeps whole search results in memory under a byte budget of its own,
//! each for a host-set lifetime, so a repeated search reads nothing. A [`TextCache`] keeps
//! passage text the same way, so a snippet or a reranker does not read a file again.
//!
//! ```
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use cosmosearch_cache::Cache;
//!
//! let dir = std::env::temp_dir().join("cosmosearch-cache-example");
//! let cache = Cache::builder(256 << 20).disk(&dir, 4 << 30).build().await?;
//! assert_eq!(cache.stats().memory_bytes(), 0);
//! cache.purge(&"index-1".parse()?).await?;
//! # Ok(())
//! # }
//! ```

mod admit;
mod block;
mod cache;
mod disk;
mod error;
mod expiring;
mod lock;
mod lru;
mod memory;
mod search_cache;
mod stats;
mod text_cache;

pub use admit::AdmitAll;
pub use cache::{Cache, CacheBuilder};
pub use error::Error;
pub use search_cache::SearchCache;
pub use stats::CacheStats;
pub use text_cache::TextCache;
