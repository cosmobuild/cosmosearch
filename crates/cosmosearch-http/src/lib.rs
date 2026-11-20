//! Embedding and reranking plug-ins for cosmosearch that call HTTP endpoints.
//!
//! The feature `embed` adds [`HttpEmbedder`], for any OpenAI-compatible `POST
//! {base}/embeddings` endpoint; the feature `rerank` adds [`HttpReranker`], for endpoints
//! that take `{model, query, documents}` at `POST {base}/rerank` and answer with `{results:
//! [{index, relevance_score}]}`, the shape most hosted and self-hosted rerank services share.
//!
//! The host names the endpoint, the model and the identity of what it returns; the plug-in
//! checks every vector or score it gets back. Requests that fail for a passing reason are
//! tried again with waits that grow, honouring the endpoint's `Retry-After`. They speak https
//! unless the host allows plain http, follow no redirects, and refuse an answer larger than
//! 64 MiB. Both block, so call them off the async runtime.
//!
#![cfg_attr(feature = "embed", doc = "```no_run")]
#![cfg_attr(not(feature = "embed"), doc = "```ignore")]
//! use cosmosearch_core::{Embedder, EmbeddingIdentity};
//! use cosmosearch_http::{ApiKey, HttpEmbedder};
//!
//! let identity = EmbeddingIdentity::builder("example-embed", "2026-01", 1024, 8192).build()?;
//! let embedder = HttpEmbedder::builder("https://api.example.com/v1", "example-embed", identity)
//!     .api_key(ApiKey::new("example-key"))
//!     .build()?;
//! let query = embedder.embed_query("refund policy for annual plans")?;
//! assert_eq!(query.vectors()[0].len(), 1024);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
#![cfg_attr(feature = "rerank", doc = "```no_run")]
#![cfg_attr(not(feature = "rerank"), doc = "```ignore")]
//! use cosmosearch_core::{Reranker, RerankerIdentity};
//! use cosmosearch_http::{ApiKey, HttpReranker};
//!
//! let identity = RerankerIdentity::new("example-rerank", "2026-01", 4096)?;
//! let reranker = HttpReranker::builder("https://api.example.com/v1", "example-rerank", identity)
//!     .api_key(ApiKey::new("example-key"))
//!     .build()?;
//! let scored = reranker.rerank(
//!     "refund policy for annual plans",
//!     &["Annual plans are refunded within 30 days.", "Parcels ship on Mondays."],
//! )?;
//! assert_eq!(scored.scores().len(), 2);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![warn(clippy::arithmetic_side_effects)]

#[cfg(any(feature = "embed", feature = "rerank"))]
mod connection;
#[cfg(feature = "embed")]
mod embed;
#[cfg(any(feature = "embed", feature = "rerank"))]
mod key;
#[cfg(feature = "rerank")]
mod rerank;

#[cfg(feature = "embed")]
pub use embed::{HttpEmbedder, HttpEmbedderBuilder};
#[cfg(any(feature = "embed", feature = "rerank"))]
pub use key::ApiKey;
#[cfg(feature = "rerank")]
pub use rerank::{HttpReranker, HttpRerankerBuilder};
