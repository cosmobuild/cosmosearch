//! A local reranking plug-in for cosmosearch, running a small cross-encoder on the CPU.
//!
//! The default model is `ettin-reranker-17m-v1`. The host reads its files from wherever it
//! keeps them and hands over the bytes; this crate never downloads anything. Each query and
//! passage pair runs through the model on its own.
//!
//! ```no_run
//! use cosmosearch_core::Reranker;
//! use cosmosearch_rerank_local::{LocalReranker, RerankerFiles, ettin_reranker_17m_v1};
//!
//! let read = |name: &str| std::fs::read(format!("models/ettin-reranker-17m-v1/{name}"));
//! let files = RerankerFiles::new(
//!     read("config.json")?,
//!     read("tokenizer.json")?,
//!     read("model.safetensors")?,
//!     read("2_Dense/model.safetensors")?,
//!     read("3_LayerNorm/model.safetensors")?,
//!     read("4_Dense/model.safetensors")?,
//! );
//! let reranker = LocalReranker::new(ettin_reranker_17m_v1()?, files)?;
//! let scored = reranker.rerank(
//!     "refund policy for annual plans",
//!     &["Annual plans are refunded within 30 days.", "Parcels ship on Mondays."],
//! )?;
//! assert!(scored.scores()[0] > scored.scores()[1]);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![warn(clippy::arithmetic_side_effects)]

mod encoder;
mod failure;
mod head;
mod local;

pub use local::{LocalReranker, RerankerFiles, ettin_reranker_17m_v1};
