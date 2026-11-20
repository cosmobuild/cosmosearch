//! A local embedding plug-in for cosmosearch, running a small model on the CPU.
//!
//! The default model is `mdbr-leaf-ir`. The host reads its files from wherever it keeps them
//! and hands over the bytes; this crate never downloads anything.
//!
//! ```no_run
//! use cosmosearch_core::Embedder;
//! use cosmosearch_embed_local::{LocalEmbedder, ModelFiles, mdbr_leaf_ir};
//!
//! let read = |name: &str| std::fs::read(format!("models/mdbr-leaf-ir/{name}"));
//! let files = ModelFiles::new(
//!     read("config.json")?,
//!     read("tokenizer.json")?,
//!     read("model.safetensors")?,
//! )
//! .dense(read("2_Dense/model.safetensors")?);
//! let embedder = LocalEmbedder::new(mdbr_leaf_ir()?, files)?;
//! let query = embedder.embed_query("refund policy for annual plans")?;
//! assert_eq!(query.vectors()[0].len(), 768);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![warn(clippy::arithmetic_side_effects)]

mod local;

pub use local::{LocalEmbedder, ModelFiles, mdbr_leaf_ir};
