//! A BM25 index plug-in for cosmosearch.
//!
//! Each segment is one file holding a tantivy index and a small hot section. Opening a segment
//! reads the end of the file, and a search then fetches only the terms and postings it needs,
//! and word positions only for an exact phrase.

#![warn(clippy::arithmetic_side_effects)]

mod bm25;
mod bundle;
mod language;

pub use bm25::Bm25;
pub use language::Language;
