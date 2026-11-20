//! A docx extractor plug-in for cosmosearch.
//!
//! Text comes from the body's paragraphs and table cells in document order, and each passage
//! is located by the paragraph it starts at. Archive sizes, entries and XML depth are held to
//! the caller's limits, a DOCTYPE is refused and no entity is expanded.
//!
//! ```
//! use cosmosearch_core::{ExtractLimits, Extractor};
//! use cosmosearch_extract_docx::Docx;
//!
//! let docx = Docx::new()?;
//! let found = docx.extract(b"not a docx", &ExtractLimits::default());
//! assert!(found.is_err());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![warn(clippy::arithmetic_side_effects)]

mod archive;
mod body;
mod docx;
#[cfg(test)]
mod fixture;
mod styles;
mod xml;

pub use docx::Docx;
