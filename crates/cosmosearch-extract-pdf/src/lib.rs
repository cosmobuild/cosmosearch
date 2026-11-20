//! A PDF extractor plug-in for cosmosearch.
//!
//! Each passage is located by its page. The parser can abort the process or run for minutes on
//! a crafted file, so a host indexing files it does not trust runs this in a separate process.

#![warn(clippy::arithmetic_side_effects)]

mod ligature;
mod markup;
mod pdf;
mod rows;
mod running;

pub use pdf::Pdf;
