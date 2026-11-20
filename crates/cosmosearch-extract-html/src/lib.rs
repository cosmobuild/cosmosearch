//! An HTML extractor plug-in for cosmosearch.
//!
//! Text comes out with tags removed and entities decoded. Each passage's byte range covers its
//! source markup, and extracting that range gives the same text back.
//!
//! ```
//! use cosmosearch_core::{ExtractLimits, Extractor};
//! use cosmosearch_extract_html::Html;
//!
//! let html = Html::new()?;
//! let file = b"<h1>Refunds</h1><p>Within 30 days.</p>";
//! let found = html.extract(file, &ExtractLimits::default())?.into_passages();
//! assert_eq!(found[0].text(), "Refunds\nWithin 30 days.");
//! assert_eq!(found[0].heading_path(), ["Refunds"]);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![warn(clippy::arithmetic_side_effects)]

mod html;

pub use html::Html;
