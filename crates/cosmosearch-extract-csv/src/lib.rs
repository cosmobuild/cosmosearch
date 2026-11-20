//! A CSV extractor plug-in for cosmosearch.
//!
//! The first record is the header, and each value is labelled with its column. Rows are
//! grouped into passages located by the records they hold.
//!
//! ```
//! use cosmosearch_core::{ExtractLimits, Extractor};
//! use cosmosearch_extract_csv::Csv;
//!
//! let csv = Csv::new()?;
//! let found = csv.extract(b"name,team\nAda,core\n", &ExtractLimits::default())?.into_passages();
//! assert_eq!(found[0].locator().to_string(), "rows=1-1");
//! assert_eq!(found[0].text(), "name: Ada\nteam: core");
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![warn(clippy::arithmetic_side_effects)]

mod csv;

pub use crate::csv::Csv;
