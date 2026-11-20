//! A JSON and JSON Lines extractor plug-in for cosmosearch.
//!
//! Values are written as lines labelled with their path, grouped into passages located by
//! the number of their first value in document order.
//!
//! ```
//! use cosmosearch_core::{ExtractLimits, Extractor};
//! use cosmosearch_extract_json::Json;
//!
//! let json = Json::new()?;
//! let file = br#"{"name": "Ada", "teams": ["core"]}"#;
//! let found = json.extract(file, &ExtractLimits::default())?.into_passages();
//! assert_eq!(found[0].locator().to_string(), "value=1");
//! assert_eq!(found[0].text(), "name: Ada\nteams/0: core");
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![warn(clippy::arithmetic_side_effects)]

mod json;
mod node;

pub use json::Json;
