//! Runs extractors safely and turns their output into chunks.
//!
//! ```
//! use cosmosearch_chunk::extract_chunks;
//! use cosmosearch_core::{
//!     ByteRange, ETag, ExtractFailure, ExtractLimits, Extraction, Extractor, ExtractorProfile,
//!     Locator, Passage,
//! };
//!
//! #[derive(Debug)]
//! struct WholeFile(ExtractorProfile);
//!
//! impl Extractor for WholeFile {
//!     fn profile(&self) -> &ExtractorProfile {
//!         &self.0
//!     }
//!
//!     fn extensions(&self) -> &[&str] {
//!         &["txt"]
//!     }
//!
//!     fn extract(&self, file: &[u8], _: &ExtractLimits) -> Result<Extraction, ExtractFailure> {
//!         let text = std::str::from_utf8(file).map_err(|_| ExtractFailure::Malformed)?;
//!         let end = u64::try_from(file.len()).map_err(|_| ExtractFailure::Malformed)?;
//!         let range = ByteRange::new(0, end).map_err(|_| ExtractFailure::NoText)?;
//!         Ok(Extraction::new(vec![Passage::new(Locator::Bytes(range), vec![], text)]))
//!     }
//! }
//!
//! let extractor = WholeFile(ExtractorProfile::new("whole-file-1")?);
//! let location = "s3://example-bucket/a.txt".parse()?;
//! let etag = ETag::new("9f3a")?;
//! let limits = ExtractLimits::default();
//! let chunks = extract_chunks(&extractor, &location, &etag, b"hello", &limits)?;
//! assert_eq!(chunks.chunks()[0].text(), "hello");
//! assert_eq!(chunks.unread_parts(), 0);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![warn(clippy::arithmetic_side_effects)]

mod error;
mod extract;
mod extractors;
mod markdown;
mod plain;
mod split;

pub use error::{Error, Result};
pub use extract::{FileChunks, extract_chunks};
pub use extractors::Extractors;
pub use markdown::Markdown;
pub use plain::PlainText;
