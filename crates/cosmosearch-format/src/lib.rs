//! On-disk tables of a cosmosearch index: the L0 file table and the chunk records, in Parquet.
//!
//! ```
//! use cosmosearch_core::{ETag, ManifestVersion};
//! use cosmosearch_format::{FileEntry, FileTable};
//!
//! let mut table = FileTable::new();
//! table.insert(FileEntry::pending("docs/a.txt", 5, ETag::new("9f3a")?, ManifestVersion::FIRST));
//! let decoded = FileTable::decode(table.encode()?.into())?;
//! assert_eq!(decoded, table);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod chunks;
mod error;
mod grouped;
mod stored;
mod table;
mod text;

pub use chunks::{ChunkGroup, ChunkIndex, ChunkPlace, ChunkRecord, ChunkTable};
pub use error::{EncodingError, Error, Result};
pub use grouped::TRAILER_BYTES;
pub use table::{FileEntry, FileGroup, FileIndex, FileReason, FileState, FileStatus, FileTable};
pub use text::{TextGroup, TextIndex, TextTable};
