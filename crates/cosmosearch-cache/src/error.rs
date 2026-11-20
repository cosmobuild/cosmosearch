use std::io;

use thiserror::Error;

/// Errors from building or purging a cache.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// The block size is zero or larger than the cache allows.
    #[error("the block size must be from 1 byte to 64 MiB, not {size} bytes")]
    InvalidBlockSize {
        /// The size asked for.
        size: u64,
    },

    /// The disk tier's directory could not be prepared, or its files removed.
    #[error("the disk tier failed")]
    Disk(#[source] io::Error),
}
