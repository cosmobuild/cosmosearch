use std::num::NonZeroU64;
use std::ops::Range;
use std::sync::Arc;

use bytes::Bytes;
use cosmosearch_core::{CachedFile, IndexId};

use crate::lru::OfIndex;

/// One aligned block of a segment file.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct BlockKey {
    pub(crate) file: Arc<CachedFile>,
    pub(crate) block: u64,
}

impl OfIndex for BlockKey {
    fn index(&self) -> &IndexId {
        self.file.index()
    }
}

/// How a file is cut into aligned blocks of one size.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Blocks {
    size: NonZeroU64,
}

impl Blocks {
    pub(crate) fn new(size: NonZeroU64) -> Self {
        Blocks { size }
    }

    /// The size of a block.
    pub(crate) fn size(self) -> u64 {
        self.size.get()
    }

    /// The blocks holding bytes `start..end` of a file.
    pub(crate) fn covering(self, start: u64, end: u64) -> Range<u64> {
        start / self.size..end.div_ceil(self.size.get())
    }

    /// The bytes of blocks `blocks` in a file of `file_size` bytes, the last block cut at its end.
    pub(crate) fn bytes(self, blocks: Range<u64>, file_size: u64) -> Range<u64> {
        let start = blocks.start.saturating_mul(self.size.get()).min(file_size);
        let end = blocks.end.saturating_mul(self.size.get()).min(file_size);
        start..end
    }

    /// The bytes of block `block` in a file of `file_size` bytes.
    pub(crate) fn block(self, block: u64, file_size: u64) -> Range<u64> {
        self.bytes(block..block.saturating_add(1), file_size)
    }
}

/// The length of `bytes` as a byte count.
pub(crate) fn len(bytes: &Bytes) -> u64 {
    u64::try_from(bytes.len()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn covers_partial_blocks_and_cuts_the_last_one() {
        let blocks = Blocks::new(NonZeroU64::new(10).unwrap());
        assert_eq!(blocks.covering(0, 1), 0..1);
        assert_eq!(blocks.covering(9, 11), 0..2);
        assert_eq!(blocks.covering(10, 20), 1..2);
        assert_eq!(blocks.covering(15, 35), 1..4);
        assert_eq!(blocks.bytes(1..4, 32), 10..32);
        assert_eq!(blocks.block(3, 32), 30..32);
        assert_eq!(blocks.block(0, 32), 0..10);
    }
}
