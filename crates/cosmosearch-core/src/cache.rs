use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;

use crate::{
    ByteRange, IndexError, IndexId, SegmentFile, SegmentFileName, SegmentId, SegmentSource,
};

/// One file of one segment of one index, as a cache keys its bytes.
///
/// Segment files are immutable, so bytes cached under this key never go stale.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CachedFile {
    index: IndexId,
    segment: SegmentId,
    name: SegmentFileName,
    size: u64,
}

impl CachedFile {
    /// `file` of `segment` in index `index`.
    pub fn new(index: IndexId, segment: SegmentId, file: &SegmentFile) -> Self {
        CachedFile {
            index,
            segment,
            name: file.name().clone(),
            size: file.size(),
        }
    }

    /// The index.
    pub fn index(&self) -> &IndexId {
        &self.index
    }

    /// The segment.
    pub fn segment(&self) -> &SegmentId {
        &self.segment
    }

    /// The file's name within its segment.
    pub fn name(&self) -> &SegmentFileName {
        &self.name
    }

    /// The file's size in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }
}

/// Where a cache keeps a block it has read from the store.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Placement {
    /// In memory, moving to disk when memory evicts it.
    Memory,
    /// On disk only.
    Disk,
    /// Nowhere: the block is used once and dropped.
    Skip,
}

/// Decides whether a cache keeps a block it has read, and in which tier.
pub trait Admission: Send + Sync + fmt::Debug {
    /// Where to keep block `block` of `file`.
    fn place(&self, file: &CachedFile, block: u64) -> Placement;
}

/// A cache of segment file bytes, shared by the readers of any number of indexes.
#[async_trait]
pub trait SegmentCache: Send + Sync + fmt::Debug {
    /// Reads `range` of `file`, from the cache where it holds the bytes and from `store` for
    /// the rest.
    ///
    /// # Errors
    ///
    /// Returns the [`IndexError`] of a read from `store` that fails.
    async fn read_range(
        &self,
        file: &CachedFile,
        range: ByteRange,
        store: &SegmentSource,
    ) -> Result<Vec<u8>, IndexError>;
}

/// A shared handle on a segment cache.
#[derive(Debug, Clone)]
pub struct SharedCache(Arc<dyn SegmentCache>);

impl SharedCache {
    /// Shares `cache`.
    pub fn new(cache: impl SegmentCache + 'static) -> Self {
        SharedCache(Arc::new(cache))
    }

    /// The cache.
    pub fn get(&self) -> &dyn SegmentCache {
        self.0.as_ref()
    }
}
