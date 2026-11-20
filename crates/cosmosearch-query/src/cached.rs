use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use cosmosearch_core::{
    ByteRange, CachedFile, IndexError, IndexId, Manifest, SegmentFileName, SegmentFiles, SegmentId,
    SegmentSource, ShardEntry, SharedCache,
};

/// The segment files of one manifest, as a cache keys them.
#[derive(Debug, Default)]
pub(crate) struct CachedFiles(BTreeMap<SegmentId, BTreeMap<SegmentFileName, CachedFile>>);

impl CachedFiles {
    pub(crate) fn of(index: &IndexId, manifest: &Manifest) -> Self {
        let mut segments: BTreeMap<SegmentId, BTreeMap<SegmentFileName, CachedFile>> =
            BTreeMap::new();
        for segment in manifest.shards().values().flat_map(ShardEntry::segments) {
            let files = segments.entry(segment.id().clone()).or_default();
            for file in segment.files() {
                let cached = CachedFile::new(index.clone(), segment.id().clone(), file);
                files.insert(file.name().clone(), cached);
            }
        }
        CachedFiles(segments)
    }

    fn get(&self, segment: &SegmentId, name: &SegmentFileName) -> Option<&CachedFile> {
        self.0.get(segment)?.get(name)
    }
}

/// A cache, and the files of the manifest one call reads through it.
#[derive(Debug, Clone)]
pub(crate) struct CacheView {
    pub(crate) cache: SharedCache,
    pub(crate) files: Arc<CachedFiles>,
}

/// Reads segment files through a cache, falling back to the store for a file the manifest
/// does not list, since the cache needs a file's size to cut it into blocks.
#[derive(Debug)]
pub(crate) struct Cached {
    pub(crate) store: SegmentSource,
    pub(crate) view: CacheView,
}

#[async_trait]
impl SegmentFiles for Cached {
    async fn read_range(
        &self,
        segment: &SegmentId,
        name: &SegmentFileName,
        range: ByteRange,
    ) -> Result<Vec<u8>, IndexError> {
        match self.view.files.get(segment, name) {
            Some(file) => {
                self.view
                    .cache
                    .get()
                    .read_range(file, range, &self.store)
                    .await
            }
            None => self.store.read_range(segment, name, range).await,
        }
    }
}
