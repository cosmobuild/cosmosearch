use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet};

use cosmosearch_core::{Level, ManifestVersion, SegmentEntry, SegmentId, ShardEntry};
use cosmosearch_format::{FileEntry, FileStatus};

use crate::indexer::shard_for;
use crate::{Indexer, Result};

/// A file as the index knows it, with the highest level its current content has reached.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileState {
    entry: FileEntry,
    level: Option<Level>,
}

impl FileState {
    /// The file's stored row: its version, status and the reason when it is not ready.
    pub fn entry(&self) -> &FileEntry {
        &self.entry
    }

    /// The highest level holding the file's current content, `None` unless it is ready. A ready
    /// file is in L1, and higher when a segment of a higher level in its shard holds it at its
    /// current content version.
    pub fn level(&self) -> Option<Level> {
        self.level
    }
}

impl Indexer {
    /// Every file the index knows, in key order within each shard.
    ///
    /// Reads each shard's file table, and the chunk table of each segment above L1 in a shard
    /// with a ready file, once however many shards share it.
    ///
    /// # Errors
    ///
    /// Returns an error if the store fails or a stored table cannot be read.
    pub async fn files(&self) -> Result<Vec<FileState>> {
        let head = self.head().await?;
        let mut holdings: BTreeMap<SegmentId, BTreeSet<(u64, ManifestVersion)>> = BTreeMap::new();
        let mut files = Vec::new();
        for shard in head.shards().values() {
            let table = self.load_table(shard).await?;
            let upper = upper_segments(shard);
            if table.iter().any(is_ready) {
                for segment in &upper {
                    if !holdings.contains_key(segment.id()) {
                        let versions = self.held_by(segment).await?;
                        holdings.insert(segment.id().clone(), versions);
                    }
                }
            }
            for entry in table.iter() {
                let version = (entry.ordinal(), entry.content_version());
                let level = is_ready(entry).then(|| {
                    upper
                        .iter()
                        .find(|segment| {
                            holdings
                                .get(segment.id())
                                .is_some_and(|files| files.contains(&version))
                        })
                        .map_or(Level::L1, |segment| segment.level())
                });
                files.push(FileState {
                    entry: entry.clone(),
                    level,
                });
            }
        }
        Ok(files)
    }

    /// One file as the index knows it: its status, the reason when it is not ready, and the
    /// level it has reached.
    ///
    /// Reads the file table of the file's shard and, for a ready file, the chunk tables of that
    /// shard's segments above L1, highest level first, until one holds the file.
    ///
    /// # Errors
    ///
    /// Returns an error if the store fails or a stored table cannot be read.
    pub async fn file(&self, key: &str) -> Result<Option<FileState>> {
        let head = self.head().await?;
        let (_, shard) = shard_for(&head, key)?;
        let Some(entry) = self.load_table(shard).await?.get(key).cloned() else {
            return Ok(None);
        };
        let mut level = is_ready(&entry).then_some(Level::L1);
        if level.is_some() {
            for segment in upper_segments(shard) {
                let holds = self.load_chunks(segment).await?.is_some_and(|chunks| {
                    !chunks
                        .file_rows(entry.ordinal(), entry.content_version())
                        .is_empty()
                });
                if holds {
                    level = Some(segment.level());
                    break;
                }
            }
        }
        Ok(Some(FileState { entry, level }))
    }

    async fn held_by(&self, segment: &SegmentEntry) -> Result<BTreeSet<(u64, ManifestVersion)>> {
        Ok(self
            .load_chunks(segment)
            .await?
            .map(|chunks| chunks.file_versions().into_iter().collect())
            .unwrap_or_default())
    }
}

fn is_ready(entry: &FileEntry) -> bool {
    entry.status() == FileStatus::Ready
}

/// The shard's segments above L1, highest level first.
fn upper_segments(shard: &ShardEntry) -> Vec<&SegmentEntry> {
    let mut upper: Vec<&SegmentEntry> = shard
        .segments()
        .iter()
        .filter(|segment| segment.level() > Level::L1)
        .collect();
    upper.sort_by_key(|segment| Reverse(segment.level()));
    upper
}
