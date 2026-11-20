use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use async_trait::async_trait;
use bytes::Bytes;
use cosmosearch_core::{
    ByteRange, Filter, IndexError, Level, Manifest, RowSet, Scope, SegmentEntry, SegmentFile,
    SegmentFileName, SegmentFiles, SegmentId, SegmentSource, ShardEntry, Usage,
};
use cosmosearch_format::{ChunkTable, FileTable};
use cosmosearch_store::IndexHandle;
use tokio::sync::watch;

use crate::Result;
use crate::cached::{CacheView, Cached};
use crate::coverage::Counts;

const CHUNKS_PREFIX: &str = "chunks-";

/// Counts the object reads of one call and the bytes they returned.
#[derive(Debug, Default)]
pub(crate) struct Counter {
    reads: AtomicU64,
    bytes: AtomicU64,
}

impl Counter {
    pub(crate) fn record(&self, bytes: usize) {
        self.reads.fetch_add(1, Ordering::Relaxed);
        let bytes = u64::try_from(bytes).unwrap_or(u64::MAX);
        self.bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    pub(crate) fn usage(&self) -> Usage {
        Usage::of_reads(
            self.reads.load(Ordering::Relaxed),
            self.bytes.load(Ordering::Relaxed),
        )
    }
}

#[derive(Debug)]
struct Counted {
    index: IndexHandle,
    counter: Arc<Counter>,
}

#[async_trait]
impl SegmentFiles for Counted {
    async fn read_range(
        &self,
        segment: &SegmentId,
        name: &SegmentFileName,
        range: ByteRange,
    ) -> Result<Vec<u8>, IndexError> {
        let read = self.index.read_range(segment, name, range).await;
        self.counter.record(read.as_ref().map_or(0, Vec::len));
        read
    }
}

/// A segment's chunk table and the file table of a shard that holds it.
type Pair = (SegmentId, SegmentId);

#[derive(Debug, Default)]
struct Cache {
    files: Slots<SegmentId, FileTable>,
    chunks: Slots<SegmentId, ChunkTable>,
    live: Slots<Pair, RowSet>,
    counts: Slots<SegmentId, Counts>,
}

/// The values of one kind, and those one call is making for all.
#[derive(Debug)]
struct Slots<K, T> {
    ready: BTreeMap<K, Arc<T>>,
    loading: BTreeMap<K, watch::Receiver<Option<Arc<T>>>>,
}

impl<K, T> Default for Slots<K, T> {
    fn default() -> Self {
        Slots {
            ready: BTreeMap::new(),
            loading: BTreeMap::new(),
        }
    }
}

type Pick<K, T> = fn(&mut Cache) -> &mut Slots<K, T>;

enum Turn<'a, K: Ord, T> {
    Ready(Arc<T>),
    Wait(watch::Receiver<Option<Arc<T>>>),
    Load(Flight<'a, K, T>),
}

/// A value this call makes for every call that wants it; dropping it lets them retry.
struct Flight<'a, K: Ord, T> {
    tables: &'a Tables,
    pick: Pick<K, T>,
    id: K,
    sender: watch::Sender<Option<Arc<T>>>,
}

impl<K: Ord + Clone, T> Flight<'_, K, T> {
    fn finish(self, table: T) -> Arc<T> {
        let table = Arc::new(table);
        (self.pick)(&mut self.tables.cache())
            .ready
            .insert(self.id.clone(), Arc::clone(&table));
        self.sender.send_replace(Some(Arc::clone(&table)));
        table
    }
}

impl<K: Ord, T> Drop for Flight<'_, K, T> {
    fn drop(&mut self) {
        (self.pick)(&mut self.tables.cache())
            .loading
            .remove(&self.id);
    }
}

/// Decoded tables of the segments a reader has used, and the live rows of each segment under
/// each file table that holds it, kept while a manifest references them.
#[derive(Debug, Default)]
pub(crate) struct Tables(Mutex<Cache>);

impl Tables {
    #[expect(
        clippy::expect_used,
        reason = "a poisoned lock means an earlier panic, and its state is not trusted"
    )]
    fn cache(&self) -> MutexGuard<'_, Cache> {
        self.0
            .lock()
            .expect("no thread panicked while holding the lock")
    }

    /// Forgets what belongs to segments `manifest` no longer references.
    pub(crate) fn retain(&self, manifest: &Manifest) {
        let referenced: BTreeSet<&SegmentId> = manifest
            .shards()
            .values()
            .flat_map(ShardEntry::segments)
            .map(SegmentEntry::id)
            .collect();
        self.retain_only(&referenced);
    }

    fn retain_only(&self, referenced: &BTreeSet<&SegmentId>) {
        let mut cache = self.cache();
        cache.files.ready.retain(|id, _| referenced.contains(id));
        cache.counts.ready.retain(|id, _| referenced.contains(id));
        cache.chunks.ready.retain(|id, _| referenced.contains(id));
        cache
            .live
            .ready
            .retain(|(chunks, files), _| referenced.contains(chunks) && referenced.contains(files));
    }

    /// The rows of `chunks` live in `files`, worked out once for each pair of segments.
    pub(crate) async fn live_rows(
        &self,
        chunks_segment: &SegmentId,
        chunks: &ChunkTable,
        files_segment: &SegmentId,
        files: &FileTable,
    ) -> Arc<RowSet> {
        let pair = (chunks_segment.clone(), files_segment.clone());
        loop {
            match self.turn(&pair, |cache| &mut cache.live) {
                Turn::Ready(rows) => return rows,
                Turn::Wait(mut receiver) => {
                    if let Some(rows) = settled(&mut receiver).await {
                        return rows;
                    }
                }
                Turn::Load(flight) => {
                    return flight.finish(chunks.rows(files, &Scope::all(), &Filter::new()));
                }
            }
        }
    }

    /// The files of `files`, read from `files_segment`, by state, worked out once for each
    /// segment.
    pub(crate) async fn counts(&self, files_segment: &SegmentId, files: &FileTable) -> Arc<Counts> {
        loop {
            match self.turn(files_segment, |cache| &mut cache.counts) {
                Turn::Ready(counts) => return counts,
                Turn::Wait(mut receiver) => {
                    if let Some(counts) = settled(&mut receiver).await {
                        return counts;
                    }
                }
                Turn::Load(flight) => {
                    return flight.finish(Counts::of(files, &Scope::all(), &Filter::new()));
                }
            }
        }
    }

    /// The value of `id`, or the work another call is running, or a claim to do it.
    fn turn<K: Ord + Clone, T>(&self, id: &K, pick: Pick<K, T>) -> Turn<'_, K, T> {
        let mut cache = self.cache();
        let slots = pick(&mut cache);
        if let Some(table) = slots.ready.get(id) {
            return Turn::Ready(Arc::clone(table));
        }
        if let Some(receiver) = slots.loading.get(id) {
            return Turn::Wait(receiver.clone());
        }
        let (sender, receiver) = watch::channel(None);
        slots.loading.insert(id.clone(), receiver);
        Turn::Load(Flight {
            tables: self,
            pick,
            id: id.clone(),
            sender,
        })
    }
}

/// Loads tables for one call, through the cache if there is one, counting what it reads from
/// the store.
#[derive(Debug, Clone)]
pub(crate) struct Loader<'a> {
    index: IndexHandle,
    source: SegmentSource,
    counter: Arc<Counter>,
    tables: &'a Tables,
}

impl<'a> Loader<'a> {
    pub(crate) fn new(index: &IndexHandle, tables: &'a Tables, cache: Option<CacheView>) -> Self {
        let counter = Arc::new(Counter::default());
        Loader {
            index: index.clone(),
            source: source(index, &counter, cache),
            counter,
            tables,
        }
    }

    /// This loader with the cache view of another manifest, still counting into one counter.
    pub(crate) fn rebind(&self, cache: Option<CacheView>) -> Self {
        Loader {
            index: self.index.clone(),
            source: source(&self.index, &self.counter, cache),
            counter: Arc::clone(&self.counter),
            tables: self.tables,
        }
    }

    /// What index plug-ins read segment files through.
    pub(crate) fn source(&self) -> &SegmentSource {
        &self.source
    }

    /// The counter this call's reads go to.
    pub(crate) fn counter(&self) -> &Counter {
        &self.counter
    }

    /// The file table of `shard`, empty if it has none yet.
    pub(crate) async fn files(&self, shard: &ShardEntry) -> Result<Arc<FileTable>> {
        let Some((segment, file)) =
            files_segment(shard).and_then(|s| Some((s, s.files().first()?)))
        else {
            return Ok(Arc::new(FileTable::new()));
        };
        self.load(
            segment.id(),
            file,
            |cache| &mut cache.files,
            |bytes| Ok(FileTable::decode(bytes)?),
        )
        .await
    }

    /// The files of `files`, read from `files_segment`, by state, shared with every call that
    /// wants them.
    pub(crate) async fn counts(&self, files_segment: &SegmentId, files: &FileTable) -> Arc<Counts> {
        self.tables.counts(files_segment, files).await
    }

    /// The rows of `chunks` live in `files`, shared with every call that wants them.
    pub(crate) async fn live_rows(
        &self,
        chunks_segment: &SegmentId,
        chunks: &ChunkTable,
        files_segment: &SegmentId,
        files: &FileTable,
    ) -> Arc<RowSet> {
        self.tables
            .live_rows(chunks_segment, chunks, files_segment, files)
            .await
    }

    /// The chunk records of `segment`, or `None` if it has none.
    pub(crate) async fn chunks(&self, segment: &SegmentEntry) -> Result<Option<Arc<ChunkTable>>> {
        let Some(file) = chunk_file(segment) else {
            return Ok(None);
        };
        let table = self
            .load(
                segment.id(),
                file,
                |cache| &mut cache.chunks,
                |bytes| Ok(ChunkTable::decode(bytes)?),
            )
            .await?;
        Ok(Some(table))
    }

    /// The table in `file` of `segment`, read and decoded by one call at a time.
    async fn load<T>(
        &self,
        segment: &SegmentId,
        file: &SegmentFile,
        pick: Pick<SegmentId, T>,
        decode: fn(Bytes) -> Result<T>,
    ) -> Result<Arc<T>> {
        loop {
            let turn = self.tables.turn(segment, pick);
            match turn {
                Turn::Ready(table) => return Ok(table),
                Turn::Wait(mut receiver) => {
                    if let Some(table) = settled(&mut receiver).await {
                        return Ok(table);
                    }
                }
                Turn::Load(flight) => {
                    let table = decode(self.whole(segment, file).await?)?;
                    return Ok(flight.finish(table));
                }
            }
        }
    }

    async fn whole(&self, segment: &SegmentId, file: &SegmentFile) -> Result<Bytes> {
        let range = ByteRange::new(0, file.size())?;
        let bytes = self.source.read_range(segment, file.name(), range).await?;
        Ok(Bytes::from(bytes))
    }
}

/// What the call making a value sent, or `None` if it gave up.
async fn settled<T>(receiver: &mut watch::Receiver<Option<Arc<T>>>) -> Option<Arc<T>> {
    receiver
        .wait_for(Option::is_some)
        .await
        .ok()
        .and_then(|sent| sent.clone())
}

fn source(index: &IndexHandle, counter: &Arc<Counter>, cache: Option<CacheView>) -> SegmentSource {
    let store = SegmentSource::new(Counted {
        index: index.clone(),
        counter: Arc::clone(counter),
    });
    match cache {
        Some(view) => SegmentSource::new(Cached { store, view }),
        None => store,
    }
}

/// The segment holding a shard's file table.
pub(crate) fn files_segment(shard: &ShardEntry) -> Option<&SegmentEntry> {
    shard
        .segments()
        .iter()
        .find(|segment| segment.level() == Level::L0)
}

/// The file holding a segment's chunk records.
pub(crate) fn chunk_file(segment: &SegmentEntry) -> Option<&SegmentFile> {
    segment
        .files()
        .iter()
        .find(|file| file.name().as_str().starts_with(CHUNKS_PREFIX))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn files(cache: &mut Cache) -> &mut Slots<SegmentId, FileTable> {
        &mut cache.files
    }

    #[test]
    fn one_call_loads_a_table_for_all_and_a_dropped_load_hands_over() {
        let tables = Tables::default();
        let id = SegmentId::new("s1").unwrap();
        let Turn::Load(first) = tables.turn(&id, files) else {
            panic!("the first call loads");
        };
        let Turn::Wait(abandoned) = tables.turn(&id, files) else {
            panic!("a second call waits");
        };
        drop(first);
        assert!(
            abandoned.has_changed().is_err(),
            "a dropped load wakes its waiters"
        );

        let Turn::Load(second) = tables.turn(&id, files) else {
            panic!("a call after a dropped load loads again");
        };
        let Turn::Wait(waiting) = tables.turn(&id, files) else {
            panic!("a call during the load waits");
        };
        let table = second.finish(FileTable::new());
        let sent = waiting.borrow().clone().unwrap();
        assert!(Arc::ptr_eq(&sent, &table));
        let Turn::Ready(ready) = tables.turn(&id, files) else {
            panic!("a loaded table is ready");
        };
        assert!(Arc::ptr_eq(&ready, &table));
        assert!(tables.cache().files.loading.is_empty());
    }

    fn live(cache: &mut Cache) -> &mut Slots<Pair, RowSet> {
        &mut cache.live
    }

    #[tokio::test]
    async fn live_rows_are_worked_out_once_per_pair_and_shared() {
        let tables = Tables::default();
        let (chunks_id, files_id, newer_id) = (
            SegmentId::new("c1").unwrap(),
            SegmentId::new("f1").unwrap(),
            SegmentId::new("f2").unwrap(),
        );
        let (chunks, files) = (ChunkTable::default(), FileTable::new());
        let (first, again) = tokio::join!(
            tables.live_rows(&chunks_id, &chunks, &files_id, &files),
            tables.live_rows(&chunks_id, &chunks, &files_id, &files),
        );
        assert!(Arc::ptr_eq(&first, &again));
        let later = tables
            .live_rows(&chunks_id, &chunks, &files_id, &files)
            .await;
        assert!(Arc::ptr_eq(&first, &later));
        let newer = tables
            .live_rows(&chunks_id, &chunks, &newer_id, &files)
            .await;
        assert!(!Arc::ptr_eq(&first, &newer));

        let pair = (chunks_id.clone(), SegmentId::new("f3").unwrap());
        let Turn::Load(flight) = tables.turn(&pair, live) else {
            panic!("the first call works the rows out");
        };
        let mut made = RowSet::new();
        made.insert(0..3);
        let (waited, finished) = tokio::join!(
            tables.live_rows(&chunks_id, &chunks, &pair.1, &files),
            async {
                tokio::task::yield_now().await;
                flight.finish(made)
            },
        );
        assert!(Arc::ptr_eq(&waited, &finished));
    }

    #[tokio::test]
    async fn live_rows_go_when_either_segment_is_no_longer_referenced() {
        let tables = Tables::default();
        let ids: Vec<SegmentId> = ["c1", "c2", "f1", "f2"]
            .into_iter()
            .map(|id| SegmentId::new(id).unwrap())
            .collect();
        let (chunks, files) = (ChunkTable::default(), FileTable::new());
        for (chunks_id, files_id) in [(0, 2), (0, 3), (1, 2)] {
            tables
                .live_rows(&ids[chunks_id], &chunks, &ids[files_id], &files)
                .await;
        }
        tables.retain_only(&BTreeSet::from([&ids[0], &ids[2]]));
        let kept: Vec<Pair> = tables.cache().live.ready.keys().cloned().collect();
        assert_eq!(kept, [(ids[0].clone(), ids[2].clone())]);
        tables.retain_only(&BTreeSet::from([&ids[0]]));
        assert!(tables.cache().live.ready.is_empty());
    }
}
