use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use async_trait::async_trait;
use bytes::Bytes;
use cosmosearch_core::{
    ByteRange, Filter, IndexError, IndexId, Level, Manifest, RowSet, Scope, SegmentEntry,
    SegmentFile, SegmentFileName, SegmentFiles, SegmentId, SegmentSource, ShardEntry, Usage,
};
use cosmosearch_format::{
    ChunkGroup, ChunkIndex, ChunkRecord, FileEntry, FileGroup, FileIndex, FileTable, TRAILER_BYTES,
    TextGroup, TextIndex,
};
use cosmosearch_store::IndexHandle;
use tokio::sync::watch;

use crate::Result;
use crate::cached::{CacheView, Cached};
use crate::coverage::Counts;
use crate::table_cache::{Part, Table, TableCache, TableKey};

const CHUNKS_PREFIX: &str = "chunks-";
const TEXT_PREFIX: &str = "text-";
const TAIL_GUESS: u64 = 64 * 1024;

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

/// What every search needs of a shard's file table: its index, and its files by state.
#[derive(Debug)]
pub(crate) struct Summary {
    pub(crate) index: Option<FileIndex>,
    pub(crate) counts: Counts,
}

impl Summary {
    fn of(index: FileIndex) -> Self {
        Summary {
            counts: Counts::of_states(index.states()),
            index: Some(index),
        }
    }

    fn empty() -> Self {
        Summary {
            index: None,
            counts: Counts::of(&FileTable::new(), &Scope::all(), &Filter::new()),
        }
    }

    /// The content version of the file with this ordinal.
    pub(crate) fn version(&self, ordinal: u64) -> Option<cosmosearch_core::ManifestVersion> {
        self.index.as_ref()?.version(ordinal)
    }
}

/// A shard's summary, and the segment its file table was read from, if it has one.
#[derive(Debug, Clone)]
pub(crate) struct ShardFiles {
    pub(crate) segment: Option<SegmentId>,
    pub(crate) summary: Arc<Summary>,
}

/// A segment's chunk index and the file table of a shard that holds it.
type Pair = (SegmentId, SegmentId);

#[derive(Debug, Default)]
struct Cache {
    indexes: Slots<SegmentId, ChunkIndex>,
    summaries: Slots<SegmentId, Summary>,
    live: Slots<Pair, RowSet>,
    files: Slots<SegmentId, FileTable>,
    file_groups: Slots<(SegmentId, usize), FileGroup>,
    groups: Slots<(SegmentId, usize), ChunkGroup>,
    texts: Slots<SegmentId, TextIndex>,
    text_groups: Slots<(SegmentId, usize), TextGroup>,
}

/// The values of one kind kept by the reader, and those one call is making for all.
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
    keep: bool,
    sender: watch::Sender<Option<Arc<T>>>,
}

impl<K: Ord + Clone, T> Flight<'_, K, T> {
    /// Hands `table` to the calls waiting for it, keeping it if this kind is kept.
    fn finish(self, table: Arc<T>) -> Arc<T> {
        if self.keep {
            (self.pick)(&mut self.tables.cache())
                .ready
                .insert(self.id.clone(), Arc::clone(&table));
        }
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

/// What a reader keeps of the segments it has used while a manifest references them: each
/// segment's chunk index, each file table's summary and the live rows of each segment under
/// each file table that holds it. Full file tables and groups of chunk records go to a
/// [`TableCache`] under its budget.
#[derive(Debug)]
pub(crate) struct Tables {
    cache: Mutex<Cache>,
    shared: TableCache,
    index: IndexId,
}

impl Tables {
    pub(crate) fn new(index: IndexId, shared: TableCache) -> Self {
        Tables {
            cache: Mutex::new(Cache::default()),
            shared,
            index,
        }
    }

    #[expect(
        clippy::expect_used,
        reason = "a poisoned lock means an earlier panic, and its state is not trusted"
    )]
    fn cache(&self) -> MutexGuard<'_, Cache> {
        self.cache
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
        cache.indexes.ready.retain(|id, _| referenced.contains(id));
        cache.texts.ready.retain(|id, _| referenced.contains(id));
        cache
            .summaries
            .ready
            .retain(|id, _| referenced.contains(id));
        cache
            .live
            .ready
            .retain(|(chunks, files), _| referenced.contains(chunks) && referenced.contains(files));
    }

    /// About how many bytes the reader holds: what it keeps, and its tables in the cache.
    pub(crate) fn resident_bytes(&self) -> u64 {
        let cache = self.cache();
        let kept = cache
            .indexes
            .ready
            .values()
            .map(|index| index.heap_bytes())
            .chain(cache.texts.ready.values().map(|texts| texts.heap_bytes()))
            .chain(cache.summaries.ready.values().map(|summary| {
                summary
                    .index
                    .as_ref()
                    .map_or(0, FileIndex::heap_bytes)
                    .saturating_add(size_of::<Summary>())
            }))
            .chain(cache.live.ready.values().map(|rows| rows.heap_bytes()))
            .fold(0usize, usize::saturating_add);
        u64::try_from(kept)
            .unwrap_or(u64::MAX)
            .saturating_add(self.shared.bytes_of(&self.index))
    }

    fn key(&self, segment: &SegmentId, part: Part) -> TableKey {
        TableKey {
            index: self.index.clone(),
            segment: segment.clone(),
            part,
        }
    }

    /// The rows of `chunks` live in `files`, worked out once for each pair of segments.
    pub(crate) async fn live_rows(
        &self,
        chunks_segment: &SegmentId,
        chunks: &ChunkIndex,
        files_segment: &SegmentId,
        files: &FileIndex,
    ) -> Arc<RowSet> {
        let pair = (chunks_segment.clone(), files_segment.clone());
        loop {
            match self.turn(&pair, |cache| &mut cache.live, true, || None) {
                Turn::Ready(rows) => return rows,
                Turn::Wait(mut receiver) => {
                    if let Some(rows) = settled(&mut receiver).await {
                        return rows;
                    }
                }
                Turn::Load(flight) => {
                    return flight.finish(Arc::new(chunks.live_rows(files)));
                }
            }
        }
    }

    /// The value of `id`, or the work another call is running, or a claim to do it. A kind
    /// the reader keeps is found in its slots, any other through `shared`.
    fn turn<K: Ord + Clone, T>(
        &self,
        id: &K,
        pick: Pick<K, T>,
        keep: bool,
        shared: impl FnOnce() -> Option<Arc<T>>,
    ) -> Turn<'_, K, T> {
        let mut cache = self.cache();
        let slots = pick(&mut cache);
        if let Some(table) = slots.ready.get(id) {
            return Turn::Ready(Arc::clone(table));
        }
        if let Some(table) = shared() {
            return Turn::Ready(table);
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
            keep,
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

    /// The full file table of `shard`, empty if it has none yet.
    pub(crate) async fn files(&self, shard: &ShardEntry) -> Result<Arc<FileTable>> {
        let Some((segment, file)) = located(shard) else {
            return Ok(Arc::new(FileTable::new()));
        };
        self.files_in(segment.id(), file).await
    }

    /// The summary of `shard`'s file table, kept by the reader.
    pub(crate) async fn summary(&self, shard: &ShardEntry) -> Result<ShardFiles> {
        let Some((segment, file)) = located(shard) else {
            return Ok(ShardFiles {
                segment: None,
                summary: Arc::new(Summary::empty()),
            });
        };
        let id = segment.id();
        loop {
            match self
                .tables
                .turn(id, |cache| &mut cache.summaries, true, || None)
            {
                Turn::Ready(summary) => {
                    return Ok(ShardFiles {
                        segment: Some(id.clone()),
                        summary,
                    });
                }
                Turn::Wait(mut receiver) => {
                    if let Some(summary) = settled(&mut receiver).await {
                        return Ok(ShardFiles {
                            segment: Some(id.clone()),
                            summary,
                        });
                    }
                }
                Turn::Load(flight) => {
                    let bytes = self.read_tail(id, file, FileIndex::locate).await?;
                    let summary = flight.finish(Arc::new(Summary::of(FileIndex::decode(bytes)?)));
                    return Ok(ShardFiles {
                        segment: Some(id.clone()),
                        summary,
                    });
                }
            }
        }
    }

    async fn files_in(&self, segment: &SegmentId, file: &SegmentFile) -> Result<Arc<FileTable>> {
        let key = self.tables.key(segment, Part::Files);
        let shared = || match self.tables.shared.get(&key) {
            Some(Table::Files(files)) => Some(files),
            Some(Table::FileGroup(_) | Table::Group(_) | Table::TextGroup(_)) | None => None,
        };
        loop {
            match self
                .tables
                .turn(segment, |cache| &mut cache.files, false, shared)
            {
                Turn::Ready(files) => return Ok(files),
                Turn::Wait(mut receiver) => {
                    if let Some(files) = settled(&mut receiver).await {
                        return Ok(files);
                    }
                }
                Turn::Load(flight) => {
                    let range = ByteRange::new(0, file.size())?;
                    let bytes = self.read(segment, file, range).await?;
                    let files = Arc::new(FileTable::decode(bytes)?);
                    self.tables
                        .shared
                        .put(key.clone(), Table::Files(Arc::clone(&files)));
                    return Ok(flight.finish(files));
                }
            }
        }
    }

    /// The rows of `chunks` live in `files`, shared with every call that wants them.
    pub(crate) async fn live_rows(
        &self,
        chunks_segment: &SegmentId,
        chunks: &ChunkIndex,
        files_segment: &SegmentId,
        files: &FileIndex,
    ) -> Arc<RowSet> {
        self.tables
            .live_rows(chunks_segment, chunks, files_segment, files)
            .await
    }

    /// The chunk index of `segment`, kept by the reader, or `None` if it has no chunk records.
    pub(crate) async fn index(&self, segment: &SegmentEntry) -> Result<Option<Arc<ChunkIndex>>> {
        let Some(file) = chunk_file(segment) else {
            return Ok(None);
        };
        let id = segment.id();
        loop {
            match self
                .tables
                .turn(id, |cache| &mut cache.indexes, true, || None)
            {
                Turn::Ready(index) => return Ok(Some(index)),
                Turn::Wait(mut receiver) => {
                    if let Some(index) = settled(&mut receiver).await {
                        return Ok(Some(index));
                    }
                }
                Turn::Load(flight) => {
                    let bytes = self.read_tail(id, file, ChunkIndex::locate).await?;
                    let index = ChunkIndex::decode(bytes)?;
                    return Ok(Some(flight.finish(Arc::new(index))));
                }
            }
        }
    }

    /// The index at the end of `file`, which `locate` finds from its trailer.
    async fn read_tail(
        &self,
        segment: &SegmentId,
        file: &SegmentFile,
        locate: fn(&[u8], u64) -> cosmosearch_format::Result<Range<u64>>,
    ) -> Result<Bytes> {
        let size = file.size();
        let tail_start = size.saturating_sub(TAIL_GUESS);
        let tail = self
            .read(segment, file, ByteRange::new(tail_start, size)?)
            .await?;
        let trailer = tail.slice(
            tail.len()
                .saturating_sub(usize::try_from(TRAILER_BYTES).unwrap_or(usize::MAX))..,
        );
        let at = locate(&trailer, size)?;
        match at.start.checked_sub(tail_start) {
            Some(offset) => {
                let start = usize::try_from(offset).unwrap_or(usize::MAX);
                let length = usize::try_from(at.end.saturating_sub(at.start)).unwrap_or(usize::MAX);
                tail.get(start..start.saturating_add(length))
                    .map(Bytes::copy_from_slice)
                    .ok_or_else(|| {
                        cosmosearch_format::Error::Malformed {
                            reason: "an index lies outside its file",
                        }
                        .into()
                    })
            }
            None => {
                self.read(segment, file, ByteRange::new(at.start, at.end)?)
                    .await
            }
        }
    }

    /// The file with key `key` in `shard`, read from the group of its file table that holds
    /// it.
    pub(crate) async fn file(&self, shard: &ShardEntry, key: &str) -> Result<Option<FileEntry>> {
        let found = self.summary(shard).await?;
        let Some(index) = found.summary.index.as_ref() else {
            return Ok(None);
        };
        for ordinal in index.ordinals_for(key) {
            let group = self.file_of(shard, &found.summary, ordinal).await?;
            if let Some(entry) = group
                .as_ref()
                .and_then(|group| group.by_ordinal(ordinal))
                .filter(|entry| entry.key() == key)
            {
                return Ok(Some(entry.clone()));
            }
        }
        Ok(None)
    }

    /// The group of files holding `ordinal` in `shard`'s file table, through the table cache.
    pub(crate) async fn file_of(
        &self,
        shard: &ShardEntry,
        summary: &Summary,
        ordinal: u64,
    ) -> Result<Option<Arc<FileGroup>>> {
        let Some(group) = summary
            .index
            .as_ref()
            .and_then(|index| index.group_of(ordinal))
        else {
            return Ok(None);
        };
        self.file_group(shard, summary, group).await
    }

    /// Group `group` of `shard`'s file table, through the table cache.
    pub(crate) async fn file_group(
        &self,
        shard: &ShardEntry,
        summary: &Summary,
        group: usize,
    ) -> Result<Option<Arc<FileGroup>>> {
        let (Some((segment, file)), Some(index)) = (located(shard), summary.index.as_ref()) else {
            return Ok(None);
        };
        let Some((_, at)) = index.group(group) else {
            return Ok(None);
        };
        let id = segment.id();
        let key = self.tables.key(id, Part::FileGroup(group));
        let cached = || match self.tables.shared.get(&key) {
            Some(Table::FileGroup(found)) => Some(found),
            Some(Table::Files(_) | Table::Group(_) | Table::TextGroup(_)) | None => None,
        };
        let slot = (id.clone(), group);
        loop {
            match self
                .tables
                .turn(&slot, |cache| &mut cache.file_groups, false, cached)
            {
                Turn::Ready(found) => return Ok(Some(found)),
                Turn::Wait(mut receiver) => {
                    if let Some(found) = settled(&mut receiver).await {
                        return Ok(Some(found));
                    }
                }
                Turn::Load(flight) => {
                    let bytes = self
                        .read(id, file, ByteRange::new(at.start, at.end)?)
                        .await?;
                    let found = Arc::new(FileGroup::decode(bytes, index, group)?);
                    self.tables
                        .shared
                        .put(key.clone(), Table::FileGroup(Arc::clone(&found)));
                    return Ok(Some(flight.finish(found)));
                }
            }
        }
    }

    /// The group of records `group` of `segment`, through the table cache.
    pub(crate) async fn group(
        &self,
        segment: &SegmentEntry,
        index: &ChunkIndex,
        group: usize,
    ) -> Result<Arc<ChunkGroup>> {
        let id = segment.id();
        let (Some(file), Some((_, at))) = (chunk_file(segment), index.group(group)) else {
            return Err(cosmosearch_format::Error::Malformed {
                reason: "a group of chunk records is missing",
            }
            .into());
        };
        let key = self.tables.key(id, Part::Group(group));
        let shared = || match self.tables.shared.get(&key) {
            Some(Table::Group(group)) => Some(group),
            Some(Table::Files(_) | Table::FileGroup(_) | Table::TextGroup(_)) | None => None,
        };
        let slot = (id.clone(), group);
        loop {
            match self
                .tables
                .turn(&slot, |cache| &mut cache.groups, false, shared)
            {
                Turn::Ready(found) => return Ok(found),
                Turn::Wait(mut receiver) => {
                    if let Some(found) = settled(&mut receiver).await {
                        return Ok(found);
                    }
                }
                Turn::Load(flight) => {
                    let bytes = self
                        .read(id, file, ByteRange::new(at.start, at.end)?)
                        .await?;
                    let found = Arc::new(ChunkGroup::decode(bytes, index, group)?);
                    self.tables
                        .shared
                        .put(key.clone(), Table::Group(Arc::clone(&found)));
                    return Ok(flight.finish(found));
                }
            }
        }
    }

    /// Gives `visit` each record in `rows` of `segment` in order, with its row, reading one
    /// group at a time, until `visit` returns false.
    pub(crate) async fn scan(
        &self,
        segment: &SegmentEntry,
        index: &ChunkIndex,
        rows: Range<usize>,
        mut visit: impl FnMut(usize, &ChunkRecord) -> bool + Send,
    ) -> Result<()> {
        let mut row = rows.start;
        while row < rows.end {
            let (Some(group), Some((held, _))) = (
                index.group_of(row),
                index.group_of(row).and_then(|group| index.group(group)),
            ) else {
                return Ok(());
            };
            let found = self.group(segment, index, group).await?;
            let end = held.end.min(rows.end);
            for at in row..end {
                let Some(record) = found.get(at) else {
                    return Ok(());
                };
                if !visit(at, record) {
                    return Ok(());
                }
            }
            row = end;
        }
        Ok(())
    }

    /// The index of `segment`'s stored text, kept by the reader, or `None` if it has none.
    pub(crate) async fn text_index(
        &self,
        segment: &SegmentEntry,
    ) -> Result<Option<Arc<TextIndex>>> {
        let Some(file) = text_file(segment) else {
            return Ok(None);
        };
        let id = segment.id();
        loop {
            match self
                .tables
                .turn(id, |cache| &mut cache.texts, true, || None)
            {
                Turn::Ready(index) => return Ok(Some(index)),
                Turn::Wait(mut receiver) => {
                    if let Some(index) = settled(&mut receiver).await {
                        return Ok(Some(index));
                    }
                }
                Turn::Load(flight) => {
                    let bytes = self.read_tail(id, file, TextIndex::locate).await?;
                    let index = TextIndex::decode(bytes)?;
                    return Ok(Some(flight.finish(Arc::new(index))));
                }
            }
        }
    }

    /// The text stored for `row` of `segment`, through the table cache, or `None` if the row
    /// has none.
    pub(crate) async fn stored_text(
        &self,
        segment: &SegmentEntry,
        texts: &TextIndex,
        row: usize,
    ) -> Result<Option<Arc<str>>> {
        let (true, Some(group)) = (texts.has_text(row), texts.group_of(row)) else {
            return Ok(None);
        };
        let found = self.text_group(segment, texts, group).await?;
        Ok(found.and_then(|found| found.get(row)))
    }

    /// Group `group` of `segment`'s stored text, through the table cache, or `None` if the
    /// segment has no such group.
    pub(crate) async fn text_group(
        &self,
        segment: &SegmentEntry,
        texts: &TextIndex,
        group: usize,
    ) -> Result<Option<Arc<TextGroup>>> {
        let (Some(file), Some((_, at))) = (text_file(segment), texts.group(group)) else {
            return Ok(None);
        };
        let id = segment.id();
        let key = self.tables.key(id, Part::TextGroup(group));
        let cached = || match self.tables.shared.get(&key) {
            Some(Table::TextGroup(found)) => Some(found),
            Some(Table::Files(_) | Table::FileGroup(_) | Table::Group(_)) | None => None,
        };
        let slot = (id.clone(), group);
        loop {
            match self
                .tables
                .turn(&slot, |cache| &mut cache.text_groups, false, cached)
            {
                Turn::Ready(found) => return Ok(Some(found)),
                Turn::Wait(mut receiver) => {
                    if let Some(found) = settled(&mut receiver).await {
                        return Ok(Some(found));
                    }
                }
                Turn::Load(flight) => {
                    let bytes = self
                        .read(id, file, ByteRange::new(at.start, at.end)?)
                        .await?;
                    let found = Arc::new(TextGroup::decode(bytes, texts, group)?);
                    self.tables
                        .shared
                        .put(key.clone(), Table::TextGroup(Arc::clone(&found)));
                    return Ok(Some(flight.finish(found)));
                }
            }
        }
    }

    async fn read(
        &self,
        segment: &SegmentId,
        file: &SegmentFile,
        range: ByteRange,
    ) -> Result<Bytes> {
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

/// The segment holding a shard's file table, and its file.
fn located(shard: &ShardEntry) -> Option<(&SegmentEntry, &SegmentFile)> {
    let segment = files_segment(shard)?;
    Some((segment, segment.files().first()?))
}

/// The segment holding a shard's file table.
pub(crate) fn files_segment(shard: &ShardEntry) -> Option<&SegmentEntry> {
    shard
        .segments()
        .iter()
        .find(|segment| segment.level() == Level::L0)
}

/// The file holding a segment's stored text, if it has one.
fn text_file(segment: &SegmentEntry) -> Option<&SegmentFile> {
    segment
        .files()
        .iter()
        .find(|file| file.name().as_str().starts_with(TEXT_PREFIX))
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
    use cosmosearch_format::ChunkTable;

    use super::*;
    use crate::table_cache::DEFAULT_TABLE_BUDGET;

    fn tables() -> Tables {
        Tables::new(
            IndexId::new("index").unwrap(),
            TableCache::new(DEFAULT_TABLE_BUDGET),
        )
    }

    fn empty_index() -> ChunkIndex {
        let bytes = Bytes::from(ChunkTable::default().encode().unwrap());
        let size = u64::try_from(bytes.len()).unwrap();
        let trailer =
            bytes.slice(usize::try_from(size.checked_sub(TRAILER_BYTES).unwrap()).unwrap()..);
        let at = ChunkIndex::locate(&trailer, size).unwrap();
        let start = usize::try_from(at.start).unwrap();
        let end = usize::try_from(at.end).unwrap();
        ChunkIndex::decode(bytes.slice(start..end)).unwrap()
    }

    fn empty_files() -> FileIndex {
        let bytes = Bytes::from(FileTable::new().encode().unwrap());
        let size = u64::try_from(bytes.len()).unwrap();
        let trailer =
            bytes.slice(usize::try_from(size.checked_sub(TRAILER_BYTES).unwrap()).unwrap()..);
        let at = FileIndex::locate(&trailer, size).unwrap();
        let start = usize::try_from(at.start).unwrap();
        let end = usize::try_from(at.end).unwrap();
        FileIndex::decode(bytes.slice(start..end)).unwrap()
    }

    fn summaries(cache: &mut Cache) -> &mut Slots<SegmentId, Summary> {
        &mut cache.summaries
    }

    fn files(cache: &mut Cache) -> &mut Slots<SegmentId, FileTable> {
        &mut cache.files
    }

    fn summary() -> Arc<Summary> {
        Arc::new(Summary::empty())
    }

    #[test]
    fn one_call_loads_a_value_for_all_and_a_dropped_load_hands_over() {
        let tables = tables();
        let id = SegmentId::new("s1").unwrap();
        let Turn::Load(first) = tables.turn(&id, summaries, true, || None) else {
            panic!("the first call loads");
        };
        let Turn::Wait(abandoned) = tables.turn(&id, summaries, true, || None) else {
            panic!("a second call waits");
        };
        drop(first);
        assert!(
            abandoned.has_changed().is_err(),
            "a dropped load wakes its waiters"
        );

        let Turn::Load(second) = tables.turn(&id, summaries, true, || None) else {
            panic!("a call after a dropped load loads again");
        };
        let Turn::Wait(waiting) = tables.turn(&id, summaries, true, || None) else {
            panic!("a call during the load waits");
        };
        let made = second.finish(summary());
        let sent = waiting.borrow().clone().unwrap();
        assert!(Arc::ptr_eq(&sent, &made));
        let Turn::Ready(ready) = tables.turn(&id, summaries, true, || None) else {
            panic!("a kept value is ready");
        };
        assert!(Arc::ptr_eq(&ready, &made));
        assert!(tables.cache().summaries.loading.is_empty());
    }

    #[test]
    fn a_table_the_reader_does_not_keep_is_found_only_through_the_cache() {
        let tables = tables();
        let id = SegmentId::new("s1").unwrap();
        let Turn::Load(flight) = tables.turn(&id, files, false, || None) else {
            panic!("the first call loads");
        };
        let made = flight.finish(Arc::new(FileTable::new()));
        assert!(matches!(
            tables.turn(&id, files, false, || None),
            Turn::Load(_)
        ));
        let Turn::Ready(found) = tables.turn(&id, files, false, || Some(Arc::clone(&made))) else {
            panic!("a table in the cache is ready");
        };
        assert!(Arc::ptr_eq(&found, &made));
        assert!(tables.cache().files.ready.is_empty());
    }

    #[tokio::test]
    async fn live_rows_are_worked_out_once_per_pair_and_shared() {
        let tables = tables();
        let (chunks_id, files_id, newer_id) = (
            SegmentId::new("c1").unwrap(),
            SegmentId::new("f1").unwrap(),
            SegmentId::new("f2").unwrap(),
        );
        let (index, versions) = (empty_index(), empty_files());
        let (first, again) = tokio::join!(
            tables.live_rows(&chunks_id, &index, &files_id, &versions),
            tables.live_rows(&chunks_id, &index, &files_id, &versions),
        );
        assert!(Arc::ptr_eq(&first, &again));
        let later = tables
            .live_rows(&chunks_id, &index, &files_id, &versions)
            .await;
        assert!(Arc::ptr_eq(&first, &later));
        let newer = tables
            .live_rows(&chunks_id, &index, &newer_id, &versions)
            .await;
        assert!(!Arc::ptr_eq(&first, &newer));
    }

    #[tokio::test]
    async fn kept_values_go_when_a_segment_is_no_longer_referenced() {
        let tables = tables();
        let ids: Vec<SegmentId> = ["c1", "c2", "f1", "f2"]
            .into_iter()
            .map(|id| SegmentId::new(id).unwrap())
            .collect();
        let (index, versions) = (empty_index(), empty_files());
        for (chunks_id, files_id) in [(0, 2), (0, 3), (1, 2)] {
            tables
                .live_rows(&ids[chunks_id], &index, &ids[files_id], &versions)
                .await;
        }
        let before = tables.resident_bytes();
        tables.retain_only(&BTreeSet::from([&ids[0], &ids[2]]));
        let kept: Vec<Pair> = tables.cache().live.ready.keys().cloned().collect();
        assert_eq!(kept, [(ids[0].clone(), ids[2].clone())]);
        assert!(tables.resident_bytes() < before);
        tables.retain_only(&BTreeSet::from([&ids[0]]));
        assert!(tables.cache().live.ready.is_empty());
    }
}
