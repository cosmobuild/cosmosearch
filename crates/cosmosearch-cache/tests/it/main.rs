//! Contract tests of the cache.

#![expect(
    clippy::unwrap_used,
    reason = "helpers in a test binary fail the test by panicking"
)]

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use cosmosearch_cache::{Cache, CacheStats};
use cosmosearch_core::{
    Admission, ByteRange, CachedFile, IndexError, IndexId, Placement, SegmentCache, SegmentFile,
    SegmentFileName, SegmentFiles, SegmentId, SegmentSource,
};
use futures_util::{StreamExt, stream};
use tempfile::TempDir;
use tokio::sync::{Notify, Semaphore};

const BLOCK: u64 = 64;
const HASH: u64 = 32;

#[derive(Debug)]
struct Gate {
    started: Notify,
    open: Semaphore,
}

impl Gate {
    fn closed() -> Gate {
        Gate {
            started: Notify::new(),
            open: Semaphore::new(0),
        }
    }
}

type Files = BTreeMap<(SegmentId, SegmentFileName), Vec<u8>>;

#[derive(Debug, Clone, Default)]
struct Store {
    files: Arc<Mutex<Files>>,
    reads: Arc<AtomicU64>,
    gate: Option<Arc<Gate>>,
}

#[async_trait]
impl SegmentFiles for Store {
    async fn read_range(
        &self,
        segment: &SegmentId,
        name: &SegmentFileName,
        range: ByteRange,
    ) -> Result<Vec<u8>, IndexError> {
        if let Some(gate) = &self.gate {
            gate.started.notify_one();
            drop(gate.open.acquire().await.unwrap());
        }
        self.reads.fetch_add(1, Ordering::SeqCst);
        let files = self.files.lock().unwrap();
        let file = files.get(&(segment.clone(), name.clone())).ok_or_else(|| {
            IndexError::MissingSegment {
                segment: segment.clone(),
            }
        })?;
        let start = usize::try_from(range.start()).unwrap();
        let end = usize::try_from(range.end()).unwrap();
        file.get(start..end)
            .map(<[u8]>::to_vec)
            .ok_or(IndexError::Malformed {
                reason: "past the end",
            })
    }
}

impl Store {
    fn source(&self) -> SegmentSource {
        SegmentSource::new(self.clone())
    }

    fn reads(&self) -> u64 {
        self.reads.load(Ordering::SeqCst)
    }

    fn add(&self, index: &str, segment: &str, size: u64) -> CachedFile {
        let segment: SegmentId = segment.parse().unwrap();
        let name: SegmentFileName = format!("lex-{segment}.bin").parse().unwrap();
        let body = contents(&segment, size);
        let replaced = self
            .files
            .lock()
            .unwrap()
            .insert((segment.clone(), name.clone()), body);
        assert!(replaced.is_none(), "segment ids are unique across indexes");
        CachedFile::new(
            index.parse().unwrap(),
            segment,
            &SegmentFile::new(name, size),
        )
    }
}

fn contents(segment: &SegmentId, size: u64) -> Vec<u8> {
    let seed = segment
        .as_str()
        .bytes()
        .fold(0usize, |seed, byte| seed * 31 + usize::from(byte))
        % 251;
    (0..size)
        .map(|i| u8::try_from((usize::try_from(i).unwrap() * 31 + seed) % 251).unwrap())
        .collect()
}

fn expected(file: &CachedFile, start: u64, end: u64) -> Vec<u8> {
    let body = contents(file.segment(), file.size());
    let range = usize::try_from(start).unwrap()..usize::try_from(end).unwrap();
    body.get(range).unwrap().to_vec()
}

async fn check(cache: &Cache, store: &Store, file: &CachedFile, start: u64, end: u64) {
    let range = ByteRange::new(start, end).unwrap();
    let read = cache
        .read_range(file, range, &store.source())
        .await
        .unwrap();
    assert_eq!(read, expected(file, start, end), "bytes {start}..{end}");
}

async fn memory_only(budget: u64) -> Cache {
    Cache::builder(budget)
        .block_size(BLOCK)
        .build()
        .await
        .unwrap()
}

async fn tiered(memory: u64, disk: u64) -> (Cache, TempDir) {
    let dir = TempDir::new().unwrap();
    let cache = tiered_in(dir.path(), memory, disk, BLOCK).await;
    (cache, dir)
}

async fn tiered_in(dir: &Path, memory: u64, disk: u64, block: u64) -> Cache {
    Cache::builder(memory)
        .disk(dir, disk)
        .block_size(block)
        .build()
        .await
        .unwrap()
}

fn put_file(path: &Path, body: &[u8]) {
    File::create(path).unwrap().write_all(body).unwrap();
}

fn block_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        if entry.file_type().unwrap().is_dir() {
            files.extend(block_files(&entry.path()));
        } else if entry.file_name() != "block-size" {
            files.push(entry.path());
        }
    }
    files
}

fn bytes_on_disk(dir: &Path) -> u64 {
    block_files(dir)
        .iter()
        .map(|path| std::fs::metadata(path).unwrap().len() - HASH)
        .sum()
}

fn moved(stats: CacheStats) -> (u64, u64, u64, u64) {
    (
        stats.memory_evictions(),
        stats.disk_writes(),
        stats.disk_evictions(),
        stats.dropped(),
    )
}

#[tokio::test]
async fn serves_any_range_from_whole_blocks_and_reads_only_what_is_missing() {
    let store = Store::default();
    let file = store.add("index-1", "s1", 1000);
    let cache = memory_only(1 << 20).await;

    check(&cache, &store, &file, 10, 300).await;
    assert_eq!((store.reads(), cache.stats().store_bytes()), (1, 320));
    check(&cache, &store, &file, 0, 640).await;
    assert_eq!((store.reads(), cache.stats().store_bytes()), (2, 640));
    check(&cache, &store, &file, 0, 1000).await;
    assert_eq!((store.reads(), cache.stats().store_bytes()), (3, 1000));
    for (start, end) in [(0, 1), (63, 65), (999, 1000), (128, 192), (5, 995)] {
        check(&cache, &store, &file, start, end).await;
    }
    assert_eq!(store.reads(), 3);
}

#[tokio::test]
async fn joins_adjacent_missing_blocks_into_one_read() {
    let store = Store::default();
    let file = store.add("index-1", "s1", 640);
    let cache = memory_only(1 << 20).await;

    check(&cache, &store, &file, 130, 140).await;
    check(&cache, &store, &file, 0, 640).await;
    assert_eq!(store.reads(), 3, "blocks 0 and 1, then 3 to 9");
    assert_eq!(cache.stats().store_bytes(), 640);
}

#[tokio::test]
async fn a_range_past_the_end_goes_to_the_store_uncached() {
    let store = Store::default();
    let file = store.add("index-1", "s1", 100);
    let cache = memory_only(1 << 20).await;
    let range = ByteRange::new(50, 150).unwrap();

    let err = cache
        .read_range(&file, range, &store.source())
        .await
        .unwrap_err();
    assert!(matches!(err, IndexError::Malformed { .. }), "{err}");
    assert_eq!(cache.stats().store_reads(), 0);
}

#[tokio::test]
async fn a_warm_repeat_reads_nothing_and_a_purge_makes_it_read_again() {
    let store = Store::default();
    let first = store.add("index-1", "s1", 500);
    let other = store.add("index-2", "s2", 500);
    let (cache, dir) = tiered(256, 1 << 20).await;

    check(&cache, &store, &first, 0, 500).await;
    check(&cache, &store, &other, 0, 500).await;
    let cold = store.reads();
    check(&cache, &store, &first, 0, 500).await;
    check(&cache, &store, &other, 0, 500).await;
    assert_eq!(store.reads(), cold, "a warm repeat reads nothing");
    assert!(cache.stats().disk_hits() > 0);

    cache.purge(first.index()).await.unwrap();
    assert!(!dir.path().join("cosmosearch-blocks/index-1").exists());
    check(&cache, &store, &other, 0, 500).await;
    assert_eq!(store.reads(), cold, "the other index keeps its blocks");
    check(&cache, &store, &first, 0, 500).await;
    assert!(store.reads() > cold, "a purged index is read again");
    let stats = cache.stats();
    assert_eq!(bytes_on_disk(dir.path()), stats.disk_bytes());
}

#[tokio::test]
async fn concurrent_misses_on_one_block_make_one_store_read() {
    let gate = Arc::new(Gate::closed());
    let store = Store {
        gate: Some(Arc::clone(&gate)),
        ..Store::default()
    };
    let file = store.add("index-1", "s1", 1000);
    let cache = memory_only(1 << 20).await;

    let mut tasks = Vec::new();
    for task in 0..16 {
        let (cache, store, file) = (cache.clone(), store.clone(), file.clone());
        let start = task % 4 * 5;
        tasks.push(tokio::spawn(async move {
            check(&cache, &store, &file, 70 + start, 100 + start).await;
        }));
    }
    gate.started.notified().await;
    for _ in 0..64 {
        tokio::task::yield_now().await;
    }
    gate.open.add_permits(Semaphore::MAX_PERMITS);
    for task in tasks {
        task.await.unwrap();
    }
    assert_eq!(store.reads(), 1);
    assert_eq!(
        cache.stats().memory_hits(),
        0,
        "every reader waited for the one read"
    );
}

#[tokio::test]
async fn eviction_moves_one_tier_at_a_time() {
    let store = Store::default();
    let files: Vec<CachedFile> = (0..4)
        .map(|n| store.add("index-1", &format!("s{n}"), 4 * BLOCK))
        .collect();
    let (cache, dir) = tiered(4 * BLOCK, 8 * BLOCK).await;

    check(&cache, &store, &files[0], 0, 4 * BLOCK).await;
    check(&cache, &store, &files[1], 0, 4 * BLOCK).await;
    assert_eq!(moved(cache.stats()), (4, 4, 0, 0), "memory spills to disk");
    check(&cache, &store, &files[2], 0, 4 * BLOCK).await;
    assert_eq!(moved(cache.stats()), (8, 8, 0, 0), "disk fills");
    check(&cache, &store, &files[3], 0, 4 * BLOCK).await;
    assert_eq!(
        moved(cache.stats()),
        (12, 12, 4, 0),
        "disk drops its oldest"
    );

    let stats = cache.stats();
    assert_eq!(
        (stats.memory_bytes(), stats.disk_bytes()),
        (4 * BLOCK, 8 * BLOCK)
    );
    assert_eq!(bytes_on_disk(dir.path()), 8 * BLOCK);
    let reads = store.reads();
    check(&cache, &store, &files[1], 0, 4 * BLOCK).await;
    assert_eq!(
        store.reads(),
        reads,
        "a block memory evicted comes back from disk"
    );
    assert_eq!(cache.stats().disk_hits(), 4);
    check(&cache, &store, &files[0], 0, 4 * BLOCK).await;
    assert_eq!(store.reads(), reads + 1, "a block disk evicted is gone");
}

#[tokio::test]
async fn without_a_disk_tier_memory_evictions_are_dropped() {
    let store = Store::default();
    let file = store.add("index-1", "s1", 8 * BLOCK);
    let cache = memory_only(4 * BLOCK).await;

    check(&cache, &store, &file, 0, 8 * BLOCK).await;
    assert_eq!(moved(cache.stats()), (4, 0, 0, 4));
    assert_eq!(cache.stats().memory_bytes(), 4 * BLOCK);
}

#[tokio::test]
async fn a_missing_or_damaged_block_file_is_a_miss() {
    let store = Store::default();
    let file = store.add("index-1", "s1", 2 * BLOCK);
    let filler = store.add("index-1", "s2", 4 * BLOCK);
    let (cache, dir) = tiered(4 * BLOCK, 1 << 20).await;

    check(&cache, &store, &file, 0, 2 * BLOCK).await;
    check(&cache, &store, &filler, 0, 4 * BLOCK).await;
    let blocks = dir.path().join("cosmosearch-blocks/index-1/s1");
    let block_file = |block: u64| {
        let prefix = format!("lex-s1.bin.{}.{block}.", 2 * BLOCK);
        std::fs::read_dir(&blocks)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .find(|path| {
                let name = path.file_name().unwrap().to_str().unwrap();
                name.starts_with(&prefix)
            })
    };
    std::fs::remove_file(block_file(0).unwrap()).unwrap();
    let damaged = block_file(1).unwrap();
    File::create(&damaged).unwrap().set_len(3).unwrap();
    let reads = store.reads();
    check(&cache, &store, &file, 0, 2 * BLOCK).await;
    assert_eq!(store.reads(), reads + 1);
    assert!(!damaged.exists(), "a damaged file is removed");
}

#[derive(Debug)]
struct Fixed(Placement);

impl Admission for Fixed {
    fn place(&self, _: &CachedFile, _: u64) -> Placement {
        self.0
    }
}

#[tokio::test]
async fn admission_chooses_the_tier() {
    for (placement, held) in [
        (Placement::Disk, (0, 4 * BLOCK)),
        (Placement::Skip, (0, 0)),
        (Placement::Memory, (4 * BLOCK, 0)),
    ] {
        let store = Store::default();
        let file = store.add("index-1", "s1", 4 * BLOCK);
        let dir = TempDir::new().unwrap();
        let cache = Cache::builder(1 << 20)
            .disk(dir.path(), 1 << 20)
            .block_size(BLOCK)
            .admission(Fixed(placement))
            .build()
            .await
            .unwrap();
        check(&cache, &store, &file, 0, 4 * BLOCK).await;
        let stats = cache.stats();
        assert_eq!(
            (stats.memory_bytes(), stats.disk_bytes()),
            held,
            "{placement:?}"
        );
        check(&cache, &store, &file, 0, 4 * BLOCK).await;
        let reads = if placement == Placement::Skip { 2 } else { 1 };
        assert_eq!(store.reads(), reads, "{placement:?}");
    }
}

#[tokio::test]
async fn many_cold_indexes_stay_within_both_budgets() {
    let store = Store::default();
    let mut files = Vec::new();
    for index in 0..30 {
        for segment in 0..2 {
            files.push(store.add(
                &format!("index-{index}"),
                &format!("i{index}s{segment}"),
                10 * BLOCK + 7,
            ));
        }
    }
    let (memory, disk) = (16 * BLOCK, 64 * BLOCK);
    let (cache, dir) = tiered(memory, disk).await;

    for file in &files {
        check(&cache, &store, file, 0, file.size()).await;
        let stats = cache.stats();
        assert!(stats.memory_bytes() <= memory, "{stats:?}");
        assert!(stats.disk_bytes() <= disk, "{stats:?}");
    }
    let stats = cache.stats();
    assert_eq!(stats.dropped(), 0);
    assert_eq!(stats.memory_evictions(), stats.disk_writes());
    assert!(stats.disk_evictions() > 0);
    assert!(bytes_on_disk(dir.path()) <= disk);
    assert_eq!(bytes_on_disk(dir.path()), stats.disk_bytes());

    for index in 0..30 {
        let index: IndexId = format!("index-{index}").parse().unwrap();
        cache.purge(&index).await.unwrap();
    }
    let stats = cache.stats();
    assert_eq!((stats.memory_bytes(), stats.disk_bytes()), (0, 0));
    assert_eq!(bytes_on_disk(dir.path()), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn budgets_hold_under_concurrent_reads() {
    let store = Store::default();
    let mut files = Vec::new();
    for index in 0..12 {
        for segment in 0..4 {
            files.push(store.add(
                &format!("index-{index}"),
                &format!("i{index}s{segment}"),
                20 * BLOCK + 3,
            ));
        }
    }
    let (memory, disk) = (24 * BLOCK, 96 * BLOCK);
    let (cache, dir) = tiered(memory, disk).await;
    let reading = Arc::new(AtomicU64::new(1));

    let watcher = {
        let (cache, reading) = (cache.clone(), Arc::clone(&reading));
        tokio::spawn(async move {
            let mut samples = 0u64;
            while reading.load(Ordering::SeqCst) == 1 {
                let stats = cache.stats();
                assert!(stats.memory_bytes() <= memory, "{stats:?}");
                assert!(stats.disk_bytes() <= disk, "{stats:?}");
                samples += 1;
                tokio::task::yield_now().await;
            }
            samples
        })
    };
    let reads: Vec<(usize, u64, u64)> = (0..1000u64)
        .map(|n| {
            let spread = if n % 3 == 0 { n * 7919 } else { n };
            let file = usize::try_from(spread % if n % 3 == 0 { 48 } else { 3 }).unwrap();
            let start = (n * 104_729) % (20 * BLOCK);
            let end = (start + 1 + (n * 31) % (6 * BLOCK)).min(20 * BLOCK + 3);
            (file, start, end)
        })
        .collect();
    stream::iter(reads)
        .map(|(file, start, end)| {
            let (cache, store, file) = (cache.clone(), store.clone(), files[file].clone());
            tokio::spawn(async move { check(&cache, &store, &file, start, end).await })
        })
        .buffer_unordered(32)
        .for_each(|joined| async move { joined.unwrap() })
        .await;
    reading.store(0, Ordering::SeqCst);
    assert!(watcher.await.unwrap() > 0);

    let stats = cache.stats();
    assert!(stats.memory_bytes() <= memory, "{stats:?}");
    assert!(stats.disk_bytes() <= disk, "{stats:?}");
    assert_eq!(bytes_on_disk(dir.path()), stats.disk_bytes());
    assert!(
        stats.disk_hits() > 0 && stats.disk_evictions() > 0,
        "{stats:?}"
    );
}

#[tokio::test]
async fn a_restart_on_the_same_directory_is_warm() {
    let store = Store::default();
    let file = store.add("index-1", "s1", 4 * BLOCK);
    let dir = TempDir::new().unwrap();
    let cache = tiered_in(dir.path(), 4 * BLOCK, 1 << 20, BLOCK).await;
    check(&cache, &store, &file, 0, 4 * BLOCK).await;
    assert_eq!(bytes_on_disk(dir.path()), 0, "memory has not evicted yet");
    cache.save().await;
    cache.save().await;
    assert_eq!(cache.stats().disk_writes(), 4, "a block is saved once");
    assert_eq!(cache.stats().memory_bytes(), 4 * BLOCK);
    drop(cache);

    let reads = store.reads();
    let restarted = tiered_in(dir.path(), 4 * BLOCK, 1 << 20, BLOCK).await;
    assert_eq!(restarted.stats().disk_bytes(), 4 * BLOCK);
    check(&restarted, &store, &file, 0, 4 * BLOCK).await;
    assert_eq!(
        store.reads(),
        reads,
        "the restart reads nothing from the store"
    );
    assert_eq!(restarted.stats().disk_hits(), 4);
}

#[tokio::test]
async fn a_folder_the_cache_did_not_write_is_refused_and_left_alone() {
    let dir = TempDir::new().unwrap();
    let folder = dir.path().join("cosmosearch-blocks");
    std::fs::create_dir(&folder).unwrap();
    let foreign = folder.join("notes.txt");
    put_file(&foreign, b"kept");
    let refused = Cache::builder(1 << 20)
        .disk(dir.path(), 1 << 20)
        .build()
        .await;
    assert!(matches!(refused, Err(cosmosearch_cache::Error::Disk(_))));
    let mut kept = Vec::new();
    File::open(&foreign)
        .unwrap()
        .read_to_end(&mut kept)
        .unwrap();
    assert_eq!(kept, b"kept");

    std::fs::remove_file(&foreign).unwrap();
    tiered_in(dir.path(), BLOCK, 1 << 20, BLOCK).await;
}

#[tokio::test]
async fn a_restart_keeps_only_sound_blocks_within_its_budget() {
    let store = Store::default();
    let file = store.add("index-1", "s1", 4 * BLOCK);
    let dir = TempDir::new().unwrap();
    let cache = tiered_in(dir.path(), 4 * BLOCK, 1 << 20, BLOCK).await;
    check(&cache, &store, &file, 0, 4 * BLOCK).await;
    cache.save().await;
    drop(cache);

    let files = block_files(dir.path());
    assert_eq!(files.len(), 4);
    let flipped = &files[0];
    let mut body = Vec::new();
    File::open(flipped).unwrap().read_to_end(&mut body).unwrap();
    body[0] ^= 0xff;
    put_file(flipped, &body);
    let folder = flipped.parent().unwrap();
    put_file(&folder.join("lex-s1.bin.256.9.1"), b"a block past the file");
    put_file(&folder.join("stray.tmp"), b"half written");

    let reads = store.reads();
    let restarted = tiered_in(dir.path(), 4 * BLOCK, 1 << 20, BLOCK).await;
    assert_eq!(
        block_files(dir.path()).len(),
        4,
        "files that are no block go"
    );
    check(&restarted, &store, &file, 0, 4 * BLOCK).await;
    assert_eq!(store.reads(), reads + 1, "the changed block is read again");
    assert_eq!(restarted.stats().disk_hits(), 3);
    restarted.save().await;
    drop(restarted);

    let smaller = tiered_in(dir.path(), 4 * BLOCK, 2 * BLOCK, BLOCK).await;
    assert_eq!(smaller.stats().disk_bytes(), 2 * BLOCK);
    assert_eq!(bytes_on_disk(dir.path()), 2 * BLOCK);
    drop(smaller);

    let recut = tiered_in(dir.path(), 4 * BLOCK, 1 << 20, 2 * BLOCK).await;
    assert_eq!(
        recut.stats().disk_bytes(),
        0,
        "another block size starts empty"
    );
    assert_eq!(block_files(dir.path()), Vec::<PathBuf>::new());
}
