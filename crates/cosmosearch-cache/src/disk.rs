use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{self, ErrorKind, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use bytes::Bytes;
use cosmosearch_core::{CachedFile, IndexId, SegmentFile};
use tracing::warn;

use crate::block::{BlockKey, Blocks, len};
use crate::lock::locked;
use crate::lru::Lru;
use crate::stats::Counters;

/// [`HASH_BYTES`] as a file length.
const HASH_LEN: u64 = 32;
const _: () = assert!(HASH_BYTES == 32, "a block file ends with a 32-byte hash");

/// The folder inside the host's directory that the disk tier owns.
const FOLDER: &str = "cosmosearch-blocks";
/// The file in that folder holding the block size its files were cut with.
const BLOCK_SIZE_FILE: &str = "block-size";
const TEMP_SUFFIX: &str = ".tmp";
/// Each block file ends with the BLAKE3 hash of its block, checked on every read, so a file
/// an earlier process left half written or a disk damaged is a miss and never wrong bytes.
const HASH_BYTES: usize = blake3::OUT_LEN;

#[derive(Debug)]
enum State {
    Writing(Bytes),
    Ready,
    Purged,
}

#[derive(Debug)]
struct Entry {
    generation: u64,
    len: u64,
    state: State,
}

#[derive(Debug, Default)]
struct Index {
    lru: Lru<BlockKey, Entry>,
    generation: u64,
    purging: u64,
}

impl Index {
    fn used(&self) -> u64 {
        self.lru.used().saturating_add(self.purging)
    }
}

/// A block a lookup found, or must read from its file.
enum Found {
    Held(Bytes),
    File {
        path: PathBuf,
        len: u64,
        generation: u64,
    },
    Missing,
}

struct Pending {
    key: BlockKey,
    bytes: Bytes,
    generation: u64,
}

/// The disk tier: one file per block under a folder per index and segment, least recently
/// used first under a byte budget that counts blocks still being written.
#[derive(Debug)]
pub(crate) struct Disk {
    root: PathBuf,
    budget: u64,
    index: Mutex<Index>,
    counters: Arc<Counters>,
}

impl Disk {
    /// A disk tier in `dir`, keeping the blocks an earlier process left there if they were
    /// cut with the same block size, newest first within `budget`, and emptying it otherwise.
    /// A folder that holds files but no block size was not written by the cache and is
    /// refused.
    pub(crate) async fn open(
        dir: PathBuf,
        budget: u64,
        blocks: Blocks,
        counters: Arc<Counters>,
    ) -> io::Result<Self> {
        let root = dir.join(FOLDER);
        let scanned = root.clone();
        let index = tokio::task::spawn_blocking(move || restore(&scanned, budget, blocks))
            .await
            .map_err(io::Error::other)??;
        Ok(Disk {
            root,
            budget,
            index: Mutex::new(index),
            counters,
        })
    }

    /// Bytes held or being written.
    pub(crate) fn used(&self) -> u64 {
        locked(&self.index).used()
    }

    /// The blocks under `keys` that the tier holds; a file that is missing or of the wrong
    /// size is a miss.
    pub(crate) async fn get(self: &Arc<Self>, keys: &[BlockKey]) -> Vec<Option<Bytes>> {
        let found = self.find(keys);
        let reads: Vec<(usize, PathBuf, u64)> = found
            .iter()
            .enumerate()
            .filter_map(|(position, found)| match found {
                Found::File { path, len, .. } => Some((position, path.clone(), *len)),
                Found::Held(_) | Found::Missing => None,
            })
            .collect();
        let mut read: Vec<Option<Bytes>> = vec![None; keys.len()];
        let mut finished = true;
        if !reads.is_empty() {
            let results = tokio::task::spawn_blocking(move || {
                let results: Vec<(usize, Option<Bytes>)> = reads
                    .into_iter()
                    .map(|(position, path, len)| (position, read_block(&path, len)))
                    .collect();
                results
            })
            .await;
            match results {
                Ok(results) => {
                    for (position, bytes) in results {
                        if let Some(slot) = read.get_mut(position) {
                            *slot = bytes;
                        }
                    }
                }
                Err(err) => {
                    warn!(error = %err, "reading cached blocks from disk did not finish");
                    finished = false;
                }
            }
        }
        let mut gone = Vec::new();
        let mut blocks = Vec::with_capacity(keys.len());
        for ((key, found), read) in keys.iter().zip(found).zip(read) {
            blocks.push(match found {
                Found::Held(bytes) => Some(bytes),
                Found::Missing => None,
                Found::File { generation, .. } => {
                    if read.is_none() && finished {
                        gone.push((key.clone(), generation));
                    }
                    read
                }
            });
        }
        if !gone.is_empty() {
            self.forget(&gone);
        }
        blocks
    }

    /// Writes `blocks`, evicting the least recently used to make room. A block the tier
    /// cannot make room for is dropped.
    pub(crate) async fn put(self: &Arc<Self>, blocks: Vec<(BlockKey, Bytes)>) {
        let (victims, pending) = self.reserve(blocks);
        if victims.is_empty() && pending.is_empty() {
            return;
        }
        let disk = Arc::clone(self);
        let written = tokio::task::spawn_blocking(move || disk.write(&victims, pending)).await;
        if let Err(err) = written {
            warn!(error = %err, "writing cached blocks to disk did not finish");
        }
    }

    /// Removes every block of index `index`.
    pub(crate) async fn purge(self: &Arc<Self>, index: &IndexId) -> io::Result<()> {
        let (paths, freed) = self.unlink(index);
        let disk = Arc::clone(self);
        let folder = self.root.join(index.as_str());
        tokio::task::spawn_blocking(move || {
            let removed = remove_files(&paths, &folder);
            let mut state = locked(&disk.index);
            state.purging = state.purging.saturating_sub(freed);
            removed
        })
        .await
        .map_err(io::Error::other)?
    }

    fn find(&self, keys: &[BlockKey]) -> Vec<Found> {
        let mut state = locked(&self.index);
        keys.iter()
            .map(|key| match state.lru.get(key) {
                Some(Entry {
                    state: State::Writing(bytes),
                    ..
                }) => Found::Held(bytes.clone()),
                Some(Entry {
                    state: State::Ready,
                    generation,
                    len,
                }) => Found::File {
                    path: self.path(key, *generation),
                    len: *len,
                    generation: *generation,
                },
                Some(Entry {
                    state: State::Purged,
                    ..
                })
                | None => Found::Missing,
            })
            .collect()
    }

    fn forget(&self, gone: &[(BlockKey, u64)]) {
        let mut state = locked(&self.index);
        for (key, generation) in gone {
            let current = state.lru.peek(key).is_some_and(|entry| {
                entry.generation == *generation && matches!(entry.state, State::Ready)
            });
            if current {
                state.lru.remove(key);
            }
        }
    }

    fn reserve(&self, blocks: Vec<(BlockKey, Bytes)>) -> (Vec<PathBuf>, Vec<Pending>) {
        let mut state = locked(&self.index);
        let mut victims = Vec::new();
        let mut pending = Vec::new();
        for (key, bytes) in blocks {
            if state.lru.get(&key).is_some() {
                continue;
            }
            let weight = len(&bytes);
            while state.used().saturating_add(weight) > self.budget {
                let Some((old, entry, _)) = state
                    .lru
                    .pop_oldest(|entry| matches!(entry.state, State::Ready))
                else {
                    break;
                };
                self.counters.disk_evictions.add(1);
                victims.push(self.path(&old, entry.generation));
            }
            if state.used().saturating_add(weight) > self.budget {
                self.counters.dropped.add(1);
                continue;
            }
            state.generation = state.generation.saturating_add(1);
            let generation = state.generation;
            let entry = Entry {
                generation,
                len: weight,
                state: State::Writing(bytes.clone()),
            };
            state.lru.insert(key.clone(), entry, weight);
            pending.push(Pending {
                key,
                bytes,
                generation,
            });
        }
        (victims, pending)
    }

    fn write(&self, victims: &[PathBuf], pending: Vec<Pending>) {
        for path in victims {
            remove_file(path);
        }
        for block in pending {
            let path = self.path(&block.key, block.generation);
            let mut temp = path.clone().into_os_string();
            temp.push(TEMP_SUFFIX);
            let temp = PathBuf::from(temp);
            let written = write_block(&path, &temp, &block.bytes);
            if let Err(err) = &written {
                warn!(error = %err, "a cached block could not be written to disk");
                remove_file(&temp);
            }
            self.settle(&block, &path, written.is_ok());
        }
    }

    fn settle(&self, block: &Pending, path: &Path, written: bool) {
        let mut state = locked(&self.index);
        let entry = state
            .lru
            .peek_mut(&block.key)
            .filter(|entry| entry.generation == block.generation);
        match entry {
            Some(entry) if written && matches!(entry.state, State::Writing(_)) => {
                entry.state = State::Ready;
                self.counters.disk_writes.add(1);
            }
            Some(_) => {
                state.lru.remove(&block.key);
                drop(state);
                if written {
                    remove_file(path);
                }
            }
            None => {
                drop(state);
                if written {
                    remove_file(path);
                }
            }
        }
    }

    fn unlink(&self, index: &IndexId) -> (Vec<PathBuf>, u64) {
        let mut state = locked(&self.index);
        let mut paths = Vec::new();
        let mut freed = 0u64;
        for key in state.lru.keys_of(index) {
            let Some(entry) = state.lru.peek_mut(&key) else {
                continue;
            };
            match entry.state {
                State::Writing(_) => entry.state = State::Purged,
                State::Purged => {}
                State::Ready => {
                    let (len, generation) = (entry.len, entry.generation);
                    state.lru.remove(&key);
                    freed = freed.saturating_add(len);
                    paths.push(self.path(&key, generation));
                }
            }
        }
        state.purging = state.purging.saturating_add(freed);
        (paths, freed)
    }

    /// Where the block under `key` is written by write `generation`. Each write has a file of
    /// its own, so removing an evicted block can never remove a later write of it.
    fn path(&self, key: &BlockKey, generation: u64) -> PathBuf {
        block_path(&self.root, key, generation)
    }
}

fn block_path(root: &Path, key: &BlockKey, generation: u64) -> PathBuf {
    let file = &key.file;
    root.join(file.index().as_str())
        .join(file.segment().as_str())
        .join(format!(
            "{}.{}.{}.{generation}",
            file.name(),
            file.size(),
            key.block
        ))
}

/// The block and write a file at `path` under `root` holds, read from its name.
fn parse_path(root: &Path, path: &Path) -> Option<(BlockKey, u64)> {
    let mut parts = path.strip_prefix(root).ok()?.iter();
    let index = parts.next()?.to_str()?.parse().ok()?;
    let segment = parts.next()?.to_str()?.parse().ok()?;
    let file_name = parts.next()?.to_str()?;
    if parts.next().is_some() {
        return None;
    }
    let mut fields = file_name.rsplitn(4, '.');
    let generation = canonical(fields.next()?)?;
    let block = canonical(fields.next()?)?;
    let size = canonical(fields.next()?)?;
    let name = fields.next()?.parse().ok()?;
    let file = CachedFile::new(index, segment, &SegmentFile::new(name, size));
    let key = BlockKey {
        file: Arc::new(file),
        block,
    };
    Some((key, generation))
}

fn canonical(digits: &str) -> Option<u64> {
    let number: u64 = digits.parse().ok()?;
    (number.to_string() == digits).then_some(number)
}

/// Prepares the folder `root` and gives the blocks it already holds.
fn restore(root: &Path, budget: u64, blocks: Blocks) -> io::Result<Index> {
    let marker = root.join(BLOCK_SIZE_FILE);
    let wanted = blocks.size().to_string();
    let found = read_text(&marker);
    if found.as_deref() == Some(wanted.as_str()) {
        return scan(root, budget, blocks);
    }
    if found.is_none() && holds_anything(root)? {
        return Err(io::Error::new(
            ErrorKind::AlreadyExists,
            "the cache folder holds files the cache did not write",
        ));
    }
    match fs::remove_dir_all(root) {
        Ok(()) => {}
        Err(err) if err.kind() == ErrorKind::NotFound => {}
        Err(err) => return Err(err),
    }
    fs::create_dir_all(root)?;
    File::create(&marker)?.write_all(wanted.as_bytes())?;
    Ok(Index::default())
}

fn holds_anything(folder: &Path) -> io::Result<bool> {
    match fs::read_dir(folder) {
        Ok(mut entries) => Ok(entries.next().is_some()),
        Err(err) if err.kind() == ErrorKind::NotFound => Ok(false),
        Err(err) => Err(err),
    }
}

fn read_text(path: &Path) -> Option<String> {
    let mut text = String::new();
    File::open(path).ok()?.read_to_string(&mut text).ok()?;
    Some(text)
}

/// Takes the files under `root` into a new index: for each block the newest write, newest
/// writes first until `budget` is full. Every other file is removed.
fn scan(root: &Path, budget: u64, blocks: Blocks) -> io::Result<Index> {
    let mut newest: BTreeMap<BlockKey, (u64, u64)> = BTreeMap::new();
    for path in block_files(root)? {
        let found = parse_path(root, &path).and_then(|(key, generation)| {
            let range = blocks.block(key.block, key.file.size());
            let len = range.end.checked_sub(range.start).filter(|len| *len > 0)?;
            let on_disk = fs::metadata(&path).ok()?.len();
            (on_disk == len.checked_add(HASH_LEN)?).then_some((key, generation, len))
        });
        let Some((key, generation, len)) = found else {
            remove_file(&path);
            continue;
        };
        match newest.get(&key) {
            Some((held, _)) if *held >= generation => remove_file(&path),
            Some((held, _)) => {
                remove_file(&block_path(root, &key, *held));
                newest.insert(key, (generation, len));
            }
            None => {
                newest.insert(key, (generation, len));
            }
        }
    }
    let mut kept: Vec<(u64, BlockKey, u64)> = newest
        .into_iter()
        .map(|(key, (generation, len))| (generation, key, len))
        .collect();
    kept.sort_unstable_by_key(|(generation, _, _)| Reverse(*generation));
    let mut index = Index {
        generation: kept.first().map_or(0, |(generation, _, _)| *generation),
        ..Index::default()
    };
    let mut fitting = Vec::with_capacity(kept.len());
    let mut used = 0u64;
    for (generation, key, len) in kept {
        match used.checked_add(len) {
            Some(total) if total <= budget => {
                used = total;
                fitting.push((generation, key, len));
            }
            Some(_) | None => remove_file(&block_path(root, &key, generation)),
        }
    }
    for (generation, key, len) in fitting.into_iter().rev() {
        let entry = Entry {
            generation,
            len,
            state: State::Ready,
        };
        index.lru.insert(key, entry, len);
    }
    Ok(index)
}

/// Every file two folders below `root`, where block files are; anything else in the way is
/// left alone.
fn block_files(root: &Path) -> io::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for index in fs::read_dir(root)? {
        let index = index?.path();
        if !index.is_dir() {
            continue;
        }
        for segment in fs::read_dir(&index)? {
            let segment = segment?.path();
            if !segment.is_dir() {
                continue;
            }
            for file in fs::read_dir(&segment)? {
                files.push(file?.path());
            }
        }
    }
    Ok(files)
}

/// The block in the file at `path`, or `None` if it is missing, not `len` bytes long before
/// its hash, or does not match its hash, in which case the file is removed.
fn read_block(path: &Path, len: u64) -> Option<Bytes> {
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(err) if err.kind() == ErrorKind::NotFound => return None,
        Err(err) => {
            warn!(error = %err, "a cached block could not be read from disk");
            return None;
        }
    };
    let stored = len.saturating_add(HASH_LEN);
    let mut bytes = Vec::with_capacity(usize::try_from(stored).unwrap_or(0));
    let read = Read::by_ref(&mut file)
        .take(stored.saturating_add(1))
        .read_to_end(&mut bytes);
    if let Err(err) = read {
        warn!(error = %err, "a cached block could not be read from disk");
        return None;
    }
    let intact = bytes
        .len()
        .checked_sub(HASH_BYTES)
        .and_then(|at| bytes.split_at_checked(at))
        .is_some_and(|(block, hash)| {
            len_of(block) == len && blake3::hash(block).as_bytes().as_slice() == hash
        });
    if !intact {
        remove_file(path);
        return None;
    }
    bytes.truncate(bytes.len().saturating_sub(HASH_BYTES));
    Some(Bytes::from(bytes))
}

fn write_block(path: &Path, temp: &Path, bytes: &[u8]) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = File::create(temp)?;
    file.write_all(bytes)?;
    file.write_all(blake3::hash(bytes).as_bytes())?;
    drop(file);
    fs::rename(temp, path)
}

fn remove_file(path: &Path) {
    match fs::remove_file(path) {
        Ok(()) => {}
        Err(err) if err.kind() == ErrorKind::NotFound => {}
        Err(err) => warn!(error = %err, "a cached block could not be removed from disk"),
    }
}

/// Removes `paths`, then the folders under `folder` they leave empty, and `folder` itself.
fn remove_files(paths: &[PathBuf], folder: &Path) -> io::Result<()> {
    let mut failed = None;
    let mut parents = BTreeSet::new();
    for path in paths {
        match fs::remove_file(path) {
            Ok(()) => {}
            Err(err) if err.kind() == ErrorKind::NotFound => {}
            Err(err) => {
                failed.get_or_insert(err);
            }
        }
        if let Some(parent) = path.parent() {
            parents.insert(parent.to_path_buf());
        }
    }
    for parent in parents.iter().map(PathBuf::as_path).chain([folder]) {
        if let Err(err) = fs::remove_dir(parent)
            && !matches!(
                err.kind(),
                ErrorKind::NotFound | ErrorKind::DirectoryNotEmpty
            )
        {
            failed.get_or_insert(err);
        }
    }
    failed.map_or(Ok(()), Err)
}

fn len_of(bytes: &[u8]) -> u64 {
    u64::try_from(bytes.len()).unwrap_or(u64::MAX)
}
