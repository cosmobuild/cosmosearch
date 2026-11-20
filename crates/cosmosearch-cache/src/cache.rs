use std::collections::HashMap;
use std::num::NonZeroU64;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use bytes::Bytes;
use cosmosearch_core::{
    Admission, ByteRange, CachedFile, IndexCause, IndexError, IndexId, Placement, SegmentCache,
    SegmentSource,
};
use futures_util::{StreamExt, stream};
use tokio::sync::watch;

use crate::Error;
use crate::admit::AdmitAll;
use crate::block::{BlockKey, Blocks};
use crate::disk::Disk;
use crate::lock::locked;
use crate::memory::Memory;
use crate::stats::{CacheStats, Counters};

/// The block size unless set. A 64 KiB block costs barely more to read or write than a
/// 4 KiB one, so few, large blocks are cached.
const DEFAULT_BLOCK_SIZE: u64 = 64 * 1024;
/// The largest block size allowed, so a single block stays a modest allocation.
const MAX_BLOCK_SIZE: u64 = 64 * 1024 * 1024;
/// Store reads one call makes at once when the blocks it misses are not adjacent.
const RUNS_IN_FLIGHT: usize = 4;

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Cache>();
};

type Flights = Mutex<HashMap<BlockKey, watch::Receiver<Option<Bytes>>>>;

/// A RAM and disk cache of segment files, cut into aligned blocks, that any number of engines
/// share. Cloning it is cheap and every clone is the same cache.
///
/// Blocks are kept by index id, segment id, file name and block number. Segment files never
/// change, so a block never goes stale and survives a manifest swap. Memory evicts the least
/// recently used blocks to disk, and disk evicts its least recently used blocks for good.
#[derive(Debug, Clone)]
pub struct Cache {
    inner: Arc<Inner>,
}

#[derive(Debug)]
struct Inner {
    blocks: Blocks,
    memory: Memory,
    disk: Option<Arc<Disk>>,
    admission: Box<dyn Admission>,
    flights: Arc<Flights>,
    counters: Arc<Counters>,
}

impl Cache {
    /// Starts building a cache holding at most `memory_budget` bytes in memory.
    pub fn builder(memory_budget: u64) -> CacheBuilder {
        CacheBuilder {
            memory_budget,
            disk: None,
            block_size: DEFAULT_BLOCK_SIZE,
            admission: Box::new(AdmitAll::new()),
        }
    }

    /// What the cache has done and holds.
    pub fn stats(&self) -> CacheStats {
        let disk = self.inner.disk.as_ref().map_or(0, |disk| disk.used());
        self.inner.counters.stats(self.inner.memory.used(), disk)
    }

    /// Writes the blocks held in memory to the disk tier, where it does not hold them yet,
    /// and leaves them in memory. A process that later starts on the same directory then
    /// finds them on disk, so a host calls this now and then and before it stops. Without a
    /// disk tier it does nothing.
    pub async fn save(&self) {
        if let Some(disk) = &self.inner.disk {
            disk.put(self.inner.memory.blocks()).await;
        }
    }

    /// Removes every block of index `index` from memory and disk. Blocks a search of that
    /// index reads afterwards are cached again, so a host purges an index it has closed.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Disk`] if a file of the index cannot be removed from disk; its
    /// blocks are gone from the cache all the same.
    pub async fn purge(&self, index: &IndexId) -> Result<(), Error> {
        self.inner.memory.purge(index);
        match &self.inner.disk {
            Some(disk) => disk.purge(index).await.map_err(Error::Disk),
            None => Ok(()),
        }
    }
}

/// Sets a cache's tiers and how it admits blocks, and builds it.
#[derive(Debug)]
pub struct CacheBuilder {
    memory_budget: u64,
    disk: Option<(PathBuf, u64)>,
    block_size: u64,
    admission: Box<dyn Admission>,
}

impl CacheBuilder {
    /// Adds a disk tier holding at most `budget` bytes of blocks, each in a file 32 bytes
    /// longer for its hash, in the folder `cosmosearch-blocks` in `dir`. The cache owns that
    /// folder, so each running process needs a directory of its own, and refuses one that
    /// holds files it did not write. A process that starts on the directory an earlier one
    /// left finds its blocks there, so its first searches read from disk and not from the
    /// store. The blocks hold index data and are created with the process's default
    /// permissions, so `dir` belongs in a folder only the host can read.
    #[must_use]
    pub fn disk(mut self, dir: impl Into<PathBuf>, budget: u64) -> Self {
        self.disk = Some((dir.into(), budget));
        self
    }

    /// Sets the size of the aligned blocks files are cached in, 64 KiB unless set.
    #[must_use]
    pub fn block_size(mut self, size: u64) -> Self {
        self.block_size = size;
        self
    }

    /// Sets what decides whether a block is kept and where; [`AdmitAll`] unless set.
    #[must_use]
    pub fn admission(mut self, admission: impl Admission + 'static) -> Self {
        self.admission = Box::new(admission);
        self
    }

    /// Builds the cache. The disk tier keeps the blocks its folder already holds if they were
    /// cut with the same block size, the newest within its budget, and is emptied otherwise.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidBlockSize`] for a block size of zero or above 64 MiB, and
    /// [`Error::Disk`] if the disk tier's folder cannot be read, emptied or created, or holds
    /// files the cache did not write.
    pub async fn build(self) -> Result<Cache, Error> {
        let size = NonZeroU64::new(self.block_size)
            .filter(|size| size.get() <= MAX_BLOCK_SIZE)
            .ok_or(Error::InvalidBlockSize {
                size: self.block_size,
            })?;
        let counters = Arc::new(Counters::default());
        let disk = match self.disk {
            Some((dir, budget)) => Some(Arc::new(
                Disk::open(dir, budget, Blocks::new(size), Arc::clone(&counters))
                    .await
                    .map_err(Error::Disk)?,
            )),
            None => None,
        };
        Ok(Cache {
            inner: Arc::new(Inner {
                blocks: Blocks::new(size),
                memory: Memory::new(self.memory_budget, size.get()),
                disk,
                admission: self.admission,
                flights: Arc::new(Mutex::new(HashMap::new())),
                counters,
            }),
        })
    }
}

#[async_trait]
impl SegmentCache for Cache {
    async fn read_range(
        &self,
        file: &CachedFile,
        range: ByteRange,
        store: &SegmentSource,
    ) -> Result<Vec<u8>, IndexError> {
        if range.end() > file.size() {
            return store.read_range(file.segment(), file.name(), range).await;
        }
        let file = Arc::new(file.clone());
        let keys: Vec<BlockKey> = self
            .inner
            .blocks
            .covering(range.start(), range.end())
            .map(|block| BlockKey {
                file: Arc::clone(&file),
                block,
            })
            .collect();
        let mut spill = Vec::new();
        let found = self.inner.fill(&keys, store, &mut spill).await;
        self.inner.spill(spill).await;
        self.inner.assemble(&keys, found?, range)
    }
}

/// A block this call fetches for every caller that wants it; dropping it lets them retry.
struct Claim {
    key: BlockKey,
    sender: watch::Sender<Option<Bytes>>,
    flights: Arc<Flights>,
}

impl Drop for Claim {
    fn drop(&mut self) {
        locked(&self.flights).remove(&self.key);
    }
}

impl Inner {
    /// Finds every block under `keys`: from memory, then disk, then the store, with one store
    /// read for each run of adjacent blocks no other call is already fetching.
    async fn fill(
        &self,
        keys: &[BlockKey],
        store: &SegmentSource,
        spill: &mut Vec<(BlockKey, Bytes)>,
    ) -> Result<Vec<Option<Bytes>>, IndexError> {
        let mut found: Vec<Option<Bytes>> = keys.iter().map(|key| self.memory.get(key)).collect();
        self.counters
            .memory_hits
            .add_len(found.iter().filter(|block| block.is_some()).count());
        if let Some(disk) = &self.disk {
            self.read_disk(disk, keys, &mut found, spill).await;
        }
        while found.iter().any(Option::is_none) {
            let (claims, waits) = self.claim(keys, &mut found);
            self.fetch(claims, store, &mut found, spill).await?;
            for (position, mut receiver) in waits {
                let sent = receiver
                    .wait_for(Option::is_some)
                    .await
                    .ok()
                    .and_then(|sent| sent.clone());
                if let (Some(bytes), Some(slot)) = (sent, found.get_mut(position)) {
                    *slot = Some(bytes);
                }
            }
        }
        Ok(found)
    }

    async fn read_disk(
        &self,
        disk: &Arc<Disk>,
        keys: &[BlockKey],
        found: &mut [Option<Bytes>],
        spill: &mut Vec<(BlockKey, Bytes)>,
    ) {
        let missing: Vec<BlockKey> = keys
            .iter()
            .zip(found.iter())
            .filter(|(_, block)| block.is_none())
            .map(|(key, _)| key.clone())
            .collect();
        if missing.is_empty() {
            return;
        }
        let mut read = disk.get(&missing).await.into_iter();
        for (key, slot) in keys.iter().zip(found.iter_mut()) {
            if slot.is_some() {
                continue;
            }
            let Some(Some(bytes)) = read.next() else {
                continue;
            };
            self.counters.disk_hits.add(1);
            if self.admission.place(&key.file, key.block) == Placement::Memory {
                self.keep_in_memory(key.clone(), bytes.clone(), spill);
            }
            *slot = Some(bytes);
        }
    }

    /// Takes over fetching each missing block no other call is fetching, and returns those
    /// claims and the blocks to wait for.
    #[expect(
        clippy::type_complexity,
        reason = "a private pair of positions with claims and positions with receivers"
    )]
    fn claim(
        &self,
        keys: &[BlockKey],
        found: &mut [Option<Bytes>],
    ) -> (
        Vec<(usize, Claim)>,
        Vec<(usize, watch::Receiver<Option<Bytes>>)>,
    ) {
        let mut claims = Vec::new();
        let mut waits = Vec::new();
        let mut flights = locked(&self.flights);
        for (position, (key, slot)) in keys.iter().zip(found.iter_mut()).enumerate() {
            if slot.is_some() {
                continue;
            }
            if let Some(bytes) = self.memory.get(key) {
                self.counters.memory_hits.add(1);
                *slot = Some(bytes);
            } else if let Some(receiver) = flights.get(key) {
                waits.push((position, receiver.clone()));
            } else {
                let (sender, receiver) = watch::channel(None);
                flights.insert(key.clone(), receiver);
                let claim = Claim {
                    key: key.clone(),
                    sender,
                    flights: Arc::clone(&self.flights),
                };
                claims.push((position, claim));
            }
        }
        (claims, waits)
    }

    async fn fetch(
        &self,
        claims: Vec<(usize, Claim)>,
        store: &SegmentSource,
        found: &mut [Option<Bytes>],
        spill: &mut Vec<(BlockKey, Bytes)>,
    ) -> Result<(), IndexError> {
        let mut runs: Vec<Vec<(usize, Claim)>> = Vec::new();
        for (position, claim) in claims {
            let adjacent = runs
                .last()
                .and_then(|run| run.last())
                .is_some_and(|(last, _)| last.checked_add(1) == Some(position));
            match runs.last_mut() {
                Some(run) if adjacent => run.push((position, claim)),
                Some(_) | None => runs.push(vec![(position, claim)]),
            }
        }
        let mut reads = stream::iter(runs)
            .map(|run| async move {
                let read = self.read_run(&run, store).await;
                (run, read)
            })
            .buffer_unordered(RUNS_IN_FLIGHT);
        while let Some((run, read)) = reads.next().await {
            let (start, bytes) = read?;
            for (position, claim) in run {
                let block = self.slice(&claim.key, start, &bytes)?;
                self.place(claim.key.clone(), block.clone(), spill);
                claim.sender.send_replace(Some(block.clone()));
                if let Some(slot) = found.get_mut(position) {
                    *slot = Some(block);
                }
            }
        }
        Ok(())
    }

    /// Reads a run of adjacent blocks in one store read, returning where it starts.
    async fn read_run(
        &self,
        run: &[(usize, Claim)],
        store: &SegmentSource,
    ) -> Result<(u64, Bytes), IndexError> {
        let (Some((_, first)), Some((_, last))) = (run.first(), run.last()) else {
            return Err(IndexError::Malformed {
                reason: "an empty run of blocks",
            });
        };
        let file = &first.key.file;
        let bytes = self.blocks.bytes(
            first.key.block..last.key.block.saturating_add(1),
            file.size(),
        );
        let range = ByteRange::new(bytes.start, bytes.end)
            .map_err(|err| IndexError::Read(IndexCause::new(err)))?;
        let read = store.read_range(file.segment(), file.name(), range).await?;
        self.counters.store_reads.add(1);
        self.counters.store_bytes.add_len(read.len());
        if len_of(&read) != range.len() {
            return Err(IndexError::Malformed {
                reason: "a store read returned the wrong number of bytes",
            });
        }
        Ok((bytes.start, Bytes::from(read)))
    }

    /// The bytes of the block under `key` within a run read from `start`, in a buffer of
    /// their own so that evicting a block frees its memory.
    fn slice(&self, key: &BlockKey, start: u64, run: &Bytes) -> Result<Bytes, IndexError> {
        let block = self.blocks.block(key.block, key.file.size());
        let from = block
            .start
            .checked_sub(start)
            .and_then(|at| usize::try_from(at).ok());
        let to = block
            .end
            .checked_sub(start)
            .and_then(|at| usize::try_from(at).ok());
        from.zip(to)
            .and_then(|(from, to)| run.get(from..to))
            .map(Bytes::copy_from_slice)
            .ok_or(IndexError::Malformed {
                reason: "a block lies outside the run read for it",
            })
    }

    fn place(&self, key: BlockKey, bytes: Bytes, spill: &mut Vec<(BlockKey, Bytes)>) {
        let placement = self.admission.place(&key.file, key.block);
        if placement == Placement::Memory {
            self.keep_in_memory(key, bytes, spill);
        } else if placement == Placement::Disk {
            spill.push((key, bytes));
        }
    }

    fn keep_in_memory(&self, key: BlockKey, bytes: Bytes, spill: &mut Vec<(BlockKey, Bytes)>) {
        let evicted = self.memory.insert(key, bytes);
        self.counters.memory_evictions.add_len(evicted.len());
        spill.extend(evicted);
    }

    /// Moves blocks memory evicted, or admitted to disk only, to disk, or drops them.
    async fn spill(&self, spill: Vec<(BlockKey, Bytes)>) {
        if spill.is_empty() {
            return;
        }
        match &self.disk {
            Some(disk) => disk.put(spill).await,
            None => self.counters.dropped.add_len(spill.len()),
        }
    }

    /// Copies `range` out of the blocks under `keys`.
    fn assemble(
        &self,
        keys: &[BlockKey],
        found: Vec<Option<Bytes>>,
        range: ByteRange,
    ) -> Result<Vec<u8>, IndexError> {
        let capacity = usize::try_from(range.len()).unwrap_or(0);
        let mut out = Vec::with_capacity(capacity);
        for (key, block) in keys.iter().zip(found) {
            let bytes = self.blocks.block(key.block, key.file.size());
            let from = range.start().max(bytes.start).saturating_sub(bytes.start);
            let to = range.end().min(bytes.end).saturating_sub(bytes.start);
            let part = block.as_ref().and_then(|block| {
                let from = usize::try_from(from).ok()?;
                let to = usize::try_from(to).ok()?;
                block.get(from..to)
            });
            let Some(part) = part else {
                return Err(IndexError::Malformed {
                    reason: "a cached block is shorter than its place in the file",
                });
            };
            out.extend_from_slice(part);
        }
        Ok(out)
    }
}

fn len_of(bytes: &[u8]) -> u64 {
    u64::try_from(bytes.len()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn refuses_a_block_size_of_zero_or_too_large() {
        for size in [0, MAX_BLOCK_SIZE + 1] {
            let err = Cache::builder(1 << 20)
                .block_size(size)
                .build()
                .await
                .unwrap_err();
            assert!(matches!(err, Error::InvalidBlockSize { .. }), "{err}");
        }
    }
}
