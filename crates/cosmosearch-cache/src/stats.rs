use std::sync::atomic::{AtomicU64, Ordering};

/// What a cache has done since it was built, and what it holds now.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheStats {
    memory_hits: u64,
    disk_hits: u64,
    store_reads: u64,
    store_bytes: u64,
    memory_evictions: u64,
    disk_writes: u64,
    disk_evictions: u64,
    dropped: u64,
    memory_bytes: u64,
    disk_bytes: u64,
}

impl CacheStats {
    /// Blocks served from memory.
    pub fn memory_hits(&self) -> u64 {
        self.memory_hits
    }

    /// Blocks served from disk.
    pub fn disk_hits(&self) -> u64 {
        self.disk_hits
    }

    /// Reads the cache made from the store, each one or more adjacent blocks.
    pub fn store_reads(&self) -> u64 {
        self.store_reads
    }

    /// Bytes those reads returned.
    pub fn store_bytes(&self) -> u64 {
        self.store_bytes
    }

    /// Blocks evicted from memory, each then written to disk or dropped.
    pub fn memory_evictions(&self) -> u64 {
        self.memory_evictions
    }

    /// Blocks written to disk.
    pub fn disk_writes(&self) -> u64 {
        self.disk_writes
    }

    /// Blocks evicted from disk, and so dropped.
    pub fn disk_evictions(&self) -> u64 {
        self.disk_evictions
    }

    /// Blocks bound for disk that it could not take: there is no disk tier, the block is
    /// larger than its budget, or every block on it is still being written.
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    /// Bytes held in memory now.
    pub fn memory_bytes(&self) -> u64 {
        self.memory_bytes
    }

    /// Bytes held or being written on disk now.
    pub fn disk_bytes(&self) -> u64 {
        self.disk_bytes
    }
}

/// The running counts behind [`CacheStats`].
#[derive(Debug, Default)]
pub(crate) struct Counters {
    pub(crate) memory_hits: Count,
    pub(crate) disk_hits: Count,
    pub(crate) store_reads: Count,
    pub(crate) store_bytes: Count,
    pub(crate) memory_evictions: Count,
    pub(crate) disk_writes: Count,
    pub(crate) disk_evictions: Count,
    pub(crate) dropped: Count,
}

impl Counters {
    pub(crate) fn stats(&self, memory_bytes: u64, disk_bytes: u64) -> CacheStats {
        CacheStats {
            memory_hits: self.memory_hits.get(),
            disk_hits: self.disk_hits.get(),
            store_reads: self.store_reads.get(),
            store_bytes: self.store_bytes.get(),
            memory_evictions: self.memory_evictions.get(),
            disk_writes: self.disk_writes.get(),
            disk_evictions: self.disk_evictions.get(),
            dropped: self.dropped.get(),
            memory_bytes,
            disk_bytes,
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct Count(AtomicU64);

impl Count {
    pub(crate) fn add(&self, n: u64) {
        self.0.fetch_add(n, Ordering::Relaxed);
    }

    pub(crate) fn add_len(&self, n: usize) {
        self.add(u64::try_from(n).unwrap_or(u64::MAX));
    }

    fn get(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }
}
