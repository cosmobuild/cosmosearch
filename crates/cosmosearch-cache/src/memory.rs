use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::{Mutex, MutexGuard};

use bytes::Bytes;
use cosmosearch_core::IndexId;

use crate::block::{BlockKey, len};
use crate::lock::locked;
use crate::lru::Lru;

/// At most this many shards, enough that concurrent searches rarely wait on one lock.
const MAX_SHARDS: u64 = 16;
/// Each shard holds at least this many blocks, so sharding does not shrink what fits.
const MIN_BLOCKS_PER_SHARD: u64 = 8;

/// The RAM tier: blocks in shards, each least recently used first under its share of the budget.
#[derive(Debug)]
pub(crate) struct Memory {
    shards: Vec<Mutex<Lru<BlockKey, Bytes>>>,
    shard_budget: u64,
}

impl Memory {
    pub(crate) fn new(budget: u64, block_size: u64) -> Self {
        let per_shard = block_size.saturating_mul(MIN_BLOCKS_PER_SHARD);
        let count = (budget / per_shard).clamp(1, MAX_SHARDS);
        let shards = (0..count).map(|_| Mutex::new(Lru::default())).collect();
        Memory {
            shards,
            shard_budget: budget / count,
        }
    }

    /// The block under `key`, now the most recently used.
    pub(crate) fn get(&self, key: &BlockKey) -> Option<Bytes> {
        self.shard(key).get(key).cloned()
    }

    /// Keeps `bytes` under `key`, returning the blocks evicted to make room, which may include
    /// this one if it is larger than a shard's budget.
    pub(crate) fn insert(&self, key: BlockKey, bytes: Bytes) -> Vec<(BlockKey, Bytes)> {
        let weight = len(&bytes);
        if weight > self.shard_budget {
            return vec![(key, bytes)];
        }
        let mut shard = self.shard(&key);
        let mut evicted = Vec::new();
        shard.remove(&key);
        while shard.used().saturating_add(weight) > self.shard_budget {
            let Some((old, bytes, _)) = shard.pop_oldest(|_| true) else {
                break;
            };
            evicted.push((old, bytes));
        }
        shard.insert(key, bytes, weight);
        evicted
    }

    /// Every block held, least recently used first within each shard.
    pub(crate) fn blocks(&self) -> Vec<(BlockKey, Bytes)> {
        let mut blocks = Vec::new();
        for shard in &self.shards {
            let shard = locked(shard);
            blocks.extend(
                shard
                    .iter()
                    .map(|(key, bytes)| (key.clone(), bytes.clone())),
            );
        }
        blocks
    }

    /// Bytes held over all shards.
    pub(crate) fn used(&self) -> u64 {
        self.shards
            .iter()
            .map(|shard| locked(shard).used())
            .fold(0, u64::saturating_add)
    }

    /// Drops every block of index `index`.
    pub(crate) fn purge(&self, index: &IndexId) {
        for shard in &self.shards {
            locked(shard).remove_index(index);
        }
    }

    #[expect(
        clippy::indexing_slicing,
        reason = "the position is a remainder of the shard count, which is at least 1"
    )]
    fn shard(&self, key: &BlockKey) -> MutexGuard<'_, Lru<BlockKey, Bytes>> {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        let count = u64::try_from(self.shards.len()).unwrap_or(1);
        let position = usize::try_from(hasher.finish() % count).unwrap_or(0);
        locked(&self.shards[position])
    }
}
