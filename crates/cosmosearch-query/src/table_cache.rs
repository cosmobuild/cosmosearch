use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

use cosmosearch_core::{IndexId, SegmentId};
use cosmosearch_format::{ChunkGroup, FileGroup, FileTable};

/// The budget of a reader's table cache unless its options give one.
pub const DEFAULT_TABLE_BUDGET: u64 = 64 * 1024 * 1024;

/// Decoded file tables and groups of chunk records, kept under a byte budget.
///
/// Readers read them for hits, scopes and filters, and the cache drops the least recently
/// used first. One cache can be shared by the readers of many indexes, so their tables share
/// one budget.
///
/// A table still in use when it is dropped stays valid for the search using it, so for a
/// moment a reader can hold more than the budget. A table larger than the whole budget is
/// not kept, so a search that needs it reads it again. Tables of segments an index no longer
/// uses leave as newer tables need their room.
#[derive(Clone)]
pub struct TableCache {
    inner: Arc<Mutex<Lru>>,
}

impl TableCache {
    /// A cache holding tables of at most about `budget` bytes.
    pub fn new(budget: u64) -> Self {
        TableCache {
            inner: Arc::new(Mutex::new(Lru::new(budget))),
        }
    }

    /// The most the cache holds, in bytes.
    pub fn budget(&self) -> u64 {
        self.lock().budget
    }

    /// About how many bytes of tables the cache holds.
    pub fn bytes(&self) -> u64 {
        self.lock().used
    }

    #[expect(
        clippy::expect_used,
        reason = "a poisoned lock means an earlier panic, and its state is not trusted"
    )]
    fn lock(&self) -> MutexGuard<'_, Lru> {
        self.inner
            .lock()
            .expect("no thread panicked while holding the lock")
    }

    pub(crate) fn get(&self, key: &TableKey) -> Option<Table> {
        self.lock().get(key)
    }

    pub(crate) fn put(&self, key: TableKey, table: Table) {
        self.lock().put(key, table);
    }

    /// About how many bytes the cache holds for `index`.
    pub(crate) fn bytes_of(&self, index: &IndexId) -> u64 {
        self.lock().by_index.get(index).copied().unwrap_or(0)
    }
}

impl fmt::Debug for TableCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let lru = self.lock();
        f.debug_struct("TableCache")
            .field("budget", &lru.budget)
            .field("bytes", &lru.used)
            .finish()
    }
}

/// Which table of a segment a cache entry is.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Part {
    Files,
    FileGroup(usize),
    Group(usize),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct TableKey {
    pub(crate) index: IndexId,
    pub(crate) segment: SegmentId,
    pub(crate) part: Part,
}

#[derive(Debug, Clone)]
pub(crate) enum Table {
    Files(Arc<FileTable>),
    FileGroup(Arc<FileGroup>),
    Group(Arc<ChunkGroup>),
}

impl Table {
    fn weight(&self) -> u64 {
        let bytes = match self {
            Table::Files(files) => files.heap_bytes(),
            Table::FileGroup(group) => group.heap_bytes(),
            Table::Group(group) => group.heap_bytes(),
        };
        u64::try_from(bytes).unwrap_or(u64::MAX)
    }
}

struct Slot {
    table: Table,
    weight: u64,
    tick: u64,
}

struct Lru {
    budget: u64,
    used: u64,
    tick: u64,
    entries: BTreeMap<TableKey, Slot>,
    order: BTreeMap<u64, TableKey>,
    by_index: BTreeMap<IndexId, u64>,
}

impl Lru {
    fn new(budget: u64) -> Self {
        Lru {
            budget,
            used: 0,
            tick: 0,
            entries: BTreeMap::new(),
            order: BTreeMap::new(),
            by_index: BTreeMap::new(),
        }
    }

    fn get(&mut self, key: &TableKey) -> Option<Table> {
        self.tick = self.tick.saturating_add(1);
        let tick = self.tick;
        let slot = self.entries.get_mut(key)?;
        self.order.remove(&slot.tick);
        slot.tick = tick;
        self.order.insert(tick, key.clone());
        Some(slot.table.clone())
    }

    fn put(&mut self, key: TableKey, table: Table) {
        self.remove(&key);
        let weight = table.weight();
        if weight > self.budget {
            return;
        }
        while self.used.saturating_add(weight) > self.budget {
            let Some((_, oldest)) = self.order.pop_first() else {
                break;
            };
            self.remove(&oldest);
        }
        self.tick = self.tick.saturating_add(1);
        self.order.insert(self.tick, key.clone());
        self.used = self.used.saturating_add(weight);
        let held = self.by_index.entry(key.index.clone()).or_default();
        *held = held.saturating_add(weight);
        self.entries.insert(
            key,
            Slot {
                table,
                weight,
                tick: self.tick,
            },
        );
    }

    fn remove(&mut self, key: &TableKey) {
        let Some(slot) = self.entries.remove(key) else {
            return;
        };
        self.order.remove(&slot.tick);
        self.used = self.used.saturating_sub(slot.weight);
        if let Some(held) = self.by_index.get_mut(&key.index) {
            *held = held.saturating_sub(slot.weight);
            if *held == 0 {
                self.by_index.remove(&key.index);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use cosmosearch_core::{ETag, ManifestVersion};
    use cosmosearch_format::FileEntry;

    use super::*;

    fn key(index: &str, segment: &str) -> TableKey {
        TableKey {
            index: IndexId::new(index).unwrap(),
            segment: SegmentId::new(segment).unwrap(),
            part: Part::Files,
        }
    }

    fn files(count: usize) -> Table {
        let mut table = FileTable::new();
        for at in 0..count {
            let etag = ETag::new("9f3a").unwrap();
            table.insert(FileEntry::pending(
                format!("docs/{at}.txt"),
                5,
                etag,
                ManifestVersion::FIRST,
            ));
        }
        Table::Files(Arc::new(table))
    }

    #[test]
    fn the_least_recently_used_table_goes_first_and_the_budget_holds() {
        let weight = files(10).weight();
        let cache = TableCache::new(weight * 2);
        cache.put(key("a", "s1"), files(10));
        cache.put(key("a", "s2"), files(10));
        assert!(cache.get(&key("a", "s1")).is_some());
        cache.put(key("b", "s3"), files(10));
        assert!(cache.get(&key("a", "s2")).is_none());
        assert!(cache.get(&key("a", "s1")).is_some());
        assert!(cache.bytes() <= cache.budget());
        assert_eq!(cache.bytes_of(&IndexId::new("b").unwrap()), weight);
    }

    #[test]
    fn a_table_larger_than_the_budget_is_not_kept() {
        let cache = TableCache::new(1);
        cache.put(key("a", "s1"), files(1));
        assert_eq!(cache.bytes(), 0);
        assert!(cache.get(&key("a", "s1")).is_none());
    }
}
