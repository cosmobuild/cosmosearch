use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::hash::Hash;

use cosmosearch_core::IndexId;

#[derive(Debug)]
struct Slot<V> {
    value: V,
    weight: u64,
    tick: u64,
}

/// A key that belongs to one index, so an index's entries can be removed together.
pub(crate) trait OfIndex: Clone + Eq + Hash {
    fn index(&self) -> &IndexId;

    /// The manifest version of its index the entry belongs to; zero for an entry of none.
    fn version(&self) -> u64 {
        0
    }
}

/// Where one index's entries are: their ticks, and how many entries hold each version.
#[derive(Debug, Default)]
struct OfOne {
    ticks: BTreeSet<u64>,
    versions: BTreeMap<u64, usize>,
}

/// Entries in least recently used order, with the sum of their weights.
#[derive(Debug)]
pub(crate) struct Lru<K, V> {
    entries: HashMap<K, Slot<V>>,
    order: BTreeMap<u64, K>,
    by_index: HashMap<IndexId, OfOne>,
    tick: u64,
    used: u64,
}

impl<K, V> Default for Lru<K, V> {
    fn default() -> Self {
        Lru {
            entries: HashMap::new(),
            order: BTreeMap::new(),
            by_index: HashMap::new(),
            tick: 0,
            used: 0,
        }
    }
}

impl<K: OfIndex, V> Lru<K, V> {
    /// The sum of the weights of all entries.
    pub(crate) fn used(&self) -> u64 {
        self.used
    }

    /// The entry under `key`, now the most recently used.
    pub(crate) fn get(&mut self, key: &K) -> Option<&mut V> {
        let tick = self.next_tick();
        let slot = self.entries.get_mut(key)?;
        self.order.remove(&slot.tick);
        if let Some(of_one) = self.by_index.get_mut(key.index()) {
            of_one.ticks.remove(&slot.tick);
            of_one.ticks.insert(tick);
        }
        slot.tick = tick;
        self.order.insert(tick, key.clone());
        Some(&mut slot.value)
    }

    /// The entry under `key`, leaving the order as it is.
    pub(crate) fn peek(&self, key: &K) -> Option<&V> {
        self.entries.get(key).map(|slot| &slot.value)
    }

    /// The entry under `key`, leaving the order as it is.
    pub(crate) fn peek_mut(&mut self, key: &K) -> Option<&mut V> {
        self.entries.get_mut(key).map(|slot| &mut slot.value)
    }

    /// The keys of every entry of index `index`, least recently used first.
    pub(crate) fn keys_of(&self, index: &IndexId) -> Vec<K> {
        self.by_index.get(index).map_or_else(Vec::new, |of_one| {
            of_one
                .ticks
                .iter()
                .filter_map(|tick| self.order.get(tick))
                .cloned()
                .collect()
        })
    }

    /// The lowest version any entry of index `index` holds.
    pub(crate) fn oldest_version(&self, index: &IndexId) -> Option<u64> {
        let of_one = self.by_index.get(index)?;
        of_one.versions.keys().next().copied()
    }

    /// Every entry, least recently used first.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.order
            .values()
            .filter_map(|key| Some((key, &self.entries.get(key)?.value)))
    }

    /// Adds `value` as the most recently used entry, replacing any under `key`.
    pub(crate) fn insert(&mut self, key: K, value: V, weight: u64) {
        self.remove(&key);
        let tick = self.next_tick();
        self.order.insert(tick, key.clone());
        self.used = self.used.saturating_add(weight);
        let of_one = self.by_index.entry(key.index().clone()).or_default();
        of_one.ticks.insert(tick);
        let held = of_one.versions.entry(key.version()).or_default();
        *held = held.saturating_add(1);
        self.entries.insert(
            key,
            Slot {
                value,
                weight,
                tick,
            },
        );
    }

    /// Removes the entry under `key`, returning it and its weight.
    pub(crate) fn remove(&mut self, key: &K) -> Option<(V, u64)> {
        let slot = self.entries.remove(key)?;
        self.order.remove(&slot.tick);
        self.used = self.used.saturating_sub(slot.weight);
        if let Some(of_one) = self.by_index.get_mut(key.index()) {
            of_one.ticks.remove(&slot.tick);
            if let Some(held) = of_one.versions.get_mut(&key.version()) {
                *held = held.saturating_sub(1);
                if *held == 0 {
                    of_one.versions.remove(&key.version());
                }
            }
            if of_one.ticks.is_empty() {
                self.by_index.remove(key.index());
            }
        }
        Some((slot.value, slot.weight))
    }

    /// Removes the least recently used entry that `evictable` accepts.
    pub(crate) fn pop_oldest(&mut self, evictable: impl Fn(&V) -> bool) -> Option<(K, V, u64)> {
        let key = self
            .order
            .values()
            .find(|key| {
                self.entries
                    .get(*key)
                    .is_some_and(|slot| evictable(&slot.value))
            })?
            .clone();
        let (value, weight) = self.remove(&key)?;
        Some((key, value, weight))
    }

    /// Removes every entry of index `index`, returning them.
    pub(crate) fn remove_index(&mut self, index: &IndexId) -> Vec<(K, V, u64)> {
        let keys = self.keys_of(index);
        let mut removed = Vec::with_capacity(keys.len());
        for key in keys {
            if let Some((value, weight)) = self.remove(&key) {
                removed.push((key, value, weight));
            }
        }
        removed
    }

    fn next_tick(&mut self) -> u64 {
        self.tick = self.tick.saturating_add(1);
        self.tick
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use cosmosearch_core::{CachedFile, SegmentFile};

    use super::*;
    use crate::block::BlockKey;

    fn key(index: &str, block: u64) -> BlockKey {
        let file = SegmentFile::new("lex-s1.bin".parse().unwrap(), 1 << 20);
        BlockKey {
            file: Arc::new(CachedFile::new(
                index.parse().unwrap(),
                "s1".parse().unwrap(),
                &file,
            )),
            block,
        }
    }

    #[test]
    fn evicts_the_least_recently_used_first() {
        let mut lru = Lru::default();
        for block in 0..3 {
            lru.insert(key("a", block), block, 10);
        }
        lru.get(&key("a", 0));
        assert_eq!(lru.used(), 30);
        let (oldest, _, _) = lru.pop_oldest(|_| true).unwrap();
        assert_eq!(oldest, key("a", 1));
        let (next, _, _) = lru.pop_oldest(|value| *value != 2).unwrap();
        assert_eq!(next, key("a", 0));
        assert_eq!(lru.used(), 10);
    }

    #[test]
    fn replacing_an_entry_keeps_the_weight_right() {
        let mut lru = Lru::default();
        lru.insert(key("a", 0), 0, 10);
        lru.insert(key("a", 0), 1, 4);
        assert_eq!(lru.used(), 4);
        assert_eq!(lru.peek(&key("a", 0)), Some(&1));
    }

    #[test]
    fn knows_each_indexs_entries_and_lets_go_of_an_index_with_none() {
        let mut lru = Lru::default();
        lru.insert(key("a", 0), 0, 10);
        lru.insert(key("b", 0), 1, 10);
        lru.insert(key("a", 1), 2, 10);
        lru.get(&key("a", 0));
        let a: IndexId = "a".parse().unwrap();
        assert_eq!(lru.keys_of(&a), [key("a", 1), key("a", 0)]);
        assert_eq!(lru.oldest_version(&a), Some(0));
        lru.pop_oldest(|_| true);
        lru.remove(&key("a", 1));
        lru.remove(&key("a", 0));
        assert_eq!(lru.keys_of(&a), Vec::new());
        assert_eq!(lru.oldest_version(&a), None);
        assert_eq!(lru.by_index.len(), 0);
    }

    #[test]
    fn removes_one_index_only() {
        let mut lru = Lru::default();
        lru.insert(key("a", 0), 0, 10);
        lru.insert(key("b", 0), 1, 10);
        lru.insert(key("a", 1), 2, 10);
        let removed = lru.remove_index(&"a".parse().unwrap());
        assert_eq!(removed.len(), 2);
        assert_eq!(lru.used(), 10);
        assert!(lru.peek(&key("b", 0)).is_some());
    }
}
