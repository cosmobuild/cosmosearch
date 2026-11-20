use std::sync::Mutex;
use std::time::{Duration, Instant};

use cosmosearch_core::IndexId;

use crate::lock::locked;
use crate::lru::{Lru, OfIndex};

/// How often within one lifetime `put` sweeps out expired values, so a cache still in use
/// holds an expired value for at most about a quarter of a lifetime longer.
const SWEEPS_PER_LIFETIME: u32 = 4;

pub(crate) type Now = Box<dyn Fn() -> Instant + Send + Sync>;

/// Values kept in memory under a byte budget, least recently used first, each returned for
/// at most a lifetime from when it was kept. An expired value is dropped when it is next
/// looked up, by [`Expiring::remove_expired`], or by a later `put` once part of a lifetime
/// has passed since the last sweep.
pub(crate) struct Expiring<K, V> {
    budget: u64,
    lifetime: Duration,
    now: Now,
    held: Mutex<Held<K, V>>,
}

struct Held<K, V> {
    entries: Lru<K, (V, Instant)>,
    swept: Instant,
}

impl<K: OfIndex, V> Held<K, V> {
    fn remove_expired(&mut self, now: Instant, lifetime: Duration) -> usize {
        let expired: Vec<K> = self
            .entries
            .iter()
            .filter(|(_, (_, kept_at))| now.saturating_duration_since(*kept_at) >= lifetime)
            .map(|(key, _)| key.clone())
            .collect();
        for key in &expired {
            self.entries.remove(key);
        }
        self.swept = now;
        expired.len()
    }
}

impl<K: OfIndex, V: Clone> Expiring<K, V> {
    pub(crate) fn new(budget: u64, lifetime: Duration, now: Now) -> Self {
        let swept = now();
        Expiring {
            budget,
            lifetime,
            now,
            held: Mutex::new(Held {
                entries: Lru::default(),
                swept,
            }),
        }
    }

    pub(crate) fn budget(&self) -> u64 {
        self.budget
    }

    pub(crate) fn lifetime(&self) -> Duration {
        self.lifetime
    }

    /// The weight of everything held.
    pub(crate) fn bytes(&self) -> u64 {
        locked(&self.held).entries.used()
    }

    /// Drops every value whose lifetime has ended, and says how many there were.
    pub(crate) fn remove_expired(&self) -> usize {
        let now = (self.now)();
        locked(&self.held).remove_expired(now, self.lifetime)
    }

    /// The value under `key`, unless its lifetime has ended, in which case it is dropped.
    pub(crate) fn get(&self, key: &K) -> Option<V> {
        let now = (self.now)();
        let mut held = locked(&self.held);
        let entries = &mut held.entries;
        let (value, kept_at) = entries.get(key)?;
        if now.saturating_duration_since(*kept_at) < self.lifetime {
            return Some(value.clone());
        }
        entries.remove(key);
        None
    }

    /// Keeps `value` under `key`, after dropping the entries of the key's index at an older
    /// version, if it holds any. A value heavier than the whole budget is not kept.
    pub(crate) fn put(&self, key: K, value: V, weight: u64) {
        let kept_at = (self.now)();
        let mut held = locked(&self.held);
        if kept_at.saturating_duration_since(held.swept) >= self.lifetime / SWEEPS_PER_LIFETIME {
            held.remove_expired(kept_at, self.lifetime);
        }
        let entries = &mut held.entries;
        let version = key.version();
        if entries
            .oldest_version(key.index())
            .is_some_and(|oldest| oldest < version)
        {
            for other in entries.keys_of(key.index()) {
                if other.version() < version {
                    entries.remove(&other);
                }
            }
        }
        entries.remove(&key);
        if weight > self.budget {
            return;
        }
        while entries.used().saturating_add(weight) > self.budget {
            if entries.pop_oldest(|_| true).is_none() {
                break;
            }
        }
        entries.insert(key, (value, kept_at), weight);
    }

    pub(crate) fn purge(&self, index: &IndexId) {
        locked(&self.held).entries.remove_index(index);
    }
}
