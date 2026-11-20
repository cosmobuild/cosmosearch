use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use cosmosearch_core::{CachedResults, IndexId, ResultCache, SearchKey};

use crate::expiring::{Expiring, Now};
use crate::lru::OfIndex;

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<SearchCache>();
};

impl OfIndex for SearchKey {
    fn index(&self) -> &IndexId {
        SearchKey::index(self)
    }

    fn version(&self) -> u64 {
        SearchKey::version(self).get()
    }
}

/// A cache of whole search results in memory, that any number of engines share. Cloning it
/// is cheap and every clone is the same cache.
///
/// Results are kept by index, manifest version and everything else that decides them, so a
/// repeated search is answered without reading anything and a commit leaves nothing stale to
/// return: keeping a result of a newer version drops the index's results of older ones.
/// Results hold refs, heading paths and snippets taken from documents, so a result is never
/// returned after a host-set lifetime from when it was kept, and the least recently used go
/// first under a byte budget. An expired result leaves memory when it is next looked up, when
/// a later result is kept, or when the host calls [`SearchCache::remove_expired`]; a host that
/// must bound how long results stay in memory calls it on an interval.
#[derive(Clone)]
pub struct SearchCache {
    inner: Arc<Expiring<SearchKey, CachedResults>>,
}

impl SearchCache {
    /// A cache holding results of at most about `budget` bytes, each for at most `lifetime`.
    pub fn new(budget: u64, lifetime: Duration) -> Self {
        SearchCache::with_clock(budget, lifetime, Box::new(Instant::now))
    }

    fn with_clock(budget: u64, lifetime: Duration, now: Now) -> Self {
        SearchCache {
            inner: Arc::new(Expiring::new(budget, lifetime, now)),
        }
    }

    /// About how many bytes of results the cache holds.
    pub fn bytes(&self) -> u64 {
        self.inner.bytes()
    }

    /// Drops every result whose lifetime has ended, and says how many there were.
    pub fn remove_expired(&self) -> usize {
        self.inner.remove_expired()
    }

    /// Removes every result of index `index`, for a host that has closed it.
    pub fn purge(&self, index: &IndexId) {
        self.inner.purge(index);
    }
}

impl fmt::Debug for SearchCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SearchCache")
            .field("budget", &self.inner.budget())
            .field("lifetime", &self.inner.lifetime())
            .field("bytes", &self.bytes())
            .finish_non_exhaustive()
    }
}

impl ResultCache for SearchCache {
    fn get(&self, key: &SearchKey) -> Option<CachedResults> {
        self.inner.get(key)
    }

    fn put(&self, key: SearchKey, results: CachedResults, weight: u64) {
        self.inner.put(key, results, weight);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use cosmosearch_core::{ManifestVersion, SearchIdentity};

    use super::*;
    use crate::lock::locked;

    const MINUTE: Duration = Duration::from_secs(60);

    fn key(index: &str, version: u64, query: &str) -> SearchKey {
        let mut at = ManifestVersion::FIRST;
        for _ in 1..version {
            at = at.next().unwrap();
        }
        SearchIdentity::new()
            .text(query)
            .key(&index.parse().unwrap(), at)
    }

    fn cache(budget: u64) -> (SearchCache, Arc<Mutex<Instant>>) {
        let time = Arc::new(Mutex::new(Instant::now()));
        let shown = Arc::clone(&time);
        let cache = SearchCache::with_clock(budget, MINUTE, Box::new(move || *locked(&shown)));
        (cache, time)
    }

    fn held(cache: &SearchCache, key: &SearchKey) -> Option<u32> {
        cache.get(key).and_then(|found| found.get::<u32>().copied())
    }

    #[test]
    fn drops_expired_results_without_a_lookup() {
        let (cache, time) = cache(1_000);
        cache.put(key("i", 1, "refund"), CachedResults::new(1u32), 10);
        *locked(&time) += MINUTE;
        assert_eq!(cache.remove_expired(), 1);
        assert_eq!(cache.bytes(), 0);
    }

    #[test]
    fn returns_what_was_kept_until_its_lifetime_ends() {
        let (cache, time) = cache(100);
        let a = key("i", 1, "a");
        cache.put(a.clone(), CachedResults::new(7u32), 10);
        assert_eq!(held(&cache, &a), Some(7));
        assert_eq!(held(&cache, &key("i", 1, "b")), None);
        *locked(&time) += Duration::from_millis(59_999);
        assert_eq!(held(&cache, &a), Some(7));
        *locked(&time) += Duration::from_millis(1);
        assert_eq!(held(&cache, &a), None);
        assert_eq!(cache.bytes(), 0);
    }

    #[test]
    fn a_newer_version_drops_the_older_results_of_its_index_only() {
        let (cache, _) = cache(100);
        cache.put(key("i", 1, "a"), CachedResults::new(1u32), 10);
        cache.put(key("i", 1, "b"), CachedResults::new(2u32), 10);
        cache.put(key("other", 1, "a"), CachedResults::new(3u32), 10);
        cache.put(key("i", 2, "a"), CachedResults::new(4u32), 10);
        assert_eq!(held(&cache, &key("i", 1, "a")), None);
        assert_eq!(held(&cache, &key("i", 1, "b")), None);
        assert_eq!(held(&cache, &key("i", 2, "a")), Some(4));
        assert_eq!(held(&cache, &key("other", 1, "a")), Some(3));
        assert_eq!(cache.bytes(), 20);
    }

    #[test]
    fn stays_within_its_budget_dropping_the_least_recently_used() {
        let (cache, _) = cache(30);
        for (query, value) in [("a", 1u32), ("b", 2), ("c", 3)] {
            cache.put(key("i", 1, query), CachedResults::new(value), 10);
        }
        assert_eq!(held(&cache, &key("i", 1, "a")), Some(1));
        cache.put(key("i", 1, "d"), CachedResults::new(4u32), 15);
        assert_eq!(held(&cache, &key("i", 1, "b")), None);
        assert_eq!(held(&cache, &key("i", 1, "c")), None);
        assert_eq!(held(&cache, &key("i", 1, "a")), Some(1));
        assert_eq!(cache.bytes(), 25);

        cache.put(key("i", 1, "a"), CachedResults::new(9u32), 31);
        assert_eq!(held(&cache, &key("i", 1, "a")), None);
        assert_eq!(cache.bytes(), 15);
    }

    #[test]
    fn purge_removes_one_index() {
        let (cache, _) = cache(100);
        cache.put(key("i", 1, "a"), CachedResults::new(1u32), 10);
        cache.put(key("other", 1, "a"), CachedResults::new(2u32), 10);
        cache.purge(&"i".parse().unwrap());
        assert_eq!(held(&cache, &key("i", 1, "a")), None);
        assert_eq!(held(&cache, &key("other", 1, "a")), Some(2));
    }
}
