use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use cosmosearch_core::{IndexId, PassageCache, SourceRef};

use crate::expiring::{Expiring, Now};
use crate::lru::OfIndex;

/// What an entry takes in memory besides its ref and text, roughly.
const ENTRY_BYTES: usize = 96;

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<TextCache>();
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PassageKey {
    index: IndexId,
    reference: String,
}

impl OfIndex for PassageKey {
    fn index(&self) -> &IndexId {
        &self.index
    }
}

/// A cache of passage text in memory, that any number of engines share. Cloning it is cheap
/// and every clone is the same cache.
///
/// A passage is kept under its index and its ref, which holds the file's ETag, so the text
/// is never that of another version of the file. It saves reading and parsing a file again
/// for a snippet or a reranker. The text is document text, so a passage is never returned
/// after a host-set lifetime from when it was kept, and the least recently used go first
/// under a byte budget. An expired passage leaves memory when it is next looked up, when a
/// later passage is kept, or when the host calls [`TextCache::remove_expired`]; a host that
/// must bound how long text stays in memory calls it on an interval.
#[derive(Clone)]
pub struct TextCache {
    inner: Arc<Expiring<PassageKey, Arc<str>>>,
}

impl TextCache {
    /// A cache holding passages of at most about `budget` bytes, each for at most `lifetime`.
    pub fn new(budget: u64, lifetime: Duration) -> Self {
        TextCache::with_clock(budget, lifetime, Box::new(Instant::now))
    }

    fn with_clock(budget: u64, lifetime: Duration, now: Now) -> Self {
        TextCache {
            inner: Arc::new(Expiring::new(budget, lifetime, now)),
        }
    }

    /// About how many bytes of passages the cache holds.
    pub fn bytes(&self) -> u64 {
        self.inner.bytes()
    }

    /// Drops every passage whose lifetime has ended, and says how many there were.
    pub fn remove_expired(&self) -> usize {
        self.inner.remove_expired()
    }

    /// Removes every passage of index `index`, for a host that has closed it.
    pub fn purge(&self, index: &IndexId) {
        self.inner.purge(index);
    }
}

impl fmt::Debug for TextCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TextCache")
            .field("budget", &self.inner.budget())
            .field("lifetime", &self.inner.lifetime())
            .field("bytes", &self.bytes())
            .finish_non_exhaustive()
    }
}

fn key(index: &IndexId, reference: &SourceRef) -> PassageKey {
    PassageKey {
        index: index.clone(),
        reference: reference.to_string(),
    }
}

impl PassageCache for TextCache {
    fn get(&self, index: &IndexId, reference: &SourceRef) -> Option<Arc<str>> {
        self.inner.get(&key(index, reference))
    }

    fn put(&self, index: &IndexId, reference: &SourceRef, text: Arc<str>) {
        let key = key(index, reference);
        let bytes = ENTRY_BYTES
            .saturating_add(key.reference.len())
            .saturating_add(text.len());
        let weight = u64::try_from(bytes).unwrap_or(u64::MAX);
        self.inner.put(key, text, weight);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::lock::locked;

    const MINUTE: Duration = Duration::from_secs(60);

    fn reference(key: &str, etag: &str) -> SourceRef {
        format!("s3://example-bucket/{key}#bytes=0-4@etag={etag}")
            .parse()
            .unwrap()
    }

    fn index(name: &str) -> IndexId {
        name.parse().unwrap()
    }

    #[test]
    fn keeps_text_by_index_ref_and_etag_until_its_lifetime_ends() {
        let time = Arc::new(Mutex::new(Instant::now()));
        let shown = Arc::clone(&time);
        let cache = TextCache::with_clock(1_000, MINUTE, Box::new(move || *locked(&shown)));
        let a = reference("a.txt", "1");
        cache.put(&index("i"), &a, Arc::from("refund policy"));
        assert_eq!(cache.get(&index("i"), &a).as_deref(), Some("refund policy"));
        assert_eq!(cache.get(&index("other"), &a), None);
        assert_eq!(cache.get(&index("i"), &reference("a.txt", "2")), None);
        assert_eq!(cache.get(&index("i"), &reference("b.txt", "1")), None);
        *locked(&time) += MINUTE;
        assert_eq!(cache.get(&index("i"), &a), None);
        assert_eq!(cache.bytes(), 0);
    }

    #[test]
    fn drops_expired_passages_without_a_lookup() {
        let time = Arc::new(Mutex::new(Instant::now()));
        let shown = Arc::clone(&time);
        let cache = TextCache::with_clock(1_000, MINUTE, Box::new(move || *locked(&shown)));
        let (a, b) = (reference("a.txt", "1"), reference("b.txt", "1"));
        cache.put(&index("i"), &a, Arc::from("refund policy"));
        *locked(&time) += MINUTE;
        assert!(cache.bytes() > 0);
        assert_eq!(cache.remove_expired(), 1);
        assert_eq!(cache.bytes(), 0);
        cache.put(&index("i"), &a, Arc::from("refund policy"));
        *locked(&time) += MINUTE;
        cache.put(&index("i"), &b, Arc::from("fees"));
        *locked(&time) += MINUTE;
        assert_eq!(
            cache.remove_expired(),
            1,
            "the later put dropped the first passage"
        );
    }

    #[test]
    fn stays_within_its_budget_and_purges_one_index() {
        let cache = TextCache::new(400, MINUTE);
        let text: Arc<str> = Arc::from("x".repeat(40));
        for name in ["a", "b", "c"] {
            cache.put(&index("i"), &reference(name, "1"), Arc::clone(&text));
        }
        assert!(cache.bytes() <= 400, "{}", cache.bytes());
        assert_eq!(cache.get(&index("i"), &reference("a", "1")), None);
        assert!(cache.get(&index("i"), &reference("c", "1")).is_some());

        cache.put(&index("other"), &reference("c", "1"), Arc::clone(&text));
        cache.purge(&index("i"));
        assert_eq!(cache.get(&index("i"), &reference("c", "1")), None);
        assert!(cache.get(&index("other"), &reference("c", "1")).is_some());
        cache.put(
            &index("i"),
            &reference("huge", "1"),
            Arc::from("y".repeat(500)),
        );
        assert_eq!(cache.get(&index("i"), &reference("huge", "1")), None);
    }
}
