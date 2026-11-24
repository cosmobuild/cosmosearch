use std::any::Any;
use std::fmt;
use std::sync::Arc;

use crate::{Filter, IndexId, ManifestVersion, Scope};

/// Everything that changes a search's results, as a cache keys them: the index, the manifest
/// version searched and a digest of the rest.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SearchKey {
    index: IndexId,
    version: ManifestVersion,
    digest: [u8; 32],
}

impl SearchKey {
    /// The index searched.
    pub fn index(&self) -> &IndexId {
        &self.index
    }

    /// The manifest version searched.
    pub fn version(&self) -> ManifestVersion {
        self.version
    }
}

/// Gathers what a search's results depend on besides the index and its version: the query,
/// the scope, the filter, the number of hits and how they are ranked.
///
/// Each part is added with its length, so two different lists of parts never give one key.
#[derive(Debug, Clone, Default)]
pub struct SearchIdentity(blake3::Hasher);

impl SearchIdentity {
    /// An identity with nothing added yet.
    pub fn new() -> Self {
        SearchIdentity::default()
    }

    /// Adds a piece of text, such as the query or a model name.
    #[must_use]
    pub fn text(mut self, text: &str) -> Self {
        self.part(text.as_bytes());
        self
    }

    /// Adds a number, such as the number of hits asked for.
    #[must_use]
    pub fn number(mut self, number: u64) -> Self {
        self.0.update(&number.to_le_bytes());
        self
    }

    /// Adds a filter, the same whatever order its conditions were given in.
    #[must_use]
    pub fn filter(mut self, filter: &Filter) -> Self {
        let parts = filter.identity_parts();
        self = self.number(len(parts.len()));
        for part in parts {
            self.part(&part);
        }
        self
    }

    /// Adds a scope, the same whatever order its prefixes were given in.
    #[must_use]
    pub fn scope(mut self, scope: &Scope) -> Self {
        self = self.number(len(scope.len()));
        for prefix in scope.sorted_prefixes() {
            self = self.text(prefix);
        }
        self
    }

    /// The key of this identity in version `version` of index `index`.
    pub fn key(&self, index: &IndexId, version: ManifestVersion) -> SearchKey {
        SearchKey {
            index: index.clone(),
            version,
            digest: *self.0.finalize().as_bytes(),
        }
    }

    fn part(&mut self, bytes: &[u8]) {
        self.0.update(&len(bytes.len()).to_le_bytes());
        self.0.update(bytes);
    }
}

fn len(len: usize) -> u64 {
    u64::try_from(len).unwrap_or(u64::MAX)
}

/// The results of one search as a cache holds them, opaque to the cache.
#[derive(Debug, Clone)]
pub struct CachedResults(Arc<dyn Any + Send + Sync>);

impl CachedResults {
    /// Wraps `results` for a cache.
    pub fn new<T: Any + Send + Sync>(results: T) -> Self {
        CachedResults(Arc::new(results))
    }

    /// The results, if they are a `T`.
    pub fn get<T: Any>(&self) -> Option<&T> {
        self.0.downcast_ref()
    }
}

/// A cache of whole search results, shared by the readers of any number of indexes.
///
/// A key holds the manifest version, so a result is never returned for another snapshot.
/// It is called on the search path, so it answers from memory and never waits on storage.
pub trait ResultCache: Send + Sync + fmt::Debug {
    /// The results kept under `key`, if any are still held.
    fn get(&self, key: &SearchKey) -> Option<CachedResults>;

    /// Keeps `results` under `key`. `weight` is about how many bytes they hold.
    fn put(&self, key: SearchKey, results: CachedResults, weight: u64);
}

/// A shared handle on a result cache.
#[derive(Debug, Clone)]
pub struct SharedResults(Arc<dyn ResultCache>);

impl SharedResults {
    /// Shares `cache`.
    pub fn new(cache: impl ResultCache + 'static) -> Self {
        SharedResults(Arc::new(cache))
    }

    /// The cache.
    pub fn get(&self) -> &dyn ResultCache {
        self.0.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(identity: &SearchIdentity) -> SearchKey {
        identity.key(&"index-1".parse().unwrap(), ManifestVersion::FIRST)
    }

    #[test]
    fn every_part_changes_the_key_and_filter_order_does_not() {
        let base = || SearchIdentity::new().text("refund policy").number(10);
        let filtered = |filter: &Filter| key(&base().filter(filter));
        let plain = filtered(&Filter::new());
        let ab = Filter::new().equals("a", "1").any_of("b", &["2", "3"]);
        let ba = Filter::new().any_of("b", &["3", "2"]).equals("a", "1");
        assert_eq!(filtered(&ab), filtered(&ba));
        let scoped = |scope: &Scope| key(&base().filter(&Filter::new()).scope(scope));
        assert_eq!(
            scoped(&Scope::prefixes(&["a/", "b/"])),
            scoped(&Scope::prefixes(&["b/", "a/", "a/x"]))
        );

        let others = [
            key(&SearchIdentity::new()
                .text("refund")
                .number(10)
                .filter(&Filter::new())),
            key(&SearchIdentity::new()
                .text("refund policy")
                .number(11)
                .filter(&Filter::new())),
            filtered(&ab),
            filtered(&Filter::new().equals("a", "1")),
            filtered(&Filter::new().equals("a", "12")),
            filtered(&Filter::new().any_of("a", &[])),
            scoped(&Scope::all()),
            scoped(&Scope::prefix("a/")),
            scoped(&Scope::prefixes(&["a/", "b/"])),
            scoped(&Scope::prefixes(&[])),
            key(&SearchIdentity::new()
                .text("refund ")
                .text("policy")
                .number(10)
                .filter(&Filter::new())),
        ];
        for (position, other) in others.iter().enumerate() {
            assert_ne!(*other, plain, "{position}");
            for later in others.iter().skip(position.saturating_add(1)) {
                assert_ne!(other, later, "{position}");
            }
        }

        let identity = base().filter(&Filter::new());
        let moved = identity.key(
            &"index-1".parse().unwrap(),
            ManifestVersion::FIRST.next().unwrap(),
        );
        assert_ne!(moved, plain);
        assert_ne!(
            identity.key(&"index-2".parse().unwrap(), ManifestVersion::FIRST),
            plain
        );
    }

    #[test]
    fn cached_results_give_back_only_their_own_type() {
        let cached = CachedResults::new(vec![1u8, 2]);
        assert_eq!(cached.get::<Vec<u8>>(), Some(&vec![1u8, 2]));
        assert_eq!(cached.get::<String>(), None);
    }
}
