use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::SystemTime;

use async_trait::async_trait;
use bytes::Bytes;
use cosmosearch_core::{ByteRange, ETag};
use futures_util::StreamExt;
use futures_util::stream::{self, BoxStream};

use crate::{Error, ObjectInfo, ObjectKey, ReadStore, Result, WriteStore};

fn byte_count(body: &Bytes) -> u64 {
    u64::try_from(body.len()).unwrap_or(u64::MAX)
}

#[derive(Debug)]
struct Stored {
    body: Bytes,
    etag: ETag,
    written: SystemTime,
}

impl Stored {
    fn info(&self, key: &ObjectKey) -> ObjectInfo {
        ObjectInfo::new(key.clone(), byte_count(&self.body), self.etag.clone())
            .with_last_modified(self.written)
    }
}

#[derive(Debug)]
struct State {
    objects: BTreeMap<ObjectKey, Stored>,
    writes: u64,
    time: SystemTime,
}

impl Default for State {
    fn default() -> Self {
        State {
            objects: BTreeMap::new(),
            writes: 0,
            time: SystemTime::UNIX_EPOCH,
        }
    }
}

impl State {
    fn store(&mut self, key: &ObjectKey, body: Bytes) -> Result<()> {
        self.writes = self.writes.saturating_add(1);
        let etag = ETag::new(self.writes.to_string())?;
        let written = self.time;
        self.objects.insert(
            key.clone(),
            Stored {
                body,
                etag,
                written,
            },
        );
        Ok(())
    }

    fn body(&self, key: &ObjectKey) -> Result<&Bytes> {
        self.objects
            .get(key)
            .map(|stored| &stored.body)
            .ok_or_else(|| Error::NotFound { key: key.clone() })
    }
}

/// A store held in memory, for tests. Clones share the same objects.
///
/// Its clock stands still at the Unix epoch until [`MemoryStore::set_time`] moves it, and
/// every write is stamped with it.
#[derive(Debug, Clone, Default)]
pub struct MemoryStore {
    state: Arc<Mutex<State>>,
}

impl MemoryStore {
    /// Creates an empty store.
    pub fn new() -> Self {
        MemoryStore::default()
    }

    /// Sets the time the store reports for the objects written from now on.
    pub fn set_time(&self, time: SystemTime) {
        self.state().time = time;
    }

    #[expect(
        clippy::expect_used,
        reason = "a poisoned lock means an earlier panic, and its state is not trusted"
    )]
    fn state(&self) -> MutexGuard<'_, State> {
        self.state
            .lock()
            .expect("no thread panicked while holding the lock")
    }

    fn snapshot(&self, prefix: Option<&ObjectKey>, after: Option<&ObjectKey>) -> Vec<ObjectInfo> {
        self.state()
            .objects
            .iter()
            .filter(|(key, _)| prefix.is_none_or(|prefix| key.is_under(prefix)))
            .filter(|(key, _)| after.is_none_or(|after| *key > after))
            .map(|(key, stored)| stored.info(key))
            .collect()
    }
}

#[async_trait]
impl ReadStore for MemoryStore {
    async fn read_range(
        &self,
        key: &ObjectKey,
        range: ByteRange,
        expected: &ETag,
    ) -> Result<Bytes> {
        let state = self.state();
        let stored = state
            .objects
            .get(key)
            .ok_or_else(|| Error::NotFound { key: key.clone() })?;
        let body = &stored.body;
        if &stored.etag != expected {
            return Err(Error::Stale {
                key: key.clone(),
                expected: expected.clone(),
            });
        }
        let start = usize::try_from(range.start()).unwrap_or(usize::MAX);
        let end = usize::try_from(range.end()).unwrap_or(usize::MAX);
        if end > body.len() {
            return Err(Error::ShortRead { key: key.clone() });
        }
        Ok(body.slice(start..end))
    }

    async fn head(&self, key: &ObjectKey) -> Result<ObjectInfo> {
        let state = self.state();
        state
            .objects
            .get(key)
            .map(|stored| stored.info(key))
            .ok_or_else(|| Error::NotFound { key: key.clone() })
    }

    fn list<'a>(
        &'a self,
        prefix: Option<&'a ObjectKey>,
        after: Option<&'a ObjectKey>,
    ) -> BoxStream<'a, Result<ObjectInfo>> {
        stream::iter(self.snapshot(prefix, after).into_iter().map(Ok)).boxed()
    }
}

#[async_trait]
impl WriteStore for MemoryStore {
    async fn get_range(&self, key: &ObjectKey, range: ByteRange) -> Result<Bytes> {
        let state = self.state();
        let body = state.body(key)?;
        let start = usize::try_from(range.start()).unwrap_or(usize::MAX);
        let end = usize::try_from(range.end()).unwrap_or(usize::MAX);
        if end > body.len() {
            return Err(Error::ShortRead { key: key.clone() });
        }
        Ok(body.slice(start..end))
    }

    async fn get(&self, key: &ObjectKey) -> Result<Bytes> {
        self.state().body(key).cloned()
    }

    async fn put_create(&self, key: &ObjectKey, body: Bytes) -> Result<()> {
        let mut state = self.state();
        if state.objects.contains_key(key) {
            return Err(Error::AlreadyExists { key: key.clone() });
        }
        state.store(key, body)
    }

    async fn put_overwrite(&self, key: &ObjectKey, body: Bytes) -> Result<()> {
        self.state().store(key, body)
    }

    async fn delete(&self, key: &ObjectKey) -> Result<()> {
        self.state().objects.remove(key);
        Ok(())
    }
}
