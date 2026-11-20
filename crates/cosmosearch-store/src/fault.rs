use std::sync::{Arc, Mutex, MutexGuard};

use async_trait::async_trait;
use bytes::Bytes;
use cosmosearch_core::{ByteRange, ETag};
use futures_util::stream::BoxStream;

use crate::{BackendError, Error, ObjectInfo, ObjectKey, ReadStore, Result, WriteStore};

/// A store operation a fault can target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Op {
    /// [`WriteStore::get`].
    Get,
    /// [`WriteStore::put_create`].
    PutCreate,
    /// [`WriteStore::put_overwrite`].
    PutOverwrite,
    /// [`WriteStore::delete`].
    Delete,
}

/// What goes wrong with one operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Fault {
    /// Nothing is applied and the call fails.
    Crash,
    /// Nothing is applied and the outcome is reported as unknown.
    Timeout,
    /// The write is applied and the outcome is reported as unknown.
    LostResponse,
    /// The outcome is reported as unknown, and [`FaultStore::release_delayed`] lands it.
    Delayed,
    /// Nothing is applied and a conflict is reported.
    InvisibleConflict,
    /// Nothing is applied and the store refuses definitely.
    Refuse,
}

#[derive(Debug)]
struct Rule {
    op: Op,
    key_contains: String,
    fault: Fault,
}

#[derive(Debug)]
enum Write {
    Create(Bytes),
    Overwrite(Bytes),
    Delete,
}

impl Write {
    fn op(&self) -> Op {
        match self {
            Write::Create(_) => Op::PutCreate,
            Write::Overwrite(_) => Op::PutOverwrite,
            Write::Delete => Op::Delete,
        }
    }

    async fn apply(self, store: &dyn WriteStore, key: &ObjectKey) -> Result<()> {
        match self {
            Write::Create(body) => store.put_create(key, body).await,
            Write::Overwrite(body) => store.put_overwrite(key, body).await,
            Write::Delete => store.delete(key).await,
        }
    }
}

#[derive(Debug, Default)]
struct State {
    rules: Vec<Rule>,
    writes_before_stop: Option<usize>,
    delayed: Vec<(ObjectKey, Write)>,
}

/// Wraps a store and injects failures, for tests. Clones share the same faults.
#[derive(Debug, Clone)]
pub struct FaultStore {
    inner: Arc<dyn WriteStore>,
    state: Arc<Mutex<State>>,
}

fn unknown(key: &ObjectKey) -> Error {
    Error::OutcomeUnknown {
        key: key.clone(),
        source: BackendError::new("injected fault"),
    }
}

fn crashed(key: &ObjectKey) -> Error {
    Error::Backend {
        key: key.to_string(),
        source: BackendError::new("injected crash"),
    }
}

impl FaultStore {
    /// Wraps `inner`.
    pub fn new(inner: impl WriteStore + 'static) -> Self {
        FaultStore {
            inner: Arc::new(inner),
            state: Arc::default(),
        }
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

    /// The next `op` on a key containing `key_contains` suffers `fault`, once.
    pub fn inject(&self, op: Op, key_contains: &str, fault: Fault) {
        self.state().rules.push(Rule {
            op,
            key_contains: key_contains.to_owned(),
            fault,
        });
    }

    /// Lets `writes` more writes through, then every later write crashes.
    pub fn stop_after_writes(&self, writes: usize) {
        self.state().writes_before_stop = Some(writes);
    }

    /// Lets writes through again after [`FaultStore::stop_after_writes`].
    pub fn resume_writes(&self) {
        self.state().writes_before_stop = None;
    }

    /// Lands every delayed write, under the same conditions it was sent with.
    ///
    /// # Errors
    ///
    /// Returns the first failure other than a create-only write finding its key taken.
    pub async fn release_delayed(&self) -> Result<()> {
        let delayed = std::mem::take(&mut self.state().delayed);
        for (key, write) in delayed {
            match write.apply(self.inner.as_ref(), &key).await {
                Ok(()) | Err(Error::AlreadyExists { .. }) => {}
                Err(err) => return Err(err),
            }
        }
        Ok(())
    }

    fn take_fault(&self, op: Op, key: &ObjectKey) -> Option<Fault> {
        let mut state = self.state();
        let position = state
            .rules
            .iter()
            .position(|rule| rule.op == op && key.as_str().contains(&rule.key_contains))?;
        Some(state.rules.remove(position).fault)
    }

    fn take_write_allowance(&self) -> bool {
        match &mut self.state().writes_before_stop {
            Some(0) => false,
            Some(left) => {
                *left = left.saturating_sub(1);
                true
            }
            None => true,
        }
    }

    async fn write(&self, key: &ObjectKey, write: Write) -> Result<()> {
        if !self.take_write_allowance() {
            return Err(crashed(key));
        }
        let store = self.inner.as_ref();
        match self.take_fault(write.op(), key) {
            None => write.apply(store, key).await,
            Some(Fault::Crash) => Err(crashed(key)),
            Some(Fault::Timeout) => Err(unknown(key)),
            Some(Fault::InvisibleConflict) => Err(Error::AlreadyExists { key: key.clone() }),
            Some(Fault::Refuse) => Err(Error::Refused {
                key: key.to_string(),
                source: BackendError::new("injected refusal"),
            }),
            Some(Fault::LostResponse) => write.apply(store, key).await.and(Err(unknown(key))),
            Some(Fault::Delayed) => {
                self.state().delayed.push((key.clone(), write));
                Err(unknown(key))
            }
        }
    }
}

#[async_trait]
impl ReadStore for FaultStore {
    async fn read_range(
        &self,
        key: &ObjectKey,
        range: ByteRange,
        expected: &ETag,
    ) -> Result<Bytes> {
        self.inner.read_range(key, range, expected).await
    }

    async fn head(&self, key: &ObjectKey) -> Result<ObjectInfo> {
        self.inner.head(key).await
    }

    fn list<'a>(
        &'a self,
        prefix: Option<&'a ObjectKey>,
        after: Option<&'a ObjectKey>,
    ) -> BoxStream<'a, Result<ObjectInfo>> {
        self.inner.list(prefix, after)
    }
}

#[async_trait]
impl WriteStore for FaultStore {
    async fn get_range(&self, key: &ObjectKey, range: ByteRange) -> Result<Bytes> {
        match self.take_fault(Op::Get, key) {
            Some(_) => Err(crashed(key)),
            None => self.inner.get_range(key, range).await,
        }
    }

    async fn get(&self, key: &ObjectKey) -> Result<Bytes> {
        match self.take_fault(Op::Get, key) {
            Some(_) => Err(crashed(key)),
            None => self.inner.get(key).await,
        }
    }

    async fn put_create(&self, key: &ObjectKey, body: Bytes) -> Result<()> {
        self.write(key, Write::Create(body)).await
    }

    async fn put_overwrite(&self, key: &ObjectKey, body: Bytes) -> Result<()> {
        self.write(key, Write::Overwrite(body)).await
    }

    async fn delete(&self, key: &ObjectKey) -> Result<()> {
        self.write(key, Write::Delete).await
    }
}
