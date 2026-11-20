use std::fmt;
use std::time::SystemTime;

use async_trait::async_trait;
use bytes::Bytes;
use cosmosearch_core::{ByteRange, ETag};
use futures_util::stream::BoxStream;

use crate::{ObjectKey, Result};

/// What a store reports about one object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectInfo {
    key: ObjectKey,
    size: u64,
    etag: ETag,
    last_modified: Option<SystemTime>,
}

impl ObjectInfo {
    /// Creates the description of an object whose written time is not known.
    pub fn new(key: ObjectKey, size: u64, etag: ETag) -> Self {
        ObjectInfo {
            key,
            size,
            etag,
            last_modified: None,
        }
    }

    /// The same description, with the time the store last wrote the object.
    #[must_use]
    pub fn with_last_modified(mut self, time: SystemTime) -> Self {
        self.last_modified = Some(time);
        self
    }

    /// The object's key.
    pub fn key(&self) -> &ObjectKey {
        &self.key
    }

    /// The object's size in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// The ETag of the object's current version.
    pub fn etag(&self) -> &ETag {
        &self.etag
    }

    /// When the store last wrote the object, by the store's clock, if it says.
    pub fn last_modified(&self) -> Option<SystemTime> {
        self.last_modified
    }
}

/// Read access to a store.
#[async_trait]
pub trait ReadStore: fmt::Debug + Send + Sync {
    /// Reads `range` of the object, only if it still has the ETag `expected`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Stale`](crate::Error::Stale) from the same request if the ETag
    /// differs, and [`Error::NotFound`](crate::Error::NotFound) if the object is gone.
    async fn read_range(&self, key: &ObjectKey, range: ByteRange, expected: &ETag)
    -> Result<Bytes>;

    /// Reports an object's size, ETag and written time.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`](crate::Error::NotFound) if there is no such object.
    async fn head(&self, key: &ObjectKey) -> Result<ObjectInfo>;

    /// Lists the objects under `prefix`, treated as a directory, or every object if it
    /// is `None`. Only keys above `after` are returned. The order is not specified.
    fn list<'a>(
        &'a self,
        prefix: Option<&'a ObjectKey>,
        after: Option<&'a ObjectKey>,
    ) -> BoxStream<'a, Result<ObjectInfo>>;
}

/// Write access to a store, including create-only writes.
#[async_trait]
pub trait WriteStore: ReadStore {
    /// Reads a whole object.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`](crate::Error::NotFound) if there is no such object.
    async fn get(&self, key: &ObjectKey) -> Result<Bytes>;

    /// Reads part of an object that never changes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`](crate::Error::NotFound) if the object is gone, and
    /// [`Error::ShortRead`](crate::Error::ShortRead) if it ends before the range does.
    async fn get_range(&self, key: &ObjectKey, range: ByteRange) -> Result<Bytes>;

    /// Writes the object only if the key is empty.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AlreadyExists`](crate::Error::AlreadyExists) if the key is taken,
    /// and [`Error::OutcomeUnknown`](crate::Error::OutcomeUnknown) if the write may still
    /// be applied.
    async fn put_create(&self, key: &ObjectKey, body: Bytes) -> Result<()>;

    /// Replaces the object atomically.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutcomeUnknown`](crate::Error::OutcomeUnknown) if the write may
    /// still be applied.
    async fn put_overwrite(&self, key: &ObjectKey, body: Bytes) -> Result<()>;

    /// Removes the object. Removing a missing object succeeds.
    ///
    /// # Errors
    ///
    /// Returns an error if the store fails.
    async fn delete(&self, key: &ObjectKey) -> Result<()>;
}
