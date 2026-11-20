use std::sync::Arc;

use bytes::Bytes;
use cosmosearch_core::ByteRange;
use futures_util::TryStreamExt;

use crate::{Error, ObjectInfo, ObjectKey, Result, WriteStore};

/// The part of an index's store lent to one plug-in, such as a cache. Keys are relative to
/// the scope, so the plug-in reaches nothing outside it. Its objects are removed with the
/// index.
#[derive(Debug, Clone)]
pub struct ScopedStore {
    backend: Arc<dyn WriteStore>,
    root: ObjectKey,
}

impl ScopedStore {
    pub(crate) fn new(backend: Arc<dyn WriteStore>, root: ObjectKey) -> Self {
        ScopedStore { backend, root }
    }

    fn key(&self, key: &ObjectKey) -> ObjectKey {
        self.root.child(key.as_str())
    }

    /// Reads part of an object that never changes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] if the object is gone, and [`Error::ShortRead`] if it
    /// ends before the range does.
    pub async fn get_range(&self, key: &ObjectKey, range: ByteRange) -> Result<Bytes> {
        self.backend.get_range(&self.key(key), range).await
    }

    /// Writes the object only if the key is empty.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AlreadyExists`] if the key is taken, and [`Error::OutcomeUnknown`]
    /// if the write may still be applied.
    pub async fn put_create(&self, key: &ObjectKey, body: Bytes) -> Result<()> {
        self.backend.put_create(&self.key(key), body).await
    }

    /// Removes the object. Removing a missing object succeeds.
    ///
    /// # Errors
    ///
    /// Returns an error if the store fails.
    pub async fn delete(&self, key: &ObjectKey) -> Result<()> {
        self.backend.delete(&self.key(key)).await
    }

    /// Lists the objects under `prefix`, treated as a directory, or every object in the
    /// scope if it is `None`. Listed keys are relative to the scope. The order is not
    /// specified.
    ///
    /// # Errors
    ///
    /// Returns an error if the store fails, or [`Error::InvalidKey`] if it lists a key
    /// outside the scope.
    pub async fn list(&self, prefix: Option<&ObjectKey>) -> Result<Vec<ObjectInfo>> {
        let prefix = prefix.map_or_else(|| self.root.clone(), |prefix| self.key(prefix));
        self.backend
            .list(Some(&prefix), None)
            .and_then(|object| async move { self.relative(&object) })
            .try_collect()
            .await
    }

    fn relative(&self, object: &ObjectInfo) -> Result<ObjectInfo> {
        let key = object
            .key()
            .as_str()
            .strip_prefix(self.root.as_str())
            .and_then(|rest| rest.strip_prefix('/'))
            .ok_or_else(|| Error::InvalidKey {
                key: object.key().to_string(),
                reason: "a listing gave a key outside its prefix",
            })?;
        let relative = ObjectInfo::new(ObjectKey::new(key)?, object.size(), object.etag().clone());
        Ok(match object.last_modified() {
            Some(time) => relative.with_last_modified(time),
            None => relative,
        })
    }
}
