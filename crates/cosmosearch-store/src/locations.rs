use std::sync::Arc;

use bytes::Bytes;
use cosmosearch_core::{ByteRange, ETag, Location};
use futures_util::stream::BoxStream;
use futures_util::{StreamExt, future};

use crate::{Error, ObjectInfo, ObjectKey, ReadStore, Result, WriteStore};

fn prefix_of(location: &Location) -> Result<Option<ObjectKey>> {
    let key = location.prefix_key();
    if !key.is_empty() {
        return Ok(Some(ObjectKey::new(key)?));
    }
    if location.key().is_empty() {
        return Ok(None);
    }
    Err(Error::InvalidKey {
        key: location.key().to_owned(),
        reason: "a prefix must not be only slashes",
    })
}

/// Where the host's files are.
#[derive(Debug, Clone)]
pub struct DataStore {
    backend: Arc<dyn ReadStore>,
    location: Location,
    prefix: Option<ObjectKey>,
    skipped: Option<ObjectKey>,
}

impl DataStore {
    /// A data location on `backend`, which must be rooted at the location's authority.
    ///
    /// # Errors
    ///
    /// Returns an error if the location's key cannot be used as a prefix.
    pub fn new(backend: impl ReadStore + 'static, location: Location) -> Result<Self> {
        let prefix = prefix_of(&location)?;
        Ok(DataStore {
            backend: Arc::new(backend),
            location,
            prefix,
            skipped: None,
        })
    }

    /// Leaves `metadata` out of listings when it sits inside this data location.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DataInsideMetadata`] if the metadata location holds this one, and
    /// an error if its key cannot be used as a prefix.
    pub fn skipping_metadata(mut self, metadata: &Location) -> Result<Self> {
        if metadata.contains(&self.location) {
            return Err(Error::DataInsideMetadata {
                data: self.location,
                metadata: metadata.clone(),
            });
        }
        self.skipped = if self.location.contains(metadata) {
            prefix_of(metadata)?
        } else {
            None
        };
        Ok(self)
    }

    /// Checks that `key` is a file of this location.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidKey`] for a key outside this location.
    pub fn check_inside(&self, key: &ObjectKey) -> Result<()> {
        let in_data = self
            .prefix
            .as_ref()
            .is_none_or(|prefix| key.is_under(prefix));
        let in_metadata = self
            .skipped
            .as_ref()
            .is_some_and(|skipped| key == skipped || key.is_under(skipped));
        if in_data && !in_metadata {
            Ok(())
        } else {
            Err(Error::InvalidKey {
                key: key.to_string(),
                reason: "the key is outside the data location",
            })
        }
    }

    /// The location of the host's files.
    pub fn location(&self) -> &Location {
        &self.location
    }

    /// Reads `range` of a file, only if it still has the ETag `expected`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Stale`] if the file changed, and [`Error::InvalidKey`] for a key
    /// outside this location.
    pub async fn read_range(
        &self,
        key: &ObjectKey,
        range: ByteRange,
        expected: &ETag,
    ) -> Result<Bytes> {
        self.check_inside(key)?;
        self.backend.read_range(key, range, expected).await
    }

    /// Reports a file's size and ETag.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] if there is no such file, and [`Error::InvalidKey`] for
    /// a key outside this location.
    pub async fn head(&self, key: &ObjectKey) -> Result<ObjectInfo> {
        self.check_inside(key)?;
        self.backend.head(key).await
    }

    /// Lists the files, leaving out the metadata location if it is inside this one.
    pub fn list(&self) -> BoxStream<'_, Result<ObjectInfo>> {
        let skipped = self.skipped.as_ref();
        self.backend
            .list(self.prefix.as_ref(), None)
            .filter(move |item| {
                let keep = match (item, skipped) {
                    (Ok(info), Some(skipped)) => {
                        info.key() != skipped && !info.key().is_under(skipped)
                    }
                    (Ok(_) | Err(_), _) => true,
                };
                future::ready(keep)
            })
            .boxed()
    }
}

/// Where everything cosmosearch derives is kept.
#[derive(Debug, Clone)]
pub struct MetadataStore {
    pub(crate) backend: Arc<dyn WriteStore>,
    location: Location,
    prefix: Option<ObjectKey>,
}

impl MetadataStore {
    /// A metadata location on `backend`, which must be rooted at the location's authority.
    ///
    /// # Errors
    ///
    /// Returns an error if the location's key cannot be used as a prefix.
    pub fn new(backend: impl WriteStore + 'static, location: Location) -> Result<Self> {
        let prefix = prefix_of(&location)?;
        Ok(MetadataStore {
            backend: Arc::new(backend),
            location,
            prefix,
        })
    }

    /// The metadata location.
    pub fn location(&self) -> &Location {
        &self.location
    }

    pub(crate) fn key(&self, name: &str) -> Result<ObjectKey> {
        match &self.prefix {
            Some(prefix) => Ok(prefix.child(name)),
            None => ObjectKey::new(name),
        }
    }

    pub(crate) fn prefix(&self) -> Option<&ObjectKey> {
        self.prefix.as_ref()
    }
}
