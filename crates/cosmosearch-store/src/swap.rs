use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use cosmosearch_core::{
    ByteRange, CommitId, IndexCause, IndexError, IndexId, Location, Manifest, ManifestObject,
    ManifestPointer, ManifestVersion, SegmentFileName, SegmentFiles, SegmentId,
};
use futures_util::TryStreamExt;
use tracing::warn;
use uuid::Uuid;

use crate::commit::{Confirmed, create_confirmed};
use crate::{Error, MetadataStore, ObjectKey, Result, ScopedStore, WriteStore};

const MAX_ROLL_FORWARD: usize = 32;

const POINTER: &str = "MANIFEST";
const MANIFESTS: &str = "manifests";
const SEGMENTS: &str = "segments";

/// A fresh id for one commit attempt.
///
/// # Errors
///
/// Returns an error if the generated id is not valid, which is a bug.
pub fn new_commit_id() -> Result<CommitId> {
    Ok(CommitId::new(random_id())?)
}

pub(crate) fn random_id() -> String {
    Uuid::new_v4().simple().to_string()
}

#[async_trait]
impl SegmentFiles for IndexHandle {
    async fn read_range(
        &self,
        segment: &SegmentId,
        name: &SegmentFileName,
        range: ByteRange,
    ) -> std::result::Result<Vec<u8>, IndexError> {
        let key = self
            .segment_key(segment, name)
            .map_err(|err| IndexError::Read(IndexCause::new(err)))?;
        match self.store.backend.get_range(&key, range).await {
            Ok(bytes) => Ok(bytes.to_vec()),
            Err(Error::NotFound { .. }) => Err(IndexError::MissingSegment {
                segment: segment.clone(),
            }),
            Err(err) => Err(IndexError::Read(IndexCause::new(err))),
        }
    }
}

/// How a commit ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CommitOutcome {
    /// The manifest is the new version.
    Committed(ManifestVersion),
    /// Another writer holds that version.
    Conflict,
    /// The write may still land.
    Unknown,
}

/// One index at a metadata location, under its own id.
#[derive(Debug, Clone)]
pub struct IndexHandle {
    store: MetadataStore,
    id: IndexId,
}

impl IndexHandle {
    pub(crate) fn new(store: MetadataStore, id: IndexId) -> Self {
        IndexHandle { store, id }
    }

    /// This index's id.
    pub fn id(&self) -> &IndexId {
        &self.id
    }

    /// The metadata location that holds this index.
    pub fn metadata_location(&self) -> &Location {
        self.store.location()
    }

    fn key(&self, name: &str) -> Result<ObjectKey> {
        self.store.key(&format!("{}/{name}", self.id))
    }

    pub(crate) fn backend(&self) -> &dyn WriteStore {
        self.store.backend.as_ref()
    }

    /// The part of this index's store lent to the plug-in `name`, such as a cache, under
    /// this index's id. Its objects are removed with the index.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidKey`] unless `name` is lowercase ASCII letters, digits and
    /// hyphens, and not a name the index itself uses.
    pub fn scope(&self, name: &str) -> Result<ScopedStore> {
        let reason = if name.is_empty() {
            Some("a scope name must not be empty")
        } else if !name
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
        {
            Some("a scope name holds only lowercase ASCII letters, digits and hyphens")
        } else if [MANIFESTS, SEGMENTS].contains(&name) {
            Some("the index itself uses this name")
        } else {
            None
        };
        if let Some(reason) = reason {
            return Err(Error::InvalidKey {
                key: name.to_owned(),
                reason,
            });
        }
        Ok(ScopedStore::new(
            Arc::clone(&self.store.backend),
            self.key(name)?,
        ))
    }

    pub(crate) fn manifest_key(&self, version: ManifestVersion) -> Result<ObjectKey> {
        self.key(&format!("{MANIFESTS}/{}", version.file_name()))
    }

    fn segment_key(&self, segment: &SegmentId, name: &SegmentFileName) -> Result<ObjectKey> {
        self.key(&format!("{SEGMENTS}/{segment}/{name}"))
    }

    /// Writes one file of a new segment. Segments are immutable, so the key must be free.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AlreadyExists`] if the file exists, and an error if the store fails.
    pub async fn write_segment_file(
        &self,
        segment: &SegmentId,
        name: &SegmentFileName,
        body: Bytes,
    ) -> Result<()> {
        let key = self.segment_key(segment, name)?;
        self.store.backend.put_create(&key, body).await
    }

    /// Whether one file of a segment is in the store.
    ///
    /// # Errors
    ///
    /// Returns an error if the store fails.
    pub async fn segment_file_exists(
        &self,
        segment: &SegmentId,
        name: &SegmentFileName,
    ) -> Result<bool> {
        match self
            .store
            .backend
            .head(&self.segment_key(segment, name)?)
            .await
        {
            Ok(_) => Ok(true),
            Err(Error::NotFound { .. }) => Ok(false),
            Err(err) => Err(err),
        }
    }

    /// Reads one file of a segment.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] if the file is missing, and an error if the store fails.
    pub async fn read_segment_file(
        &self,
        segment: &SegmentId,
        name: &SegmentFileName,
    ) -> Result<Bytes> {
        self.store
            .backend
            .get(&self.segment_key(segment, name)?)
            .await
    }

    /// Finds the newest manifest, or `None` if no manifest resolves.
    ///
    /// # Errors
    ///
    /// Returns an error if the store fails or the newest manifest does not decode.
    pub async fn head(&self) -> Result<Option<Manifest>> {
        let start = match self.read_pointer().await? {
            Some(version) if self.slot_exists(version).await? => Some(version),
            Some(_) | None => None,
        };
        let head = match start {
            Some(version) => Some(self.roll_forward(version).await?),
            None => self.highest_listed(None).await?,
        };
        let Some(head) = head else {
            return Ok(None);
        };
        let key = self.manifest_key(head)?;
        match self.store.backend.get(&key).await {
            Ok(bytes) => Ok(ManifestObject::decode(&bytes)?.into_manifest()),
            Err(Error::NotFound { .. }) => Ok(None),
            Err(err) => Err(err),
        }
    }

    async fn read_pointer(&self) -> Result<Option<ManifestVersion>> {
        let bytes = match self.store.backend.get(&self.key(POINTER)?).await {
            Ok(bytes) => bytes,
            Err(Error::NotFound { .. }) => return Ok(None),
            Err(err) => return Err(err),
        };
        match ManifestPointer::decode(&bytes) {
            Ok(pointer) => Ok(Some(pointer.version())),
            Err(err) => {
                warn!(error = %err, "ignoring an unreadable manifest pointer");
                Ok(None)
            }
        }
    }

    async fn slot_exists(&self, version: ManifestVersion) -> Result<bool> {
        match self.store.backend.head(&self.manifest_key(version)?).await {
            Ok(_) => Ok(true),
            Err(Error::NotFound { .. }) => Ok(false),
            Err(err) => Err(err),
        }
    }

    async fn roll_forward(&self, from: ManifestVersion) -> Result<ManifestVersion> {
        let mut current = from;
        for _ in 0..MAX_ROLL_FORWARD {
            let next = current.next()?;
            if !self.slot_exists(next).await? {
                return Ok(current);
            }
            current = next;
        }
        Ok(self.highest_listed(Some(current)).await?.unwrap_or(current))
    }

    async fn highest_listed(
        &self,
        after: Option<ManifestVersion>,
    ) -> Result<Option<ManifestVersion>> {
        let prefix = self.key(MANIFESTS)?;
        let after_key = after
            .map(|version| self.manifest_key(version))
            .transpose()?;
        self.store
            .backend
            .list(Some(&prefix), after_key.as_ref())
            .try_fold(after, |highest, info| async move {
                let found = ManifestVersion::from_file_name(info.key().file_name()).ok();
                Ok(highest.max(found))
            })
            .await
    }

    /// Publishes `manifest` by a create-only write of its version's slot.
    ///
    /// # Errors
    ///
    /// Returns [`Error::WrongIndex`] for a manifest of another index,
    /// [`Error::ParentMissing`] if the previous version does not exist, and an error if the
    /// store definitely refused the write.
    pub async fn commit(&self, manifest: &Manifest) -> Result<CommitOutcome> {
        if manifest.index_id() != &self.id {
            return Err(Error::WrongIndex {
                expected: self.id.clone(),
                found: manifest.index_id().clone(),
            });
        }
        if let Some(parent) = manifest.version().previous()
            && !self.slot_exists(parent).await?
        {
            return Err(Error::ParentMissing { version: parent });
        }
        let key = self.manifest_key(manifest.version())?;
        let body = Bytes::from(manifest.encode()?);
        let is_mine =
            |bytes: &[u8]| Ok(ManifestObject::decode(bytes)?.commit_id() == manifest.commit_id());
        match create_confirmed(self.store.backend.as_ref(), &key, &body, is_mine).await? {
            Confirmed::Mine => {
                self.move_pointer(manifest.version()).await;
                Ok(CommitOutcome::Committed(manifest.version()))
            }
            Confirmed::Other => Ok(CommitOutcome::Conflict),
            Confirmed::Unknown => Ok(CommitOutcome::Unknown),
        }
    }

    pub(crate) async fn move_pointer(&self, version: ManifestVersion) {
        if let Err(err) = self.try_move_pointer(version).await {
            warn!(error = %err, version = version.get(), "could not move the manifest pointer");
        }
    }

    async fn try_move_pointer(&self, version: ManifestVersion) -> Result<()> {
        let body = Bytes::from(ManifestPointer::new(version).encode()?);
        self.store
            .backend
            .put_overwrite(&self.key(POINTER)?, body)
            .await
    }
}
