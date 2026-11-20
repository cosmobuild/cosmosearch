use bytes::Bytes;
use cosmosearch_core::{
    CommitId, FileSource, IndexId, IndexRecord, Location, Manifest, ManifestObject, ManifestVersion,
};
use futures_util::TryStreamExt;

use crate::commit::{Confirmed, create_confirmed};
use crate::swap::random_id;
use crate::{Error, IndexHandle, MetadataStore, ObjectKey, Result};

const DELETE_CONCURRENCY: usize = 16;

const INDEX: &str = "INDEX";

fn retry_stable_commit_id(id: &IndexId) -> Result<CommitId> {
    Ok(CommitId::new(id.as_str())?)
}

/// How creating an index ended.
#[derive(Debug)]
#[non_exhaustive]
pub enum CreateOutcome {
    /// The index now exists.
    Created(IndexHandle),
    /// The location was not empty, or another create won.
    Conflict,
    /// Call [`MetadataStore::retry_create`] with this id to find out.
    Unknown(IndexId),
}

/// An index that opened successfully, with its newest manifest.
#[derive(Debug)]
pub struct ReadyIndex {
    handle: IndexHandle,
    head: Box<Manifest>,
}

impl ReadyIndex {
    /// The index.
    pub fn handle(&self) -> &IndexHandle {
        &self.handle
    }

    /// Its newest manifest.
    pub fn head(&self) -> &Manifest {
        &self.head
    }

    /// Takes its newest manifest.
    pub fn into_head(self) -> Manifest {
        *self.head
    }

    /// Takes the index and its newest manifest.
    pub fn into_parts(self) -> (IndexHandle, Manifest) {
        (self.handle, *self.head)
    }
}

/// What a metadata location holds.
#[derive(Debug)]
#[non_exhaustive]
pub enum OpenState {
    /// The location is empty.
    NotIndexed,
    /// `INDEX` exists and its manifest resolves.
    Ready(ReadyIndex),
    /// Anything else, such as a partial delete or a crashed create. `destroy` clears it.
    Incomplete,
}

impl MetadataStore {
    fn refuse_data_inside(&self, data: &Location) -> Result<()> {
        if self.location().contains(data) {
            return Err(Error::DataInsideMetadata {
                data: data.clone(),
                metadata: self.location().clone(),
            });
        }
        Ok(())
    }

    async fn is_empty(&self) -> Result<bool> {
        let mut listing = self.backend.list(self.prefix(), None);
        Ok(listing.try_next().await?.is_none())
    }

    async fn holds_only(&self, id: &IndexId) -> Result<bool> {
        let subtree = self.key(id.as_str())?;
        self.backend
            .list(self.prefix(), None)
            .try_fold(true, |only, info| {
                let inside = info.key().is_under(&subtree);
                async move { Ok(only && inside) }
            })
            .await
    }

    async fn read_index_record(&self) -> Result<Option<IndexRecord>> {
        match self.backend.get(&self.key(INDEX)?).await {
            Ok(bytes) => Ok(Some(IndexRecord::decode(&bytes)?)),
            Err(Error::NotFound { .. }) => Ok(None),
            Err(err) => Err(err),
        }
    }

    /// The id of the index this location holds now, read from `INDEX`, or `None` if there is
    /// none.
    ///
    /// # Errors
    ///
    /// Returns an error if the store fails or `INDEX` does not decode.
    pub async fn current_index(&self) -> Result<Option<IndexId>> {
        let record = self.read_index_record().await?;
        Ok(record.map(|record| record.index_id().clone()))
    }

    /// Creates an index for the files at `data`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DataInsideMetadata`] if `destroy` would remove the host's files,
    /// [`Error::CreateOnlyIgnored`] for a store that accepts a second create-only write of
    /// the same key, and an error if the store refuses a write.
    pub async fn create(&self, data: &Location, source: FileSource) -> Result<CreateOutcome> {
        self.refuse_data_inside(data)?;
        if !self.is_empty().await? {
            return Ok(CreateOutcome::Conflict);
        }
        self.write_new_index(IndexId::new(random_id())?, data, source)
            .await
    }

    /// Finishes a create that ended [`CreateOutcome::Unknown`], with the same id.
    ///
    /// # Errors
    ///
    /// As for [`MetadataStore::create`].
    pub async fn retry_create(
        &self,
        id: &IndexId,
        data: &Location,
        source: FileSource,
    ) -> Result<CreateOutcome> {
        self.refuse_data_inside(data)?;
        match self.read_index_record().await? {
            Some(record) if record.index_id() == id => Ok(CreateOutcome::Created(
                IndexHandle::new(self.clone(), id.clone()),
            )),
            None if self.holds_only(id).await? => {
                self.write_new_index(id.clone(), data, source).await
            }
            Some(_) | None => Ok(CreateOutcome::Conflict),
        }
    }

    async fn write_new_index(
        &self,
        id: IndexId,
        data: &Location,
        source: FileSource,
    ) -> Result<CreateOutcome> {
        let handle = IndexHandle::new(self.clone(), id.clone());
        let commit_id = retry_stable_commit_id(&id)?;
        let manifest = Manifest::first(id.clone(), commit_id, data.clone(), source)?;
        let key = handle.manifest_key(ManifestVersion::FIRST)?;
        let body = Bytes::from(manifest.encode()?);
        match self.place_first_manifest(&key, body, &id).await? {
            Confirmed::Mine => {}
            Confirmed::Other | Confirmed::Unknown => return Ok(CreateOutcome::Unknown(id)),
        }

        handle.move_pointer(ManifestVersion::FIRST).await;

        let body = Bytes::from(IndexRecord::new(id.clone()).encode()?);
        let is_mine = |bytes: &[u8]| Ok(IndexRecord::decode(bytes)?.index_id() == &id);
        let index_key = self.key(INDEX)?;
        match create_confirmed(self.backend.as_ref(), &index_key, &body, is_mine).await? {
            Confirmed::Mine => Ok(CreateOutcome::Created(handle)),
            Confirmed::Other => Ok(CreateOutcome::Conflict),
            Confirmed::Unknown => Ok(CreateOutcome::Unknown(id)),
        }
    }

    async fn place_first_manifest(
        &self,
        key: &ObjectKey,
        body: Bytes,
        id: &IndexId,
    ) -> Result<Confirmed> {
        match self.backend.put_create(key, body.clone()).await {
            Ok(()) => self.probe_create_only(key, body).await,
            Err(Error::AlreadyExists { .. }) => self.confirm_first_manifest(key, id).await,
            Err(Error::OutcomeUnknown { .. }) => Ok(Confirmed::Unknown),
            Err(err) => Err(err),
        }
    }

    async fn probe_create_only(&self, key: &ObjectKey, body: Bytes) -> Result<Confirmed> {
        match self.backend.put_create(key, body).await {
            Err(Error::AlreadyExists { .. }) => Ok(Confirmed::Mine),
            Ok(()) => Err(Error::CreateOnlyIgnored {
                location: self.location().clone(),
            }),
            Err(Error::OutcomeUnknown { .. }) => Ok(Confirmed::Unknown),
            Err(err) => Err(err),
        }
    }

    async fn confirm_first_manifest(&self, key: &ObjectKey, id: &IndexId) -> Result<Confirmed> {
        match self.backend.get(key).await {
            Ok(bytes) if ManifestObject::decode(&bytes)?.index_id() == id => Ok(Confirmed::Mine),
            Ok(_) | Err(Error::NotFound { .. }) => Ok(Confirmed::Unknown),
            Err(err) => Err(err),
        }
    }

    /// Reports what this location holds.
    ///
    /// # Errors
    ///
    /// Returns [`Error::WrongDataLocation`] or [`Error::WrongFileSource`] if the index was
    /// created differently, and an error if the store fails.
    pub async fn open(&self, data: &Location, source: FileSource) -> Result<OpenState> {
        let record = match self.read_index_record().await {
            Ok(record) => record,
            Err(Error::Core(err)) if err.is_damaged_data() => return Ok(OpenState::Incomplete),
            Err(err) => return Err(err),
        };
        let Some(record) = record else {
            return Ok(if self.is_empty().await? {
                OpenState::NotIndexed
            } else {
                OpenState::Incomplete
            });
        };
        let handle = IndexHandle::new(self.clone(), record.index_id().clone());
        let head = match handle.head().await {
            Ok(Some(head)) if head.index_id() == handle.id() => head,
            Ok(Some(_) | None) => return Ok(OpenState::Incomplete),
            Err(Error::Core(err)) if err.is_damaged_data() => return Ok(OpenState::Incomplete),
            Err(err) => return Err(err),
        };
        if !head.data_location().same_prefix(data) {
            return Err(Error::WrongDataLocation {
                recorded: head.data_location().clone(),
                given: data.clone(),
            });
        }
        if head.file_source() != source {
            return Err(Error::WrongFileSource {
                recorded: head.file_source(),
                given: source,
            });
        }
        Ok(OpenState::Ready(ReadyIndex {
            handle,
            head: Box::new(head),
        }))
    }

    /// Removes everything at this location, from any state. Safe to repeat.
    ///
    /// # Errors
    ///
    /// Returns an error if the store fails. Calling again finishes the work.
    pub async fn destroy(&self) -> Result<()> {
        let backend = self.backend.as_ref();
        backend.delete(&self.key(INDEX)?).await?;
        backend
            .list(self.prefix(), None)
            .map_ok(|info| async move { backend.delete(info.key()).await })
            .try_buffer_unordered(DELETE_CONCURRENCY)
            .try_collect::<()>()
            .await
    }
}
