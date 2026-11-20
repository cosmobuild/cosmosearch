use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, SystemTime};

use bytes::Bytes;
use cosmosearch_core::{
    Manifest, ManifestObject, ManifestVersion, ModelId, SegmentEntry, SegmentId,
};
use futures_util::{StreamExt, TryStreamExt, stream};
use tracing::warn;

use crate::{Error, IndexHandle, Result};

const SLOT_CONCURRENCY: usize = 8;

/// What the retained manifests of an index reference. The newest manifest is retained, and so
/// is every other one superseded no longer than the retention time ago.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Retained {
    segments: BTreeMap<SegmentId, SegmentEntry>,
    model_ids: BTreeSet<ModelId>,
}

impl Retained {
    pub(crate) fn add(&mut self, manifest: &Manifest) {
        for shard in manifest.shards().values() {
            for segment in shard.segments() {
                self.segments
                    .entry(segment.id().clone())
                    .or_insert_with(|| segment.clone());
            }
        }
        self.model_ids.extend(manifest.model_ids().iter().cloned());
    }

    /// Every segment a retained manifest references.
    pub fn segments(&self) -> &BTreeMap<SegmentId, SegmentEntry> {
        &self.segments
    }

    /// The model ids of the retained manifests.
    pub fn model_ids(&self) -> &BTreeSet<ModelId> {
        &self.model_ids
    }
}

pub(crate) enum Content {
    Manifest(Box<Manifest>),
    Stub,
    Unreadable,
}

struct Slot {
    version: ManifestVersion,
    written: Option<SystemTime>,
    content: Content,
}

/// The manifests of an index, read from the newest down to the first expired one.
pub(crate) struct Survey {
    pub(crate) head: ManifestVersion,
    pub(crate) retained: Retained,
    pub(crate) kept: u64,
    pub(crate) unreadable: Vec<ManifestVersion>,
    pub(crate) newest_expired: Option<ManifestVersion>,
}

#[derive(Debug, Default)]
pub(crate) struct Stubbing {
    pub(crate) stubbed: u64,
    pub(crate) unreadable: u64,
}

pub(crate) fn older_than(written: SystemTime, age: Duration, now: SystemTime) -> bool {
    now.duration_since(written)
        .is_ok_and(|elapsed| elapsed > age)
}

impl IndexHandle {
    /// What the retained manifests reference, at `now`, or `None` if no manifest resolves.
    /// A manifest is retained until it was superseded longer than `retention` ago, by the
    /// store's written time of the version after it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ManifestUnreadable`] if a retained manifest is missing or does not
    /// decode, and an error if the store fails.
    pub async fn retained(&self, retention: Duration, now: SystemTime) -> Result<Option<Retained>> {
        let Some(survey) = self.survey(retention, now).await? else {
            return Ok(None);
        };
        if let Some(version) = survey.unreadable.first() {
            return Err(Error::ManifestUnreadable { version: *version });
        }
        Ok(Some(survey.retained))
    }

    /// Reads from the head down until a stub or an expired manifest. A stub is written only
    /// over an expired manifest, and every manifest older than an expired one is expired too.
    pub(crate) async fn survey(
        &self,
        retention: Duration,
        now: SystemTime,
    ) -> Result<Option<Survey>> {
        let Some(head) = self.head().await? else {
            return Ok(None);
        };
        let mut superseded = self.written(head.version()).await?;
        let mut survey = Survey {
            head: head.version(),
            retained: Retained::default(),
            kept: 1,
            unreadable: Vec::new(),
            newest_expired: None,
        };
        survey.retained.add(&head);
        let older = (1..head.version().get()).rev();
        let mut slots = stream::iter(older)
            .map(|version| async move { self.read_slot(ManifestVersion::new(version)?).await })
            .buffered(SLOT_CONCURRENCY);
        while let Some(slot) = slots.try_next().await? {
            let expired = superseded.is_some_and(|time| older_than(time, retention, now));
            match slot.content {
                Content::Stub => break,
                Content::Manifest(_) | Content::Unreadable if expired => {
                    survey.newest_expired = Some(slot.version);
                    break;
                }
                Content::Manifest(manifest) => {
                    survey.retained.add(&manifest);
                    survey.kept = survey.kept.saturating_add(1);
                }
                Content::Unreadable => survey.unreadable.push(slot.version),
            }
            superseded = slot.written;
        }
        Ok(Some(survey))
    }

    /// Overwrites every expired manifest up to `newest_expired` with its stub, oldest first,
    /// so stubs always fill the versions from the first one up.
    pub(crate) async fn stub_through(&self, newest_expired: ManifestVersion) -> Result<Stubbing> {
        let mut done = Stubbing::default();
        let mut version = self.lowest_unstubbed(newest_expired).await?;
        loop {
            match self.read_content(version).await? {
                Content::Manifest(manifest) => {
                    let body = Bytes::from(manifest.stub().encode()?);
                    let key = self.manifest_key(version)?;
                    self.backend().put_overwrite(&key, body).await?;
                    done.stubbed = done.stubbed.saturating_add(1);
                }
                Content::Stub => {}
                Content::Unreadable => {
                    warn!(
                        version = version.get(),
                        "an expired manifest cannot be read, so it is left as it is"
                    );
                    done.unreadable = done.unreadable.saturating_add(1);
                }
            }
            if version >= newest_expired {
                return Ok(done);
            }
            version = version.next()?;
        }
    }

    async fn lowest_unstubbed(&self, newest_expired: ManifestVersion) -> Result<ManifestVersion> {
        let mut low = ManifestVersion::FIRST.get();
        let mut high = newest_expired.get();
        while low < high {
            let middle = low.saturating_add(high.saturating_sub(low) / 2);
            match self.read_content(ManifestVersion::new(middle)?).await? {
                Content::Stub => low = middle.saturating_add(1),
                Content::Manifest(_) | Content::Unreadable => high = middle,
            }
        }
        Ok(ManifestVersion::new(low)?)
    }

    async fn written(&self, version: ManifestVersion) -> Result<Option<SystemTime>> {
        match self.backend().head(&self.manifest_key(version)?).await {
            Ok(info) => Ok(info.last_modified()),
            Err(Error::NotFound { .. }) => Ok(None),
            Err(err) => Err(err),
        }
    }

    async fn read_slot(&self, version: ManifestVersion) -> Result<Slot> {
        let written = self.written(version).await?;
        let content = self.read_content(version).await?;
        Ok(Slot {
            version,
            written,
            content,
        })
    }

    pub(crate) async fn read_content(&self, version: ManifestVersion) -> Result<Content> {
        let bytes = match self.backend().get(&self.manifest_key(version)?).await {
            Ok(bytes) => bytes,
            Err(Error::NotFound { .. }) => {
                warn!(version = version.get(), "a manifest version is missing");
                return Ok(Content::Unreadable);
            }
            Err(err) => return Err(err),
        };
        let object = match ManifestObject::decode(&bytes) {
            Ok(object) if object.index_id() == self.id() => object,
            Ok(_) => {
                warn!(
                    version = version.get(),
                    "a manifest belongs to another index"
                );
                return Ok(Content::Unreadable);
            }
            Err(err) => {
                warn!(version = version.get(), error = %err, "a manifest does not decode");
                return Ok(Content::Unreadable);
            }
        };
        Ok(match object.into_manifest() {
            Some(manifest) if manifest.version() == version => {
                Content::Manifest(Box::new(manifest))
            }
            Some(_) => Content::Unreadable,
            None => Content::Stub,
        })
    }
}
