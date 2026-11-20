use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, SystemTime};

use cosmosearch_core::{IndexId, ManifestVersion, SegmentId};
use futures_util::{StreamExt, TryStreamExt, stream};
use tracing::{info, warn};

use crate::retention::{Content, Retained, Survey, older_than};
use crate::{IndexHandle, MetadataStore, ObjectInfo, ObjectKey, Result};

const DELETE_CONCURRENCY: usize = 16;
/// Bounds what one run holds in memory; the rest waits for the next run.
const MAX_ORPHAN_SEGMENTS: usize = 4096;
/// A segment has a handful of files. One run tracks at most this many of an unreferenced
/// segment's files and leaves the rest for the next run.
const MAX_FILES_PER_SEGMENT: usize = 64;
const MAX_OTHER_INDEXES: usize = 64;
/// `INDEX` is read again before each batch of deletes in another index's subtree.
const DELETE_BATCH: usize = 256;

/// How long garbage collection keeps what may still be needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CollectionSettings {
    retention: Duration,
    grace: Duration,
}

impl CollectionSettings {
    /// Keeps a manifest for `retention` after it is superseded, and any unreferenced object
    /// until it is older than `grace`.
    pub fn new(retention: Duration, grace: Duration) -> Self {
        CollectionSettings { retention, grace }
    }

    /// How long a superseded manifest is kept before it becomes a stub.
    pub fn retention(&self) -> Duration {
        self.retention
    }

    /// How old an unreferenced object must be before it is removed.
    pub fn grace(&self) -> Duration {
        self.grace
    }
}

/// What one garbage collection run did and left.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CollectionReport {
    manifests_stubbed: u64,
    manifests_retained: u64,
    manifests_unreadable: u64,
    segments_deleted: u64,
    bytes_deleted: u64,
    segments_young: u64,
    segments_deferred: u64,
    sweep_skipped: bool,
    indexes_removed: u64,
    indexes_young: u64,
}

impl CollectionReport {
    /// Expired manifests overwritten with their stub.
    pub fn manifests_stubbed(&self) -> u64 {
        self.manifests_stubbed
    }

    /// Manifests kept whole: the newest and those inside the retention time.
    pub fn manifests_retained(&self) -> u64 {
        self.manifests_retained
    }

    /// Manifests left as they are because they are missing or do not decode.
    pub fn manifests_unreadable(&self) -> u64 {
        self.manifests_unreadable
    }

    /// Segments no retained manifest references, deleted.
    pub fn segments_deleted(&self) -> u64 {
        self.segments_deleted
    }

    /// The bytes of the deleted segments.
    pub fn bytes_deleted(&self) -> u64 {
        self.bytes_deleted
    }

    /// Unreferenced segments left because they are inside the grace period, or the store
    /// gave no written time.
    pub fn segments_young(&self) -> u64 {
        self.segments_young
    }

    /// Unreferenced segments left for a later run, beyond what one run holds.
    pub fn segments_deferred(&self) -> u64 {
        self.segments_deferred
    }

    /// Whether no segment was deleted because what a retained manifest references was not
    /// known for sure.
    pub fn sweep_skipped(&self) -> bool {
        self.sweep_skipped
    }

    /// Subtrees of other index ids removed.
    pub fn indexes_removed(&self) -> u64 {
        self.indexes_removed
    }

    /// Subtrees of other index ids left, wholly or in part, because they hold an object
    /// inside the grace period.
    pub fn indexes_young(&self) -> u64 {
        self.indexes_young
    }
}

#[derive(Debug, Default)]
struct Dated {
    newest: Option<SystemTime>,
    undated: bool,
}

impl Dated {
    fn see(&mut self, object: &ObjectInfo) {
        match object.last_modified() {
            Some(time) => self.newest = self.newest.max(Some(time)),
            None => self.undated = true,
        }
    }

    fn older_than(&self, age: Duration, now: SystemTime) -> bool {
        !self.undated && self.newest.is_some_and(|time| older_than(time, age, now))
    }
}

#[derive(Debug, Default)]
struct Orphan {
    dated: Dated,
    bytes: u64,
    files: Vec<ObjectKey>,
}

#[derive(Debug, Default)]
struct Scan {
    orphans: BTreeMap<SegmentId, Orphan>,
    deferred: BTreeSet<SegmentId>,
    others: BTreeMap<IndexId, Dated>,
}

impl Scan {
    /// Notes one object. Under the current index only segments matter, and only when what the
    /// retained manifests reference is known.
    fn see(
        &mut self,
        relative: &str,
        object: &ObjectInfo,
        current: Option<&IndexId>,
        retained: Option<&Retained>,
    ) {
        let Some((first, rest)) = relative.split_once('/') else {
            return;
        };
        if current.is_some_and(|id| id.as_str() == first) {
            if let Some(retained) = retained {
                self.see_own(rest, object, retained);
            }
            return;
        }
        let Ok(id) = IndexId::new(first) else {
            return;
        };
        if !self.others.contains_key(&id) && self.others.len() >= MAX_OTHER_INDEXES {
            return;
        }
        self.others.entry(id).or_default().see(object);
    }

    fn see_own(&mut self, rest: &str, object: &ObjectInfo, retained: &Retained) {
        let Some((segment, _)) = rest
            .strip_prefix("segments/")
            .and_then(|rest| rest.split_once('/'))
        else {
            return;
        };
        let Ok(segment) = SegmentId::new(segment) else {
            return;
        };
        if retained.segments().contains_key(&segment) {
            return;
        }
        if !self.orphans.contains_key(&segment) && self.orphans.len() >= MAX_ORPHAN_SEGMENTS {
            if self.deferred.len() < MAX_ORPHAN_SEGMENTS {
                self.deferred.insert(segment);
            }
            return;
        }
        let orphan = self.orphans.entry(segment).or_default();
        orphan.dated.see(object);
        if orphan.files.len() < MAX_FILES_PER_SEGMENT {
            orphan.bytes = orphan.bytes.saturating_add(object.size());
            orphan.files.push(object.key().clone());
        }
    }
}

fn count(len: usize) -> u64 {
    u64::try_from(len).unwrap_or(u64::MAX)
}

impl MetadataStore {
    /// Collects the garbage at this location, at the host's time `now`. Safe to repeat, to run
    /// from several processes at once and to cancel.
    ///
    /// For the index `INDEX` names, a manifest superseded longer than the retention time ago
    /// is overwritten with its stub, never deleted, and the newest is never a stub. A segment
    /// is deleted once no retained manifest references it and its newest file is older than
    /// the grace period; the head is read again just before, so a commit that landed during
    /// the run keeps its segments. A subtree of another index id is removed only when its
    /// newest object is older than the grace period, and `INDEX` is read again before each
    /// batch of deletes in it. Anything whose age or references are not known is kept, and
    /// a location with no `INDEX` is left as it is: nothing there says which subtree is
    /// current, so finishing a half-done delete is `destroy`'s to do.
    ///
    /// # Errors
    ///
    /// Returns an error if the store fails or `INDEX` does not decode. What was done before
    /// the failure stands, and the next run finishes the rest.
    pub async fn collect_garbage(
        &self,
        settings: &CollectionSettings,
        now: SystemTime,
    ) -> Result<CollectionReport> {
        let mut report = CollectionReport::default();
        let Some(current) = self.current_index().await? else {
            return Ok(report);
        };
        let index = Some(IndexHandle::new(self.clone(), current));
        let survey = match &index {
            Some(index) => index.survey(settings.retention, now).await?,
            None => None,
        };
        if let (Some(index), Some(survey)) = (&index, &survey) {
            report.manifests_retained = survey.kept;
            report.manifests_unreadable = count(survey.unreadable.len());
            if let Some(newest_expired) = survey.newest_expired {
                let stubbing = index.stub_through(newest_expired).await?;
                report.manifests_stubbed = stubbing.stubbed;
                report.manifests_unreadable = report
                    .manifests_unreadable
                    .saturating_add(stubbing.unreadable);
            }
        }
        let retained = survey
            .as_ref()
            .filter(|survey| survey.unreadable.is_empty())
            .map(|survey| &survey.retained);
        report.sweep_skipped = index.is_some() && retained.is_none();
        let scan = self
            .scan(index.as_ref().map(IndexHandle::id), retained)
            .await?;
        report.segments_deferred = count(scan.deferred.len());
        if let (Some(index), Some(survey)) = (&index, &survey)
            && retained.is_some()
        {
            sweep_segments(index, survey, scan.orphans, settings, now, &mut report).await?;
        }
        for (id, dated) in scan.others {
            if !dated.older_than(settings.grace, now) {
                report.indexes_young = report.indexes_young.saturating_add(1);
                continue;
            }
            match self.remove_subtree(&id, settings.grace, now).await? {
                Removal::Removed => {
                    report.indexes_removed = report.indexes_removed.saturating_add(1);
                }
                Removal::Young => report.indexes_young = report.indexes_young.saturating_add(1),
                Removal::Live => {}
            }
        }
        info!(
            stubbed = report.manifests_stubbed,
            segments = report.segments_deleted,
            bytes = report.bytes_deleted,
            indexes = report.indexes_removed,
            "garbage collected"
        );
        Ok(report)
    }

    async fn scan(&self, current: Option<&IndexId>, retained: Option<&Retained>) -> Result<Scan> {
        let mut scan = Scan::default();
        let mut listing = self.backend.list(self.prefix(), None);
        while let Some(object) = listing.try_next().await? {
            if let Some(relative) = self.relative(object.key()) {
                scan.see(relative, &object, current, retained);
            }
        }
        Ok(scan)
    }

    fn relative<'k>(&self, key: &'k ObjectKey) -> Option<&'k str> {
        match self.prefix() {
            Some(prefix) => key
                .as_str()
                .strip_prefix(prefix.as_str())
                .and_then(|rest| rest.strip_prefix('/')),
            None => Some(key.as_str()),
        }
    }

    async fn remove_subtree(
        &self,
        id: &IndexId,
        grace: Duration,
        now: SystemTime,
    ) -> Result<Removal> {
        let prefix = self.key(id.as_str())?;
        let backend = self.backend.as_ref();
        let mut listing = backend.list(Some(&prefix), None);
        let mut batch = Vec::with_capacity(DELETE_BATCH);
        let mut young = false;
        loop {
            let next = listing.try_next().await?;
            if let Some(object) = &next {
                if object
                    .last_modified()
                    .is_some_and(|time| older_than(time, grace, now))
                {
                    batch.push(object.key().clone());
                } else {
                    young = true;
                }
            }
            let full = batch.len() >= DELETE_BATCH;
            if full || (next.is_none() && !batch.is_empty()) {
                let current = self.current_index().await?;
                if current.as_ref().is_none_or(|current| current == id) {
                    warn!(index = %id, "an index thought replaced is current or `INDEX` is gone, so it is kept");
                    return Ok(Removal::Live);
                }
                stream::iter(batch.drain(..))
                    .map(|key| async move { backend.delete(&key).await })
                    .buffer_unordered(DELETE_CONCURRENCY)
                    .try_collect::<()>()
                    .await?;
            }
            if next.is_none() {
                return Ok(if young {
                    Removal::Young
                } else {
                    Removal::Removed
                });
            }
        }
    }
}

enum Removal {
    Removed,
    Young,
    Live,
}

async fn sweep_segments(
    index: &IndexHandle,
    survey: &Survey,
    orphans: BTreeMap<SegmentId, Orphan>,
    settings: &CollectionSettings,
    now: SystemTime,
    report: &mut CollectionReport,
) -> Result<()> {
    let (mut old, young): (BTreeMap<_, _>, BTreeMap<_, _>) = orphans
        .into_iter()
        .partition(|(_, orphan)| orphan.dated.older_than(settings.grace, now));
    report.segments_young = count(young.len());
    if old.is_empty() {
        return Ok(());
    }
    let Some(landed) = landed_since(index, survey.head).await? else {
        report.sweep_skipped = true;
        return Ok(());
    };
    old.retain(|id, _| !landed.segments().contains_key(id));
    let deleted = count(old.len());
    let bytes = old
        .values()
        .fold(0u64, |sum, orphan| sum.saturating_add(orphan.bytes));
    let backend = index.backend();
    stream::iter(old.into_values().flat_map(|orphan| orphan.files))
        .map(|key| async move { backend.delete(&key).await })
        .buffer_unordered(DELETE_CONCURRENCY)
        .try_collect::<()>()
        .await?;
    report.segments_deleted = deleted;
    report.bytes_deleted = bytes;
    Ok(())
}

/// What the manifests committed after `surveyed` reference, read just before deleting, or
/// `None` if one of them cannot be read.
async fn landed_since(index: &IndexHandle, surveyed: ManifestVersion) -> Result<Option<Retained>> {
    let mut landed = Retained::default();
    let Some(head) = index.head().await? else {
        return Ok(None);
    };
    let mut version = surveyed;
    while version < head.version() {
        version = version.next()?;
        match index.read_content(version).await? {
            Content::Manifest(manifest) => landed.add(&manifest),
            Content::Stub | Content::Unreadable => return Ok(None),
        }
    }
    landed.add(&head);
    Ok(Some(landed))
}
