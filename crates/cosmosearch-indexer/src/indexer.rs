use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroU64;

use bytes::Bytes;
use cosmosearch_core::{
    Attributes, BuiltBy, ETag, FileSource, KeyRange, Level, Manifest, ManifestVersion, MediaType,
    SegmentEntry, SegmentFile, SegmentFileName, SegmentId, ShardEntry, ShardId,
};
use cosmosearch_format::{FileEntry, FileReason, FileStatus, FileTable};
use cosmosearch_store::{CommitOutcome, DataStore, IndexHandle, ObjectKey, new_commit_id};
use futures_util::{StreamExt, TryStreamExt, stream};

use crate::{Applied, AttributeChange, ChangeOutcome, Error, FileChange, FileUpsert, Result};

const HEAD_CONCURRENCY: usize = 16;
pub(crate) const MAX_COMMIT_ATTEMPTS: usize = 8;
/// The source bytes a shard may hold before it splits, unless set. Shards are meant to hold
/// 100 MB to 1 GB, and splitting at the top of that range keeps segment files large enough
/// to read efficiently.
const DEFAULT_SHARD_BYTES: NonZeroU64 = match NonZeroU64::new(1024 * 1024 * 1024) {
    Some(bytes) => bytes,
    None => NonZeroU64::MIN,
};
const CUT_WINDOW_LOW_PERCENT: u64 = 40;
const CUT_WINDOW_HIGH_PERCENT: u64 = 60;
const PERCENT: u64 = 100;

/// What one commit changes in one shard.
pub(crate) struct ShardUpdate {
    pub(crate) table: FileTable,
    pub(crate) add: Vec<SegmentEntry>,
    pub(crate) remove: BTreeSet<SegmentId>,
}

impl ShardUpdate {
    fn table_only(table: FileTable) -> Self {
        ShardUpdate {
            table,
            add: Vec::new(),
            remove: BTreeSet::new(),
        }
    }
}

/// Records file changes in an index, each call as one commit.
#[derive(Debug, Clone)]
pub struct Indexer {
    index: IndexHandle,
    data: DataStore,
    shard_bytes: NonZeroU64,
}

#[derive(Debug)]
struct Incoming {
    key: String,
    size: u64,
    etag: ETag,
    display_name: Option<String>,
    media_type: Option<MediaType>,
    attributes: Attributes,
    expected: Expected,
}

/// The state a file must be in for a step to apply to it.
#[derive(Debug, Clone, Copy)]
enum Expected {
    Any,
    At(ManifestVersion),
    Absent,
}

impl From<Option<ManifestVersion>> for Expected {
    fn from(version: Option<ManifestVersion>) -> Self {
        version.map_or(Expected::Any, Expected::At)
    }
}

#[derive(Debug)]
enum Step {
    Upsert(Incoming),
    Remove {
        key: String,
        expected: Option<ManifestVersion>,
    },
    Retry {
        key: String,
        expected: Option<ManifestVersion>,
    },
    Describe(AttributeChange),
    Record {
        key: String,
        planned: ManifestVersion,
        reason: FileReason,
    },
    Skip,
    Missing,
}

impl Step {
    fn key(&self) -> Option<&str> {
        match self {
            Step::Upsert(incoming) => Some(&incoming.key),
            Step::Describe(change) => Some(&change.key),
            Step::Remove { key, .. } | Step::Retry { key, .. } | Step::Record { key, .. } => {
                Some(key)
            }
            Step::Skip | Step::Missing => None,
        }
    }
}

impl Indexer {
    /// An indexer for `index`, whose files are at `data`. A metadata location inside the data
    /// location is left out of `data`, so the index never indexes itself.
    ///
    /// # Errors
    ///
    /// Returns an error if the metadata location holds the data location. Every later call
    /// returns [`cosmosearch_store::Error::WrongDataLocation`] if `data` is not the location
    /// the index was created for.
    pub fn new(index: IndexHandle, data: DataStore) -> Result<Self> {
        let data = data.skipping_metadata(index.metadata_location())?;
        Ok(Indexer {
            index,
            data,
            shard_bytes: DEFAULT_SHARD_BYTES,
        })
    }

    /// Sets the source bytes a shard may hold before a commit that touches it splits it in
    /// two near its middle; 1 GiB unless set. Shards are the unit of heat and promotion, so a
    /// smaller size spends more finely and makes more, smaller segments to read. Every
    /// indexer of an index should use the same size.
    #[must_use]
    pub fn shard_bytes(mut self, max: NonZeroU64) -> Self {
        self.shard_bytes = max;
        self
    }

    /// Records changes named by the host as a single commit, and never lists the data.
    ///
    /// An index that lists its data takes only retries here. An attribute change reads nothing
    /// and replaces only the file table of the shard it touches.
    ///
    /// # Errors
    ///
    /// Returns [`Error::WrongFileSource`] for an addition, removal or attribute change unless
    /// the index takes host-named files, an error for a key outside the data location,
    /// [`Error::CommitUnknown`] if the commit may have landed, in which case sending the same
    /// changes again is safe, and [`Error::TooManyConflicts`] if other writers kept winning.
    pub async fn apply(&self, changes: Vec<FileChange>) -> Result<Applied> {
        let only_retries = changes
            .iter()
            .all(|change| matches!(change, FileChange::Retry { .. }));
        if !only_retries {
            self.require(FileSource::HostNamed).await?;
        }
        let steps = self.resolve(changes).await?;
        self.commit_steps(&steps).await
    }

    /// Lists the data location and records every addition, change and removal as a single commit.
    ///
    /// Each change is checked against the file as it was before the listing, so a newer
    /// commit, such as one from a sync that started later, is never undone; such a change is
    /// reported [`ChangeOutcome::Stale`] and the next sync picks it up.
    ///
    /// # Errors
    ///
    /// Returns [`Error::WrongFileSource`] unless the index lists its data, and otherwise as for
    /// [`Indexer::apply`].
    pub async fn sync_listing(&self) -> Result<Applied> {
        self.require(FileSource::Listing).await?;
        let known = self.entries().await?;
        let listed: BTreeMap<String, (u64, ETag)> = self
            .data
            .list()
            .map_ok(|info| {
                (
                    info.key().as_str().to_owned(),
                    (info.size(), info.etag().clone()),
                )
            })
            .try_collect()
            .await?;
        let mut steps = Vec::new();
        for entry in &known {
            if !listed.contains_key(entry.key()) {
                steps.push(Step::Remove {
                    key: entry.key().to_owned(),
                    expected: Some(entry.content_version()),
                });
            }
        }
        let known: BTreeMap<&str, &FileEntry> = known.iter().map(|e| (e.key(), e)).collect();
        for (key, (size, etag)) in listed {
            let expected = match known.get(key.as_str()) {
                Some(entry) if entry.etag() == &etag => continue,
                Some(entry) => Expected::At(entry.content_version()),
                None => Expected::Absent,
            };
            steps.push(Step::Upsert(Incoming {
                key,
                size,
                etag,
                display_name: None,
                media_type: None,
                attributes: Attributes::new(),
                expected,
            }));
        }
        self.commit_steps(&steps).await
    }

    pub(crate) async fn entries(&self) -> Result<Vec<FileEntry>> {
        let head = self.head().await?;
        let mut files = Vec::new();
        for shard in head.shards().values() {
            files.extend(self.load_table(shard).await?.iter().cloned());
        }
        Ok(files)
    }

    pub(crate) async fn record_reasons(
        &self,
        reasons: Vec<(String, ManifestVersion, Option<FileReason>)>,
    ) -> Result<Applied> {
        let steps: Vec<Step> = reasons
            .into_iter()
            .map(|(key, planned, reason)| match reason {
                Some(reason) => Step::Record {
                    key,
                    planned,
                    reason,
                },
                None => Step::Skip,
            })
            .collect();
        self.commit_steps(&steps).await
    }

    pub(crate) fn index(&self) -> &IndexHandle {
        &self.index
    }

    pub(crate) fn data(&self) -> &DataStore {
        &self.data
    }

    /// Checks that `key` is a whole object key inside the data location.
    pub(crate) fn check_key(&self, key: &str) -> Result<ObjectKey> {
        let object = ObjectKey::new(key)?;
        self.data.check_inside(&object)?;
        Ok(object)
    }

    pub(crate) async fn head(&self) -> Result<Manifest> {
        let head = self.index.head().await?.ok_or(Error::NoManifest)?;
        if !head.data_location().same_prefix(self.data.location()) {
            return Err(cosmosearch_store::Error::WrongDataLocation {
                recorded: head.data_location().clone(),
                given: self.data.location().clone(),
            }
            .into());
        }
        Ok(head)
    }

    async fn require(&self, needed: FileSource) -> Result<()> {
        let actual = self.head().await?.file_source();
        if actual == needed {
            Ok(())
        } else {
            Err(Error::WrongFileSource { needed, actual })
        }
    }

    async fn resolve(&self, changes: Vec<FileChange>) -> Result<Vec<Step>> {
        stream::iter(changes.into_iter().map(|change| self.resolve_one(change)))
            .buffered(HEAD_CONCURRENCY)
            .try_collect()
            .await
    }

    async fn resolve_one(&self, change: FileChange) -> Result<Step> {
        match change {
            FileChange::Upsert(upsert) => self.resolve_upsert(upsert).await,
            FileChange::Remove { key, expected } => {
                self.check_key(&key)?;
                Ok(Step::Remove { key, expected })
            }
            FileChange::Retry { key, expected } => {
                self.check_key(&key)?;
                Ok(Step::Retry { key, expected })
            }
            FileChange::Attributes(change) => {
                self.check_key(&change.key)?;
                Ok(Step::Describe(change))
            }
        }
    }

    async fn resolve_upsert(&self, upsert: FileUpsert) -> Result<Step> {
        let object = self.check_key(&upsert.key)?;
        let (size, etag) = if let Some(fields) = upsert.fields {
            (fields.size, fields.etag)
        } else {
            match self.data.head(&object).await {
                Ok(info) => (info.size(), info.etag().clone()),
                Err(cosmosearch_store::Error::NotFound { .. }) => return Ok(Step::Missing),
                Err(err) => return Err(err.into()),
            }
        };
        Ok(Step::Upsert(Incoming {
            key: upsert.key,
            size,
            etag,
            display_name: upsert.display_name,
            media_type: upsert.media_type,
            attributes: upsert.attributes,
            expected: upsert.expected.into(),
        }))
    }

    async fn commit_steps(&self, steps: &[Step]) -> Result<Applied> {
        for _ in 0..MAX_COMMIT_ATTEMPTS {
            let head = self.head().await?;
            let version = head.version().next()?;
            let mut tables = self.load_touched(&head, steps).await?;
            let mut changed = BTreeSet::new();
            let mut outcomes = Vec::with_capacity(steps.len());
            for step in steps {
                let outcome = apply_step(&head, &mut tables, &mut changed, step, version)?;
                outcomes.push(outcome);
            }
            if changed.is_empty() {
                return Ok(Applied {
                    version: None,
                    outcomes,
                });
            }
            let updates = changed
                .iter()
                .filter_map(|id| {
                    let table = tables.remove(id)?;
                    Some((id.clone(), ShardUpdate::table_only(table)))
                })
                .collect();
            let next = self.next_manifest(&head, updates).await?;
            let outcome = self.index.commit(&next).await?;
            if let CommitOutcome::Committed(version) = outcome {
                return Ok(Applied {
                    version: Some(version),
                    outcomes,
                });
            }
            if outcome != CommitOutcome::Conflict {
                return Err(Error::CommitUnknown);
            }
        }
        Err(Error::TooManyConflicts {
            attempts: MAX_COMMIT_ATTEMPTS,
        })
    }

    async fn load_touched(
        &self,
        head: &Manifest,
        steps: &[Step],
    ) -> Result<BTreeMap<ShardId, FileTable>> {
        let mut tables = BTreeMap::new();
        for key in steps.iter().filter_map(Step::key) {
            let (id, shard) = shard_for(head, key)?;
            if !tables.contains_key(id) {
                tables.insert(id.clone(), self.load_table(shard).await?);
            }
        }
        Ok(tables)
    }

    pub(crate) async fn load_table(&self, shard: &ShardEntry) -> Result<FileTable> {
        let Some(segment) = l0_segment(shard) else {
            return Ok(FileTable::new());
        };
        let Some(file) = segment.files().first() else {
            return Ok(FileTable::new());
        };
        let bytes = self
            .index
            .read_segment_file(segment.id(), file.name())
            .await?;
        Ok(FileTable::decode(bytes)?)
    }

    pub(crate) async fn next_manifest(
        &self,
        head: &Manifest,
        updates: BTreeMap<ShardId, ShardUpdate>,
    ) -> Result<Manifest> {
        let mut shards = head.shards().clone();
        let mut model_ids = head.model_ids().clone();
        for (id, update) in updates {
            model_ids.extend(
                update
                    .add
                    .iter()
                    .filter_map(|segment| segment.model_id().cloned()),
            );
            let Some(shard) = shards.remove(&id) else {
                continue;
            };
            let kept: Vec<SegmentEntry> = shard
                .segments()
                .iter()
                .filter(|segment| {
                    segment.level() != Level::L0 && !update.remove.contains(segment.id())
                })
                .cloned()
                .chain(update.add)
                .collect();
            let parts = split_oversized(shard.range().clone(), update.table, self.shard_bytes)?;
            let divided = parts.len() > 1;
            for (range, table) in parts {
                let segment = self.write_table(&table).await?;
                let segments = std::iter::once(segment)
                    .chain(kept.iter().cloned())
                    .collect();
                let part_id = if divided {
                    ShardId::new(new_commit_id()?.as_str())?
                } else {
                    id.clone()
                };
                shards.insert(part_id, ShardEntry::new(range, segments));
            }
        }
        Ok(head.successor(new_commit_id()?, shards, model_ids)?)
    }

    async fn write_table(&self, table: &FileTable) -> Result<SegmentEntry> {
        let id = SegmentId::new(new_commit_id()?.as_str())?;
        let name = SegmentFileName::new(format!("meta-{id}.parquet"))?;
        let body = Bytes::from(table.encode()?);
        let size = u64::try_from(body.len()).unwrap_or(u64::MAX);
        self.index.write_segment_file(&id, &name, body).await?;
        Ok(SegmentEntry::new(
            id,
            Level::L0,
            BuiltBy::new(None, BTreeSet::new()),
            None,
            vec![SegmentFile::new(name, size)],
        ))
    }
}

fn split_oversized(
    range: KeyRange,
    table: FileTable,
    max: NonZeroU64,
) -> Result<Vec<(KeyRange, FileTable)>> {
    let mut finished = Vec::new();
    let mut waiting = vec![(range, table)];
    while let Some((range, table)) = waiting.pop() {
        let cut = (table.total_size() > max.get())
            .then(|| find_cut(&table))
            .flatten();
        let Some(cut) = cut else {
            finished.push((range, table));
            continue;
        };
        let (left, right) = table.split_at(&cut);
        let left_range = KeyRange::new(range.start(), Some(cut.clone()))?;
        let right_range = KeyRange::new(cut, range.end().map(str::to_owned))?;
        waiting.push((left_range, left));
        waiting.push((right_range, right));
    }
    Ok(finished)
}

fn shared_folders(a: &str, b: &str) -> usize {
    a.bytes()
        .zip(b.bytes())
        .take_while(|(x, y)| x == y)
        .filter(|(x, _)| *x == b'/')
        .count()
}

/// The key to divide a table at: near the byte median, at the shallowest path boundary there.
fn find_cut(table: &FileTable) -> Option<String> {
    let total = table.total_size();
    let half = total / 2;
    let low = (total / PERCENT).saturating_mul(CUT_WINDOW_LOW_PERCENT);
    let high = (total / PERCENT).saturating_mul(CUT_WINDOW_HIGH_PERCENT);
    let mut before = 0u64;
    let mut previous: Option<&str> = None;
    let mut candidates = Vec::new();
    for file in table.iter() {
        if let Some(previous) = previous {
            candidates.push((
                shared_folders(previous, file.key()),
                before.abs_diff(half),
                before,
                file.key(),
            ));
        }
        before = before.saturating_add(file.size());
        previous = Some(file.key());
    }
    let in_window = candidates
        .iter()
        .filter(|(_, _, at, _)| (low..=high).contains(at))
        .min_by_key(|(depth, distance, _, _)| (*depth, *distance));
    let nearest = || {
        candidates
            .iter()
            .min_by_key(|(_, distance, _, _)| *distance)
    };
    in_window
        .or_else(nearest)
        .map(|(_, _, _, key)| (*key).to_owned())
}

fn l0_segment(shard: &ShardEntry) -> Option<&SegmentEntry> {
    shard
        .segments()
        .iter()
        .find(|segment| segment.level() == Level::L0)
}

pub(crate) fn shard_for<'a>(
    head: &'a Manifest,
    key: &str,
) -> Result<(&'a ShardId, &'a ShardEntry)> {
    head.shard_for(key).ok_or_else(|| Error::NoShard {
        key: key.to_owned(),
    })
}

fn apply_step(
    head: &Manifest,
    tables: &mut BTreeMap<ShardId, FileTable>,
    changed: &mut BTreeSet<ShardId>,
    step: &Step,
    version: ManifestVersion,
) -> Result<ChangeOutcome> {
    let Some(key) = step.key() else {
        return Ok(match step {
            Step::Skip => ChangeOutcome::Unchanged,
            Step::Upsert(_)
            | Step::Remove { .. }
            | Step::Retry { .. }
            | Step::Describe(_)
            | Step::Record { .. }
            | Step::Missing => ChangeOutcome::NotFound,
        });
    };
    let (id, _) = shard_for(head, key)?;
    let Some(table) = tables.get_mut(id) else {
        return Err(Error::NoShard {
            key: key.to_owned(),
        });
    };
    let outcome = match step {
        Step::Upsert(incoming) => apply_upsert(table, incoming, version),
        Step::Remove { key, expected } => apply_remove(table, key, *expected),
        Step::Retry { key, expected } => apply_retry(table, key, *expected, version),
        Step::Describe(change) => apply_describe(table, change),
        Step::Record {
            key,
            planned,
            reason,
        } => apply_record(table, key, *planned, *reason),
        Step::Skip => ChangeOutcome::Unchanged,
        Step::Missing => ChangeOutcome::NotFound,
    };
    if outcome == ChangeOutcome::Applied {
        changed.insert(id.clone());
    }
    Ok(outcome)
}

fn moved_on(table: &FileTable, key: &str, expected: Option<ManifestVersion>) -> bool {
    expected.is_some_and(|version| {
        table
            .get(key)
            .is_none_or(|entry| entry.content_version() != version)
    })
}

fn apply_upsert(
    table: &mut FileTable,
    incoming: &Incoming,
    version: ManifestVersion,
) -> ChangeOutcome {
    let current = table.get(&incoming.key);
    let same = current.is_some_and(|entry| {
        entry.etag() == &incoming.etag && entry.media_type() == incoming.media_type.as_ref()
    });
    let moved = match incoming.expected {
        Expected::Any => false,
        Expected::At(version) => moved_on(table, &incoming.key, Some(version)),
        Expected::Absent => current.is_some() && !same,
    };
    if moved {
        return ChangeOutcome::Stale;
    }
    if same {
        return ChangeOutcome::Unchanged;
    }
    let entry = FileEntry::pending(
        incoming.key.clone(),
        incoming.size,
        incoming.etag.clone(),
        version,
    )
    .with_display_name(incoming.display_name.clone())
    .with_media_type(incoming.media_type.clone())
    .with_attributes(incoming.attributes.clone());
    table.insert(entry);
    ChangeOutcome::Applied
}

fn apply_remove(
    table: &mut FileTable,
    key: &str,
    expected: Option<ManifestVersion>,
) -> ChangeOutcome {
    if moved_on(table, key, expected) {
        ChangeOutcome::Stale
    } else if table.remove(key) {
        ChangeOutcome::Applied
    } else {
        ChangeOutcome::Unchanged
    }
}

fn apply_retry(
    table: &mut FileTable,
    key: &str,
    expected: Option<ManifestVersion>,
    version: ManifestVersion,
) -> ChangeOutcome {
    if table.get(key).is_none() {
        ChangeOutcome::NotFound
    } else if moved_on(table, key, expected) {
        ChangeOutcome::Stale
    } else {
        table.make_pending(key, version);
        ChangeOutcome::Applied
    }
}

fn apply_describe(table: &mut FileTable, change: &AttributeChange) -> ChangeOutcome {
    let Some(entry) = table.get(&change.key) else {
        return ChangeOutcome::NotFound;
    };
    if moved_on(table, &change.key, change.expected) {
        ChangeOutcome::Stale
    } else if entry.display_name() == change.display_name.as_deref()
        && entry.attributes() == &change.attributes
    {
        ChangeOutcome::Unchanged
    } else {
        table.describe(
            &change.key,
            change.display_name.clone(),
            change.attributes.clone(),
        );
        ChangeOutcome::Applied
    }
}

fn apply_record(
    table: &mut FileTable,
    key: &str,
    planned: ManifestVersion,
    reason: FileReason,
) -> ChangeOutcome {
    let Some(entry) = table.get(key) else {
        return ChangeOutcome::Stale;
    };
    if entry.content_version() != planned {
        ChangeOutcome::Stale
    } else if entry.reason() == Some(reason) {
        ChangeOutcome::Unchanged
    } else if entry.status() == FileStatus::Pending {
        table.record(key, reason);
        ChangeOutcome::Applied
    } else {
        ChangeOutcome::Stale
    }
}
