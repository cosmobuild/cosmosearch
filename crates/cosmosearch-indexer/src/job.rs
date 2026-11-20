use std::collections::{BTreeMap, BTreeSet};

use bytes::Bytes;
use cosmosearch_chunk::{Extractors, extract_chunks};
use cosmosearch_core::{
    BuiltBy, ByteRange, Chunk, ETag, ExtractFailure, ExtractLimits, ExtractorProfile, IndexId,
    IndexKind, Level, LimitKind, Location, Manifest, ManifestVersion, ModelId, SegmentEntry,
    SegmentFile, SegmentFileName, SegmentId, ShardId, SharedIndex, Usage,
};
use cosmosearch_format::{ChunkRecord, ChunkTable, FileEntry, FileReason, FileStatus, FileTable};
use cosmosearch_store::{CommitOutcome, ObjectKey, new_commit_id};
use serde::{Deserialize, Serialize};

use crate::indexer::{MAX_COMMIT_ATTEMPTS, ShardUpdate};
use crate::promotion::{Guard, appeared};
use crate::{Applied, Error, Indexer, Result};

const CHUNKS_PREFIX: &str = "chunks-";

/// One file to extract, as it was when the job was planned. Attributes are never part of a job.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileJob {
    index_id: IndexId,
    shard: ShardId,
    key: String,
    ordinal: u64,
    size: u64,
    etag: ETag,
    content_version: ManifestVersion,
}

impl FileJob {
    pub(crate) fn of(index_id: &IndexId, shard: &ShardId, entry: &FileEntry) -> Self {
        FileJob {
            index_id: index_id.clone(),
            shard: shard.clone(),
            key: entry.key().to_owned(),
            ordinal: entry.ordinal(),
            size: entry.size(),
            etag: entry.etag().clone(),
            content_version: entry.content_version(),
        }
    }

    /// The file's key.
    pub fn key(&self) -> &str {
        &self.key
    }

    /// The content version this job was planned for.
    pub fn content_version(&self) -> ManifestVersion {
        self.content_version
    }
}

/// Files of one shard, which can go into one segment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Batch {
    shard: ShardId,
    jobs: Vec<FileJob>,
}

impl Batch {
    /// The shard these files belong to.
    pub fn shard(&self) -> &ShardId {
        &self.shard
    }

    /// The files.
    pub fn jobs(&self) -> &[FileJob] {
        &self.jobs
    }
}

/// The work waiting at one moment. It can be stored and picked up by another process.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Plan {
    batches: Vec<Batch>,
}

impl Plan {
    /// The work, one batch per shard.
    pub fn batches(&self) -> &[Batch] {
        &self.batches
    }

    /// Whether there is nothing to do.
    pub fn is_empty(&self) -> bool {
        self.batches.is_empty()
    }
}

/// Segments of one index plug-in that hold chunks no longer live, with their live files.
///
/// Built with that plug-in and published like new work, it replaces them with a segment
/// holding only the live files.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rebuild {
    kind: IndexKind,
    model_id: Option<ModelId>,
    replaces: Vec<SegmentId>,
    jobs: Vec<FileJob>,
}

impl Rebuild {
    /// The kind of the index plug-in that built the segments, and must build their
    /// replacement.
    pub fn kind(&self) -> &IndexKind {
        &self.kind
    }

    /// The embedding identity of that plug-in, for an index with vectors.
    pub fn model_id(&self) -> Option<&ModelId> {
        self.model_id.as_ref()
    }

    /// The segments to replace.
    pub fn replaces(&self) -> &[SegmentId] {
        &self.replaces
    }

    /// The live files to extract again, possibly none.
    pub fn jobs(&self) -> &[FileJob] {
        &self.jobs
    }
}

/// What a rebuild groups segments by: the shards that reference them and what built them.
type RebuildKey = (BTreeSet<ShardId>, IndexKind, Option<ModelId>);

#[derive(Default)]
struct RebuildGroup {
    replaces: Vec<SegmentId>,
    jobs: BTreeMap<String, FileJob>,
}

/// What extracting one file gave.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ExtractOutcome {
    /// The file's chunks.
    Chunks {
        /// The extractor that produced them.
        profile: ExtractorProfile,
        /// The chunks, in reading order.
        chunks: Vec<Chunk>,
        /// How many parts of the file, such as PDF pages that hold only an image, the
        /// extractor could not read.
        unread_parts: u32,
    },
    /// The file could not be extracted.
    Failed(ExtractFailure),
    /// The file no longer has the planned ETag, so nothing was read. It stays pending.
    SourceMoved,
}

/// The result of one extraction job, returned to the host and never stored by the library.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtractedFile {
    job: FileJob,
    outcome: ExtractOutcome,
}

impl ExtractedFile {
    /// A failure found outside [`Indexer::extract`], for example by a host whose separate
    /// extraction process died on this file.
    pub fn failed(job: FileJob, failure: ExtractFailure) -> Self {
        ExtractedFile {
            job,
            outcome: ExtractOutcome::Failed(failure),
        }
    }

    /// The job this answers.
    pub fn job(&self) -> &FileJob {
        &self.job
    }

    /// What happened.
    pub fn outcome(&self) -> &ExtractOutcome {
        &self.outcome
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct BuiltFile {
    key: String,
    ordinal: u64,
    content_version: ManifestVersion,
    unread_parts: u32,
}

/// Work ready to publish: most often a segment whose files are written.
///
/// It can also be a rebuild that only takes segments away because none of their files is
/// live, or a demotion, which takes away the segments of one index plug-in above L1.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BuiltSegment {
    index_id: IndexId,
    entry: Option<SegmentEntry>,
    files: Vec<BuiltFile>,
    replaces: Vec<SegmentId>,
    guard: Option<Guard>,
    drops: bool,
    usage: Usage,
}

impl BuiltSegment {
    pub(crate) fn dropping(index_id: IndexId, replaces: Vec<SegmentId>, guard: Guard) -> Self {
        BuiltSegment {
            index_id,
            entry: None,
            files: Vec::new(),
            replaces,
            guard: Some(guard),
            drops: true,
            usage: Usage::new(),
        }
    }

    /// The new segment's id, or `None` for a rebuild that only takes segments away.
    pub fn id(&self) -> Option<&SegmentId> {
        self.entry.as_ref().map(SegmentEntry::id)
    }

    /// The new segment's entry, as it will appear in the manifest.
    pub fn entry(&self) -> Option<&SegmentEntry> {
        self.entry.as_ref()
    }

    /// What the index plug-in spent building it, such as embedding tokens.
    pub fn usage(&self) -> Usage {
        self.usage
    }

    /// The keys of the files it holds.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.files.iter().map(|file| file.key.as_str())
    }
}

/// How publishing a segment ended.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum PublishOutcome {
    /// The segment is in the index at this version.
    Published(ManifestVersion),
    /// This result is already reflected, so nothing changed.
    AlreadyDone,
    /// These files have moved on. Nothing changed; build again without them.
    Stale(Vec<String>),
    /// A segment this one replaces is gone, or for a promotion another segment of its
    /// plug-in has appeared, so nothing changed. Plan again.
    Superseded,
    /// A rebuild leaves out these files, which are still live in a segment it replaces.
    /// Nothing changed; build again with them.
    Incomplete(Vec<String>),
    /// The segment's files are no longer in the store, most likely removed by garbage
    /// collection because publishing came later than its grace period. Nothing changed;
    /// build again.
    Expired,
}

enum Verdict {
    Valid,
    AlreadyDone,
    Stale,
}

fn verdict(table: &FileTable, file: &BuiltFile, over_ready: bool) -> Verdict {
    match table.get(&file.key) {
        Some(entry)
            if entry.content_version() == file.content_version
                && entry.ordinal() == file.ordinal =>
        {
            if over_ready || entry.status() == FileStatus::Pending {
                Verdict::Valid
            } else {
                Verdict::AlreadyDone
            }
        }
        Some(_) | None => Verdict::Stale,
    }
}

fn references(head: &Manifest, segment: &SegmentId) -> bool {
    head.shards()
        .values()
        .any(|shard| shard.segments().iter().any(|entry| entry.id() == segment))
}

/// Every segment holding chunks, with the shards that reference it.
pub(crate) fn chunk_segments(
    head: &Manifest,
) -> BTreeMap<SegmentId, (SegmentEntry, BTreeSet<ShardId>)> {
    let mut found: BTreeMap<SegmentId, (SegmentEntry, BTreeSet<ShardId>)> = BTreeMap::new();
    for (shard_id, shard) in head.shards() {
        for segment in shard.segments() {
            if segment.level() == Level::L0 {
                continue;
            }
            found
                .entry(segment.id().clone())
                .or_insert_with(|| (segment.clone(), BTreeSet::new()))
                .1
                .insert(shard_id.clone());
        }
    }
    found
}

/// The file whose chunks at `version` are live under one of `shards`, and that shard.
pub(crate) fn live_file<'a>(
    tables: &'a BTreeMap<ShardId, FileTable>,
    shards: &'a BTreeSet<ShardId>,
    ordinal: u64,
    version: ManifestVersion,
) -> Option<(&'a ShardId, &'a FileEntry)> {
    shards.iter().find_map(|shard| {
        let entry = tables.get(shard)?.by_ordinal(ordinal)?;
        (entry.content_version() == version).then_some((shard, entry))
    })
}

fn chunk_file(segment: &SegmentEntry) -> Option<&SegmentFile> {
    segment
        .files()
        .iter()
        .find(|file| file.name().as_str().starts_with(CHUNKS_PREFIX))
}

impl Indexer {
    pub(crate) fn check_index(&self, found: &IndexId) -> Result<()> {
        let expected = self.index().id();
        if found == expected {
            Ok(())
        } else {
            Err(Error::WrongIndex {
                expected: expected.clone(),
                found: found.clone(),
            })
        }
    }

    /// Lists up to `limit` pending files, grouped by shard. A file whose recorded version is
    /// known to be gone is left out until a newer change arrives.
    ///
    /// # Errors
    ///
    /// Returns an error if the store fails or a stored table cannot be read.
    pub async fn plan(&self, limit: usize) -> Result<Plan> {
        let head = self.head().await?;
        let index_id = head.index_id().clone();
        let mut remaining = limit;
        let mut batches = Vec::new();
        for (shard_id, shard) in head.shards() {
            if remaining == 0 {
                break;
            }
            let table = self.load_table(shard).await?;
            let jobs: Vec<FileJob> = table
                .iter()
                .filter(|entry| entry.status() == FileStatus::Pending && entry.reason().is_none())
                .take(remaining)
                .map(|entry| FileJob::of(&index_id, shard_id, entry))
                .collect();
            remaining = remaining.saturating_sub(jobs.len());
            if !jobs.is_empty() {
                batches.push(Batch {
                    shard: shard_id.clone(),
                    jobs,
                });
            }
        }
        Ok(Plan { batches })
    }

    /// Finds the segments that hold chunks no longer live, of files removed or whose content
    /// moved on, and plans one rebuild for each set of shards referencing them and each index
    /// plug-in that built them.
    ///
    /// # Errors
    ///
    /// Returns an error if the store fails or a stored table cannot be read.
    pub async fn plan_rebuilds(&self) -> Result<Vec<Rebuild>> {
        let head = self.head().await?;
        let index_id = head.index_id().clone();
        let segments = chunk_segments(&head);
        let tables = self.load_holders(&head, segments.values()).await?;
        let mut groups: BTreeMap<RebuildKey, RebuildGroup> = BTreeMap::new();
        for (id, (segment, shards)) in &segments {
            let Some(kind) = segment.built_by().index() else {
                continue;
            };
            let Some(chunks) = self.load_chunks(segment).await? else {
                continue;
            };
            let mut dead = false;
            let mut jobs = Vec::new();
            for (ordinal, version) in chunks.file_versions() {
                let Some((shard, entry)) = live_file(&tables, shards, ordinal, version) else {
                    dead = true;
                    continue;
                };
                jobs.push(FileJob::of(&index_id, shard, entry));
            }
            if dead {
                let key = (shards.clone(), kind.clone(), segment.model_id().cloned());
                let group = groups.entry(key).or_default();
                group.replaces.push(id.clone());
                group
                    .jobs
                    .extend(jobs.into_iter().map(|job| (job.key.clone(), job)));
            }
        }
        Ok(groups
            .into_iter()
            .map(|((_, kind, model_id), group)| Rebuild {
                kind,
                model_id,
                replaces: group.replaces,
                jobs: group.jobs.into_values().collect(),
            })
            .collect())
    }

    /// Plans, for each set of shards and each index plug-in with more than `more_than`
    /// segments there, one rebuild that replaces them all with a single segment of their live
    /// files. This is how a shard with many small segments, which every search must open, is
    /// made cheap to read again.
    ///
    /// # Errors
    ///
    /// Returns an error if the store fails or a stored table cannot be read.
    pub async fn plan_merges(&self, more_than: usize) -> Result<Vec<Rebuild>> {
        let head = self.head().await?;
        let index_id = head.index_id().clone();
        let segments = chunk_segments(&head);
        let mut counts: BTreeMap<RebuildKey, usize> = BTreeMap::new();
        for (segment, shards) in segments.values() {
            if let Some(kind) = segment.built_by().index() {
                let key = (shards.clone(), kind.clone(), segment.model_id().cloned());
                let count = counts.entry(key).or_default();
                *count = count.saturating_add(1);
            }
        }
        counts.retain(|_, count| *count > more_than);
        let crowded = segments.values().filter(|(segment, shards)| {
            segment.built_by().index().is_some_and(|kind| {
                counts.contains_key(&(shards.clone(), kind.clone(), segment.model_id().cloned()))
            })
        });
        let tables = self.load_holders(&head, crowded).await?;
        let mut groups: BTreeMap<RebuildKey, RebuildGroup> = BTreeMap::new();
        for (id, (segment, shards)) in &segments {
            let Some(kind) = segment.built_by().index() else {
                continue;
            };
            let key = (shards.clone(), kind.clone(), segment.model_id().cloned());
            if !counts.contains_key(&key) {
                continue;
            }
            let group = groups.entry(key).or_default();
            group.replaces.push(id.clone());
            let Some(chunks) = self.load_chunks(segment).await? else {
                continue;
            };
            for (ordinal, version) in chunks.file_versions() {
                if let Some((shard, entry)) = live_file(&tables, shards, ordinal, version) {
                    let job = FileJob::of(&index_id, shard, entry);
                    group.jobs.insert(job.key.clone(), job);
                }
            }
        }
        Ok(groups
            .into_iter()
            .map(|((_, kind, model_id), group)| Rebuild {
                kind,
                model_id,
                replaces: group.replaces,
                jobs: group.jobs.into_values().collect(),
            })
            .collect())
    }

    pub(crate) async fn load_holders(
        &self,
        head: &Manifest,
        segments: impl Iterator<Item = &(SegmentEntry, BTreeSet<ShardId>)>,
    ) -> Result<BTreeMap<ShardId, FileTable>> {
        let mut tables = BTreeMap::new();
        for (_, shards) in segments {
            for id in shards {
                if tables.contains_key(id) {
                    continue;
                }
                if let Some(shard) = head.shards().get(id) {
                    tables.insert(id.clone(), self.load_table(shard).await?);
                }
            }
        }
        Ok(tables)
    }

    pub(crate) async fn load_chunks(&self, segment: &SegmentEntry) -> Result<Option<ChunkTable>> {
        let Some(file) = chunk_file(segment) else {
            return Ok(None);
        };
        let bytes = self
            .index()
            .read_segment_file(segment.id(), file.name())
            .await?;
        Ok(Some(ChunkTable::decode(bytes)?))
    }

    /// Extracts one file with the extractor that reads its type, reading exactly the version the
    /// job was planned for. A file no extractor reads fails as unsupported, and one larger than
    /// the limit on file bytes fails as over that limit; in both cases nothing is read.
    ///
    /// # Errors
    ///
    /// Returns [`Error::WrongIndex`] for a job of another index, and an error if reading fails
    /// for a reason that may pass. Nothing is recorded against the file then: it stays pending,
    /// and [`Error::is_retryable`] says whether running the job again can work.
    pub async fn extract(
        &self,
        job: FileJob,
        extractors: &Extractors,
        limits: &ExtractLimits,
    ) -> Result<ExtractedFile> {
        self.check_index(&job.index_id)?;
        let Some(extractor) = extractors.for_key(&job.key) else {
            return Ok(ExtractedFile::failed(job, ExtractFailure::NoExtractor));
        };
        if job.size > limits.file_bytes() {
            let failure = ExtractFailure::LimitExceeded(LimitKind::FileBytes);
            return Ok(ExtractedFile::failed(job, failure));
        }
        let Some(bytes) = self.read_planned(&job).await? else {
            return Ok(ExtractedFile {
                job,
                outcome: ExtractOutcome::SourceMoved,
            });
        };
        let data = self.data().location();
        let location = Location::new(data.scheme(), data.authority(), job.key.as_str())?;
        let extractor = extractor.clone();
        let limits = *limits;
        let etag = job.etag.clone();
        let result = tokio::task::spawn_blocking(move || {
            let chunks = extract_chunks(extractor.get(), &location, &etag, &bytes, &limits);
            (extractor.get().profile().clone(), chunks)
        })
        .await
        .map_err(Error::Worker)?;
        let outcome = match result {
            (profile, Ok(found)) => ExtractOutcome::Chunks {
                profile,
                unread_parts: found.unread_parts(),
                chunks: found.into_chunks(),
            },
            (_, Err(cosmosearch_chunk::Error::File(failure))) => ExtractOutcome::Failed(failure),
            (_, Err(err)) => return Err(err.into()),
        };
        Ok(ExtractedFile { job, outcome })
    }

    /// Records, as a single commit, what stopped each file: a failure makes it failed or
    /// unsupported, and a source that moved keeps it pending with that reason. Files that gave
    /// chunks are left for [`Indexer::build`]. A result applies only while its file is pending
    /// at the planned version, and is reported stale otherwise. Safe to repeat.
    ///
    /// # Errors
    ///
    /// Returns [`Error::WrongIndex`] for a job of another index, [`Error::CommitUnknown`] if
    /// the commit may have landed, in which case recording again is safe, and
    /// [`Error::TooManyConflicts`] if other writers kept winning.
    pub async fn record(&self, files: &[ExtractedFile]) -> Result<Applied> {
        let mut reasons = Vec::with_capacity(files.len());
        for file in files {
            self.check_index(&file.job.index_id)?;
            let reason = match &file.outcome {
                ExtractOutcome::Chunks { .. } => None,
                ExtractOutcome::Failed(failure) => Some(FileReason::Extraction(*failure)),
                ExtractOutcome::SourceMoved => Some(FileReason::SourceMoved),
            };
            reasons.push((file.job.key.clone(), file.job.content_version, reason));
        }
        self.record_reasons(reasons).await
    }

    async fn read_planned(&self, job: &FileJob) -> Result<Option<Bytes>> {
        let Ok(range) = ByteRange::new(0, job.size) else {
            return Ok(Some(Bytes::new()));
        };
        let key = ObjectKey::new(&job.key)?;
        match self.data().read_range(&key, range, &job.etag).await {
            Ok(bytes) => Ok(Some(bytes)),
            Err(
                cosmosearch_store::Error::Stale { .. }
                | cosmosearch_store::Error::NotFound { .. }
                | cosmosearch_store::Error::ShortRead { .. },
            ) => Ok(None),
            Err(err) => Err(err.into()),
        }
    }

    /// Builds one segment from extracted files of one shard with `index` and writes its files.
    /// The segment is recorded at the plug-in's level and with its embedding identity.
    ///
    /// Files that gave no chunks are left out. `replaces` names the segments a rebuild takes
    /// over from, and is empty for new work; a rebuild is built with the plug-in that
    /// [`Rebuild::kind`] names. A rebuild may hold files of every shard that shares the
    /// segments it replaces, and with no chunks it only takes them away. New work with no
    /// chunks returns `None`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::WrongLevel`] for new work with a plug-in that is not at L1, since
    /// every file is first made searchable at L1 and reaches higher levels only by
    /// promotion, [`Error::MixedShards`] for new work with files of different shards, or files
    /// of different shards that share an ordinal, [`Error::WrongIndex`] for a job of another
    /// index, and an error if building or storing fails.
    pub async fn build(
        &self,
        index: &SharedIndex,
        files: Vec<ExtractedFile>,
        replaces: Vec<SegmentId>,
    ) -> Result<Option<BuiltSegment>> {
        let level = index.get().level();
        if replaces.is_empty() && level != Level::L1 {
            return Err(Error::WrongLevel { level });
        }
        self.build_segment(index, files, replaces, None).await
    }

    pub(crate) async fn build_segment(
        &self,
        index: &SharedIndex,
        files: Vec<ExtractedFile>,
        replaces: Vec<SegmentId>,
        guard: Option<Guard>,
    ) -> Result<Option<BuiltSegment>> {
        let mut shard: Option<ShardId> = None;
        let mut ordinals: BTreeMap<u64, ShardId> = BTreeMap::new();
        let mut profiles = BTreeSet::new();
        let mut chunks = Vec::new();
        let mut records = Vec::new();
        let mut built_files = Vec::new();
        for file in files {
            self.check_index(&file.job.index_id)?;
            match &shard {
                Some(first) if *first != file.job.shard && replaces.is_empty() => {
                    return Err(Error::MixedShards {
                        first: first.clone(),
                        second: file.job.shard,
                    });
                }
                Some(_) => {}
                None => shard = Some(file.job.shard.clone()),
            }
            let owner = ordinals
                .entry(file.job.ordinal)
                .or_insert_with(|| file.job.shard.clone());
            if *owner != file.job.shard {
                return Err(Error::MixedShards {
                    first: owner.clone(),
                    second: file.job.shard,
                });
            }
            let ExtractOutcome::Chunks {
                profile,
                chunks: found,
                unread_parts,
            } = file.outcome
            else {
                continue;
            };
            for chunk in &found {
                records.push(ChunkRecord::new(
                    file.job.ordinal,
                    file.job.content_version,
                    profile.clone(),
                    chunk,
                ));
            }
            profiles.insert(profile);
            chunks.extend(found);
            built_files.push(BuiltFile {
                key: file.job.key,
                ordinal: file.job.ordinal,
                content_version: file.job.content_version,
                unread_parts,
            });
        }
        if chunks.is_empty() {
            return Ok((!replaces.is_empty()).then(|| BuiltSegment {
                index_id: self.index().id().clone(),
                entry: None,
                files: Vec::new(),
                replaces,
                guard,
                drops: false,
                usage: Usage::new(),
            }));
        }
        let id = SegmentId::new(new_commit_id()?.as_str())?;
        let table = ChunkTable::new(records)?;
        let plugin = index.get();
        let built = plugin.build(&id, &chunks).await?;
        let usage = built.usage();
        let mut files = built.into_files();
        let name = SegmentFileName::new(format!("{CHUNKS_PREFIX}{id}.parquet"))?;
        files.push(cosmosearch_core::BuiltFile::new(name, table.encode()?));
        let mut stored = Vec::with_capacity(files.len());
        for file in files {
            let name = file.name().clone();
            let size = u64::try_from(file.body().len()).unwrap_or(u64::MAX);
            self.index()
                .write_segment_file(&id, &name, file.into_body().into())
                .await?;
            stored.push(SegmentFile::new(name, size));
        }
        let built_by = BuiltBy::new(Some(plugin.kind().clone()), profiles);
        let entry = SegmentEntry::new(
            id,
            plugin.level(),
            built_by,
            plugin.model_id().cloned(),
            stored,
        );
        Ok(Some(BuiltSegment {
            index_id: self.index().id().clone(),
            entry: Some(entry),
            files: built_files,
            replaces,
            guard,
            drops: false,
            usage,
        }))
    }

    /// Makes a built segment part of the index, if it still applies. Safe to repeat.
    ///
    /// # Errors
    ///
    /// Returns [`Error::WrongIndex`] for a segment of another index, [`Error::WrongPlugin`] for
    /// a rebuild built with another plug-in than the segments it replaces,
    /// [`Error::CommitUnknown`] if the commit may have landed, in which case publishing again
    /// is safe, and [`Error::TooManyConflicts`] if other writers kept winning.
    pub async fn publish(&self, built: &BuiltSegment) -> Result<PublishOutcome> {
        self.check_index(&built.index_id)?;
        let mut files_checked = false;
        for _ in 0..MAX_COMMIT_ATTEMPTS {
            let head = self.head().await?;
            if built.id().is_some_and(|id| references(&head, id)) {
                return Ok(PublishOutcome::AlreadyDone);
            }
            if !built.replaces.iter().all(|id| references(&head, id)) {
                return Ok(PublishOutcome::Superseded);
            }
            check_plugin(&head, built)?;
            if appeared(&head, built.guard.as_ref(), built.keys(), &built.replaces) {
                return Ok(PublishOutcome::Superseded);
            }
            let updates = match self.prepare_publish(&head, built).await? {
                Prepared::Updates(updates) => updates,
                Prepared::Finished(outcome) => return Ok(outcome),
            };
            if !files_checked {
                if !self.files_exist(built).await? {
                    return Ok(PublishOutcome::Expired);
                }
                files_checked = true;
            }
            let next = self.next_manifest(&head, updates).await?;
            let outcome = self.index().commit(&next).await?;
            if let CommitOutcome::Committed(version) = outcome {
                return Ok(PublishOutcome::Published(version));
            }
            if outcome != CommitOutcome::Conflict {
                return Err(Error::CommitUnknown);
            }
        }
        Err(Error::TooManyConflicts {
            attempts: MAX_COMMIT_ATTEMPTS,
        })
    }

    async fn files_exist(&self, built: &BuiltSegment) -> Result<bool> {
        let Some(entry) = &built.entry else {
            return Ok(true);
        };
        for file in entry.files() {
            if !self
                .index()
                .segment_file_exists(entry.id(), file.name())
                .await?
            {
                return Ok(false);
            }
        }
        Ok(true)
    }

    async fn prepare_publish(&self, head: &Manifest, built: &BuiltSegment) -> Result<Prepared> {
        let over_ready = !built.replaces.is_empty() || built.guard.is_some();
        let mut tables: BTreeMap<ShardId, FileTable> = BTreeMap::new();
        let mut stale = Vec::new();
        let mut done = 0usize;
        for file in &built.files {
            let (id, shard) = head.shard_for(&file.key).ok_or_else(|| Error::NoShard {
                key: file.key.clone(),
            })?;
            if !tables.contains_key(id) {
                tables.insert(id.clone(), self.load_table(shard).await?);
            }
            let Some(table) = tables.get_mut(id) else {
                continue;
            };
            match verdict(table, file, over_ready) {
                Verdict::Valid => {
                    table.mark_ready(&file.key, file.unread_parts);
                }
                Verdict::AlreadyDone => {
                    done = done.saturating_add(1);
                    stale.push(file.key.clone());
                }
                Verdict::Stale => stale.push(file.key.clone()),
            }
        }
        if !over_ready && done == built.files.len() {
            return Ok(Prepared::Finished(PublishOutcome::AlreadyDone));
        }
        if !stale.is_empty() {
            stale.sort();
            return Ok(Prepared::Finished(PublishOutcome::Stale(stale)));
        }
        let touched: BTreeSet<ShardId> = built
            .files
            .iter()
            .filter_map(|file| head.shard_for(&file.key).map(|(id, _)| id.clone()))
            .collect();
        check_ordinals(head, &tables, &touched, built)?;
        let replaced: BTreeSet<SegmentId> = built.replaces.iter().cloned().collect();
        for (id, shard) in head.shards() {
            let holds_replaced = shard.segments().iter().any(|s| replaced.contains(s.id()));
            if holds_replaced && !tables.contains_key(id) {
                tables.insert(id.clone(), self.load_table(shard).await?);
            }
        }
        let missing = if built.drops {
            Vec::new()
        } else {
            self.left_out(head, &tables, built).await?
        };
        if !missing.is_empty() {
            return Ok(Prepared::Finished(PublishOutcome::Incomplete(missing)));
        }
        let updates = tables
            .into_iter()
            .map(|(id, table)| {
                let add = match &built.entry {
                    Some(entry) if touched.contains(&id) => vec![entry.clone()],
                    Some(_) | None => Vec::new(),
                };
                let update = ShardUpdate {
                    table,
                    add,
                    remove: replaced.clone(),
                };
                (id, update)
            })
            .collect();
        Ok(Prepared::Updates(updates))
    }

    /// The live files of the segments a rebuild replaces that the rebuild does not hold.
    async fn left_out(
        &self,
        head: &Manifest,
        tables: &BTreeMap<ShardId, FileTable>,
        built: &BuiltSegment,
    ) -> Result<Vec<String>> {
        let included: BTreeSet<(&str, ManifestVersion)> = built
            .files
            .iter()
            .map(|file| (file.key.as_str(), file.content_version))
            .collect();
        let segments = chunk_segments(head);
        let mut missing = BTreeSet::new();
        for id in &built.replaces {
            let Some((segment, shards)) = segments.get(id) else {
                continue;
            };
            let Some(chunks) = self.load_chunks(segment).await? else {
                continue;
            };
            for (ordinal, version) in chunks.file_versions() {
                if let Some((_, entry)) = live_file(tables, shards, ordinal, version)
                    && !included.contains(&(entry.key(), version))
                {
                    missing.insert(entry.key().to_owned());
                }
            }
        }
        Ok(missing.into_iter().collect())
    }
}

/// Refuses a rebuild built by another plug-in than one of the segments it replaces, which
/// would leave the chunks of that plug-in's kind out of its searches.
fn check_plugin(head: &Manifest, built: &BuiltSegment) -> Result<()> {
    let Some(entry) = &built.entry else {
        return Ok(());
    };
    let replaced: BTreeSet<&SegmentId> = built.replaces.iter().collect();
    for shard in head.shards().values() {
        for segment in shard.segments() {
            if !replaced.contains(segment.id()) || same_plugin(segment, entry) {
                continue;
            }
            return Err(match segment.built_by().index() {
                Some(kind) => Error::WrongPlugin { kind: kind.clone() },
                None => Error::WrongLevel {
                    level: segment.level(),
                },
            });
        }
    }
    Ok(())
}

fn same_plugin(a: &SegmentEntry, b: &SegmentEntry) -> bool {
    a.built_by().index() == b.built_by().index() && a.model_id() == b.model_id()
}

/// Refuses a segment whose ordinals would name a different file in a shard it joins, since
/// chunk records find their file by ordinal.
fn check_ordinals(
    head: &Manifest,
    tables: &BTreeMap<ShardId, FileTable>,
    touched: &BTreeSet<ShardId>,
    built: &BuiltSegment,
) -> Result<()> {
    for file in &built.files {
        let Some((own, _)) = head.shard_for(&file.key) else {
            continue;
        };
        for shard in touched {
            let other = tables
                .get(shard)
                .and_then(|table| table.by_ordinal(file.ordinal));
            if other.is_some_and(|entry| entry.key() != file.key) {
                return Err(Error::MixedShards {
                    first: own.clone(),
                    second: shard.clone(),
                });
            }
        }
    }
    Ok(())
}

enum Prepared {
    Updates(BTreeMap<ShardId, ShardUpdate>),
    Finished(PublishOutcome),
}
