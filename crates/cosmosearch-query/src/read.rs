use std::collections::BTreeSet;
use std::sync::Arc;

use bytes::Bytes;
use cosmosearch_chunk::extract_chunks;
use cosmosearch_core::{
    ByteRange, Chunk, ETag, ExtractFailure, ExtractLimits, Extractor, ExtractorProfile, Filter,
    Level, LimitKind, Location, Locator, Manifest, Passage, Scope, SegmentEntry, ShardEntry,
    SharedExtractor, SourceRef, Usage,
};
use cosmosearch_format::{ChunkIndex, ChunkRecord, FileEntry};
use cosmosearch_store::{DataStore, ObjectKey};

use crate::tables::Loader;
use crate::{Error, Plugin, Plugins, Result};

const PASSAGE_SEPARATOR: &str = "\n\n";

/// Text read back from a file, as chunks with their refs, and what reading it cost.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reading {
    chunks: Vec<Chunk>,
    usage: Usage,
}

impl Reading {
    pub(crate) fn new(chunks: Vec<Chunk>, usage: Usage) -> Self {
        Reading { chunks, usage }
    }

    /// The chunks, in reading order.
    pub fn chunks(&self) -> &[Chunk] {
        &self.chunks
    }

    /// Takes the chunks.
    pub fn into_chunks(self) -> Vec<Chunk> {
        self.chunks
    }

    /// What the read cost in storage.
    pub fn usage(&self) -> Usage {
        self.usage
    }
}

/// How many neighbouring chunks of the same file to read before and after the one named.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Reach {
    pub(crate) before: usize,
    pub(crate) after: usize,
}

/// The files a caller may read: those in a scope that pass a filter.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Access<'a> {
    pub(crate) scope: &'a Scope,
    pub(crate) filter: &'a Filter,
}

/// What reading needs besides the snapshot.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Sources<'a> {
    pub(crate) loader: &'a Loader<'a>,
    pub(crate) plugins: &'a Plugins,
    pub(crate) data: &'a DataStore,
    pub(crate) limits: ExtractLimits,
}

/// Where a ref's passage was found: the segment and chunk index holding it, its first and
/// last row, and the extractor that found it.
struct Found {
    segment: SegmentEntry,
    index: Arc<ChunkIndex>,
    first: usize,
    last: usize,
    profile: ExtractorProfile,
}

/// What to read: the passage a ref names with its neighbours, or a run of a file's chunks
/// in reading order.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Target<'a> {
    Passage {
        reference: &'a SourceRef,
        reach: Reach,
    },
    File {
        key: &'a str,
        first: usize,
        limit: usize,
    },
}

impl Target<'_> {
    pub(crate) fn key(&self) -> &str {
        match self {
            Target::Passage { reference, .. } => reference.location().key(),
            Target::File { key, .. } => key,
        }
    }

    pub(crate) fn not_in_index(&self) -> Error {
        match self {
            Target::Passage { reference, .. } => Error::NotInIndex {
                reference: (*reference).clone(),
            },
            Target::File { key, .. } => Error::FileNotInIndex {
                key: (*key).to_owned(),
            },
        }
    }
}

/// Reads what `target` names. A file outside `access` is reported as not in the index,
/// before anything else about it is checked.
pub(crate) async fn run(
    sources: Sources<'_>,
    manifest: &Manifest,
    target: Target<'_>,
    access: Option<Access<'_>>,
) -> Result<Vec<Chunk>> {
    match target {
        Target::Passage { reference, reach } => {
            passage(sources, manifest, reference, reach, access).await
        }
        Target::File { key, first, limit } => {
            file_chunks(sources, manifest, key, first, limit, access).await
        }
    }
}

async fn passage(
    sources: Sources<'_>,
    manifest: &Manifest,
    reference: &SourceRef,
    reach: Reach,
    access: Option<Access<'_>>,
) -> Result<Vec<Chunk>> {
    let location = reference.location();
    let key = inside_data(sources.data, reference)?;
    let not_in_index = || Error::NotInIndex {
        reference: reference.clone(),
    };
    if access.is_some_and(|access| !access.scope.contains(location.key())) {
        return Err(not_in_index());
    }
    let (_, shard) = manifest
        .shard_for(location.key())
        .ok_or_else(not_in_index)?;
    let entry = sources
        .loader
        .file(shard, location.key())
        .await?
        .ok_or_else(not_in_index)?;
    let entry = &entry;
    if access.is_some_and(|access| !access.filter.matches(entry.attributes(), entry.modified())) {
        return Err(not_in_index());
    }
    if !reference.is_current(entry.etag()) {
        return Err(Error::StaleRef {
            reference: reference.clone(),
        });
    }
    let found = find(sources.loader, shard, entry, reference.locator())
        .await?
        .ok_or_else(|| Error::NoPassage {
            reference: reference.clone(),
        })?;
    let wanted = wanted(sources.loader, &found, reach).await?;
    if wanted.is_empty() {
        return Err(Error::NoPassage {
            reference: reference.clone(),
        });
    }
    read_records(sources, location, &key, entry, &wanted, reference).await
}

/// Reads up to `limit` chunks of the file `key`, from its chunk at position `first` in reading
/// order; none once `first` is past its last chunk.
async fn file_chunks(
    sources: Sources<'_>,
    manifest: &Manifest,
    key: &str,
    first: usize,
    limit: usize,
    access: Option<Access<'_>>,
) -> Result<Vec<Chunk>> {
    let object = file_inside(sources.data, key)?;
    let not_in_index = || Error::FileNotInIndex {
        key: key.to_owned(),
    };
    if access.is_some_and(|access| !access.scope.contains(key)) {
        return Err(not_in_index());
    }
    let (_, shard) = manifest.shard_for(key).ok_or_else(not_in_index)?;
    let entry = sources
        .loader
        .file(shard, key)
        .await?
        .ok_or_else(not_in_index)?;
    let entry = &entry;
    if access.is_some_and(|access| !access.filter.matches(entry.attributes(), entry.modified())) {
        return Err(not_in_index());
    }
    let wanted = file_records(sources.loader, shard, entry, first, limit).await?;
    let Some(start) = wanted.first() else {
        return Ok(Vec::new());
    };
    let data = sources.data.location();
    let location = Location::new(data.scheme(), data.authority(), key)?;
    let reference = SourceRef::new(location.clone(), start.locator(), entry.etag().clone())?;
    read_records(sources, &location, &object, entry, &wanted, &reference).await
}

/// The records of up to `limit` chunks of the file at `entry`, one for each locator in reading
/// order, from position `first`.
async fn file_records(
    loader: &Loader<'_>,
    shard: &ShardEntry,
    entry: &FileEntry,
    first: usize,
    limit: usize,
) -> Result<Vec<ChunkRecord>> {
    for segment in shard.segments() {
        if segment.level() == Level::L0 {
            continue;
        }
        let Some(index) = loader.index(segment).await? else {
            continue;
        };
        let rows = index.file_rows(entry.ordinal(), entry.content_version());
        if rows.is_empty() {
            continue;
        }
        let mut last: Option<Locator> = None;
        let mut distinct = 0usize;
        let mut records = Vec::new();
        loader
            .scan(segment, &index, rows, |_, record| {
                if last == Some(record.locator()) {
                    return true;
                }
                last = Some(record.locator());
                let at = distinct;
                distinct = distinct.saturating_add(1);
                if at >= first.saturating_add(limit) {
                    return false;
                }
                if at >= first {
                    records.push(record.clone());
                }
                true
            })
            .await?;
        return Ok(records);
    }
    Ok(Vec::new())
}

/// Reads the text of `wanted`, records of one file in reading order, with one read pinned to
/// the file's ETag and one extraction by the extractor that indexed the first of them.
async fn read_records(
    sources: Sources<'_>,
    location: &Location,
    key: &ObjectKey,
    entry: &FileEntry,
    wanted: &[ChunkRecord],
    reference: &SourceRef,
) -> Result<Vec<Chunk>> {
    let Some(profile) = wanted.first().map(ChunkRecord::profile) else {
        return Ok(Vec::new());
    };
    let extractor = sources
        .plugins
        .extractors()
        .get(profile)
        .cloned()
        .ok_or_else(|| {
            Error::MissingPlugins(BTreeSet::from([Plugin::Extractor(profile.clone())]))
        })?;
    let locators: Vec<Locator> = wanted.iter().map(ChunkRecord::locator).collect();
    let span = span_of(&locators, entry.size())?;
    let bytes = read_pinned(sources, key, span, entry, reference).await?;
    let texts = extract_texts(
        extractor,
        location.clone(),
        entry.etag().clone(),
        bytes,
        span.start(),
        locators,
        sources.limits,
    )
    .await?;
    let mut chunks = Vec::with_capacity(texts.len());
    for (record, text) in wanted.iter().zip(texts) {
        let passage = Passage::new(record.locator(), record.heading_path().to_vec(), text);
        chunks.push(passage.into_chunk(location, entry.etag())?);
    }
    Ok(chunks)
}

/// Reads the passages `references` name, which are all of one file at one ETag, with one
/// read of the file and one extraction. A passage the index no longer holds, or one indexed
/// by another extractor than the first found, has no text.
pub(crate) async fn passages(
    sources: Sources<'_>,
    manifest: &Manifest,
    references: &[&SourceRef],
) -> Result<Vec<Option<String>>> {
    let Some(first) = references.first() else {
        return Ok(Vec::new());
    };
    let location = first.location();
    let key = inside_data(sources.data, first)?;
    let not_in_index = || Error::NotInIndex {
        reference: (*first).clone(),
    };
    let (_, shard) = manifest
        .shard_for(location.key())
        .ok_or_else(not_in_index)?;
    let entry = sources
        .loader
        .file(shard, location.key())
        .await?
        .ok_or_else(not_in_index)?;
    let entry = &entry;
    if !first.is_current(entry.etag()) {
        return Err(Error::StaleRef {
            reference: (*first).clone(),
        });
    }
    let mut profile: Option<ExtractorProfile> = None;
    let mut locators: Vec<Locator> = Vec::new();
    let mut found: Vec<Option<Locator>> = Vec::with_capacity(references.len());
    for reference in references {
        let locator = reference.locator();
        let at = find(sources.loader, shard, entry, locator).await?;
        let indexed_by = at.as_ref().map(|at| &at.profile);
        let readable = match (indexed_by, &profile) {
            (Some(by), Some(chosen)) => by == chosen,
            (Some(by), None) => {
                profile = Some(by.clone());
                true
            }
            (None, _) => false,
        };
        if readable && !locators.contains(&locator) {
            locators.push(locator);
        }
        found.push(readable.then_some(locator));
    }
    let Some(profile) = profile else {
        return Ok(vec![None; references.len()]);
    };
    let extractor = sources
        .plugins
        .extractors()
        .get(&profile)
        .cloned()
        .ok_or_else(|| {
            Error::MissingPlugins(BTreeSet::from([Plugin::Extractor(profile.clone())]))
        })?;
    let span = span_of(&locators, entry.size())?;
    let bytes = read_pinned(sources, &key, span, entry, first).await?;
    let texts = extract_texts(
        extractor,
        location.clone(),
        entry.etag().clone(),
        bytes,
        span.start(),
        locators.clone(),
        sources.limits,
    )
    .await?;
    let text_at = |locator: Locator| {
        let index = locators.iter().position(|l| *l == locator)?;
        texts.get(index).cloned()
    };
    Ok(found
        .into_iter()
        .map(|locator| locator.and_then(text_at))
        .collect())
}

pub(crate) fn file_inside(data: &DataStore, key: &str) -> Result<ObjectKey> {
    let outside = || Error::FileOutsideData {
        key: key.to_owned(),
    };
    let object = ObjectKey::new(key).map_err(|_cause| outside())?;
    data.check_inside(&object).map_err(|_cause| outside())?;
    Ok(object)
}

fn inside_data(data: &DataStore, reference: &SourceRef) -> Result<ObjectKey> {
    let outside = || Error::OutsideData {
        reference: reference.clone(),
    };
    if !data.location().contains(reference.location()) {
        return Err(outside());
    }
    let key = ObjectKey::new(reference.location().key()).map_err(|_cause| outside())?;
    data.check_inside(&key).map_err(|_cause| outside())?;
    Ok(key)
}

async fn find(
    loader: &Loader<'_>,
    shard: &ShardEntry,
    entry: &FileEntry,
    locator: Locator,
) -> Result<Option<Found>> {
    for segment in shard.segments() {
        if segment.level() == Level::L0 {
            continue;
        }
        let Some(index) = loader.index(segment).await? else {
            continue;
        };
        let rows = index.file_rows(entry.ordinal(), entry.content_version());
        let mut matching: Option<(usize, usize)> = None;
        loader
            .scan(segment, &index, rows, |row, record| {
                if record.locator() != locator {
                    return matching.is_none();
                }
                matching = Some(matching.map_or((row, row), |(first, _)| (first, row)));
                true
            })
            .await?;
        if let Some((first, last)) = matching {
            let Some(place) = index.place(first) else {
                continue;
            };
            let profile = place.profile.clone();
            return Ok(Some(Found {
                segment: segment.clone(),
                first,
                last,
                profile,
                index,
            }));
        }
    }
    Ok(None)
}

/// The records to read, one for each locator in reading order.
async fn wanted(loader: &Loader<'_>, found: &Found, reach: Reach) -> Result<Vec<ChunkRecord>> {
    let start = found.index.neighbours(found.first, reach.before, 0).start;
    let end = found.index.neighbours(found.last, 0, reach.after).end;
    let mut wanted: Vec<ChunkRecord> = Vec::new();
    for record in loader
        .records(&found.segment, &found.index, start..end)
        .await?
    {
        if wanted
            .last()
            .is_none_or(|last| last.locator() != record.locator())
        {
            wanted.push(record);
        }
    }
    Ok(wanted)
}

/// The bytes to read: the whole file if any locator needs it parsed, else the ranges' span.
fn span_of(locators: &[Locator], size: u64) -> Result<ByteRange> {
    let mut span: Option<(u64, u64)> = None;
    for locator in locators {
        let Locator::Bytes(range) = locator else {
            return Ok(ByteRange::new(0, size)?);
        };
        span = Some(match span {
            Some((start, end)) => (start.min(range.start()), end.max(range.end())),
            None => (range.start(), range.end()),
        });
    }
    let (start, end) = span.unwrap_or((0, size));
    Ok(ByteRange::new(start, end)?)
}

async fn read_pinned(
    sources: Sources<'_>,
    key: &ObjectKey,
    span: ByteRange,
    entry: &FileEntry,
    reference: &SourceRef,
) -> Result<Bytes> {
    if span.len() > sources.limits.file_bytes() {
        let failure = ExtractFailure::LimitExceeded(LimitKind::FileBytes);
        return Err(Error::Extract(cosmosearch_chunk::Error::File(failure)));
    }
    let read = sources.data.read_range(key, span, entry.etag()).await;
    sources
        .loader
        .counter()
        .record(read.as_ref().map_or(0, Bytes::len));
    match read {
        Ok(bytes) => Ok(bytes),
        Err(
            cosmosearch_store::Error::Stale { .. }
            | cosmosearch_store::Error::NotFound { .. }
            | cosmosearch_store::Error::ShortRead { .. },
        ) => Err(Error::StaleRef {
            reference: reference.clone(),
        }),
        Err(err) => Err(err.into()),
    }
}

async fn extract_texts(
    extractor: SharedExtractor,
    location: Location,
    etag: ETag,
    bytes: Bytes,
    offset: u64,
    locators: Vec<Locator>,
    limits: ExtractLimits,
) -> Result<Vec<String>> {
    tokio::task::spawn_blocking(move || {
        texts_at(
            extractor.get(),
            &location,
            &etag,
            &bytes,
            offset,
            &locators,
            &limits,
        )
    })
    .await
    .map_err(Error::Worker)?
}

/// The text at each locator. A byte range is extracted on its own, which gives back the
/// passage; any other locator is found by extracting the whole file once.
fn texts_at(
    extractor: &dyn Extractor,
    location: &Location,
    etag: &ETag,
    bytes: &[u8],
    offset: u64,
    locators: &[Locator],
    limits: &ExtractLimits,
) -> Result<Vec<String>> {
    let mut whole: Option<Vec<Chunk>> = None;
    let mut texts = Vec::with_capacity(locators.len());
    for locator in locators {
        let found: Vec<Chunk> = if let Locator::Bytes(range) = locator {
            let part =
                slice(bytes, offset, *range).ok_or(cosmosearch_core::Error::InvalidRange {
                    start: range.start(),
                    end: range.end(),
                })?;
            extract_chunks(extractor, location, etag, part, limits)?.into_chunks()
        } else {
            if whole.is_none() {
                whole =
                    Some(extract_chunks(extractor, location, etag, bytes, limits)?.into_chunks());
            }
            whole
                .iter()
                .flatten()
                .filter(|chunk| chunk.source().locator() == *locator)
                .cloned()
                .collect()
        };
        let parts: Vec<&str> = found.iter().map(Chunk::text).collect();
        texts.push(parts.join(PASSAGE_SEPARATOR));
    }
    Ok(texts)
}

fn slice(bytes: &[u8], offset: u64, range: ByteRange) -> Option<&[u8]> {
    let start = usize::try_from(range.start().checked_sub(offset)?).ok()?;
    let end = usize::try_from(range.end().checked_sub(offset)?).ok()?;
    bytes.get(start..end)
}
