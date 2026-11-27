use std::collections::BTreeSet;
use std::ops::Range;
use std::sync::Arc;

use bytes::Bytes;
use cosmosearch_chunk::extract_chunks;
use cosmosearch_core::{
    ByteRange, Chunk, ETag, ExtractFailure, ExtractLimits, Extractor, ExtractorProfile, Filter,
    Level, LimitKind, Location, Locator, Manifest, Passage, Scope, SegmentEntry, ShardEntry,
    SharedExtractor, SourceRef, Usage,
};
use cosmosearch_format::{ChunkIndex, ChunkRecord, FileEntry, TextGroup};
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

/// The rows of one segment holding the chunks of one locator, with the first one's record.
#[derive(Debug, Clone)]
struct Wanted {
    record: ChunkRecord,
    rows: Range<usize>,
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
    let read = Read {
        location,
        key: &key,
        entry,
        segment: &found.segment,
        reference,
    };
    read_records(sources, &read, &wanted).await
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
    let Some((segment, wanted)) = file_records(sources.loader, shard, entry, first, limit).await?
    else {
        return Ok(Vec::new());
    };
    let Some(start) = wanted.first() else {
        return Ok(Vec::new());
    };
    let data = sources.data.location();
    let location = Location::new(data.scheme(), data.authority(), key)?;
    let reference = SourceRef::new(
        location.clone(),
        start.record.locator(),
        entry.etag().clone(),
    )?;
    let read = Read {
        location: &location,
        key: &object,
        entry,
        segment: &segment,
        reference: &reference,
    };
    read_records(sources, &read, &wanted).await
}

/// The rows of up to `limit` chunks of the file at `entry`, one run for each locator in
/// reading order, from position `first`, and the segment holding them.
async fn file_records(
    loader: &Loader<'_>,
    shard: &ShardEntry,
    entry: &FileEntry,
    first: usize,
    limit: usize,
) -> Result<Option<(SegmentEntry, Vec<Wanted>)>> {
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
        let mut runs: Vec<Wanted> = Vec::new();
        let mut last: Option<Locator> = None;
        let mut distinct = 0usize;
        let end = first.saturating_add(limit);
        loader
            .scan(segment, &index, rows, |row, record| {
                let locator = record.locator();
                if last == Some(locator) {
                    if let Some(run) = runs.last_mut()
                        && run.rows.end == row
                    {
                        run.rows.end = row.saturating_add(1);
                    }
                    return true;
                }
                last = Some(locator);
                let at = distinct;
                distinct = distinct.saturating_add(1);
                if at >= end {
                    return false;
                }
                if at >= first {
                    runs.push(Wanted {
                        record: record.clone(),
                        rows: row..row.saturating_add(1),
                    });
                }
                true
            })
            .await?;
        return Ok(Some((segment.clone(), runs)));
    }
    Ok(None)
}

/// Reads the text of `wanted`, records of one file in reading order, with one read pinned to
/// the file's ETag and one extraction by the extractor that indexed the first of them.
/// What one read is of: a file as the index holds it, the segment holding its chunks, and
/// the ref a stale file is reported with.
struct Read<'a> {
    location: &'a Location,
    key: &'a ObjectKey,
    entry: &'a FileEntry,
    segment: &'a SegmentEntry,
    reference: &'a SourceRef,
}

/// The chunks of `wanted`, one for each run: from the segment's stored text where it holds
/// every row, after checking that the file is still the version indexed unless the read asks
/// for that version; otherwise read from the file and extracted.
async fn read_records(
    sources: Sources<'_>,
    read: &Read<'_>,
    wanted: &[Wanted],
) -> Result<Vec<Chunk>> {
    let texts = match stored_texts(sources.loader, read.segment, wanted).await? {
        Some(texts) => {
            check_current(sources, read).await?;
            texts
        }
        None => extracted(sources, read, wanted).await?,
    };
    let mut chunks = Vec::with_capacity(texts.len());
    for (wanted, text) in wanted.iter().zip(texts) {
        let record = &wanted.record;
        let passage = Passage::new(record.locator(), record.heading_path().to_vec(), text);
        chunks.push(passage.into_chunk(read.location, read.entry.etag())?);
    }
    Ok(chunks)
}

/// The text of each of `wanted` from `segment`'s stored text, or `None` unless it holds every
/// row of them.
async fn stored_texts(
    loader: &Loader<'_>,
    segment: &SegmentEntry,
    wanted: &[Wanted],
) -> Result<Option<Vec<String>>> {
    let Some(index) = loader.text_index(segment).await? else {
        return Ok(None);
    };
    let complete = wanted
        .iter()
        .all(|wanted| wanted.rows.clone().all(|row| index.has_text(row)));
    if !complete {
        return Ok(None);
    }
    let mut texts = Vec::with_capacity(wanted.len());
    let mut held: Option<(usize, Arc<TextGroup>)> = None;
    for wanted in wanted {
        let mut parts = Vec::with_capacity(wanted.rows.len());
        for row in wanted.rows.clone() {
            let Some(group) = index.group_of(row) else {
                return Ok(None);
            };
            let found = match &held {
                Some((at, found)) if *at == group => Arc::clone(found),
                _ => {
                    let Some(found) = loader.text_group(segment, &index, group).await? else {
                        return Ok(None);
                    };
                    held = Some((group, Arc::clone(&found)));
                    found
                }
            };
            match found.get(row) {
                Some(text) => parts.push(text),
                None => return Ok(None),
            }
        }
        let parts: Vec<&str> = parts.iter().map(AsRef::as_ref).collect();
        texts.push(parts.join(PASSAGE_SEPARATOR));
    }
    Ok(Some(texts))
}

/// Checks with one request that the file is still the version the index holds.
async fn check_current(sources: Sources<'_>, read: &Read<'_>) -> Result<()> {
    let found = sources.data.head(read.key).await;
    sources.loader.counter().record(0);
    let stale = || Error::StaleRef {
        reference: read.reference.clone(),
    };
    match found {
        Ok(info) if info.etag() == read.entry.etag() => Ok(()),
        Ok(_) | Err(cosmosearch_store::Error::NotFound { .. }) => Err(stale()),
        Err(err) => Err(err.into()),
    }
}

/// The text of each of `wanted`, read from the file and extracted.
async fn extracted(
    sources: Sources<'_>,
    read: &Read<'_>,
    wanted: &[Wanted],
) -> Result<Vec<String>> {
    let Some(profile) = wanted.first().map(|wanted| wanted.record.profile()) else {
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
    let locators: Vec<Locator> = wanted
        .iter()
        .map(|wanted| wanted.record.locator())
        .collect();
    let span = span_of(&locators, read.entry.size())?;
    let bytes = read_pinned(sources, read.key, span, read.entry, read.reference).await?;
    extract_texts(
        extractor,
        read.location.clone(),
        read.entry.etag().clone(),
        bytes,
        span.start(),
        locators,
        sources.limits,
    )
    .await
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

/// The rows to read, one run for each locator in reading order.
async fn wanted(loader: &Loader<'_>, found: &Found, reach: Reach) -> Result<Vec<Wanted>> {
    let start = found.index.neighbours(found.first, reach.before, 0).start;
    let end = found.index.neighbours(found.last, 0, reach.after).end;
    let mut wanted: Vec<Wanted> = Vec::new();
    loader
        .scan(&found.segment, &found.index, start..end, |row, record| {
            match wanted.last_mut() {
                Some(last) if last.record.locator() == record.locator() => {
                    last.rows.end = row.saturating_add(1);
                }
                Some(_) | None => wanted.push(Wanted {
                    record: record.clone(),
                    rows: row..row.saturating_add(1),
                }),
            }
            true
        })
        .await?;
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
