use std::ops::Range;
use std::sync::Arc;

use arrow_array::builder::{ListBuilder, StringBuilder};
use arrow_array::{Array, ArrayRef, FixedSizeBinaryArray, ListArray, StringArray, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use cosmosearch_core::{
    Chunk, ChunkHash, ExtractorProfile, Filter, Location, Locator, ManifestVersion, RowSet, Scope,
    SourceRef,
};

use crate::error::malformed;
use crate::grouped::{self, GROUP_ROWS, Groups, as_u64, as_usize, slice};
use crate::stored::{self, required};
use crate::{Error, FileEntry, FileIndex, FileTable, Result};

const FORMAT: u32 = 3;
const HASH_BYTES: i32 = 32;
const MAGIC: [u8; 8] = *b"CSCHUNKS";

const FILE_ORDINAL: &str = "file_ordinal";
const CONTENT_VERSION: &str = "content_version";
const PROFILE: &str = "profile";
const LOCATOR: &str = "locator";
const HEADING_PATH: &str = "heading_path";
const HASH: &str = "hash";
const FIRST_ROW: &str = "first_row";
const ROWS: &str = "rows";
const SPAN_BYTES: usize = size_of::<FileSpan>() + size_of::<(u64, usize)>();

/// Where one chunk came from. It holds no text, only what is needed to find it again.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkRecord {
    file_ordinal: u64,
    content_version: ManifestVersion,
    profile: ExtractorProfile,
    locator: Locator,
    heading_path: Vec<String>,
    hash: ChunkHash,
}

impl ChunkRecord {
    /// A record of `chunk`, which the extractor with `profile` found in the file with this
    /// ordinal at this content version.
    pub fn new(
        file_ordinal: u64,
        content_version: ManifestVersion,
        profile: ExtractorProfile,
        chunk: &Chunk,
    ) -> Self {
        ChunkRecord {
            file_ordinal,
            content_version,
            profile,
            locator: chunk.source().locator(),
            heading_path: chunk.heading_path().to_vec(),
            hash: chunk.hash(),
        }
    }

    /// The file's ordinal in its shard's table.
    pub fn file_ordinal(&self) -> u64 {
        self.file_ordinal
    }

    /// The content version of the file this chunk was built from.
    pub fn content_version(&self) -> ManifestVersion {
        self.content_version
    }

    /// The extractor that found the chunk, which can find it again.
    pub fn profile(&self) -> &ExtractorProfile {
        &self.profile
    }

    /// Where the chunk sits in its file.
    pub fn locator(&self) -> Locator {
        self.locator
    }

    /// The headings above the chunk, outermost first.
    pub fn heading_path(&self) -> &[String] {
        &self.heading_path
    }

    /// The hash of the chunk's normalized text.
    pub fn hash(&self) -> ChunkHash {
        self.hash
    }

    /// The ref of this chunk in `file`, or `None` if it is not a live chunk of that file.
    ///
    /// # Errors
    ///
    /// Returns an error if the location or the ref cannot be formed.
    pub fn source_ref(&self, file: &FileEntry, data: &Location) -> Result<Option<SourceRef>> {
        if file.ordinal() != self.file_ordinal || file.content_version() != self.content_version {
            return Ok(None);
        }
        let location = Location::new(data.scheme(), data.authority(), file.key())?;
        let source = SourceRef::new(location, self.locator, file.etag().clone())?;
        Ok(Some(source))
    }

    fn heap_bytes(&self) -> usize {
        let headings: usize = self
            .heading_path
            .iter()
            .map(|heading| heading.capacity().saturating_add(size_of::<String>()))
            .sum();
        size_of::<Self>()
            .saturating_add(self.profile.as_str().len())
            .saturating_add(headings)
    }
}

/// The rows of one file's chunks, which share a content version.
#[derive(Debug, Clone, PartialEq, Eq)]
struct FileSpan {
    file_ordinal: u64,
    content_version: ManifestVersion,
    profile: usize,
    rows: Range<usize>,
}

/// Where a chunk sits: its file, the content version and extractor it came from, and its
/// position among the file's chunks in the segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct ChunkPlace<'a> {
    /// The file's ordinal in its shard's table.
    pub file_ordinal: u64,
    /// The content version of the file the chunk was built from.
    pub content_version: ManifestVersion,
    /// The extractor that found the chunk.
    pub profile: &'a ExtractorProfile,
    /// The chunk's position among its file's chunks, from 0.
    pub position: usize,
}

/// Where each file's chunks are in a segment, by row.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Spans {
    each: Vec<FileSpan>,
    profiles: Vec<ExtractorProfile>,
    by_file: Vec<(u64, usize)>,
}

impl Spans {
    fn of(records: &[ChunkRecord]) -> Result<Self> {
        let mut spans: Vec<FileSpan> = Vec::new();
        let mut profiles = Vec::new();
        for (row, record) in records.iter().enumerate() {
            if let Some(span) = spans.last_mut()
                && span.file_ordinal == record.file_ordinal
            {
                if span.content_version != record.content_version {
                    return Err(malformed("a file has two content versions in one segment"));
                }
                if profiles.get(span.profile) != Some(&record.profile) {
                    return Err(malformed("a file has two extractors in one segment"));
                }
                span.rows.end = row.saturating_add(1);
                continue;
            }
            spans.push(FileSpan {
                file_ordinal: record.file_ordinal,
                content_version: record.content_version,
                profile: intern(&mut profiles, &record.profile),
                rows: row..row.saturating_add(1),
            });
        }
        Spans::new(spans, profiles)
    }

    /// Spans that follow each other from row 0, at most one for each file.
    fn new(spans: Vec<FileSpan>, profiles: Vec<ExtractorProfile>) -> Result<Self> {
        let mut end = 0;
        for span in &spans {
            if span.profile >= profiles.len() {
                return Err(malformed("a file's extractor is missing"));
            }
            if span.rows.start != end || span.rows.is_empty() {
                return Err(malformed("a file's chunks are not where the index says"));
            }
            end = span.rows.end;
        }
        let mut by_file: Vec<(u64, usize)> = spans
            .iter()
            .enumerate()
            .map(|(at, span)| (span.file_ordinal, at))
            .collect();
        by_file.sort_unstable();
        if by_file
            .windows(2)
            .any(|pair| matches!(pair, [a, b] if a.0 == b.0))
        {
            return Err(malformed("a file's chunks are not adjacent"));
        }
        Ok(Spans {
            each: spans,
            profiles,
            by_file,
        })
    }

    fn rows(&self) -> usize {
        self.each.last().map_or(0, |span| span.rows.end)
    }

    fn span_of(&self, row: usize) -> Option<&FileSpan> {
        let after = self.each.partition_point(|span| span.rows.start <= row);
        let span = self.each.get(after.checked_sub(1)?)?;
        span.rows.contains(&row).then_some(span)
    }

    fn place(&self, row: usize) -> Option<ChunkPlace<'_>> {
        let span = self.span_of(row)?;
        Some(ChunkPlace {
            file_ordinal: span.file_ordinal,
            content_version: span.content_version,
            profile: self.profiles.get(span.profile)?,
            position: row.checked_sub(span.rows.start)?,
        })
    }

    fn neighbours(&self, row: usize, before: usize, after: usize) -> Range<usize> {
        let Some(span) = self.span_of(row) else {
            return row..row;
        };
        let start = row.saturating_sub(before).max(span.rows.start);
        let end = row
            .saturating_add(after)
            .saturating_add(1)
            .min(span.rows.end);
        start..end
    }

    fn file_rows(&self, file_ordinal: u64, content_version: ManifestVersion) -> Range<usize> {
        let found = self
            .by_file
            .binary_search_by_key(&file_ordinal, |(ordinal, _)| *ordinal)
            .ok()
            .and_then(|at| self.by_file.get(at))
            .and_then(|(_, span)| self.each.get(*span));
        found
            .filter(|span| span.content_version == content_version)
            .map_or(0..0, |span| span.rows.clone())
    }

    fn file_versions(&self) -> Vec<(u64, ManifestVersion)> {
        self.each
            .iter()
            .map(|span| (span.file_ordinal, span.content_version))
            .collect()
    }

    fn matching(&self, live: impl Fn(&FileSpan) -> bool) -> RowSet {
        let mut rows = RowSet::new();
        for span in self.each.iter().filter(|span| live(span)) {
            rows.insert(as_u64(span.rows.start)..as_u64(span.rows.end));
        }
        rows
    }

    fn filtered(&self, files: &FileTable, scope: &Scope, filter: &Filter) -> RowSet {
        self.matching(|span| {
            files.by_ordinal(span.file_ordinal).is_some_and(|file| {
                file.content_version() == span.content_version
                    && scope.contains(file.key())
                    && filter.matches(file.attributes(), file.modified())
            })
        })
    }

    fn live(&self, files: &FileIndex) -> RowSet {
        self.matching(|span| files.version(span.file_ordinal) == Some(span.content_version))
    }

    fn heap_bytes(&self) -> usize {
        let profiles: usize = self
            .profiles
            .iter()
            .map(|profile| {
                profile
                    .as_str()
                    .len()
                    .saturating_add(size_of::<ExtractorProfile>())
            })
            .sum();
        self.each
            .len()
            .saturating_mul(SPAN_BYTES)
            .saturating_add(profiles)
    }
}

/// The chunks of one segment. A file's chunks are adjacent and in reading order.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChunkTable {
    records: Vec<ChunkRecord>,
    spans: Spans,
}

impl ChunkTable {
    /// A table of `records`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Malformed`] unless each file's records are adjacent and share one
    /// content version.
    pub fn new(records: Vec<ChunkRecord>) -> Result<Self> {
        let spans = Spans::of(&records)?;
        Ok(ChunkTable { records, spans })
    }

    /// The record in this row.
    pub fn get(&self, row: usize) -> Option<&ChunkRecord> {
        self.records.get(row)
    }

    /// How many chunks the segment holds.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the segment holds no chunks.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// The rows of the chunks of the file with this ordinal at this content version, empty if
    /// the segment holds none.
    pub fn file_rows(&self, file_ordinal: u64, content_version: ManifestVersion) -> Range<usize> {
        self.spans.file_rows(file_ordinal, content_version)
    }

    /// Each file the segment holds, as its ordinal and the content version its chunks came
    /// from, in row order.
    pub fn file_versions(&self) -> Vec<(u64, ManifestVersion)> {
        self.spans.file_versions()
    }

    /// The bytes stored for this table: its records in Parquet groups of up to 512 rows, then
    /// an index of where each file's chunks and each group are, then a trailer of
    /// [`TRAILER_BYTES`](crate::TRAILER_BYTES) giving the index's length.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Encoding`] if encoding fails.
    pub fn encode(&self) -> Result<Vec<u8>> {
        let groups = self
            .records
            .chunks(GROUP_ROWS)
            .map(encode_records)
            .collect::<Result<Vec<_>>>()?;
        grouped::assemble(
            &groups,
            GROUP_ROWS,
            |metadata| encode_index(&self.spans, metadata),
            MAGIC,
        )
    }

    /// Reads a stored table.
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnsupportedFormat`] for another format, and an error for bytes that do
    /// not decode or a table that is inconsistent.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "taken by value like the other tables' decoders"
    )]
    pub fn decode(bytes: Bytes) -> Result<Self> {
        let index = ChunkIndex::decode(grouped::index_of(&bytes, MAGIC)?)?;
        let mut records = Vec::new();
        for group in 0..index.groups.len() {
            let (_, at) = index
                .group(group)
                .ok_or_else(|| malformed("a group is missing"))?;
            let decoded = ChunkGroup::decode(slice(&bytes, at)?, &index, group)?;
            records.extend(decoded.records);
        }
        let table = ChunkTable::new(records)?;
        if table.spans != index.spans {
            return Err(malformed("the index does not match the records"));
        }
        Ok(table)
    }
}

/// Where each file's chunks and each group of records are in a segment's chunk file: what a
/// reader keeps to search the segment, without its records.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkIndex {
    spans: Spans,
    groups: Groups,
}

impl ChunkIndex {
    /// The byte range of the index in a chunk file of `size` bytes, from its last
    /// [`TRAILER_BYTES`](crate::TRAILER_BYTES) bytes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Malformed`] if the trailer is not one.
    pub fn locate(trailer: &[u8], size: u64) -> Result<Range<u64>> {
        grouped::locate(trailer, size, MAGIC)
    }

    /// Reads the index of a chunk file, the bytes [`ChunkIndex::locate`] gave.
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnsupportedFormat`] for another format, and an error for bytes that do
    /// not decode or an index that is inconsistent.
    pub fn decode(bytes: Bytes) -> Result<Self> {
        let stored = stored::read(bytes, FORMAT)?;
        let mut spans = Vec::new();
        let mut profiles = Vec::new();
        for batch in &stored.batches {
            let ordinals = required::<UInt64Array>(batch, FILE_ORDINAL)?;
            let versions = required::<UInt64Array>(batch, CONTENT_VERSION)?;
            let named = required::<StringArray>(batch, PROFILE)?;
            let firsts = required::<UInt64Array>(batch, FIRST_ROW)?;
            let counts = required::<UInt64Array>(batch, ROWS)?;
            for row in 0..batch.num_rows() {
                let start = as_usize(firsts.value(row))?;
                let end = start
                    .checked_add(as_usize(counts.value(row))?)
                    .ok_or_else(|| malformed("a file's rows are too many"))?;
                let profile = ExtractorProfile::new(named.value(row))?;
                spans.push(FileSpan {
                    file_ordinal: ordinals.value(row),
                    content_version: ManifestVersion::new(versions.value(row))?,
                    profile: intern(&mut profiles, &profile),
                    rows: start..end,
                });
            }
        }
        let spans = Spans::new(spans, profiles)?;
        let groups = Groups::of(&stored.metadata, spans.rows(), GROUP_ROWS)?;
        Ok(ChunkIndex { spans, groups })
    }

    /// How many chunks the segment holds.
    pub fn len(&self) -> usize {
        self.spans.rows()
    }

    /// Whether the segment holds no chunks.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The group holding `row`, if the segment has that row.
    pub fn group_of(&self, row: usize) -> Option<usize> {
        self.groups.of_row(row, self.len())
    }

    /// The rows of group `group` and its byte range in the chunk file.
    pub fn group(&self, group: usize) -> Option<(Range<usize>, Range<u64>)> {
        self.groups.get(group, self.len())
    }

    /// Where the chunk in `row` sits, if the segment has that row.
    pub fn place(&self, row: usize) -> Option<ChunkPlace<'_>> {
        self.spans.place(row)
    }

    /// The rows around `row` in the same file, up to `before` and `after` further chunks.
    pub fn neighbours(&self, row: usize, before: usize, after: usize) -> Range<usize> {
        self.spans.neighbours(row, before, after)
    }

    /// The rows of the chunks of the file with this ordinal at this content version, empty if
    /// the segment holds none.
    pub fn file_rows(&self, file_ordinal: u64, content_version: ManifestVersion) -> Range<usize> {
        self.spans.file_rows(file_ordinal, content_version)
    }

    /// Each file the segment holds, as its ordinal and the content version its chunks came
    /// from, in row order.
    pub fn file_versions(&self) -> Vec<(u64, ManifestVersion)> {
        self.spans.file_versions()
    }

    /// The rows whose chunks are live in `files` and whose file is in `scope` and passes
    /// `filter`.
    pub fn rows(&self, files: &FileTable, scope: &Scope, filter: &Filter) -> RowSet {
        self.spans.filtered(files, scope, filter)
    }

    /// The rows whose chunks are of the content version `files` holds for their file.
    pub fn live_rows(&self, files: &FileIndex) -> RowSet {
        self.spans.live(files)
    }

    /// About how many bytes the index holds in memory.
    pub fn heap_bytes(&self) -> usize {
        self.spans
            .heap_bytes()
            .saturating_add(self.groups.heap_bytes())
            .saturating_add(size_of::<Self>())
    }
}

/// The records of one group of rows of a segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkGroup {
    first: usize,
    records: Vec<ChunkRecord>,
}

impl ChunkGroup {
    /// Reads group `group` of the segment `index` describes, from the bytes
    /// [`ChunkIndex::group`] located.
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnsupportedFormat`] for another format, and an error for bytes that do
    /// not decode or that hold other records than the index says: another number of them, or
    /// one of another file, content version or extractor.
    pub fn decode(bytes: Bytes, index: &ChunkIndex, group: usize) -> Result<Self> {
        let (rows, _) = index
            .group(group)
            .ok_or_else(|| malformed("a group is missing"))?;
        let records = decode_records(bytes)?;
        let matches = records.len() == rows.len()
            && rows.clone().zip(&records).all(|(row, record)| {
                index.place(row).is_some_and(|place| {
                    place.file_ordinal == record.file_ordinal
                        && place.content_version == record.content_version
                        && *place.profile == record.profile
                })
            });
        if !matches {
            return Err(malformed("a group holds other records than the index says"));
        }
        Ok(ChunkGroup {
            first: rows.start,
            records,
        })
    }

    /// The record in this row of the segment, if the group holds it.
    pub fn get(&self, row: usize) -> Option<&ChunkRecord> {
        self.records.get(row.checked_sub(self.first)?)
    }

    /// About how many bytes the group holds in memory.
    pub fn heap_bytes(&self) -> usize {
        self.records
            .iter()
            .map(ChunkRecord::heap_bytes)
            .sum::<usize>()
            .saturating_add(size_of::<Self>())
    }
}

fn encode_records(records: &[ChunkRecord]) -> Result<Vec<u8>> {
    let mut paths = ListBuilder::new(StringBuilder::new());
    for record in records {
        paths.append_value(record.heading_path.iter().map(Some));
    }
    let paths = paths.finish();
    let hashes = FixedSizeBinaryArray::try_from_sparse_iter_with_size(
        records.iter().map(|r| Some(*r.hash.as_bytes())),
        HASH_BYTES,
    )
    .map_err(|source| Error::Encoding(crate::error::EncodingError::new(source)))?;
    let schema = Schema::new(vec![
        Field::new(FILE_ORDINAL, DataType::UInt64, false),
        Field::new(CONTENT_VERSION, DataType::UInt64, false),
        Field::new(PROFILE, DataType::Utf8, false),
        Field::new(LOCATOR, DataType::Utf8, false),
        Field::new(HEADING_PATH, paths.data_type().clone(), false),
        Field::new(HASH, DataType::FixedSizeBinary(HASH_BYTES), false),
    ]);
    let columns: Vec<ArrayRef> = vec![
        Arc::new(UInt64Array::from_iter_values(
            records.iter().map(|r| r.file_ordinal),
        )),
        Arc::new(UInt64Array::from_iter_values(
            records.iter().map(|r| r.content_version.get()),
        )),
        Arc::new(StringArray::from_iter_values(
            records.iter().map(|r| r.profile.as_str()),
        )),
        Arc::new(StringArray::from_iter_values(
            records.iter().map(|r| r.locator.to_string()),
        )),
        Arc::new(paths),
        Arc::new(hashes),
    ];
    stored::write(schema, columns, FORMAT, vec![])
}

fn decode_records(bytes: Bytes) -> Result<Vec<ChunkRecord>> {
    let stored = stored::read(bytes, FORMAT)?;
    let mut records = Vec::new();
    for batch in &stored.batches {
        let ordinals = required::<UInt64Array>(batch, FILE_ORDINAL)?;
        let versions = required::<UInt64Array>(batch, CONTENT_VERSION)?;
        let profiles = required::<StringArray>(batch, PROFILE)?;
        let locators = required::<StringArray>(batch, LOCATOR)?;
        let paths = required::<ListArray>(batch, HEADING_PATH)?;
        let hashes = required::<FixedSizeBinaryArray>(batch, HASH)?;
        for row in 0..batch.num_rows() {
            let path = paths.value(row);
            let path = path
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| malformed("a heading path has the wrong type"))?;
            let heading_path = path
                .iter()
                .map(|heading| heading.map(str::to_owned))
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| malformed("a heading path holds a null"))?;
            let hash = <[u8; 32]>::try_from(hashes.value(row))
                .map_err(|_cause| malformed("a chunk hash has the wrong length"))?;
            records.push(ChunkRecord {
                file_ordinal: ordinals.value(row),
                content_version: ManifestVersion::new(versions.value(row))?,
                profile: ExtractorProfile::new(profiles.value(row))?,
                locator: locators.value(row).parse()?,
                heading_path,
                hash: ChunkHash::from_bytes(hash),
            });
        }
    }
    Ok(records)
}

fn encode_index(index: &Spans, metadata: Vec<(&'static str, String)>) -> Result<Vec<u8>> {
    let spans = &index.each;
    let schema = Schema::new(vec![
        Field::new(FILE_ORDINAL, DataType::UInt64, false),
        Field::new(CONTENT_VERSION, DataType::UInt64, false),
        Field::new(PROFILE, DataType::Utf8, false),
        Field::new(FIRST_ROW, DataType::UInt64, false),
        Field::new(ROWS, DataType::UInt64, false),
    ]);
    let column = |value: fn(&FileSpan) -> u64| -> ArrayRef {
        Arc::new(UInt64Array::from_iter_values(spans.iter().map(value)))
    };
    let profiles = StringArray::from_iter_values(spans.iter().map(|span| {
        index
            .profiles
            .get(span.profile)
            .map_or("", ExtractorProfile::as_str)
    }));
    let columns = vec![
        column(|span| span.file_ordinal),
        column(|span| span.content_version.get()),
        Arc::new(profiles),
        column(|span| as_u64(span.rows.start)),
        column(|span| as_u64(span.rows.len())),
    ];
    stored::write(schema, columns, FORMAT, metadata)
}

fn intern(profiles: &mut Vec<ExtractorProfile>, profile: &ExtractorProfile) -> usize {
    if let Some(at) = profiles.iter().position(|known| known == profile) {
        return at;
    }
    profiles.push(profile.clone());
    profiles.len().saturating_sub(1)
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use cosmosearch_core::{AttributeValue, Attributes, ByteRange, ETag};
    use proptest::prelude::*;

    use super::*;
    use crate::TRAILER_BYTES;

    fn version(value: u64) -> ManifestVersion {
        ManifestVersion::new(value).unwrap()
    }

    fn bytes(start: u64) -> Locator {
        Locator::Bytes(ByteRange::new(start, start + 5).unwrap())
    }

    fn record(file: u64, content: u64, locator: Locator) -> ChunkRecord {
        ChunkRecord {
            file_ordinal: file,
            content_version: version(content),
            profile: ExtractorProfile::new(if matches!(locator, Locator::Page(_)) {
                "pdf-1"
            } else {
                "text-1"
            })
            .unwrap(),
            locator,
            heading_path: vec!["Terms".to_owned(), "Refunds".to_owned()],
            hash: ChunkHash::of("text"),
        }
    }

    fn table(records: Vec<ChunkRecord>) -> ChunkTable {
        ChunkTable::new(records).unwrap()
    }

    fn sample() -> ChunkTable {
        table(vec![
            record(0, 2, bytes(0)),
            record(0, 2, bytes(10)),
            record(0, 2, bytes(20)),
            record(1, 3, Locator::Page(NonZeroU32::new(4).unwrap())),
            record(1, 3, Locator::Page(NonZeroU32::new(5).unwrap())),
        ])
    }

    fn files() -> FileTable {
        let mut files = FileTable::new();
        for (key, content) in [("docs/a.txt", 2), ("docs/b.pdf", 3)] {
            files.insert(FileEntry::pending(
                key,
                5,
                ETag::new("9f3a").unwrap(),
                version(content),
            ));
        }
        files
    }

    fn data() -> Location {
        "s3://example-bucket/docs/".parse().unwrap()
    }

    fn indexed(table: &ChunkTable) -> ChunkIndex {
        index_of(&Bytes::from(table.encode().unwrap()))
    }

    fn source_ref(chunks: &ChunkTable, row: usize) -> Option<SourceRef> {
        let record = chunks.get(row)?;
        let files = files();
        let file = files.by_ordinal(record.file_ordinal)?;
        record.source_ref(file, &data()).unwrap()
    }

    #[test]
    fn a_table_round_trips() {
        let original = sample();
        let bytes = Bytes::from(original.encode().unwrap());
        assert_eq!(ChunkTable::decode(bytes).unwrap(), original);
    }

    #[test]
    fn an_empty_table_round_trips() {
        let empty = table(vec![]);
        let bytes = Bytes::from(empty.encode().unwrap());
        assert_eq!(ChunkTable::decode(bytes).unwrap(), empty);
    }

    #[test]
    fn records_resolve_to_refs_for_each_locator_kind() {
        let cases = [
            (1, "s3://example-bucket/docs/a.txt#bytes=10-15@etag=9f3a"),
            (4, "s3://example-bucket/docs/b.pdf#page=5@etag=9f3a"),
        ];
        for (row, expected) in cases {
            let found = source_ref(&sample(), row).unwrap();
            assert_eq!(found.to_string(), expected);
        }
    }

    #[test]
    fn a_chunk_of_an_older_version_or_a_removed_file_is_not_live() {
        let chunks = table(vec![record(0, 1, bytes(0)), record(9, 2, bytes(0))]);
        assert_eq!(source_ref(&chunks, 0), None);
        assert_eq!(source_ref(&chunks, 1), None);
        assert_eq!(source_ref(&chunks, 99), None);
    }

    #[test]
    fn lists_each_file_once_in_row_order() {
        assert_eq!(sample().file_versions(), [(0, version(2)), (1, version(3))]);
        assert_eq!(table(vec![]).file_versions(), []);
    }

    #[test]
    fn rows_are_the_live_chunks_of_files_that_pass_the_filter() {
        let mut files = files();
        let labelled = files
            .get("docs/b.pdf")
            .unwrap()
            .clone()
            .with_attributes(Attributes::from([(
                "label".to_owned(),
                AttributeValue::One("legal".to_owned()),
            )]));
        files.insert(labelled);
        let chunks = indexed(&table(vec![
            record(0, 2, bytes(0)),
            record(0, 2, bytes(10)),
            record(1, 3, bytes(0)),
            record(7, 3, bytes(0)),
        ]));
        let members = |filter: &Filter| -> Vec<u64> {
            let rows = chunks.rows(&files, &Scope::all(), filter);
            (0..5).filter(|row| rows.contains(*row)).collect()
        };
        assert_eq!(members(&Filter::new()), [0, 1, 2]);
        assert_eq!(members(&Filter::new().equals("label", "legal")), [2]);
        assert_eq!(
            members(&Filter::new().equals("label", "hr")),
            Vec::<u64>::new()
        );

        let older = indexed(&table(vec![record(0, 1, bytes(0))]));
        assert!(older.rows(&files, &Scope::all(), &Filter::new()).is_empty());
        let scoped = |scope: &Scope| -> Vec<u64> {
            let rows = chunks.rows(&files, scope, &Filter::new());
            (0..5).filter(|row| rows.contains(*row)).collect()
        };
        assert_eq!(scoped(&Scope::prefix("docs/b")), [2]);
        assert_eq!(scoped(&Scope::prefix("docs/c")), Vec::<u64>::new());
    }

    #[test]
    fn finds_the_rows_of_one_file_at_one_version() {
        let chunks = sample();
        assert_eq!(chunks.file_rows(0, version(2)), 0..3);
        assert_eq!(chunks.file_rows(1, version(3)), 3..5);
        assert_eq!(chunks.file_rows(1, version(2)), 0..0);
        assert_eq!(chunks.file_rows(7, version(3)), 0..0);
    }

    #[test]
    fn neighbours_stay_inside_the_file() {
        let chunks = indexed(&sample());
        let cases = [
            ((1, 1, 1), 0..3),
            ((0, 5, 0), 0..1),
            ((2, 0, 5), 2..3),
            ((3, 5, 0), 3..4),
            ((3, 0, 5), 3..5),
            ((1, 0, 0), 1..2),
            ((99, 1, 1), 99..99),
        ];
        for ((row, before, after), expected) in cases {
            assert_eq!(chunks.neighbours(row, before, after), expected, "{row}");
        }
    }

    #[test]
    fn refuses_records_that_break_the_layout() {
        for records in [
            vec![
                record(0, 2, bytes(0)),
                record(1, 2, bytes(0)),
                record(0, 2, bytes(10)),
            ],
            vec![record(0, 2, bytes(0)), record(0, 3, bytes(10))],
            vec![record(0, 2, bytes(0)), {
                let mut other = record(0, 2, bytes(10));
                other.profile = ExtractorProfile::new("other-1").unwrap();
                other
            }],
        ] {
            ChunkTable::new(records).unwrap_err();
        }
    }

    #[test]
    fn a_record_keeps_no_text() {
        let location: Location = "s3://example-bucket/a.txt".parse().unwrap();
        let chunk = cosmosearch_core::Passage::new(bytes(0), vec![], "confidential words")
            .into_chunk(&location, &ETag::new("9f3a").unwrap())
            .unwrap();
        let profile = ExtractorProfile::new("text-1").unwrap();
        let record = ChunkRecord::new(0, version(2), profile, &chunk);
        assert_eq!(record.hash(), chunk.hash());
        let encoded = table(vec![record]).encode().unwrap();
        let found = encoded
            .windows(b"confidential".len())
            .any(|window| window == b"confidential");
        assert!(!found);
    }

    fn walked_rows(
        chunks: &ChunkTable,
        files: &FileTable,
        scope: &Scope,
        filter: &Filter,
    ) -> RowSet {
        let mut rows = RowSet::new();
        for (row, record) in chunks.records.iter().enumerate() {
            let passes = files.by_ordinal(record.file_ordinal).is_some_and(|file| {
                file.content_version() == record.content_version
                    && scope.contains(file.key())
                    && filter.matches(file.attributes(), file.modified())
            });
            if passes {
                rows.insert(as_u64(row)..as_u64(row) + 1);
            }
        }
        rows
    }

    #[derive(Debug, Clone, Copy)]
    enum InTable {
        Live,
        Older,
        Removed,
        Missing,
    }

    fn in_table() -> impl Strategy<Value = InTable> {
        prop_oneof![
            4 => Just(InTable::Live),
            1 => Just(InTable::Older),
            1 => Just(InTable::Removed),
            1 => Just(InTable::Missing),
        ]
    }

    fn generated(
        files: &[(usize, InTable, Option<bool>)],
        order: &[usize],
    ) -> (ChunkTable, FileTable) {
        let mut records = Vec::new();
        for at in order {
            let (chunks, _, _) = files[*at];
            for chunk in 0..chunks {
                records.push(record(*at as u64, 3, bytes(chunk as u64 * 10)));
            }
        }
        let mut table = FileTable::new();
        for (at, (_, state, label)) in files.iter().enumerate() {
            let key = format!("docs/{at:03}.txt");
            let content = if matches!(state, InTable::Older) {
                2
            } else {
                3
            };
            let mut entry =
                FileEntry::pending(&key, 5, ETag::new("9f3a").unwrap(), version(content));
            if let Some(legal) = label {
                let value = if *legal { "legal" } else { "hr" };
                entry = entry.with_attributes(Attributes::from([(
                    "label".to_owned(),
                    AttributeValue::One(value.to_owned()),
                )]));
            }
            table.insert(entry);
        }
        for (at, (_, state, _)) in files.iter().enumerate() {
            if matches!(state, InTable::Removed | InTable::Missing) {
                table.remove(&format!("docs/{at:03}.txt"));
            }
        }
        (table_of(records), table)
    }

    fn table_of(records: Vec<ChunkRecord>) -> ChunkTable {
        ChunkTable::new(records).unwrap()
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        #[test]
        fn rows_from_spans_equal_a_walk_over_every_record(
            (files, order) in prop::collection::vec(
                (1usize..5, in_table(), prop::option::of(any::<bool>())),
                0..12,
            )
            .prop_flat_map(|files| {
                let order = Just((0..files.len()).collect::<Vec<_>>()).prop_shuffle();
                (Just(files), order)
            }),
            kept in prop::collection::vec(any::<bool>(), 12),
        ) {
            let order: Vec<usize> = order
                .into_iter()
                .filter(|at| kept[*at])
                .collect();
            let (chunks, table) = generated(&files, &order);
            let index = indexed(&chunks);
            for filter in [
                Filter::new(),
                Filter::new().equals("label", "legal"),
                Filter::new().equals("label", "hr"),
            ] {
                for scope in [Scope::all(), Scope::prefix("docs/00"), Scope::prefixes(&["docs/001", "docs/01"])] {
                    prop_assert_eq!(
                        index.rows(&table, &scope, &filter),
                        walked_rows(&chunks, &table, &scope, &filter)
                    );
                }
            }
            for file in 0..files.len() as u64 {
                for content in [2, 3] {
                    let walked: Vec<usize> = (0..chunks.len())
                        .filter(|at| {
                            let record = chunks.get(*at).unwrap();
                            record.file_ordinal == file && record.content_version == version(content)
                        })
                        .collect();
                    let found: Vec<usize> = chunks.file_rows(file, version(content)).collect();
                    prop_assert_eq!(found, walked);
                }
            }
            prop_assert!(index.neighbours(chunks.len(), 1, 1).is_empty());
            for row in 0..chunks.len() {
                for (before, after) in [(0, 0), (1, 2), (9, 9)] {
                    let found = index.neighbours(row, before, after);
                    prop_assert!(found.contains(&row));
                    for at in found.clone() {
                        prop_assert_eq!(
                            chunks.get(at).map(ChunkRecord::file_ordinal),
                            chunks.get(row).map(ChunkRecord::file_ordinal)
                        );
                    }
                    let same = |at: usize| {
                        chunks.get(at).map(ChunkRecord::file_ordinal)
                            == chunks.get(row).map(ChunkRecord::file_ordinal)
                    };
                    let reached_start = found.start + before == row
                        || found.start == 0
                        || !same(found.start - 1);
                    let reached_end = found.end == row + after + 1 || !same(found.end);
                    prop_assert!(reached_start && reached_end);
                }
            }
        }
    }

    fn many(files: u64, each: u64) -> ChunkTable {
        let records = (0..files)
            .flat_map(|file| (0..each).map(move |at| record(file, 2, bytes(at * 10))))
            .collect();
        table(records)
    }

    fn index_of(encoded: &Bytes) -> ChunkIndex {
        let size = as_u64(encoded.len());
        let trailer = slice(encoded, size - TRAILER_BYTES..size).unwrap();
        let at = ChunkIndex::locate(&trailer, size).unwrap();
        ChunkIndex::decode(slice(encoded, at).unwrap()).unwrap()
    }

    #[test]
    fn a_table_of_several_groups_round_trips_and_each_group_reads_alone() {
        let original = many(7, 300);
        let encoded = Bytes::from(original.encode().unwrap());
        assert_eq!(ChunkTable::decode(encoded.clone()).unwrap(), original);
        let index = index_of(&encoded);
        assert_eq!(index.len(), 2100);
        assert_eq!(index.file_versions(), original.file_versions());
        assert_eq!(
            index.file_rows(3, version(2)),
            original.file_rows(3, version(2))
        );
        assert_eq!(index.neighbours(899, 5, 5), 894..900);
        let place = index.place(905).unwrap();
        assert_eq!((place.file_ordinal, place.position), (3, 5));
        assert_eq!(place.profile, original.get(905).unwrap().profile());
        assert_eq!(index.place(2100), None);
        assert_eq!(index.group_of(2100), None);
        let mut groups = 0;
        for row in [0, 511, 512, 1023, 2099] {
            let group = index.group_of(row).unwrap();
            let (rows, at) = index.group(group).unwrap();
            assert!(rows.contains(&row));
            let decoded = ChunkGroup::decode(slice(&encoded, at).unwrap(), &index, group).unwrap();
            assert_eq!(decoded.get(row), original.get(row));
            groups = groups.max(group + 1);
        }
        assert_eq!(groups, 5);
    }

    #[test]
    fn live_rows_follow_the_versions_of_the_file_table() {
        let table = sample();
        let index = index_of(&Bytes::from(table.encode().unwrap()));
        let mut files = files();
        files.make_pending("docs/b.pdf", version(4));
        let encoded = Bytes::from(files.encode().unwrap());
        let file_index =
            FileIndex::decode(grouped::index_of(&encoded, *b"CSFILETB").unwrap()).unwrap();
        let live = index.live_rows(&file_index);
        assert_eq!(live, index.rows(&files, &Scope::all(), &Filter::new()));
        assert!(live.covers(0..3) && !live.contains(3));
    }

    #[test]
    fn a_file_without_a_trailer_or_with_a_long_index_is_refused() {
        let encoded = Bytes::from(sample().encode().unwrap());
        let size = as_u64(encoded.len());
        let trailer = slice(&encoded, size - TRAILER_BYTES..size).unwrap();
        ChunkIndex::locate(trailer.get(..8).unwrap(), size).unwrap_err();
        ChunkIndex::locate(&trailer, 8).unwrap_err();
        let mut long = trailer.to_vec();
        long.splice(..8, u64::MAX.to_le_bytes());
        ChunkIndex::locate(&long, size).unwrap_err();
        let mut other = trailer.to_vec();
        if let Some(last) = other.last_mut() {
            *last ^= 1;
        }
        ChunkIndex::locate(&other, size).unwrap_err();
    }

    #[test]
    fn a_group_that_does_not_match_its_index_is_refused() {
        let encoded = Bytes::from(sample().encode().unwrap());
        let index = index_of(&encoded);
        let (_, at) = index.group(0).unwrap();
        ChunkGroup::decode(slice(&encoded, at.clone()).unwrap(), &index, 1).unwrap_err();
        let other = Bytes::from(
            table(vec![
                record(7, 2, bytes(0)),
                record(7, 2, bytes(10)),
                record(7, 2, bytes(20)),
                record(8, 3, bytes(0)),
                record(8, 3, bytes(10)),
            ])
            .encode()
            .unwrap(),
        );
        let (_, theirs) = index_of(&other).group(0).unwrap();
        ChunkGroup::decode(slice(&other, theirs).unwrap(), &index, 0).unwrap_err();
        ChunkGroup::decode(slice(&encoded, at).unwrap(), &index, 0).unwrap();
    }

    #[test]
    fn an_index_claiming_more_rows_than_its_groups_hold_is_refused() {
        let huge = Spans::new(
            vec![FileSpan {
                file_ordinal: 0,
                content_version: version(2),
                profile: 0,
                rows: 0..1 << 40,
            }],
            vec![ExtractorProfile::new("text-1").unwrap()],
        )
        .unwrap();
        let bytes = grouped::assemble(
            &[vec![0; 10]],
            GROUP_ROWS,
            |metadata| encode_index(&huge, metadata),
            MAGIC,
        )
        .unwrap();
        ChunkTable::decode(Bytes::from(bytes)).unwrap_err();
    }

    #[test]
    fn damaged_bytes_give_an_error_and_never_a_panic() {
        ChunkTable::decode(Bytes::from_static(b"garbage")).unwrap_err();
        let good = sample().encode().unwrap();
        for at in 0..good.len() {
            let mut bad = good.clone();
            bad[at] ^= 0xff;
            let _outcome = ChunkTable::decode(Bytes::from(bad));
        }
    }
}
