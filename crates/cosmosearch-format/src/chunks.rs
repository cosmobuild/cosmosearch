use std::collections::BTreeMap;
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
use crate::stored::{self, required};
use crate::{Error, FileTable, Result};

const FORMAT: u32 = 2;
const HASH_BYTES: i32 = 32;

const FILE_ORDINAL: &str = "file_ordinal";
const CONTENT_VERSION: &str = "content_version";
const PROFILE: &str = "profile";
const LOCATOR: &str = "locator";
const HEADING_PATH: &str = "heading_path";
const HASH: &str = "hash";

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
}

/// The chunks of one segment. A file's chunks are adjacent and in reading order.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChunkTable {
    records: Vec<ChunkRecord>,
    spans: Vec<FileSpan>,
    by_file: BTreeMap<u64, usize>,
}

/// The rows of one file's chunks, which share a content version.
#[derive(Debug, Clone, PartialEq, Eq)]
struct FileSpan {
    file_ordinal: u64,
    content_version: ManifestVersion,
    rows: Range<usize>,
}

impl ChunkTable {
    /// A table of `records`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Malformed`] unless each file's records are adjacent and share one
    /// content version.
    pub fn new(records: Vec<ChunkRecord>) -> Result<Self> {
        let mut spans: Vec<FileSpan> = Vec::new();
        let mut by_file = BTreeMap::new();
        for (row, record) in records.iter().enumerate() {
            if let Some(span) = spans.last_mut()
                && span.file_ordinal == record.file_ordinal
            {
                if span.content_version != record.content_version {
                    return Err(malformed("a file has two content versions in one segment"));
                }
                span.rows.end = row.saturating_add(1);
                continue;
            }
            if by_file.insert(record.file_ordinal, spans.len()).is_some() {
                return Err(malformed("a file's chunks are not adjacent"));
            }
            spans.push(FileSpan {
                file_ordinal: record.file_ordinal,
                content_version: record.content_version,
                rows: row..row.saturating_add(1),
            });
        }
        Ok(ChunkTable {
            records,
            spans,
            by_file,
        })
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

    fn span_of(&self, row: usize) -> Option<&FileSpan> {
        let after = self.spans.partition_point(|span| span.rows.start <= row);
        let span = self.spans.get(after.checked_sub(1)?)?;
        span.rows.contains(&row).then_some(span)
    }

    /// The rows around `row` in the same file, up to `before` and `after` further chunks.
    pub fn neighbours(&self, row: usize, before: usize, after: usize) -> Range<usize> {
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

    /// The rows of the chunks of the file with this ordinal at this content version, empty if
    /// the segment holds none.
    pub fn file_rows(&self, file_ordinal: u64, content_version: ManifestVersion) -> Range<usize> {
        self.by_file
            .get(&file_ordinal)
            .and_then(|at| self.spans.get(*at))
            .filter(|span| span.content_version == content_version)
            .map_or(0..0, |span| span.rows.clone())
    }

    /// Each file the segment holds, as its ordinal and the content version its chunks came
    /// from, in row order.
    pub fn file_versions(&self) -> Vec<(u64, ManifestVersion)> {
        self.spans
            .iter()
            .map(|span| (span.file_ordinal, span.content_version))
            .collect()
    }

    /// The rows whose chunks are live in `files` and whose file is in `scope` and passes
    /// `filter`.
    pub fn rows(&self, files: &FileTable, scope: &Scope, filter: &Filter) -> RowSet {
        let mut rows = RowSet::new();
        for span in &self.spans {
            let passes = files.by_ordinal(span.file_ordinal).is_some_and(|file| {
                file.content_version() == span.content_version
                    && scope.contains(file.key())
                    && filter.matches(file.attributes())
            });
            if passes {
                rows.insert(as_u64(span.rows.start)..as_u64(span.rows.end));
            }
        }
        rows
    }

    /// The ref of the chunk in `row`, or `None` if the chunk is not live in `files`.
    ///
    /// # Errors
    ///
    /// Returns an error if the location or the ref cannot be formed.
    pub fn source_ref(
        &self,
        row: usize,
        files: &FileTable,
        data: &Location,
    ) -> Result<Option<SourceRef>> {
        let Some(record) = self.get(row) else {
            return Ok(None);
        };
        let Some(file) = files.by_ordinal(record.file_ordinal) else {
            return Ok(None);
        };
        if file.content_version() != record.content_version {
            return Ok(None);
        }
        let location = Location::new(data.scheme(), data.authority(), file.key())?;
        let source = SourceRef::new(location, record.locator, file.etag().clone())?;
        Ok(Some(source))
    }

    /// The Parquet bytes stored for this table.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Encoding`] if encoding fails.
    pub fn encode(&self) -> Result<Vec<u8>> {
        let records = &self.records;
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

    /// Reads a stored table.
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnsupportedFormat`] for a newer format, and an error for bytes that do
    /// not decode or a table that is inconsistent.
    pub fn decode(bytes: Bytes) -> Result<Self> {
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
        ChunkTable::new(records)
    }
}

fn as_u64(row: usize) -> u64 {
    u64::try_from(row).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use cosmosearch_core::{AttributeValue, Attributes, ByteRange, ETag};
    use proptest::prelude::*;

    use super::*;
    use crate::FileEntry;

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
            let found = sample()
                .source_ref(row, &files(), &data())
                .unwrap()
                .unwrap();
            assert_eq!(found.to_string(), expected);
        }
    }

    #[test]
    fn a_chunk_of_an_older_version_or_a_removed_file_is_not_live() {
        let chunks = table(vec![record(0, 1, bytes(0)), record(9, 2, bytes(0))]);
        assert_eq!(chunks.source_ref(0, &files(), &data()).unwrap(), None);
        assert_eq!(chunks.source_ref(1, &files(), &data()).unwrap(), None);
        assert_eq!(chunks.source_ref(99, &files(), &data()).unwrap(), None);
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
        let chunks = table(vec![
            record(0, 2, bytes(0)),
            record(0, 2, bytes(10)),
            record(1, 3, bytes(0)),
            record(7, 3, bytes(0)),
        ]);
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

        let older = table(vec![record(0, 1, bytes(0))]);
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
        let chunks = sample();
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
                    && filter.matches(file.attributes())
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
            for filter in [
                Filter::new(),
                Filter::new().equals("label", "legal"),
                Filter::new().equals("label", "hr"),
            ] {
                for scope in [Scope::all(), Scope::prefix("docs/00"), Scope::prefixes(&["docs/001", "docs/01"])] {
                    prop_assert_eq!(
                        chunks.rows(&table, &scope, &filter),
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
            prop_assert!(chunks.neighbours(chunks.len(), 1, 1).is_empty());
            for row in 0..chunks.len() {
                for (before, after) in [(0, 0), (1, 2), (9, 9)] {
                    let found = chunks.neighbours(row, before, after);
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
