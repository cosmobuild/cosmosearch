use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{
    CommitId, Error, ExtractorProfile, IndexId, IndexKind, Level, Location, ManifestVersion,
    ModelId, Result, SegmentFileName, SegmentId, ShardId,
};

const FORMAT: u32 = 2;
const RECORD_FORMAT: u32 = 1;

fn check_format(format: u32, expected: u32) -> Result<()> {
    if format == expected {
        Ok(())
    } else {
        Err(Error::UnsupportedFormat { format })
    }
}

#[derive(Deserialize)]
struct FormatHeader {
    format: u32,
}

fn check_encoded_format(bytes: &[u8], expected: u32) -> Result<()> {
    let header: FormatHeader = serde_json::from_slice(bytes).map_err(Error::ManifestJson)?;
    check_format(header.format, expected)
}

fn invalid(reason: impl Into<String>) -> Error {
    Error::InvalidManifest {
        reason: reason.into(),
    }
}

/// How files reach an index, fixed when the index is created.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum FileSource {
    /// The library lists the data location.
    Listing,
    /// The data location is never listed; only files the host names are in the index.
    HostNamed,
}

/// A half-open range of object keys. `end` of `None` means the end of the key space.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KeyRange {
    start: String,
    end: Option<String>,
}

impl KeyRange {
    /// The whole key space.
    pub fn all() -> Self {
        KeyRange {
            start: String::new(),
            end: None,
        }
    }

    /// Creates the range `start..end`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidManifest`] unless `start` is below `end`.
    pub fn new(start: impl Into<String>, end: Option<String>) -> Result<Self> {
        let range = KeyRange {
            start: start.into(),
            end,
        };
        range.check()?;
        Ok(range)
    }

    fn check(&self) -> Result<()> {
        match &self.end {
            Some(end) if *end <= self.start => Err(invalid(format!(
                "key range `{}`..`{end}` is empty",
                self.start
            ))),
            Some(_) | None => Ok(()),
        }
    }

    /// First key, inclusive.
    pub fn start(&self) -> &str {
        &self.start
    }

    /// End of the range, exclusive.
    pub fn end(&self) -> Option<&str> {
        self.end.as_deref()
    }

    /// Whether `key` falls in this range.
    pub fn contains(&self, key: &str) -> bool {
        key >= self.start.as_str() && self.end.as_deref().is_none_or(|end| key < end)
    }
}

/// One file of a segment and its size.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentFile {
    name: SegmentFileName,
    size: u64,
}

impl SegmentFile {
    /// Creates a file entry.
    pub fn new(name: SegmentFileName, size: u64) -> Self {
        SegmentFile { name, size }
    }

    /// The file name, relative to the segment's directory.
    pub fn name(&self) -> &SegmentFileName {
        &self.name
    }

    /// The file's size in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }
}

/// The plug-ins a segment was built with, which an engine needs to use it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BuiltBy {
    index: Option<IndexKind>,
    extractors: BTreeSet<ExtractorProfile>,
}

impl BuiltBy {
    /// Records the index kind, which only an L0 segment may leave out, and the extractors.
    pub fn new(index: Option<IndexKind>, extractors: BTreeSet<ExtractorProfile>) -> Self {
        BuiltBy { index, extractors }
    }

    /// The kind of index the segment was built for.
    pub fn index(&self) -> Option<&IndexKind> {
        self.index.as_ref()
    }

    /// The extractors that produced the segment's text.
    pub fn extractors(&self) -> &BTreeSet<ExtractorProfile> {
        &self.extractors
    }
}

/// One immutable group of index files.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentEntry {
    id: SegmentId,
    level: Level,
    built_by: BuiltBy,
    model_id: Option<ModelId>,
    files: Vec<SegmentFile>,
}

impl SegmentEntry {
    /// Creates a segment entry. `model_id` names the embedding identity of a vector segment.
    pub fn new(
        id: SegmentId,
        level: Level,
        built_by: BuiltBy,
        model_id: Option<ModelId>,
        files: Vec<SegmentFile>,
    ) -> Self {
        SegmentEntry {
            id,
            level,
            built_by,
            model_id,
            files,
        }
    }

    /// The segment id, unique within its index.
    pub fn id(&self) -> &SegmentId {
        &self.id
    }

    /// The level this segment belongs to.
    pub fn level(&self) -> Level {
        self.level
    }

    /// The plug-ins this segment was built with.
    pub fn built_by(&self) -> &BuiltBy {
        &self.built_by
    }

    /// The embedding identity of a vector segment.
    pub fn model_id(&self) -> Option<&ModelId> {
        self.model_id.as_ref()
    }

    /// Every file of the segment.
    pub fn files(&self) -> &[SegmentFile] {
        &self.files
    }
}

/// A shard: its key range and its segments.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardEntry {
    range: KeyRange,
    segments: Vec<SegmentEntry>,
}

impl ShardEntry {
    /// Creates a shard entry.
    pub fn new(range: KeyRange, segments: Vec<SegmentEntry>) -> Self {
        ShardEntry { range, segments }
    }

    /// The keys this shard covers.
    pub fn range(&self) -> &KeyRange {
        &self.range
    }

    /// The shard's segments.
    pub fn segments(&self) -> &[SegmentEntry] {
        &self.segments
    }
}

/// A snapshot of a whole index at one version.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "ManifestFields")]
pub struct Manifest {
    format: u32,
    version: ManifestVersion,
    index_id: IndexId,
    commit_id: CommitId,
    data_location: Location,
    file_source: FileSource,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    stored_text: bool,
    shards: BTreeMap<ShardId, ShardEntry>,
    model_ids: BTreeSet<ModelId>,
}

#[derive(Deserialize)]
struct ManifestFields {
    format: u32,
    version: ManifestVersion,
    index_id: IndexId,
    commit_id: CommitId,
    data_location: Location,
    file_source: FileSource,
    #[serde(default)]
    stored_text: bool,
    shards: BTreeMap<ShardId, ShardEntry>,
    model_ids: BTreeSet<ModelId>,
}

/// How an index is built, chosen when it is created and kept by every version after.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct IndexOptions {
    stored_text: bool,
}

impl IndexOptions {
    /// The options of an index created without any: no stored text.
    pub fn new() -> Self {
        IndexOptions::default()
    }

    /// Stores the text of each chunk that cannot be read back by a byte range, such as a PDF
    /// page or a docx paragraph, with its segment, so reading it needs no parse of its file.
    /// The index then holds document text at rest.
    #[must_use]
    pub fn stored_text(mut self, stored: bool) -> Self {
        self.stored_text = stored;
        self
    }

    /// Whether the index stores the text of chunks that cannot be read by a byte range.
    pub fn has_stored_text(&self) -> bool {
        self.stored_text
    }
}

impl TryFrom<ManifestFields> for Manifest {
    type Error = Error;

    fn try_from(fields: ManifestFields) -> Result<Self> {
        let manifest = Manifest {
            format: fields.format,
            version: fields.version,
            index_id: fields.index_id,
            commit_id: fields.commit_id,
            data_location: fields.data_location,
            file_source: fields.file_source,
            stored_text: fields.stored_text,
            shards: fields.shards,
            model_ids: fields.model_ids,
        };
        manifest.validate()?;
        Ok(manifest)
    }
}

impl Manifest {
    /// The first manifest of a new index: one shard covering every key, with no segments.
    ///
    /// # Errors
    ///
    /// Returns an error if the fixed first shard id is not valid, which is a bug.
    pub fn first(
        index_id: IndexId,
        commit_id: CommitId,
        data_location: Location,
        file_source: FileSource,
    ) -> Result<Self> {
        Manifest::first_with(
            index_id,
            commit_id,
            data_location,
            file_source,
            IndexOptions::new(),
        )
    }

    /// The first manifest of a new index built with `options`.
    ///
    /// # Errors
    ///
    /// Returns an error if the fixed first shard id is not valid, which is a bug.
    pub fn first_with(
        index_id: IndexId,
        commit_id: CommitId,
        data_location: Location,
        file_source: FileSource,
        options: IndexOptions,
    ) -> Result<Self> {
        let shard = ShardEntry::new(KeyRange::all(), Vec::new());
        Ok(Manifest {
            format: FORMAT,
            version: ManifestVersion::FIRST,
            index_id,
            commit_id,
            data_location,
            file_source,
            stored_text: options.stored_text,
            shards: BTreeMap::from([(ShardId::new("0")?, shard)]),
            model_ids: BTreeSet::new(),
        })
    }

    /// The next manifest of this index, with new contents.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidManifest`] if the contents are inconsistent, and
    /// [`Error::InvalidVersion`] if there is no next version.
    pub fn successor(
        &self,
        commit_id: CommitId,
        shards: BTreeMap<ShardId, ShardEntry>,
        model_ids: BTreeSet<ModelId>,
    ) -> Result<Self> {
        let manifest = Manifest {
            format: FORMAT,
            version: self.version.next()?,
            index_id: self.index_id.clone(),
            commit_id,
            data_location: self.data_location.clone(),
            file_source: self.file_source,
            stored_text: self.stored_text,
            shards,
            model_ids,
        };
        manifest.validate()?;
        Ok(manifest)
    }

    fn validate(&self) -> Result<()> {
        check_format(self.format, FORMAT)?;
        self.validate_coverage()?;
        for (shard_id, shard) in &self.shards {
            let mut seen = BTreeSet::new();
            for segment in &shard.segments {
                if !seen.insert(&segment.id) {
                    return Err(invalid(format!(
                        "shard `{shard_id}` lists segment `{}` twice",
                        segment.id
                    )));
                }
                if segment.level != Level::L0 && segment.built_by.index.is_none() {
                    return Err(invalid(format!(
                        "segment `{}` at {} names no index kind",
                        segment.id, segment.level
                    )));
                }
                if let Some(model_id) = &segment.model_id
                    && !self.model_ids.contains(model_id)
                {
                    return Err(invalid(format!(
                        "segment `{}` names unlisted model `{model_id}`",
                        segment.id
                    )));
                }
            }
        }
        Ok(())
    }

    fn validate_coverage(&self) -> Result<()> {
        let mut ranges: Vec<&KeyRange> = self.shards.values().map(|s| &s.range).collect();
        ranges.sort_by_key(|range| range.start.as_str());
        let mut next_start = Some("");
        for range in ranges {
            range.check()?;
            if next_start != Some(range.start.as_str()) {
                return Err(invalid(format!(
                    "shard ranges do not meet at `{}`",
                    range.start
                )));
            }
            next_start = range.end.as_deref();
        }
        if next_start.is_some() {
            return Err(invalid(
                "shard ranges do not reach the end of the key space",
            ));
        }
        Ok(())
    }

    /// This manifest's version.
    pub fn version(&self) -> ManifestVersion {
        self.version
    }

    /// The index this manifest belongs to.
    pub fn index_id(&self) -> &IndexId {
        &self.index_id
    }

    /// The commit attempt that wrote this manifest.
    pub fn commit_id(&self) -> &CommitId {
        &self.commit_id
    }

    /// The files this index covers.
    pub fn data_location(&self) -> &Location {
        &self.data_location
    }

    /// How files reach this index.
    pub fn file_source(&self) -> FileSource {
        self.file_source
    }

    /// The options the index was created with.
    pub fn options(&self) -> IndexOptions {
        IndexOptions {
            stored_text: self.stored_text,
        }
    }

    /// The shards, by id.
    pub fn shards(&self) -> &BTreeMap<ShardId, ShardEntry> {
        &self.shards
    }

    /// Every embedding identity a segment references.
    pub fn model_ids(&self) -> &BTreeSet<ModelId> {
        &self.model_ids
    }

    /// The shard holding `key`.
    pub fn shard_for(&self, key: &str) -> Option<(&ShardId, &ShardEntry)> {
        self.shards.iter().find(|(_, s)| s.range.contains(key))
    }

    /// The JSON stored for this manifest.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ManifestJson`] if encoding fails.
    pub fn encode(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(&Encoded::Manifest(self)).map_err(Error::ManifestJson)
    }

    /// The stub that replaces this manifest once it expires.
    pub fn stub(&self) -> ManifestStub {
        ManifestStub {
            format: FORMAT,
            version: self.version,
            index_id: self.index_id.clone(),
            commit_id: self.commit_id.clone(),
        }
    }
}

/// The small object that replaces an expired manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestStub {
    format: u32,
    version: ManifestVersion,
    index_id: IndexId,
    commit_id: CommitId,
}

impl ManifestStub {
    /// The version whose slot this stub occupies.
    pub fn version(&self) -> ManifestVersion {
        self.version
    }

    /// The index this stub belongs to.
    pub fn index_id(&self) -> &IndexId {
        &self.index_id
    }

    /// The commit attempt that wrote the original manifest.
    pub fn commit_id(&self) -> &CommitId {
        &self.commit_id
    }

    /// The JSON stored for this stub.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ManifestJson`] if encoding fails.
    pub fn encode(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(&Encoded::Stub(self)).map_err(Error::ManifestJson)
    }
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Encoded<'a> {
    Manifest(&'a Manifest),
    Stub(&'a ManifestStub),
}

/// What a version slot holds: a manifest, or the stub of an expired one.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ManifestObject {
    /// A full manifest.
    Manifest(Box<Manifest>),
    /// The stub of an expired manifest.
    Stub(ManifestStub),
}

impl ManifestObject {
    /// Decodes and validates what a version slot holds.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ManifestJson`] if the bytes do not decode or the contents are
    /// inconsistent.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        check_encoded_format(bytes, FORMAT)?;
        serde_json::from_slice(bytes).map_err(Error::ManifestJson)
    }

    /// The manifest in this slot, or `None` for a stub.
    pub fn into_manifest(self) -> Option<Manifest> {
        match self {
            ManifestObject::Manifest(manifest) => Some(*manifest),
            ManifestObject::Stub(_) => None,
        }
    }

    /// The commit attempt that wrote this slot.
    pub fn commit_id(&self) -> &CommitId {
        match self {
            ManifestObject::Manifest(manifest) => manifest.commit_id(),
            ManifestObject::Stub(stub) => stub.commit_id(),
        }
    }

    /// The index this slot belongs to.
    pub fn index_id(&self) -> &IndexId {
        match self {
            ManifestObject::Manifest(manifest) => manifest.index_id(),
            ManifestObject::Stub(stub) => stub.index_id(),
        }
    }
}

/// The content of `MANIFEST`: a recent version, only ever a hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestPointer {
    format: u32,
    version: ManifestVersion,
}

impl ManifestPointer {
    /// A pointer to `version`.
    pub fn new(version: ManifestVersion) -> Self {
        ManifestPointer {
            format: RECORD_FORMAT,
            version,
        }
    }

    /// The version pointed at.
    pub fn version(self) -> ManifestVersion {
        self.version
    }

    /// The JSON stored for this pointer.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ManifestJson`] if encoding fails.
    pub fn encode(self) -> Result<Vec<u8>> {
        serde_json::to_vec(&self).map_err(Error::ManifestJson)
    }

    /// Decodes a pointer.
    ///
    /// # Errors
    ///
    /// Returns an error for bytes that do not decode or an unknown format.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        check_encoded_format(bytes, RECORD_FORMAT)?;
        serde_json::from_slice(bytes).map_err(Error::ManifestJson)
    }
}

/// The content of `INDEX`: which index id currently lives at a metadata location.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexRecord {
    format: u32,
    index_id: IndexId,
}

impl IndexRecord {
    /// A record naming `index_id`.
    pub fn new(index_id: IndexId) -> Self {
        IndexRecord {
            format: RECORD_FORMAT,
            index_id,
        }
    }

    /// The index id.
    pub fn index_id(&self) -> &IndexId {
        &self.index_id
    }

    /// The JSON stored for this record.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ManifestJson`] if encoding fails.
    pub fn encode(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(Error::ManifestJson)
    }

    /// Decodes a record.
    ///
    /// # Errors
    ///
    /// Returns an error for bytes that do not decode or an unknown format.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        check_encoded_format(bytes, RECORD_FORMAT)?;
        serde_json::from_slice(bytes).map_err(Error::ManifestJson)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::*;

    fn first() -> Manifest {
        Manifest::first(
            IndexId::new("idx").unwrap(),
            CommitId::new("c1").unwrap(),
            "s3://example-bucket/docs/".parse().unwrap(),
            FileSource::Listing,
        )
        .unwrap()
    }

    fn segment(id: &str, model: Option<&str>) -> SegmentEntry {
        SegmentEntry::new(
            SegmentId::new(id).unwrap(),
            Level::L1,
            BuiltBy::new(
                Some(IndexKind::new("bm25").unwrap()),
                BTreeSet::from([ExtractorProfile::new("pdf").unwrap()]),
            ),
            model.map(|m| ModelId::new(m).unwrap()),
            vec![SegmentFile::new(
                SegmentFileName::new(format!("lex-{id}.bin")).unwrap(),
                42,
            )],
        )
    }

    fn range(start: &str, end: Option<&str>) -> KeyRange {
        KeyRange::new(start, end.map(str::to_owned)).unwrap()
    }

    fn shards(entries: Vec<(&str, KeyRange, Vec<SegmentEntry>)>) -> BTreeMap<ShardId, ShardEntry> {
        entries
            .into_iter()
            .map(|(id, r, s)| (ShardId::new(id).unwrap(), ShardEntry::new(r, s)))
            .collect()
    }

    #[test]
    fn the_first_manifest_covers_every_key_with_no_segments() {
        let manifest = first();
        assert_eq!(manifest.version(), ManifestVersion::FIRST);
        let (_, shard) = manifest.shard_for("any/key").unwrap();
        assert_eq!(shard.segments(), Vec::<SegmentEntry>::new());
    }

    #[test]
    fn a_successor_keeps_identity_and_moves_one_version() {
        let parent = first();
        let next = parent
            .successor(
                CommitId::new("c2").unwrap(),
                shards(vec![
                    ("a", range("", Some("m")), vec![segment("s1", None)]),
                    ("b", range("m", None), vec![]),
                ]),
                BTreeSet::new(),
            )
            .unwrap();
        assert_eq!(next.version().get(), 2);
        assert_eq!(next.index_id(), parent.index_id());
        assert_eq!(next.data_location(), parent.data_location());
        assert_eq!(next.shard_for("n").unwrap().0.as_str(), "b");
    }

    #[test]
    fn stored_text_is_kept_by_successors_and_an_older_format_is_refused() {
        let manifest = Manifest::first_with(
            IndexId::new("idx").unwrap(),
            CommitId::new("c1").unwrap(),
            "s3://example-bucket/docs/".parse().unwrap(),
            FileSource::Listing,
            IndexOptions::new().stored_text(true),
        )
        .unwrap();
        let encoded = String::from_utf8(manifest.encode().unwrap()).unwrap();
        assert!(encoded.contains(r#""stored_text":true"#), "{encoded}");
        let next = manifest
            .successor(
                CommitId::new("c2").unwrap(),
                manifest.shards().clone(),
                BTreeSet::new(),
            )
            .unwrap();
        assert!(next.options().has_stored_text());
        let older = r#"{"kind":"manifest","format":1,"version":1,"index_id":"idx","commit_id":"c1","data_location":"s3://example-bucket/docs/","file_source":"listing","shards":{"0":{"range":{"start":"","end":null},"segments":[]}},"model_ids":[]}"#;
        let err = ManifestObject::decode(older.as_bytes()).unwrap_err();
        assert!(
            matches!(err, Error::UnsupportedFormat { format: 1 }),
            "{err}"
        );
        let without = older.replace(r#""format":1"#, r#""format":2"#);
        let ManifestObject::Manifest(decoded) = ManifestObject::decode(without.as_bytes()).unwrap()
        else {
            panic!("a manifest");
        };
        assert!(!decoded.options().has_stored_text());
    }

    #[test]
    fn manifest_and_stub_round_trip() {
        let manifest = first();
        let decoded = ManifestObject::decode(&manifest.encode().unwrap()).unwrap();
        assert_eq!(
            decoded,
            ManifestObject::Manifest(Box::new(manifest.clone()))
        );

        let stub = manifest.stub();
        let decoded = ManifestObject::decode(&stub.encode().unwrap()).unwrap();
        assert_eq!(decoded.commit_id(), manifest.commit_id());
        assert_eq!(decoded, ManifestObject::Stub(stub));
    }

    #[test]
    fn pointer_and_index_record_round_trip() {
        let pointer = ManifestPointer::new(ManifestVersion::new(7).unwrap());
        assert_eq!(
            ManifestPointer::decode(&pointer.encode().unwrap()).unwrap(),
            pointer
        );
        let record = IndexRecord::new(IndexId::new("idx").unwrap());
        assert_eq!(
            IndexRecord::decode(&record.encode().unwrap()).unwrap(),
            record
        );
    }

    fn check_rejected(path: &str, replacement: Value) {
        let mut value: Value = serde_json::from_slice(&first().encode().unwrap()).unwrap();
        *value.pointer_mut(path).unwrap() = replacement;
        ManifestObject::decode(value.to_string().as_bytes()).unwrap_err();
    }

    #[test]
    fn rejects_inconsistent_manifests() {
        let whole = json!({"start": "", "end": null});
        let built_by = json!({"index": "bm25", "extractors": []});
        let seg =
            json!({"id": "s1", "level": "L1", "built_by": built_by, "model_id": null, "files": []});
        let cases = [
            ("/format", json!(1)),
            ("/version", json!(0)),
            ("/kind", json!("other")),
            ("/shards", json!({})),
            ("/shards/0/range", json!({"start": "a", "end": null})),
            ("/shards/0/range", json!({"start": "", "end": "z"})),
            ("/shards/0/range", json!({"start": "b", "end": "a"})),
            (
                "/shards",
                json!({"0": {"range": whole, "segments": []}, "1": {"range": whole, "segments": []}}),
            ),
            ("/shards/0/segments", json!([seg, seg])),
            (
                "/shards/0/segments",
                json!([{"id": "s1", "level": "L2", "built_by": built_by, "model_id": "m", "files": []}]),
            ),
            (
                "/shards/0/segments",
                json!([{"id": "s1", "level": "L1", "built_by": {"index": null, "extractors": []}, "model_id": null, "files": []}]),
            ),
            (
                "/shards/0/segments",
                json!([{"id": "s1", "level": "L1", "built_by": built_by, "model_id": null, "files": [{"name": "a/b", "size": 1}]}]),
            ),
        ];
        for (path, replacement) in cases {
            check_rejected(path, replacement);
        }
    }

    #[test]
    fn a_future_format_is_reported_even_if_its_shape_changed() {
        let future = br#"{"kind": "manifest_v3", "format": 3, "everything": "different"}"#;
        let err = ManifestObject::decode(future).unwrap_err();
        assert!(
            matches!(err, Error::UnsupportedFormat { format: 3 }),
            "{err}"
        );
        assert!(!err.is_damaged_data());
        assert!(
            ManifestObject::decode(b"garbage")
                .unwrap_err()
                .is_damaged_data()
        );
    }

    #[test]
    fn successor_rejects_a_gap() {
        first()
            .successor(
                CommitId::new("c2").unwrap(),
                shards(vec![("a", range("", Some("m")), vec![])]),
                BTreeSet::new(),
            )
            .unwrap_err();
    }

    #[test]
    fn an_l0_segment_needs_no_index_kind() {
        let entry = SegmentEntry::new(
            SegmentId::new("s1").unwrap(),
            Level::L0,
            BuiltBy::new(None, BTreeSet::new()),
            None,
            vec![],
        );
        let next = first()
            .successor(
                CommitId::new("c2").unwrap(),
                shards(vec![("a", KeyRange::all(), vec![entry.clone()])]),
                BTreeSet::new(),
            )
            .unwrap();
        assert_eq!(next.shard_for("x").unwrap().1.segments(), [entry]);
    }

    #[test]
    fn on_disk_forms_are_stable() {
        let manifest = first();
        insta::assert_snapshot!(serde_json::to_string(&segment("s1", None)).unwrap(), @r#"{"id":"s1","level":"L1","built_by":{"index":"bm25","extractors":["pdf"]},"model_id":null,"files":[{"name":"lex-s1.bin","size":42}]}"#);
        insta::assert_snapshot!(String::from_utf8(manifest.encode().unwrap()).unwrap(), @r#"{"kind":"manifest","format":2,"version":1,"index_id":"idx","commit_id":"c1","data_location":"s3://example-bucket/docs/","file_source":"listing","shards":{"0":{"range":{"start":"","end":null},"segments":[]}},"model_ids":[]}"#);
        insta::assert_snapshot!(String::from_utf8(manifest.stub().encode().unwrap()).unwrap(), @r#"{"kind":"stub","format":2,"version":1,"index_id":"idx","commit_id":"c1"}"#);
        insta::assert_snapshot!(String::from_utf8(ManifestPointer::new(ManifestVersion::FIRST).encode().unwrap()).unwrap(), @r#"{"format":1,"version":1}"#);
        insta::assert_snapshot!(String::from_utf8(IndexRecord::new(IndexId::new("idx").unwrap()).encode().unwrap()).unwrap(), @r#"{"format":1,"index_id":"idx"}"#);
    }
}
