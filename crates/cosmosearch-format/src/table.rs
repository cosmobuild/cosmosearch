use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Int64Array, StringArray, UInt32Array, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use cosmosearch_core::{
    AttributeValue, Attributes, ETag, ExtractFailure, ManifestVersion, MediaType, Timestamp,
};
use serde::{Deserialize, Serialize};

use crate::error::{EncodingError, malformed};
use crate::grouped::{self, GROUP_ROWS, Groups, slice};
use crate::stored::{self, optional, required};
use crate::{Error, Result};

const FORMAT: u32 = 4;
const MAGIC: [u8; 8] = *b"CSFILETB";
const MAP_ENTRY_BYTES: usize = 48;
const NEXT_ORDINAL: &str = "cosmosearch.next_ordinal";

const ORDINAL: &str = "ordinal";
const KEY: &str = "key";
const SIZE: &str = "size";
const ETAG: &str = "etag";
const CONTENT_VERSION: &str = "content_version";
const STATUS: &str = "status";
const REASON: &str = "reason";
const DISPLAY_NAME: &str = "display_name";
const MEDIA_TYPE: &str = "media_type";
const MODIFIED: &str = "modified";
const ATTRIBUTES: &str = "attributes";
const UNREAD_PARTS: &str = "unread_parts";
const KEY_HASH: &str = "key_hash";

/// How far a file has got.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FileStatus {
    /// The change is recorded and waits for a job.
    Pending,
    /// The file's current content is in the index.
    Ready,
    /// The file's own problem stopped it, such as a limit or a parser failure.
    Failed,
    /// No extractor can read the file, or it has no text. Not a failure.
    Unsupported,
}

impl FileStatus {
    fn as_str(self) -> &'static str {
        match self {
            FileStatus::Pending => "pending",
            FileStatus::Ready => "ready",
            FileStatus::Failed => "failed",
            FileStatus::Unsupported => "unsupported",
        }
    }

    fn parse(text: &str) -> Result<Self> {
        match text {
            "pending" => Ok(FileStatus::Pending),
            "ready" => Ok(FileStatus::Ready),
            "failed" => Ok(FileStatus::Failed),
            "unsupported" => Ok(FileStatus::Unsupported),
            _ => Err(malformed("a file has an unknown status")),
        }
    }
}

/// Why a file is not ready.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum FileReason {
    /// Extraction ended this way.
    Extraction(ExtractFailure),
    /// The recorded version is gone from the data location, so the file waits for a newer change.
    SourceMoved,
}

impl FileReason {
    fn status(self) -> FileStatus {
        match self {
            FileReason::Extraction(failure) if failure.is_unsupported() => FileStatus::Unsupported,
            FileReason::Extraction(_) => FileStatus::Failed,
            FileReason::SourceMoved => FileStatus::Pending,
        }
    }
}

impl fmt::Display for FileReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FileReason::Extraction(failure) => failure.fmt(f),
            FileReason::SourceMoved => f.write_str("the recorded version of the file is gone"),
        }
    }
}

/// One file known to an index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileEntry {
    ordinal: u64,
    key: String,
    size: u64,
    etag: ETag,
    content_version: ManifestVersion,
    status: FileStatus,
    reason: Option<FileReason>,
    display_name: Option<String>,
    media_type: Option<MediaType>,
    modified: Option<Timestamp>,
    attributes: Attributes,
    unread_parts: u32,
}

impl FileEntry {
    /// A file waiting for a job, whose content last changed at `content_version`.
    pub fn pending(
        key: impl Into<String>,
        size: u64,
        etag: ETag,
        content_version: ManifestVersion,
    ) -> Self {
        FileEntry {
            ordinal: 0,
            key: key.into(),
            size,
            etag,
            content_version,
            status: FileStatus::Pending,
            reason: None,
            display_name: None,
            media_type: None,
            modified: None,
            attributes: Attributes::new(),
            unread_parts: 0,
        }
    }

    /// Sets the name a host shows for this file.
    #[must_use]
    pub fn with_display_name(mut self, name: Option<String>) -> Self {
        self.display_name = name;
        self
    }

    /// Sets the media type a host named for this file.
    #[must_use]
    pub fn with_media_type(mut self, media_type: Option<MediaType>) -> Self {
        self.media_type = media_type;
        self
    }

    /// Sets when the recorded version of the file was last modified.
    #[must_use]
    pub fn with_modified(mut self, modified: Option<Timestamp>) -> Self {
        self.modified = modified;
        self
    }

    /// Sets the file's attributes.
    #[must_use]
    pub fn with_attributes(mut self, attributes: Attributes) -> Self {
        self.attributes = attributes;
        self
    }

    /// The file's number within its table, assigned when it is added and kept for its life.
    pub fn ordinal(&self) -> u64 {
        self.ordinal
    }

    /// The object key.
    pub fn key(&self) -> &str {
        &self.key
    }

    /// The size in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// The ETag of the version recorded.
    pub fn etag(&self) -> &ETag {
        &self.etag
    }

    /// The manifest version of the latest change to the content.
    pub fn content_version(&self) -> ManifestVersion {
        self.content_version
    }

    /// How far the file has got.
    pub fn status(&self) -> FileStatus {
        self.status
    }

    /// Why the file is not ready, when something is known.
    pub fn reason(&self) -> Option<FileReason> {
        self.reason
    }

    /// The name a host shows for this file.
    pub fn display_name(&self) -> Option<&str> {
        self.display_name.as_deref()
    }

    /// The media type a host named for this file, which chooses its extractor ahead of the
    /// key's extension.
    pub fn media_type(&self) -> Option<&MediaType> {
        self.media_type.as_ref()
    }

    /// When the recorded version of the file was last modified, as the store or the host
    /// reported it.
    pub fn modified(&self) -> Option<Timestamp> {
        self.modified
    }

    /// The file's attributes.
    pub fn attributes(&self) -> &Attributes {
        &self.attributes
    }

    /// How many parts of the file, such as PDF pages that hold only an image, its extraction
    /// could not read; zero unless the file is ready.
    pub fn unread_parts(&self) -> u32 {
        self.unread_parts
    }

    fn heap_bytes(&self) -> usize {
        let text = |value: &str| value.len().saturating_add(size_of::<String>());
        let attributes: usize = self
            .attributes
            .iter()
            .map(|(name, value)| {
                let held = match value {
                    AttributeValue::One(one) => text(one),
                    AttributeValue::Many(many) => many.iter().map(|one| text(one)).sum(),
                    AttributeValue::Number(_) | AttributeValue::Time(_) | _ => 0,
                };
                text(name)
                    .saturating_add(held)
                    .saturating_add(size_of::<AttributeValue>())
                    .saturating_add(MAP_ENTRY_BYTES)
            })
            .sum();
        size_of::<Self>()
            .saturating_add(text(&self.key).saturating_mul(3))
            .saturating_add(self.etag.as_str().len())
            .saturating_add(self.display_name.as_deref().map_or(0, str::len))
            .saturating_add(
                self.media_type
                    .as_ref()
                    .map_or(0, |media| media.as_str().len()),
            )
            .saturating_add(attributes)
            .saturating_add(MAP_ENTRY_BYTES.saturating_mul(2))
    }
}

/// The files of one shard, by key.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FileTable {
    files: BTreeMap<String, FileEntry>,
    keys_by_ordinal: BTreeMap<u64, String>,
    next_ordinal: u64,
}

impl FileTable {
    /// An empty table.
    pub fn new() -> Self {
        FileTable::default()
    }

    /// The file with this key.
    pub fn get(&self, key: &str) -> Option<&FileEntry> {
        self.files.get(key)
    }

    /// The file with this ordinal.
    pub fn by_ordinal(&self, ordinal: u64) -> Option<&FileEntry> {
        self.files.get(self.keys_by_ordinal.get(&ordinal)?)
    }

    /// Adds a file, or replaces the one with the same key and keeps its ordinal.
    pub fn insert(&mut self, mut entry: FileEntry) {
        if let Some(existing) = self.files.get(&entry.key) {
            entry.ordinal = existing.ordinal;
        } else {
            entry.ordinal = self.next_ordinal;
            self.next_ordinal = self.next_ordinal.saturating_add(1);
            self.keys_by_ordinal
                .insert(entry.ordinal, entry.key.clone());
        }
        self.files.insert(entry.key.clone(), entry);
    }

    /// Removes a file, reporting whether it was there. Its ordinal is never used again.
    pub fn remove(&mut self, key: &str) -> bool {
        let Some(entry) = self.files.remove(key) else {
            return false;
        };
        self.keys_by_ordinal.remove(&entry.ordinal);
        true
    }

    /// Records that a file's current content is in the index, with how many of its parts the
    /// extraction could not read, reporting whether the file was there.
    pub fn mark_ready(&mut self, key: &str, unread_parts: u32) -> bool {
        let Some(entry) = self.files.get_mut(key) else {
            return false;
        };
        entry.status = FileStatus::Ready;
        entry.reason = None;
        entry.unread_parts = unread_parts;
        true
    }

    /// Records why a file is not ready, which also sets its status, reporting whether it was
    /// there.
    pub fn record(&mut self, key: &str, reason: FileReason) -> bool {
        let Some(entry) = self.files.get_mut(key) else {
            return false;
        };
        entry.status = reason.status();
        entry.reason = Some(reason);
        entry.unread_parts = 0;
        true
    }

    /// Makes a file wait for a job again at a new content version, reporting whether it was
    /// there.
    pub fn make_pending(&mut self, key: &str, content_version: ManifestVersion) -> bool {
        let Some(entry) = self.files.get_mut(key) else {
            return false;
        };
        entry.content_version = content_version;
        entry.status = FileStatus::Pending;
        entry.reason = None;
        entry.unread_parts = 0;
        true
    }

    /// Replaces a file's display name and attributes and leaves its content version and status
    /// alone, reporting whether it was there.
    pub fn describe(
        &mut self,
        key: &str,
        display_name: Option<String>,
        attributes: Attributes,
    ) -> bool {
        let Some(entry) = self.files.get_mut(key) else {
            return false;
        };
        entry.display_name = display_name;
        entry.attributes = attributes;
        true
    }

    /// About how many bytes the table holds in memory.
    pub fn heap_bytes(&self) -> usize {
        self.files
            .values()
            .map(FileEntry::heap_bytes)
            .sum::<usize>()
            .saturating_add(size_of::<Self>())
    }

    /// The total size of the files, in bytes.
    pub fn total_size(&self) -> u64 {
        self.files
            .values()
            .fold(0, |total, file| total.saturating_add(file.size))
    }

    /// Divides the table at `key`, which starts the second part. Ordinals carry over unchanged,
    /// and neither part can reuse one, so segments the two parts share stay valid.
    pub fn split_at(&self, key: &str) -> (FileTable, FileTable) {
        let mut parts = (self.empty_like(), self.empty_like());
        for (name, entry) in &self.files {
            let part = if name.as_str() < key {
                &mut parts.0
            } else {
                &mut parts.1
            };
            part.keys_by_ordinal.insert(entry.ordinal, name.clone());
            part.files.insert(name.clone(), entry.clone());
        }
        parts
    }

    fn empty_like(&self) -> FileTable {
        FileTable {
            next_ordinal: self.next_ordinal,
            ..FileTable::default()
        }
    }

    /// Every file, in key order.
    pub fn iter(&self) -> impl Iterator<Item = &FileEntry> {
        self.files.values()
    }

    /// How many files the table holds.
    pub fn len(&self) -> usize {
        self.files.len()
    }

    /// Whether the table holds no files.
    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    /// The bytes stored for this table: its files in Parquet groups of up to 512 by ordinal,
    /// then an index of each file's ordinal, content version and state and of where each
    /// group is, then a trailer of [`TRAILER_BYTES`](crate::TRAILER_BYTES) giving the index's
    /// length.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Encoding`](crate::Error::Encoding) if encoding fails.
    pub fn encode(&self) -> Result<Vec<u8>> {
        let files: Vec<&FileEntry> = self
            .keys_by_ordinal
            .values()
            .filter_map(|key| self.files.get(key))
            .collect();
        let groups = files
            .chunks(GROUP_ROWS)
            .map(encode_entries)
            .collect::<Result<Vec<_>>>()?;
        grouped::assemble(
            &groups,
            GROUP_ROWS,
            |mut metadata| {
                metadata.push((NEXT_ORDINAL, self.next_ordinal.to_string()));
                encode_index(&files, metadata)
            },
            MAGIC,
        )
    }

    /// Reads a stored table.
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnsupportedFormat`](crate::Error::UnsupportedFormat) for another
    /// format, and an error for bytes that do not decode or a table that is inconsistent.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "taken by value like the other tables' decoders"
    )]
    pub fn decode(bytes: Bytes) -> Result<Self> {
        let index = FileIndex::decode(grouped::index_of(&bytes, MAGIC)?)?;
        let mut table = FileTable {
            next_ordinal: index.next_ordinal,
            ..FileTable::default()
        };
        for group in 0..index.groups.len() {
            let (_, at) = index
                .group(group)
                .ok_or_else(|| malformed("a group is missing"))?;
            let found = FileGroup::decode(slice(&bytes, at)?, &index, group)?;
            for entry in found.entries {
                table.restore(entry)?;
            }
        }
        Ok(table)
    }

    fn restore(&mut self, entry: FileEntry) -> Result<()> {
        if entry.ordinal >= self.next_ordinal {
            return Err(malformed("a file's ordinal is past the next ordinal"));
        }
        if self.keys_by_ordinal.contains_key(&entry.ordinal) {
            return Err(malformed("two files share an ordinal"));
        }
        if self.files.contains_key(&entry.key) {
            return Err(malformed("two files share a key"));
        }
        self.keys_by_ordinal
            .insert(entry.ordinal, entry.key.clone());
        self.files.insert(entry.key.clone(), entry);
        Ok(())
    }
}

/// What every search needs of a shard's file table, without the files themselves: each
/// file's ordinal, content version and state, and where each group of files is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileIndex {
    states: Vec<FileState>,
    groups: Groups,
    next_ordinal: u64,
}

/// A file's ordinal, content version and state, as a file table's index holds them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileState {
    ordinal: u64,
    key_hash: u64,
    content_version: ManifestVersion,
    status: FileStatus,
    reason: Option<FileReason>,
    unread_parts: u32,
}

impl FileState {
    /// The file's ordinal in its table.
    pub fn ordinal(&self) -> u64 {
        self.ordinal
    }

    /// The version whose content the file's chunks come from.
    pub fn content_version(&self) -> ManifestVersion {
        self.content_version
    }

    /// The file's status.
    pub fn status(&self) -> FileStatus {
        self.status
    }

    /// Why the file is in its state, if a job recorded a reason.
    pub fn reason(&self) -> Option<FileReason> {
        self.reason
    }

    /// How many parts of a ready file had no text that could be read.
    pub fn unread_parts(&self) -> u32 {
        self.unread_parts
    }
}

impl FileIndex {
    /// The byte range of the index in a file table of `size` bytes, from its last
    /// [`TRAILER_BYTES`](crate::TRAILER_BYTES) bytes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Malformed`](crate::Error::Malformed) if the trailer is not one.
    pub fn locate(trailer: &[u8], size: u64) -> Result<std::ops::Range<u64>> {
        grouped::locate(trailer, size, MAGIC)
    }

    /// Reads the index of a file table, the bytes [`FileIndex::locate`] gave.
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnsupportedFormat`](crate::Error::UnsupportedFormat) for another
    /// format, and an error for bytes that do not decode or an index that is inconsistent.
    pub fn decode(bytes: Bytes) -> Result<Self> {
        let stored = stored::read(bytes, FORMAT)?;
        let next_ordinal = stored
            .metadata
            .get(NEXT_ORDINAL)
            .and_then(|value| value.parse::<u64>().ok())
            .ok_or_else(|| malformed("the next ordinal is missing"))?;
        let mut found: Vec<FileState> = Vec::new();
        for batch in &stored.batches {
            let ordinals = required::<UInt64Array>(batch, ORDINAL)?;
            let hashes = required::<UInt64Array>(batch, KEY_HASH)?;
            let versions = required::<UInt64Array>(batch, CONTENT_VERSION)?;
            let statuses = required::<StringArray>(batch, STATUS)?;
            let reasons = optional::<StringArray>(batch, REASON)?;
            let unread = required::<UInt32Array>(batch, UNREAD_PARTS)?;
            for row in 0..batch.num_rows() {
                let ordinal = ordinals.value(row);
                if ordinal >= next_ordinal
                    || found.last().is_some_and(|last| last.ordinal >= ordinal)
                {
                    return Err(malformed("a file table's ordinals are out of order"));
                }
                let status = FileStatus::parse(statuses.value(row))?;
                let reason = reason_of(reasons, row)?;
                let unread_parts = unread.value(row);
                check_state(status, reason, unread_parts)?;
                found.push(FileState {
                    ordinal,
                    key_hash: hashes.value(row),
                    content_version: ManifestVersion::new(versions.value(row))?,
                    status,
                    reason,
                    unread_parts,
                });
            }
        }
        let groups = Groups::of(&stored.metadata, found.len(), GROUP_ROWS)?;
        Ok(FileIndex {
            states: found,
            groups,
            next_ordinal,
        })
    }

    /// Each file's state, by ordinal.
    pub fn states(&self) -> &[FileState] {
        &self.states
    }

    /// How many files the table holds.
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Whether the table holds no files.
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    fn position(&self, ordinal: u64) -> Option<usize> {
        self.states
            .binary_search_by_key(&ordinal, |state| state.ordinal)
            .ok()
    }

    /// The ordinals of the files whose key may be `key`: those whose key hashes as it does.
    pub fn ordinals_for(&self, key: &str) -> impl Iterator<Item = u64> + '_ {
        let hash = key_hash(key);
        self.states
            .iter()
            .filter(move |state| state.key_hash == hash)
            .map(|state| state.ordinal)
    }

    /// The content version of the file with this ordinal.
    pub fn version(&self, ordinal: u64) -> Option<ManifestVersion> {
        let state = self.states.get(self.position(ordinal)?)?;
        Some(state.content_version)
    }

    /// The group holding the file with this ordinal.
    pub fn group_of(&self, ordinal: u64) -> Option<usize> {
        self.groups.of_row(self.position(ordinal)?, self.len())
    }

    /// The positions of group `group`'s files among the table's, and its byte range in the
    /// file table.
    pub fn group(&self, group: usize) -> Option<(std::ops::Range<usize>, std::ops::Range<u64>)> {
        self.groups.get(group, self.len())
    }

    /// About how many bytes the index holds in memory.
    pub fn heap_bytes(&self) -> usize {
        self.states
            .len()
            .saturating_mul(size_of::<FileState>())
            .saturating_add(self.groups.heap_bytes())
            .saturating_add(size_of::<Self>())
    }
}

/// The files of one group of a file table, by ordinal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileGroup {
    entries: Vec<FileEntry>,
}

impl FileGroup {
    /// Reads group `group` of the table `index` describes, from the bytes
    /// [`FileIndex::group`] located.
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnsupportedFormat`](crate::Error::UnsupportedFormat) for another
    /// format, and an error for bytes that do not decode or that hold other files than the
    /// index says.
    pub fn decode(bytes: Bytes, index: &FileIndex, group: usize) -> Result<Self> {
        let (rows, _) = index
            .group(group)
            .ok_or_else(|| malformed("a group is missing"))?;
        let expected = index
            .states
            .get(rows)
            .ok_or_else(|| malformed("a group is missing"))?;
        let entries = decode_entries(bytes)?;
        let matches = entries.len() == expected.len()
            && entries.iter().zip(expected).all(|(entry, state)| {
                entry.ordinal == state.ordinal
                    && key_hash(&entry.key) == state.key_hash
                    && entry.content_version == state.content_version
                    && entry.status == state.status
                    && entry.reason == state.reason
                    && entry.unread_parts == state.unread_parts
            });
        if !matches {
            return Err(malformed("a group holds other files than the index says"));
        }
        Ok(FileGroup { entries })
    }

    /// The file with this ordinal, if the group holds it.
    pub fn by_ordinal(&self, ordinal: u64) -> Option<&FileEntry> {
        let at = self
            .entries
            .binary_search_by_key(&ordinal, |entry| entry.ordinal)
            .ok()?;
        self.entries.get(at)
    }

    /// About how many bytes the group holds in memory.
    pub fn heap_bytes(&self) -> usize {
        self.entries
            .iter()
            .map(FileEntry::heap_bytes)
            .sum::<usize>()
            .saturating_add(size_of::<Self>())
    }
}

/// A file key's 64-bit FNV-1a hash, which the index keeps to find a file by its key.
fn key_hash(key: &str) -> u64 {
    key.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
    })
}

fn reason_of(reasons: &StringArray, row: usize) -> Result<Option<FileReason>> {
    if reasons.is_null(row) {
        return Ok(None);
    }
    let reason: FileReason = serde_json::from_str(reasons.value(row))
        .map_err(|_cause| malformed("a file's reason does not decode"))?;
    Ok(Some(reason))
}

fn check_state(status: FileStatus, reason: Option<FileReason>, unread_parts: u32) -> Result<()> {
    if reason.map_or(status, FileReason::status) != status
        || (reason.is_none() && !matches!(status, FileStatus::Pending | FileStatus::Ready))
    {
        return Err(malformed("a file's reason does not fit its status"));
    }
    if unread_parts > 0 && status != FileStatus::Ready {
        return Err(malformed("a file that is not ready has unread parts"));
    }
    Ok(())
}

fn reasons_of(files: &[&FileEntry]) -> Result<Vec<Option<String>>> {
    files
        .iter()
        .map(|file| file.reason.as_ref().map(serde_json::to_string).transpose())
        .collect::<serde_json::Result<Vec<_>>>()
        .map_err(|source| Error::Encoding(EncodingError::new(source)))
}

fn encode_index(files: &[&FileEntry], metadata: Vec<(&'static str, String)>) -> Result<Vec<u8>> {
    let schema = Schema::new(vec![
        Field::new(ORDINAL, DataType::UInt64, false),
        Field::new(KEY_HASH, DataType::UInt64, false),
        Field::new(CONTENT_VERSION, DataType::UInt64, false),
        Field::new(STATUS, DataType::Utf8, false),
        Field::new(REASON, DataType::Utf8, true),
        Field::new(UNREAD_PARTS, DataType::UInt32, false),
    ]);
    let columns: Vec<ArrayRef> = vec![
        Arc::new(UInt64Array::from_iter_values(
            files.iter().map(|f| f.ordinal),
        )),
        Arc::new(UInt64Array::from_iter_values(
            files.iter().map(|f| key_hash(&f.key)),
        )),
        Arc::new(UInt64Array::from_iter_values(
            files.iter().map(|f| f.content_version.get()),
        )),
        Arc::new(StringArray::from_iter_values(
            files.iter().map(|f| f.status.as_str()),
        )),
        Arc::new(StringArray::from(reasons_of(files)?)),
        Arc::new(UInt32Array::from_iter_values(
            files.iter().map(|f| f.unread_parts),
        )),
    ];
    stored::write(schema, columns, FORMAT, metadata)
}

fn encode_entries(files: &[&FileEntry]) -> Result<Vec<u8>> {
    let attributes = files
        .iter()
        .map(|file| serde_json::to_string(&file.attributes))
        .collect::<serde_json::Result<Vec<_>>>()
        .map_err(|source| Error::Encoding(EncodingError::new(source)))?;
    let display_names: Vec<Option<&str>> = files
        .iter()
        .map(|file| file.display_name.as_deref())
        .collect();
    let media_types: Vec<Option<&str>> = files
        .iter()
        .map(|file| file.media_type.as_ref().map(MediaType::as_str))
        .collect();
    let columns: Vec<ArrayRef> = vec![
        Arc::new(UInt64Array::from_iter_values(
            files.iter().map(|f| f.ordinal),
        )),
        Arc::new(StringArray::from_iter_values(files.iter().map(|f| &f.key))),
        Arc::new(UInt64Array::from_iter_values(files.iter().map(|f| f.size))),
        Arc::new(StringArray::from_iter_values(
            files.iter().map(|f| f.etag.as_str()),
        )),
        Arc::new(UInt64Array::from_iter_values(
            files.iter().map(|f| f.content_version.get()),
        )),
        Arc::new(StringArray::from_iter_values(
            files.iter().map(|f| f.status.as_str()),
        )),
        Arc::new(StringArray::from(reasons_of(files)?)),
        Arc::new(StringArray::from(display_names)),
        Arc::new(StringArray::from(media_types)),
        Arc::new(Int64Array::from(
            files
                .iter()
                .map(|f| f.modified.map(Timestamp::millis))
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from_iter_values(attributes)),
        Arc::new(UInt32Array::from_iter_values(
            files.iter().map(|f| f.unread_parts),
        )),
    ];
    stored::write(schema(), columns, FORMAT, vec![])
}

fn decode_entries(bytes: Bytes) -> Result<Vec<FileEntry>> {
    let stored = stored::read(bytes, FORMAT)?;
    let mut entries = Vec::new();
    for batch in &stored.batches {
        let ordinals = required::<UInt64Array>(batch, ORDINAL)?;
        let keys = required::<StringArray>(batch, KEY)?;
        let sizes = required::<UInt64Array>(batch, SIZE)?;
        let etags = required::<StringArray>(batch, ETAG)?;
        let versions = required::<UInt64Array>(batch, CONTENT_VERSION)?;
        let statuses = required::<StringArray>(batch, STATUS)?;
        let reasons = optional::<StringArray>(batch, REASON)?;
        let names = optional::<StringArray>(batch, DISPLAY_NAME)?;
        let media_types = optional::<StringArray>(batch, MEDIA_TYPE)?;
        let modified = optional::<Int64Array>(batch, MODIFIED)?;
        let attributes = required::<StringArray>(batch, ATTRIBUTES)?;
        let unread = required::<UInt32Array>(batch, UNREAD_PARTS)?;
        for row in 0..batch.num_rows() {
            let attributes: Attributes = serde_json::from_str(attributes.value(row))
                .map_err(|_cause| malformed("a file's attributes do not decode"))?;
            let status = FileStatus::parse(statuses.value(row))?;
            let reason = reason_of(reasons, row)?;
            let unread_parts = unread.value(row);
            check_state(status, reason, unread_parts)?;
            entries.push(FileEntry {
                ordinal: ordinals.value(row),
                key: keys.value(row).to_owned(),
                size: sizes.value(row),
                etag: ETag::new(etags.value(row))?,
                content_version: ManifestVersion::new(versions.value(row))?,
                status,
                reason,
                display_name: (!names.is_null(row)).then(|| names.value(row).to_owned()),
                media_type: if media_types.is_null(row) {
                    None
                } else {
                    Some(MediaType::new(media_types.value(row))?)
                },
                modified: (!modified.is_null(row))
                    .then(|| Timestamp::from_millis(modified.value(row))),
                attributes,
                unread_parts,
            });
        }
    }
    Ok(entries)
}

fn schema() -> Schema {
    Schema::new(vec![
        Field::new(ORDINAL, DataType::UInt64, false),
        Field::new(KEY, DataType::Utf8, false),
        Field::new(SIZE, DataType::UInt64, false),
        Field::new(ETAG, DataType::Utf8, false),
        Field::new(CONTENT_VERSION, DataType::UInt64, false),
        Field::new(STATUS, DataType::Utf8, false),
        Field::new(REASON, DataType::Utf8, true),
        Field::new(DISPLAY_NAME, DataType::Utf8, true),
        Field::new(MEDIA_TYPE, DataType::Utf8, true),
        Field::new(MODIFIED, DataType::Int64, true),
        Field::new(ATTRIBUTES, DataType::Utf8, false),
        Field::new(UNREAD_PARTS, DataType::UInt32, false),
    ])
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use cosmosearch_core::{LimitKind, Number};

    use super::*;

    fn entry(key: &str) -> FileEntry {
        FileEntry::pending(
            key,
            5,
            ETag::new("9f3a").unwrap(),
            ManifestVersion::new(3).unwrap(),
        )
    }

    fn round_trip(table: &FileTable) -> FileTable {
        FileTable::decode(Bytes::from(table.encode().unwrap())).unwrap()
    }

    #[test]
    fn a_table_round_trips_with_names_media_types_and_attributes() {
        let mut table = FileTable::new();
        let attributes = Attributes::from([
            ("owner".to_owned(), AttributeValue::One("legal".to_owned())),
            (
                "labels".to_owned(),
                AttributeValue::Many(vec!["a".to_owned(), "b".to_owned()]),
            ),
        ]);
        table.insert(
            entry("docs/a.txt")
                .with_display_name(Some("Contract A".to_owned()))
                .with_media_type(Some(MediaType::new("text/markdown").unwrap()))
                .with_modified(Some(Timestamp::from_millis(1_700_000_000_000)))
                .with_attributes(attributes),
        );
        table.insert(entry("docs/b.txt"));
        assert_eq!(round_trip(&table), table);
    }

    #[test]
    fn an_empty_table_round_trips() {
        assert_eq!(round_trip(&FileTable::new()), FileTable::new());
    }

    #[test]
    fn inserting_replaces_and_removing_reports() {
        let mut table = FileTable::new();
        table.insert(entry("a"));
        table.insert(entry("a"));
        assert_eq!(table.len(), 1);
        assert!(table.remove("a"));
        assert!(!table.remove("a"));
        assert!(table.is_empty());
    }

    #[test]
    fn keeps_files_in_key_order() {
        let mut table = FileTable::new();
        for key in ["c", "a", "b"] {
            table.insert(entry(key));
        }
        let keys: Vec<_> = table.iter().map(FileEntry::key).collect();
        assert_eq!(keys, ["a", "b", "c"]);
    }

    #[test]
    fn a_file_keeps_its_ordinal_when_it_changes() {
        let mut table = FileTable::new();
        table.insert(entry("a"));
        table.insert(entry("b"));
        let b = table.get("b").unwrap().ordinal();
        table.insert(entry("b").with_display_name(Some("new".to_owned())));
        assert_eq!(table.get("b").unwrap().ordinal(), b);
        assert_eq!(table.by_ordinal(b).unwrap().display_name(), Some("new"));
    }

    #[test]
    fn an_ordinal_is_never_used_again_even_after_a_round_trip() {
        let mut table = FileTable::new();
        table.insert(entry("a"));
        table.insert(entry("b"));
        let last = table.get("b").unwrap().ordinal();
        assert!(table.remove("b"));
        assert_eq!(table.by_ordinal(last), None);
        let mut reloaded = round_trip(&table);
        reloaded.insert(entry("c"));
        assert!(reloaded.get("c").unwrap().ordinal() > last);
    }

    #[test]
    fn a_ready_file_round_trips() {
        let mut table = FileTable::new();
        table.insert(entry("a"));
        assert!(table.mark_ready("a", 0));
        assert!(!table.mark_ready("gone", 0));
        assert_eq!(
            round_trip(&table).get("a").unwrap().status(),
            FileStatus::Ready
        );
    }

    #[test]
    fn unread_parts_round_trip_and_clear_when_the_file_changes() {
        let mut table = FileTable::new();
        table.insert(entry("a"));
        assert!(table.mark_ready("a", 3));
        assert_eq!(round_trip(&table).get("a").unwrap().unread_parts(), 3);
        assert!(table.make_pending("a", ManifestVersion::new(2).unwrap()));
        assert_eq!(table.get("a").unwrap().unread_parts(), 0);
    }

    #[test]
    fn a_reason_sets_the_status_and_round_trips() {
        let limit = FileReason::Extraction(ExtractFailure::LimitExceeded(LimitKind::Pages));
        let scan = FileReason::Extraction(ExtractFailure::NoText);
        let cases = [
            (limit, FileStatus::Failed),
            (scan, FileStatus::Unsupported),
            (FileReason::SourceMoved, FileStatus::Pending),
        ];
        for (reason, status) in cases {
            let mut table = FileTable::new();
            table.insert(entry("a"));
            assert!(table.record("a", reason));
            assert!(!table.record("gone", reason));
            let stored = round_trip(&table);
            let file = stored.get("a").unwrap();
            assert_eq!((file.status(), file.reason()), (status, Some(reason)));
        }
    }

    #[test]
    fn ready_and_pending_again_both_clear_the_reason() {
        let mut table = FileTable::new();
        table.insert(entry("a"));
        table.insert(entry("b"));
        for key in ["a", "b"] {
            assert!(table.record(key, FileReason::Extraction(ExtractFailure::Malformed)));
        }
        assert!(table.mark_ready("a", 0));
        let later = ManifestVersion::new(9).unwrap();
        assert!(table.make_pending("b", later));
        assert!(!table.make_pending("gone", later));
        let stored = round_trip(&table);
        let a = stored.get("a").unwrap();
        assert_eq!((a.status(), a.reason()), (FileStatus::Ready, None));
        let b = stored.get("b").unwrap();
        assert_eq!((b.status(), b.reason()), (FileStatus::Pending, None));
        assert_eq!(b.content_version(), later);
    }

    #[test]
    fn describing_a_file_keeps_its_content_version_status_and_ordinal() {
        let mut table = FileTable::new();
        table.insert(entry("a"));
        assert!(table.mark_ready("a", 0));
        let before = table.get("a").unwrap().clone();
        let attributes =
            Attributes::from([("owner".to_owned(), AttributeValue::One("legal".to_owned()))]);
        assert!(table.describe("a", Some("Contract A".to_owned()), attributes.clone()));
        assert!(!table.describe("gone", None, Attributes::new()));
        let after = round_trip(&table).get("a").unwrap().clone();
        assert_eq!(
            (after.ordinal(), after.content_version(), after.status()),
            (before.ordinal(), before.content_version(), before.status())
        );
        assert_eq!(after.display_name(), Some("Contract A"));
        assert_eq!(after.attributes(), &attributes);
    }

    #[test]
    fn the_reasons_on_disk_are_stable() {
        let cases = [
            (FileReason::SourceMoved, r#""SourceMoved""#),
            (
                FileReason::Extraction(ExtractFailure::Malformed),
                r#"{"Extraction":"Malformed"}"#,
            ),
            (
                FileReason::Extraction(ExtractFailure::BadLocator),
                r#"{"Extraction":"BadLocator"}"#,
            ),
            (
                FileReason::Extraction(ExtractFailure::NoText),
                r#"{"Extraction":"NoText"}"#,
            ),
            (
                FileReason::Extraction(ExtractFailure::Panicked),
                r#"{"Extraction":"Panicked"}"#,
            ),
            (
                FileReason::Extraction(ExtractFailure::LimitExceeded(LimitKind::Depth)),
                r#"{"Extraction":{"LimitExceeded":"Depth"}}"#,
            ),
        ];
        for (reason, stored) in cases {
            assert_eq!(serde_json::to_string(&reason).unwrap(), stored);
            assert_eq!(serde_json::from_str::<FileReason>(stored).unwrap(), reason);
        }
    }

    #[test]
    fn a_reason_reads_clearly() {
        assert_eq!(
            FileReason::Extraction(ExtractFailure::NoText).to_string(),
            "the file has no extractable text"
        );
        assert_eq!(
            FileReason::SourceMoved.to_string(),
            "the recorded version of the file is gone"
        );
    }

    #[test]
    fn splitting_keeps_ordinals_and_neither_part_reuses_one() {
        let mut table = FileTable::new();
        for key in ["a", "b", "c"] {
            table.insert(entry(key));
        }
        let (mut left, mut right) = table.split_at("b");
        assert_eq!(left.len(), 1);
        assert_eq!(right.len(), 2);
        assert_eq!(
            right.get("c").unwrap().ordinal(),
            table.get("c").unwrap().ordinal()
        );
        left.insert(entry("a2"));
        right.insert(entry("d"));
        assert!(left.get("a2").unwrap().ordinal() >= 3);
        assert!(right.get("d").unwrap().ordinal() >= 3);
    }

    #[test]
    fn the_columns_on_disk_are_stable() {
        let names: Vec<String> = schema()
            .fields()
            .iter()
            .map(|f| format!("{}:{}", f.name(), f.data_type()))
            .collect();
        assert_eq!(
            names,
            [
                "ordinal:UInt64",
                "key:Utf8",
                "size:UInt64",
                "etag:Utf8",
                "content_version:UInt64",
                "status:Utf8",
                "reason:Utf8",
                "display_name:Utf8",
                "media_type:Utf8",
                "modified:Int64",
                "attributes:Utf8",
                "unread_parts:UInt32",
            ]
        );
    }

    #[test]
    fn reports_a_newer_format() {
        let bytes = stored::write(schema(), empty_columns(), FORMAT + 1, vec![]).unwrap();
        let err = FileIndex::decode(Bytes::from(bytes)).unwrap_err();
        assert!(
            matches!(err, Error::UnsupportedFormat { format } if format == FORMAT + 1),
            "{err}"
        );
    }

    fn empty_columns() -> Vec<ArrayRef> {
        vec![
            Arc::new(UInt64Array::from_iter_values([])),
            Arc::new(StringArray::from_iter_values(Vec::<String>::new())),
            Arc::new(UInt64Array::from_iter_values([])),
            Arc::new(StringArray::from_iter_values(Vec::<String>::new())),
            Arc::new(UInt64Array::from_iter_values([])),
            Arc::new(StringArray::from_iter_values(Vec::<String>::new())),
            Arc::new(StringArray::from(Vec::<Option<&str>>::new())),
            Arc::new(StringArray::from(Vec::<Option<&str>>::new())),
            Arc::new(StringArray::from(Vec::<Option<&str>>::new())),
            Arc::new(Int64Array::from(Vec::<Option<i64>>::new())),
            Arc::new(StringArray::from_iter_values(Vec::<String>::new())),
            Arc::new(UInt32Array::from_iter_values([])),
        ]
    }

    #[test]
    fn damaged_bytes_give_an_error_and_never_a_panic() {
        FileTable::decode(Bytes::from_static(b"garbage")).unwrap_err();
        let mut table = FileTable::new();
        table.insert(
            entry("docs/a.txt")
                .with_display_name(Some("name".to_owned()))
                .with_media_type(Some(MediaType::new("text/plain").unwrap()))
                .with_modified(Some(Timestamp::from_millis(-3)))
                .with_attributes(Attributes::from([(
                    "amount".to_owned(),
                    AttributeValue::Number(Number::new(2.5).unwrap()),
                )])),
        );
        let good = table.encode().unwrap();
        FileTable::decode(Bytes::copy_from_slice(&good[..good.len() / 2])).unwrap_err();
        for at in 0..good.len() {
            let mut bad = good.clone();
            bad[at] ^= 0xff;
            let _outcome = FileTable::decode(Bytes::from(bad));
        }
    }

    fn attribute_value() -> impl Strategy<Value = AttributeValue> {
        prop_oneof![
            ".{0,8}".prop_map(AttributeValue::One),
            prop::collection::vec(".{0,8}", 0..3).prop_map(AttributeValue::Many),
            any::<f64>()
                .prop_filter_map("finite", Number::new)
                .prop_map(AttributeValue::Number),
            any::<i64>().prop_map(|millis| AttributeValue::Time(Timestamp::from_millis(millis))),
        ]
    }

    #[test]
    fn a_stored_media_type_that_is_not_valid_is_an_error() {
        let mut columns = empty_columns();
        let one = |values: Vec<Option<&str>>| -> ArrayRef { Arc::new(StringArray::from(values)) };
        columns[0] = Arc::new(UInt64Array::from_iter_values([0]));
        columns[1] = Arc::new(StringArray::from_iter_values(["docs/a.txt"]));
        columns[2] = Arc::new(UInt64Array::from_iter_values([5]));
        columns[3] = Arc::new(StringArray::from_iter_values(["9f3a"]));
        columns[4] = Arc::new(UInt64Array::from_iter_values([3]));
        columns[5] = Arc::new(StringArray::from_iter_values(["pending"]));
        columns[6] = one(vec![None]);
        columns[7] = one(vec![None]);
        columns[8] = one(vec![Some("not a media type")]);
        columns[9] = Arc::new(Int64Array::from(vec![None::<i64>]));
        columns[10] = Arc::new(StringArray::from_iter_values(["{}"]));
        columns[11] = Arc::new(UInt32Array::from_iter_values([0]));
        let bytes = stored::write(schema(), columns, FORMAT, vec![]).unwrap();
        let err = decode_entries(Bytes::from(bytes)).unwrap_err();
        assert!(matches!(err, Error::Core(_)), "{err}");
    }

    fn index_of(encoded: &Bytes) -> FileIndex {
        FileIndex::decode(grouped::index_of(encoded, MAGIC).unwrap()).unwrap()
    }

    fn many(count: usize) -> FileTable {
        let mut table = FileTable::new();
        for at in 0..count {
            table.insert(entry(&format!("docs/{at:05}.txt")));
        }
        for gone in (0..count).step_by(7) {
            table.remove(&format!("docs/{gone:05}.txt"));
        }
        table
    }

    #[test]
    fn a_table_of_several_groups_round_trips_and_each_group_reads_alone() {
        let table = many(1300);
        let encoded = Bytes::from(table.encode().unwrap());
        assert_eq!(FileTable::decode(encoded.clone()).unwrap(), table);
        let index = index_of(&encoded);
        assert_eq!(index.len(), table.len());
        assert_eq!(index.version(7), None, "a removed file is not in the index");
        for ordinal in [1, 600, 1299] {
            let expected = table.by_ordinal(ordinal).unwrap();
            assert_eq!(index.version(ordinal), Some(expected.content_version()));
            let group = index.group_of(ordinal).unwrap();
            let (_, at) = index.group(group).unwrap();
            let found = FileGroup::decode(slice(&encoded, at).unwrap(), &index, group).unwrap();
            assert_eq!(found.by_ordinal(ordinal), Some(expected));
        }
        let (_, first) = index.group(0).unwrap();
        FileGroup::decode(slice(&encoded, first).unwrap(), &index, 1).unwrap_err();
    }

    #[test]
    fn the_index_holds_what_counting_files_needs() {
        let mut table = many(20);
        table.mark_ready("docs/00001.txt", 2);
        let index = index_of(&Bytes::from(table.encode().unwrap()));
        let found: Vec<u64> = index.ordinals_for("docs/00005.txt").collect();
        assert_eq!(found, [5]);
        assert_eq!(index.ordinals_for("docs/00007.txt").count(), 0);
        let state = index.states().first().unwrap();
        assert_eq!(state.ordinal(), 1);
        assert_eq!(state.status(), FileStatus::Ready);
        assert_eq!(state.unread_parts(), 2);
        assert_eq!(state.reason(), None);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        #[test]
        fn any_table_round_trips(
            files in prop::collection::btree_map(
                "[a-z/é]{1,20}",
                (
                    any::<u64>(),
                    "[a-f0-9]{1,16}",
                    prop::option::of(".{0,12}"),
                    prop::option::of("[a-z]{1,8}/[a-z0-9.+-]{1,12}"),
                    prop::option::of(any::<i64>()),
                    prop::collection::btree_map("[a-z]{1,6}", attribute_value(), 0..4),
                ),
                0..40,
            ),
        ) {
            let mut table = FileTable::new();
            for (key, (size, tag, name, media_type, modified, attributes)) in files {
                let etag = ETag::new(tag).unwrap();
                let version = ManifestVersion::new(7).unwrap();
                table.insert(
                    FileEntry::pending(key, size, etag, version)
                        .with_display_name(name)
                        .with_media_type(media_type.map(|value| MediaType::new(value).unwrap()))
                        .with_modified(modified.map(Timestamp::from_millis))
                        .with_attributes(attributes),
                );
            }
            prop_assert_eq!(round_trip(&table), table);
        }
    }
}
