use std::fmt;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{Chunk, ETag, Error, Location, Locator, Result, SourceRef};

/// The most bytes of text in one chunk, about 500 tokens.
pub const MAX_CHUNK_BYTES: usize = 2_000;

const DEFAULT_PAGES: usize = 20_000;
const DEFAULT_CHARS: usize = 32 * 1024 * 1024;
const DEFAULT_CHUNKS: usize = 200_000;
const DEFAULT_FILE_BYTES: u64 = 128 * 1024 * 1024;
const DEFAULT_ARCHIVE_ENTRIES: usize = 10_000;
const DEFAULT_ARCHIVE_BYTES: u64 = 64 * 1024 * 1024;
const DEFAULT_ARCHIVE_RATIO: u32 = 200;
const DEFAULT_DEPTH: usize = 256;

/// One of the limits a caller sets on extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum LimitKind {
    /// Pages in a document.
    Pages,
    /// Extracted characters in a file.
    Chars,
    /// Chunks from a file.
    Chunks,
    /// Bytes of one file, read whole to extract it.
    FileBytes,
    /// Entries in an archive.
    ArchiveEntries,
    /// Bytes an archive entry expands to.
    ArchiveBytes,
    /// How many times larger than its compressed size an archive entry may be.
    ArchiveRatio,
    /// Nesting of elements in XML, or of values in JSON.
    Depth,
}

impl fmt::Display for LimitKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            LimitKind::Pages => "pages",
            LimitKind::Chars => "extracted characters",
            LimitKind::Chunks => "chunks",
            LimitKind::FileBytes => "file bytes",
            LimitKind::ArchiveEntries => "archive entries",
            LimitKind::ArchiveBytes => "archive bytes",
            LimitKind::ArchiveRatio => "archive ratio",
            LimitKind::Depth => "nesting depth",
        })
    }
}

/// Why one file could not be extracted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ExtractFailure {
    /// The file went over a limit.
    #[error("the limit on {0} was exceeded")]
    LimitExceeded(LimitKind),
    /// The file could not be parsed. A parser's own message may quote the file, so it is dropped.
    #[error("the file could not be parsed")]
    Malformed,
    /// The extractor reported a location outside the file, which is a bug in the extractor.
    #[error("the extractor reported a location outside the file")]
    BadLocator,
    /// The file has no text to index, such as an empty file.
    #[error("the file has no extractable text")]
    NoText,
    /// The file has parts that hold text but none could be read, such as a PDF of scanned
    /// pages. It counts as failed, not unsupported, so a search over it is never complete.
    #[error("no part of the file could be read, such as a scan")]
    Unreadable,
    /// The extractor panicked. Its message may quote the file, so it is dropped.
    #[error("the extractor panicked")]
    Panicked,
    /// No registered extractor reads this type of file.
    #[error("no registered extractor reads this type of file")]
    NoExtractor,
}

impl ExtractFailure {
    /// Whether the file is unsupported rather than failed. An unsupported file cannot be
    /// searched, so it does not make a search's coverage incomplete.
    pub fn is_unsupported(self) -> bool {
        matches!(self, ExtractFailure::NoText | ExtractFailure::NoExtractor)
    }
}

/// The limits a caller sets on extracting one file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExtractLimits {
    pages: usize,
    chars: usize,
    chunks: usize,
    file_bytes: u64,
    archive_entries: usize,
    archive_bytes: u64,
    archive_ratio: u32,
    depth: usize,
}

impl Default for ExtractLimits {
    fn default() -> Self {
        ExtractLimits {
            pages: DEFAULT_PAGES,
            chars: DEFAULT_CHARS,
            chunks: DEFAULT_CHUNKS,
            file_bytes: DEFAULT_FILE_BYTES,
            archive_entries: DEFAULT_ARCHIVE_ENTRIES,
            archive_bytes: DEFAULT_ARCHIVE_BYTES,
            archive_ratio: DEFAULT_ARCHIVE_RATIO,
            depth: DEFAULT_DEPTH,
        }
    }
}

impl ExtractLimits {
    /// Starts from the default limits.
    pub fn builder() -> ExtractLimitsBuilder {
        ExtractLimitsBuilder(ExtractLimits::default())
    }

    /// The most pages in a document.
    pub fn pages(&self) -> usize {
        self.pages
    }

    /// The most extracted characters in a file.
    pub fn chars(&self) -> usize {
        self.chars
    }

    /// The most chunks from a file.
    pub fn chunks(&self) -> usize {
        self.chunks
    }

    /// The largest file, in bytes, that is read to extract it.
    pub fn file_bytes(&self) -> u64 {
        self.file_bytes
    }

    /// The most entries in an archive.
    pub fn archive_entries(&self) -> usize {
        self.archive_entries
    }

    /// The most bytes an archive entry may expand to.
    pub fn archive_bytes(&self) -> u64 {
        self.archive_bytes
    }

    /// How many times larger than its compressed size an archive entry may be.
    pub fn archive_ratio(&self) -> u32 {
        self.archive_ratio
    }

    /// The deepest nesting of elements in XML, or of values in JSON.
    pub fn depth(&self) -> usize {
        self.depth
    }
}

/// Builds [`ExtractLimits`].
#[derive(Debug, Clone, Copy)]
pub struct ExtractLimitsBuilder(ExtractLimits);

impl ExtractLimitsBuilder {
    /// Sets the most pages in a document.
    #[must_use]
    pub fn pages(mut self, value: usize) -> Self {
        self.0.pages = value;
        self
    }

    /// Sets the most extracted characters in a file.
    #[must_use]
    pub fn chars(mut self, value: usize) -> Self {
        self.0.chars = value;
        self
    }

    /// Sets the most chunks from a file.
    #[must_use]
    pub fn chunks(mut self, value: usize) -> Self {
        self.0.chunks = value;
        self
    }

    /// Sets the largest file, in bytes, that is read to extract it.
    #[must_use]
    pub fn file_bytes(mut self, value: u64) -> Self {
        self.0.file_bytes = value;
        self
    }

    /// Sets the most entries in an archive.
    #[must_use]
    pub fn archive_entries(mut self, value: usize) -> Self {
        self.0.archive_entries = value;
        self
    }

    /// Sets the most bytes an archive entry may expand to.
    #[must_use]
    pub fn archive_bytes(mut self, value: u64) -> Self {
        self.0.archive_bytes = value;
        self
    }

    /// Sets how many times larger than its compressed size an archive entry may be.
    #[must_use]
    pub fn archive_ratio(mut self, value: u32) -> Self {
        self.0.archive_ratio = value;
        self
    }

    /// Sets the deepest nesting of elements in XML, or of values in JSON.
    #[must_use]
    pub fn depth(mut self, value: usize) -> Self {
        self.0.depth = value;
        self
    }

    /// Checks the limits.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidLimit`] for a limit of zero, which no file could meet.
    pub fn build(self) -> Result<ExtractLimits> {
        let limits = self.0;
        let checks = [
            (limits.pages == 0, LimitKind::Pages),
            (limits.chars == 0, LimitKind::Chars),
            (limits.chunks == 0, LimitKind::Chunks),
            (limits.file_bytes == 0, LimitKind::FileBytes),
            (limits.archive_entries == 0, LimitKind::ArchiveEntries),
            (limits.archive_bytes == 0, LimitKind::ArchiveBytes),
            (limits.archive_ratio == 0, LimitKind::ArchiveRatio),
            (limits.depth == 0, LimitKind::Depth),
        ];
        match checks.into_iter().find(|(is_zero, _)| *is_zero) {
            Some((_, limit)) => Err(Error::InvalidLimit { limit }),
            None => Ok(limits),
        }
    }
}

/// Text an extractor found at one place in a file, in reading order.
#[derive(Clone, PartialEq, Eq)]
pub struct Passage {
    locator: Locator,
    heading_path: Vec<String>,
    text: String,
}

impl Passage {
    /// Creates a passage. `heading_path` lists the headings above it, outermost first.
    pub fn new(locator: Locator, heading_path: Vec<String>, text: impl Into<String>) -> Self {
        Passage {
            locator,
            heading_path,
            text: text.into(),
        }
    }

    /// Where the text sits in the file.
    pub fn locator(&self) -> Locator {
        self.locator
    }

    /// The headings above this text, outermost first.
    pub fn heading_path(&self) -> &[String] {
        &self.heading_path
    }

    /// The text.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Whether the text holds nothing but whitespace.
    pub fn is_blank(&self) -> bool {
        self.text.trim().is_empty()
    }

    /// Turns this passage into a chunk of the file at `location` with ETag `etag`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidRef`] if the location has no key.
    pub fn into_chunk(self, location: &Location, etag: &ETag) -> Result<Chunk> {
        let source = SourceRef::new(location.clone(), self.locator, etag.clone())?;
        Ok(Chunk::new(source, self.heading_path, self.text))
    }
}

impl fmt::Debug for Passage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Passage")
            .field("locator", &self.locator)
            .field("text_bytes", &self.text.len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ByteRange;

    fn passage(text: &str) -> Passage {
        Passage::new(
            Locator::Bytes(ByteRange::new(0, 5).unwrap()),
            vec!["Secret heading".to_owned()],
            text,
        )
    }

    #[test]
    fn the_defaults_are_valid_and_overrides_apply() {
        let limits = ExtractLimits::builder()
            .pages(3)
            .archive_ratio(7)
            .build()
            .unwrap();
        assert_eq!(limits.pages(), 3);
        assert_eq!(limits.archive_ratio(), 7);
        assert_eq!(limits.chars(), ExtractLimits::default().chars());
    }

    #[test]
    fn refuses_a_limit_of_zero() {
        let b = ExtractLimits::builder;
        let cases = [
            (b().pages(0), LimitKind::Pages),
            (b().chars(0), LimitKind::Chars),
            (b().chunks(0), LimitKind::Chunks),
            (b().file_bytes(0), LimitKind::FileBytes),
            (b().archive_entries(0), LimitKind::ArchiveEntries),
            (b().archive_bytes(0), LimitKind::ArchiveBytes),
            (b().archive_ratio(0), LimitKind::ArchiveRatio),
            (b().depth(0), LimitKind::Depth),
        ];
        for (builder, expected) in cases {
            let err = builder.build().unwrap_err();
            assert!(
                matches!(err, Error::InvalidLimit { limit } if limit == expected),
                "{err}"
            );
        }
    }

    #[test]
    fn only_a_file_with_no_text_or_no_extractor_is_unsupported() {
        assert!(ExtractFailure::NoText.is_unsupported());
        assert!(ExtractFailure::NoExtractor.is_unsupported());
        for failure in [
            ExtractFailure::LimitExceeded(LimitKind::Pages),
            ExtractFailure::Malformed,
            ExtractFailure::BadLocator,
            ExtractFailure::Unreadable,
            ExtractFailure::Panicked,
        ] {
            assert!(!failure.is_unsupported(), "{failure}");
        }
    }

    #[test]
    fn a_failure_reads_clearly() {
        assert_eq!(
            ExtractFailure::LimitExceeded(LimitKind::Depth).to_string(),
            "the limit on nesting depth was exceeded"
        );
    }

    #[test]
    fn a_chunk_carries_its_ref_and_hash() {
        let location: Location = "s3://example-bucket/a.txt".parse().unwrap();
        let etag = ETag::new("9f3a").unwrap();
        let chunk = passage("hello").into_chunk(&location, &etag).unwrap();
        assert_eq!(
            chunk.source().to_string(),
            "s3://example-bucket/a.txt#bytes=0-5@etag=9f3a"
        );
        assert_eq!(chunk.heading_path(), ["Secret heading"]);
        assert_eq!(chunk.text(), "hello");
        assert_eq!(chunk.hash(), crate::ChunkHash::of("Secret heading\nhello"));
    }

    #[test]
    fn a_chunk_needs_a_file() {
        let bucket: Location = "s3://example-bucket/".parse().unwrap();
        passage("hello")
            .into_chunk(&bucket, &ETag::new("9f3a").unwrap())
            .unwrap_err();
    }

    #[test]
    fn debug_output_never_shows_document_text() {
        let location: Location = "s3://example-bucket/a.txt".parse().unwrap();
        let chunk = passage("confidential words")
            .into_chunk(&location, &ETag::new("9f3a").unwrap())
            .unwrap();
        for output in [
            format!("{:?}", passage("confidential words")),
            format!("{chunk:?}"),
        ] {
            assert!(!output.contains("confidential"), "{output}");
            assert!(!output.contains("Secret heading"), "{output}");
        }
    }

    #[test]
    fn blank_means_only_whitespace() {
        assert!(passage(" \n\t").is_blank());
        assert!(!passage(" a ").is_blank());
    }
}

/// What an extractor read from one file: its passages and the parts it could not read.
///
/// A part is, for example, a PDF page that holds only an image. A file with unread parts is
/// searched only in part, and coverage says so.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Extraction {
    passages: Vec<Passage>,
    unread_parts: u32,
}

impl Extraction {
    /// The passages of a file read whole.
    pub fn new(passages: Vec<Passage>) -> Self {
        Extraction {
            passages,
            unread_parts: 0,
        }
    }

    /// Records that `parts` parts of the file could not be read.
    #[must_use]
    pub fn with_unread_parts(mut self, parts: u32) -> Self {
        self.unread_parts = parts;
        self
    }

    /// The passages, in reading order.
    pub fn passages(&self) -> &[Passage] {
        &self.passages
    }

    /// The passages, in reading order.
    pub fn into_passages(self) -> Vec<Passage> {
        self.passages
    }

    /// How many parts of the file could not be read.
    pub fn unread_parts(&self) -> u32 {
        self.unread_parts
    }
}
