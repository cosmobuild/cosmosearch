use std::error::Error as StdError;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use thiserror::Error;

use crate::{ByteRange, SegmentFileName, SegmentId, Usage};

/// The cause of a failure inside an index plug-in or its storage.
#[derive(Debug)]
pub struct IndexCause(Box<dyn StdError + Send + Sync>);

impl IndexCause {
    /// Wraps a library's own error.
    pub fn new(source: impl Into<Box<dyn StdError + Send + Sync>>) -> Self {
        IndexCause(source.into())
    }
}

impl fmt::Display for IndexCause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl StdError for IndexCause {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.0.source()
    }
}

/// Errors from building or searching an index.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum IndexError {
    /// A segment file could not be read.
    #[error("a segment file could not be read")]
    Read(#[source] IndexCause),

    /// A file of a segment is not in the store, for example because it was swept.
    #[error("a file of segment `{segment}` is missing")]
    MissingSegment {
        /// The segment.
        segment: SegmentId,
    },

    /// The index library failed.
    #[error("the index library failed")]
    Engine(#[source] IndexCause),

    /// A stored segment is damaged.
    #[error("a stored segment is damaged: {reason}")]
    Malformed {
        /// What is wrong with it.
        reason: &'static str,
    },
}

/// What an index plug-in reads its segment files through.
#[async_trait]
pub trait SegmentFiles: Send + Sync + fmt::Debug {
    /// Reads part of one file of a segment.
    ///
    /// # Errors
    ///
    /// Returns [`IndexError::MissingSegment`] if the file is not in the store, and
    /// [`IndexError::Read`] if the bytes cannot be read.
    async fn read_range(
        &self,
        segment: &SegmentId,
        name: &SegmentFileName,
        range: ByteRange,
    ) -> Result<Vec<u8>, IndexError>;
}

/// A shared handle on the segment files of one index.
#[derive(Debug, Clone)]
pub struct SegmentSource(Arc<dyn SegmentFiles>);

impl SegmentSource {
    /// A source reading through `files`.
    pub fn new(files: impl SegmentFiles + 'static) -> Self {
        SegmentSource(Arc::new(files))
    }

    /// Reads part of one file of a segment.
    ///
    /// # Errors
    ///
    /// Returns [`IndexError::MissingSegment`] if the file is not in the store, and
    /// [`IndexError::Read`] if the bytes cannot be read.
    pub async fn read_range(
        &self,
        segment: &SegmentId,
        name: &SegmentFileName,
        range: ByteRange,
    ) -> Result<Vec<u8>, IndexError> {
        self.0.read_range(segment, name, range).await
    }
}

/// One file a plug-in built for a segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuiltFile {
    name: SegmentFileName,
    body: Vec<u8>,
}

impl BuiltFile {
    /// A built file.
    pub fn new(name: SegmentFileName, body: Vec<u8>) -> Self {
        BuiltFile { name, body }
    }

    /// The file's name within its segment.
    pub fn name(&self) -> &SegmentFileName {
        &self.name
    }

    /// The file's bytes.
    pub fn body(&self) -> &[u8] {
        &self.body
    }

    /// Takes the file's bytes.
    pub fn into_body(self) -> Vec<u8> {
        self.body
    }
}

/// The files an index plug-in built for a segment, and what building them spent.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Built {
    files: Vec<BuiltFile>,
    usage: Usage,
}

impl Built {
    /// `files`, built with `usage`.
    pub fn new(files: Vec<BuiltFile>, usage: Usage) -> Self {
        Built { files, usage }
    }

    /// The files.
    pub fn files(&self) -> &[BuiltFile] {
        &self.files
    }

    /// Takes the files.
    pub fn into_files(self) -> Vec<BuiltFile> {
        self.files
    }

    /// What building spent, such as the tokens an embedder read.
    pub fn usage(&self) -> Usage {
        self.usage
    }
}

/// What a caller is looking for.
#[derive(Clone, PartialEq, Eq)]
pub struct Query {
    text: String,
    exact_phrases: bool,
}

impl Query {
    /// A query for `text`, read as plain words. Its length is not limited, and what a search
    /// costs grows with it, so a host taking queries from users bounds them.
    pub fn new(text: impl Into<String>) -> Self {
        Query {
            text: text.into(),
            exact_phrases: false,
        }
    }

    /// The same query with text in double quotes read as exact phrases, by the index plug-ins
    /// that support them, such as BM25. Without it, quotes are ordinary text, so prose that
    /// quotes something is searched like any other prose.
    #[must_use]
    pub fn with_exact_phrases(mut self) -> Self {
        self.exact_phrases = true;
        self
    }

    /// The text.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Whether text in double quotes is read as exact phrases.
    pub fn exact_phrases(&self) -> bool {
        self.exact_phrases
    }
}

impl fmt::Debug for Query {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Query")
            .field("text_bytes", &self.text.len())
            .field("exact_phrases", &self.exact_phrases)
            .finish()
    }
}

/// One match: a chunk of a segment and its score.
#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    segment: SegmentId,
    row: u64,
    score: f32,
}

impl Hit {
    /// A match on the chunk in `row` of `segment`.
    pub fn new(segment: SegmentId, row: u64, score: f32) -> Self {
        Hit {
            segment,
            row,
            score,
        }
    }

    /// The segment holding the chunk.
    pub fn segment(&self) -> &SegmentId {
        &self.segment
    }

    /// The chunk's row in its segment.
    pub fn row(&self) -> u64 {
        self.row
    }

    /// The score, higher being better.
    pub fn score(&self) -> f32 {
        self.score
    }
}

/// The matches an index plug-in found, and what finding them spent.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Hits {
    hits: Vec<Hit>,
    usage: Usage,
}

impl Hits {
    /// `hits`, found with `usage`.
    pub fn new(hits: Vec<Hit>, usage: Usage) -> Self {
        Hits { hits, usage }
    }

    /// The matches.
    pub fn hits(&self) -> &[Hit] {
        &self.hits
    }

    /// Takes the matches.
    pub fn into_hits(self) -> Vec<Hit> {
        self.hits
    }

    /// What finding them spent, such as the tokens of embedding the query.
    pub fn usage(&self) -> Usage {
        self.usage
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_output_never_shows_the_query() {
        let output = format!("{:?}", Query::new("confidential question"));
        assert!(!output.contains("confidential"), "{output}");
    }
}
