use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;

use crate::{
    AllowedRows, Built, Chunk, ExtractFailure, ExtractLimits, Extraction, ExtractorProfile, Hits,
    IndexError, IndexKind, Level, ModelId, Query, SegmentEntry, SegmentId, SegmentSource,
};

/// A plug-in that turns files of some format into text an index can use.
pub trait Extractor: Send + Sync + fmt::Debug {
    /// Identifies this extractor and its output; recorded with every segment it builds.
    fn profile(&self) -> &ExtractorProfile;

    /// The file name extensions this extractor reads, in lower case and without the dot.
    fn extensions(&self) -> &[&str];

    /// Extracts the passages of one file, in reading order, and counts the parts it could
    /// not read. Blocks for as long as parsing takes.
    ///
    /// # Errors
    ///
    /// Returns an [`ExtractFailure`] if this file cannot be extracted within `limits`.
    fn extract(&self, file: &[u8], limits: &ExtractLimits) -> Result<Extraction, ExtractFailure>;
}

/// A plug-in that builds and searches one kind of index.
///
/// Both calls run on the host's async runtime, so a plug-in moves blocking work, such as
/// embedding or building an index in memory, off it with `spawn_blocking`.
#[async_trait]
pub trait Index: Send + Sync + fmt::Debug {
    /// The kind of index; recorded with every segment it builds.
    fn kind(&self) -> &IndexKind;

    /// The level this index serves; recorded with every segment it builds.
    fn level(&self) -> Level;

    /// The embedding identity of the vectors it builds, recorded with every segment it
    /// builds; `None` for an index without vectors.
    fn model_id(&self) -> Option<&ModelId> {
        None
    }

    /// The settings that change which hits a search returns, such as how much of a segment
    /// it reads; a result cache keeps results apart by them. Empty unless the plug-in has
    /// such settings.
    fn search_settings(&self) -> Vec<u64> {
        Vec::new()
    }

    /// Builds the files of a segment holding `chunks`, whose rows are their positions, and
    /// reports what building spent.
    ///
    /// # Errors
    ///
    /// Returns an [`IndexError`] if the files cannot be built.
    async fn build(&self, segment: &SegmentId, chunks: &[Chunk]) -> Result<Built, IndexError>;

    /// Finds the best `limit` matches across those of `segments` this plug-in built, among
    /// the rows `allowed` names. Other rows are never scored, so they cannot take a place.
    /// The usage reported leaves out reads through `source`, which the caller counts.
    ///
    /// # Errors
    ///
    /// Returns an [`IndexError`] if a segment cannot be read or searched.
    async fn search(
        &self,
        source: &SegmentSource,
        segments: &[SegmentEntry],
        allowed: &AllowedRows,
        query: &Query,
        limit: usize,
    ) -> Result<Hits, IndexError>;
}

/// A shared handle on an extractor, for running it on another thread.
#[derive(Debug, Clone)]
pub struct SharedExtractor(Arc<dyn Extractor>);

impl SharedExtractor {
    /// Shares `extractor`.
    pub fn new(extractor: impl Extractor + 'static) -> Self {
        SharedExtractor(Arc::new(extractor))
    }

    /// The extractor.
    pub fn get(&self) -> &dyn Extractor {
        self.0.as_ref()
    }
}

/// A shared handle on an index plug-in, for running it on another thread.
#[derive(Debug, Clone)]
pub struct SharedIndex(Arc<dyn Index>);

impl SharedIndex {
    /// Shares `index`.
    pub fn new(index: impl Index + 'static) -> Self {
        SharedIndex(Arc::new(index))
    }

    /// The index plug-in.
    pub fn get(&self) -> &dyn Index {
        self.0.as_ref()
    }
}

impl<T: Index + 'static> From<T> for SharedIndex {
    fn from(index: T) -> Self {
        SharedIndex::new(index)
    }
}
