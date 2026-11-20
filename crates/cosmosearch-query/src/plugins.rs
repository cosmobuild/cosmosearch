use std::collections::{BTreeMap, BTreeSet};

use cosmosearch_chunk::Extractors;
use cosmosearch_core::{
    ExtractorProfile, IndexKind, Manifest, RerankerIdentity, SegmentEntry, ShardEntry,
    SharedExtractor, SharedIndex, SharedReranker,
};

use crate::{Error, Plugin, Result};

/// The extractors, index plug-ins and reranker a reader works with.
#[derive(Debug, Clone, Default)]
pub struct Plugins {
    extractors: Extractors,
    indexes: BTreeMap<IndexKind, SharedIndex>,
    reranker: Option<SharedReranker>,
}

impl Plugins {
    /// Starts from `extractors`, no index plug-in and no reranker.
    pub fn new(extractors: Extractors) -> Self {
        Plugins {
            extractors,
            indexes: BTreeMap::new(),
            reranker: None,
        }
    }

    /// Adds an extractor, which reads the extensions it names in place of any added before.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DuplicatePlugin`] if an extractor with the same profile is there.
    pub fn add_extractor(&mut self, extractor: SharedExtractor) -> Result<()> {
        let profile = extractor.get().profile().clone();
        if self.extractors.insert(extractor) {
            Ok(())
        } else {
            Err(Error::DuplicatePlugin(Plugin::Extractor(profile)))
        }
    }

    /// Adds an index plug-in.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DuplicatePlugin`] if a plug-in of the same kind is there.
    pub fn add_index(&mut self, index: SharedIndex) -> Result<()> {
        let kind = index.get().kind().clone();
        if self.indexes.contains_key(&kind) {
            return Err(Error::DuplicatePlugin(Plugin::Index(kind)));
        }
        self.indexes.insert(kind, index);
        Ok(())
    }

    /// Sets the reranker that rescores the shortlist of every search, in place of any set
    /// before.
    pub fn set_reranker(&mut self, reranker: SharedReranker) {
        self.reranker = Some(reranker);
    }

    /// The extractors.
    pub fn extractors(&self) -> &Extractors {
        &self.extractors
    }

    /// The profiles of the extractors, in order.
    pub fn extractor_profiles(&self) -> impl Iterator<Item = &ExtractorProfile> {
        self.extractors.profiles()
    }

    /// The kinds of the index plug-ins, in order.
    pub fn index_kinds(&self) -> impl Iterator<Item = &IndexKind> {
        self.indexes.keys()
    }

    /// What gives the reranker's scores, if there is a reranker.
    pub fn reranker(&self) -> Option<&RerankerIdentity> {
        self.reranker
            .as_ref()
            .map(|reranker| reranker.get().identity())
    }

    pub(crate) fn shared_reranker(&self) -> Option<&SharedReranker> {
        self.reranker.as_ref()
    }

    pub(crate) fn index(&self, kind: &IndexKind) -> Option<&SharedIndex> {
        self.indexes.get(kind)
    }

    /// Every plug-in that built a segment of `manifest` and is not here.
    pub fn missing(&self, manifest: &Manifest) -> BTreeSet<Plugin> {
        let mut missing = BTreeSet::new();
        let segments = manifest.shards().values().flat_map(ShardEntry::segments);
        for built_by in segments.map(SegmentEntry::built_by) {
            if let Some(kind) = built_by.index()
                && !self.indexes.contains_key(kind)
            {
                missing.insert(Plugin::Index(kind.clone()));
            }
            for profile in built_by.extractors() {
                if !self.extractors.contains(profile) {
                    missing.insert(Plugin::Extractor(profile.clone()));
                }
            }
        }
        missing
    }
}
