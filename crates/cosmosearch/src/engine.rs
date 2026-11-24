use std::sync::Arc;
use std::time::Duration;

use cosmosearch_chunk::Extractors;
use cosmosearch_core::{
    ExtractLimits, Extractor, ExtractorProfile, FileSource, Filter, HeatSink, IndexId, IndexKind,
    Manifest, ManifestVersion, PassageCache, Reranker, RerankerIdentity, ResultCache, Scope,
    SegmentCache, SharedCache, SharedExtractor, SharedHeat, SharedIndex, SharedPassages,
    SharedReranker, SharedResults, SourceRef,
};
use cosmosearch_format::FileEntry;
use cosmosearch_query::{Plugins, Reader, ReaderOptions, Reading, SearchRequest, SearchResults};
use cosmosearch_store::{DataStore, MetadataStore};

use crate::Result;

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Engine>();
    assert_send_sync::<EngineBuilder>();
};

/// An open index, searched and read through the manifest snapshot it holds until refreshed.
#[derive(Debug)]
pub struct Engine {
    reader: Reader,
}

impl Engine {
    /// Starts building an engine over `data` and `metadata`, with the built-in text and
    /// Markdown extractors.
    pub fn builder(data: DataStore, metadata: MetadataStore) -> EngineBuilder {
        EngineBuilder {
            data,
            metadata,
            file_source: FileSource::Listing,
            extractors: Vec::new(),
            indexes: Vec::new(),
            reranker: None,
            options: ReaderOptions::new(),
        }
    }

    /// The manifest this engine searches.
    pub fn manifest(&self) -> Arc<Manifest> {
        self.reader.manifest()
    }

    /// The id of the index this engine searches. To close an index and drop its cached files,
    /// drop its engines and purge this id from the cache.
    pub fn index_id(&self) -> &IndexId {
        self.reader.index_id()
    }

    /// The extractors this engine has, built in or added.
    pub fn extractors(&self) -> impl Iterator<Item = &ExtractorProfile> {
        self.reader.plugins().extractor_profiles()
    }

    /// The indexes this engine has.
    pub fn indexes(&self) -> impl Iterator<Item = &IndexKind> {
        self.reader.plugins().index_kinds()
    }

    /// What gives the reranker's scores, if the engine has a reranker.
    pub fn reranker(&self) -> Option<&RerankerIdentity> {
        self.reader.plugins().reranker()
    }

    /// Finds the best passages for `request` among the files in its scope that pass its filter,
    /// opening only the shards its scope reaches, within its budget. With a reranker, the
    /// shortlist of the best fused hits is rescored and comes first; if the reranker fails, the
    /// hits keep their fused order and the results say so.
    ///
    /// # Errors
    ///
    /// As for [`Reader::search`].
    pub async fn search(&self, request: &SearchRequest) -> Result<SearchResults> {
        self.reader.search(request).await
    }

    /// Reads the passage `reference` names, from the file version it was made from.
    ///
    /// Any ref to an indexed passage inside the data location is read, whatever scope or
    /// filter the search that returned it had. A host that takes refs from users checks their
    /// access to the ref's file itself, or calls [`Engine::read_within`].
    ///
    /// # Errors
    ///
    /// As for [`Reader::expand`].
    pub async fn read(&self, reference: &SourceRef) -> Result<Reading> {
        self.reader.read(reference).await
    }

    /// Reads the passage `reference` names with up to `before` and `after` neighbouring chunks
    /// of the same file. Access is not checked, as for [`Engine::read`].
    ///
    /// # Errors
    ///
    /// As for [`Reader::expand`].
    pub async fn expand(
        &self,
        reference: &SourceRef,
        before: usize,
        after: usize,
    ) -> Result<Reading> {
        self.reader.expand(reference, before, after).await
    }

    /// Reads the passage `reference` names only if its file is in `scope` and passes
    /// `filter`, so the scope and filter that limit a caller's searches also limit what the
    /// caller reads.
    ///
    /// # Errors
    ///
    /// As for [`Reader::read_within`].
    pub async fn read_within(
        &self,
        reference: &SourceRef,
        scope: &Scope,
        filter: &Filter,
    ) -> Result<Reading> {
        self.reader.read_within(reference, scope, filter).await
    }

    /// Reads the passage `reference` names with its neighbours only if its file is in `scope`
    /// and passes `filter`.
    ///
    /// # Errors
    ///
    /// As for [`Reader::read_within`].
    pub async fn expand_within(
        &self,
        reference: &SourceRef,
        before: usize,
        after: usize,
        scope: &Scope,
        filter: &Filter,
    ) -> Result<Reading> {
        self.reader
            .expand_within(reference, before, after, scope, filter)
            .await
    }

    /// The file with this key as the index holds it: its status and why it is not ready, its
    /// display name, media type, modified time and attributes. Access is not checked, as for
    /// [`Engine::read`]; [`Engine::file_within`] checks it.
    ///
    /// # Errors
    ///
    /// As for [`Reader::file`].
    pub async fn file(&self, key: &str) -> Result<Option<FileEntry>> {
        self.reader.file(key).await
    }

    /// The file with this key, as [`Engine::file`] gives it, only if it is in `scope` and
    /// passes `filter`; `None` otherwise, as for a file the index does not hold.
    ///
    /// # Errors
    ///
    /// As for [`Reader::file_within`].
    pub async fn file_within(
        &self,
        key: &str,
        scope: &Scope,
        filter: &Filter,
    ) -> Result<Option<FileEntry>> {
        self.reader.file_within(key, scope, filter).await
    }

    /// Reads up to `limit` chunks of the file `key` in reading order, from its chunk at
    /// position `first`, counting from 0, so a whole file can be read page by page without a
    /// search hit. Past the last chunk, the reading holds none, as it does for a file with no
    /// chunks indexed yet; [`Engine::file`] gives its status. Each call reads the file again,
    /// the whole of it for formats that must be parsed, such as PDF. Access is not checked, as
    /// for [`Engine::read`].
    ///
    /// # Errors
    ///
    /// As for [`Reader::chunks`].
    pub async fn chunks(&self, key: &str, first: usize, limit: usize) -> Result<Reading> {
        self.reader.chunks(key, first, limit).await
    }

    /// Reads chunks of the file `key`, as [`Engine::chunks`] does, only if the file is in
    /// `scope` and passes `filter`.
    ///
    /// # Errors
    ///
    /// As for [`Reader::chunks_within`].
    pub async fn chunks_within(
        &self,
        key: &str,
        first: usize,
        limit: usize,
        scope: &Scope,
        filter: &Filter,
    ) -> Result<Reading> {
        self.reader
            .chunks_within(key, first, limit, scope, filter)
            .await
    }

    /// Re-reads `INDEX` and moves to the newest manifest, returning its version.
    ///
    /// # Errors
    ///
    /// As for [`Reader::refresh`].
    pub async fn refresh(&self) -> Result<ManifestVersion> {
        self.reader.refresh().await
    }
}

/// Chooses an engine's plug-ins and settings, and opens it.
#[derive(Debug)]
pub struct EngineBuilder {
    data: DataStore,
    metadata: MetadataStore,
    file_source: FileSource,
    extractors: Vec<SharedExtractor>,
    indexes: Vec<SharedIndex>,
    reranker: Option<SharedReranker>,
    options: ReaderOptions,
}

impl EngineBuilder {
    /// Says how files reach this index. Listing is the default.
    #[must_use]
    pub fn file_source(mut self, file_source: FileSource) -> Self {
        self.file_source = file_source;
        self
    }

    /// Adds an extractor, which reads the extensions it names in place of a built-in one.
    #[must_use]
    pub fn extractor(mut self, extractor: impl Extractor + 'static) -> Self {
        self.extractors.push(SharedExtractor::new(extractor));
        self
    }

    /// Adds an index: a plug-in, or a [`SharedIndex`] the host also hands its indexer, so
    /// both use one instance and its caches.
    #[must_use]
    pub fn index(mut self, index: impl Into<SharedIndex>) -> Self {
        self.indexes.push(index.into());
        self
    }

    /// Sets the reranker that rescores the shortlist of every search, in place of any set
    /// before. Without one, hits keep their fused order.
    #[must_use]
    pub fn reranker(mut self, reranker: impl Reranker + 'static) -> Self {
        self.reranker = Some(SharedReranker::new(reranker));
        self
    }

    /// Sets how many of a search's best fused hits the reranker rescores; 20 unless set, and
    /// zero turns reranking off.
    #[must_use]
    pub fn rerank_shortlist(mut self, size: usize) -> Self {
        self.options = self.options.rerank_shortlist(size);
        self
    }

    /// Sets the rank constant k of reciprocal rank fusion, 60 unless set.
    #[must_use]
    pub fn fusion_k(mut self, k: f64) -> Self {
        self.options = self.options.fusion_k(k);
        self
    }

    /// Sets the weight of index kind `kind` in fusion, 1 unless set: a hit at rank r in its
    /// list adds `weight / (k + r)` to its score.
    #[must_use]
    pub fn fusion_weight(mut self, kind: IndexKind, weight: f64) -> Self {
        self.options = self.options.fusion_weight(kind, weight);
        self
    }

    /// Sets the limits on extracting a file again when reading it.
    #[must_use]
    pub fn limits(mut self, limits: ExtractLimits) -> Self {
        self.options = self.options.limits(limits);
        self
    }

    /// Refreshes before a call once `interval` has passed since the last refresh or attempt,
    /// as for [`ReaderOptions::refresh_interval`]. Without it the engine refreshes only when
    /// asked. Nothing runs in the background.
    #[must_use]
    pub fn refresh_interval(mut self, interval: Duration) -> Self {
        self.options = self.options.refresh_interval(interval);
        self
    }

    /// Reads segment files through `cache`, such as a clone of one `Cache` that every engine
    /// of the process shares, in place of any set before. Without one, every search reads
    /// what it needs from the store.
    #[must_use]
    pub fn cache(mut self, cache: impl SegmentCache + 'static) -> Self {
        self.options = self.options.cache(SharedCache::new(cache));
        self
    }

    /// Keeps search results in `results`, such as a clone of one `SearchCache` that every
    /// engine of the process shares, and answers a repeated search from it, in place of any
    /// set before. Without one, every search is run.
    #[must_use]
    pub fn result_cache(mut self, results: impl ResultCache + 'static) -> Self {
        self.options = self.options.result_cache(SharedResults::new(results));
        self
    }

    /// Gives each hit a snippet, its passage with the query's words marked, reading at most
    /// `max_files` files for them in one search; zero, the default, gives none.
    #[must_use]
    pub fn snippets(mut self, max_files: usize) -> Self {
        self.options = self.options.snippets(max_files);
        self
    }

    /// Keeps passage text read for snippets and reranking in `passages`, such as a clone of
    /// one `TextCache` that every engine of the process shares, in place of any set before.
    #[must_use]
    pub fn passage_cache(mut self, passages: impl PassageCache + 'static) -> Self {
        self.options = self.options.passage_cache(SharedPassages::new(passages));
        self
    }

    /// Records heat in `heat`, such as a `HeatRecorder` that every engine of the process shares:
    /// each search counts the shards it covered and each read or expand the shard of its ref.
    /// Without it nothing is recorded.
    #[must_use]
    pub fn heat(mut self, heat: impl HeatSink + 'static) -> Self {
        self.options = self.options.heat(SharedHeat::new(heat));
        self
    }

    /// Opens the index, reading only `INDEX` and its manifest.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DuplicatePlugin`](crate::Error::DuplicatePlugin) if a plug-in was added
    /// twice, [`Error::InvalidFusion`](crate::Error::InvalidFusion) for a fusion k or weight
    /// that is not finite and above zero, and otherwise as for [`Reader::open`].
    pub async fn open(self) -> Result<Engine> {
        let mut plugins = Plugins::new(Extractors::builtin()?);
        for extractor in self.extractors {
            plugins.add_extractor(extractor)?;
        }
        for index in self.indexes {
            plugins.add_index(index)?;
        }
        if let Some(reranker) = self.reranker {
            plugins.set_reranker(reranker);
        }
        let reader = Reader::open(
            self.data,
            self.metadata,
            self.file_source,
            plugins,
            self.options,
        )
        .await?;
        Ok(Engine { reader })
    }
}
