use std::collections::BTreeSet;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use cosmosearch_core::{
    CachedResults, ExtractLimits, FileSource, Filter, IndexError, IndexId, IndexKind, Manifest,
    ManifestVersion, ModelId, Query, Scope, SearchIdentity, SegmentId, ShardId, SharedCache,
    SharedHeat, SharedPassages, SharedReranker, SharedResults, Signal, SourceRef, Status, Usage,
};
use cosmosearch_format::FileEntry;
use cosmosearch_store::{DataStore, IndexHandle, MetadataStore, OpenState};
use tokio::time;
use tracing::warn;

use crate::boost;
use crate::cached::{CacheView, CachedFiles};
use crate::deadline;
use crate::fuse::Fusion;
use crate::passages::Passages;
use crate::read::{Access, Reach, Sources, Target};
use crate::search::{Reached, Searched};
use crate::snippet;
use crate::table_cache::{DEFAULT_TABLE_BUDGET, TableCache};
use crate::tables::{Loader, Tables};
use crate::{
    Error, Plugins, Reading, Reranking, Result, SearchHit, SearchRequest, SearchResults, Snippet,
    read, rerank, search,
};

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Reader>();
};

/// How many of a search's best fused hits a reranker rescores unless set: few enough for a
/// small local cross-encoder to score quickly.
const DEFAULT_SHORTLIST: usize = 20;

/// How a reader extracts, fuses, reranks, refreshes, caches and records heat.
///
/// It sets the limits on extracting text again, the fusion of its index kinds' lists, how
/// many hits a reranker rescores, how often it looks for a newer manifest, what it caches
/// segment files in and where it records heat.
#[derive(Debug, Clone)]
pub struct ReaderOptions {
    limits: ExtractLimits,
    fusion: Fusion,
    shortlist: usize,
    refresh_interval: Option<Duration>,
    cache: Option<SharedCache>,
    results: Option<SharedResults>,
    passages: Option<SharedPassages>,
    snippet_files: usize,
    heat: Option<SharedHeat>,
    tables: Option<TableCache>,
}

impl Default for ReaderOptions {
    fn default() -> Self {
        ReaderOptions {
            limits: ExtractLimits::default(),
            fusion: Fusion::default(),
            shortlist: DEFAULT_SHORTLIST,
            refresh_interval: None,
            cache: None,
            results: None,
            passages: None,
            snippet_files: 0,
            heat: None,
            tables: None,
        }
    }
}

impl ReaderOptions {
    /// The default limits, fusion with k 60 and a weight of 1 for every index kind, a
    /// shortlist of 20, no refresh until the host asks for one, no cache and no heat.
    pub fn new() -> Self {
        ReaderOptions::default()
    }

    /// Sets the rank constant k of reciprocal rank fusion, 60 unless set; a larger k evens out
    /// the first ranks. [`Reader::open`] refuses a k that is not finite and above zero.
    #[must_use]
    pub fn fusion_k(mut self, k: f64) -> Self {
        self.fusion.set_k(k);
        self
    }

    /// Sets the weight of the list of index kind `kind` in fusion, 1 unless set: a hit at rank
    /// r in it adds `weight / (k + r)` to its score. [`Reader::open`] refuses a weight that is
    /// not finite and above zero.
    #[must_use]
    pub fn fusion_weight(mut self, kind: IndexKind, weight: f64) -> Self {
        self.fusion.set_weight(kind, weight);
        self
    }

    /// Sets how many of a search's best fused hits a reranker rescores, whatever the number
    /// of hits asked for; zero turns reranking off.
    #[must_use]
    pub fn rerank_shortlist(mut self, size: usize) -> Self {
        self.shortlist = size;
        self
    }

    /// Sets the limits on extracting a file again at read time.
    #[must_use]
    pub fn limits(mut self, limits: ExtractLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Refreshes before a call once `interval` has passed since the last refresh or attempt.
    /// One call refreshes while others go on with the snapshot held. Nothing runs in the
    /// background.
    #[must_use]
    pub fn refresh_interval(mut self, interval: Duration) -> Self {
        self.refresh_interval = Some(interval);
        self
    }

    /// Reads segment files through `cache`, which readers of any index can share. A search's
    /// usage still counts only the reads that reach the store.
    #[must_use]
    pub fn cache(mut self, cache: SharedCache) -> Self {
        self.cache = Some(cache);
        self
    }

    /// Keeps the results of searches in `results`, which readers of any index can share, and
    /// answers a repeated search from it without reading anything. A result is kept under
    /// the manifest version searched and everything that decides it: the query as given, the
    /// scope, the filter, the number of hits, the index kinds with each plug-in's model and
    /// search settings, fusion, the reranker with its shortlist, and the extraction limits
    /// when snippets or reranking read passages. Results with a shard left unread or a
    /// reranker that failed are not kept.
    #[must_use]
    pub fn result_cache(mut self, results: SharedResults) -> Self {
        self.results = Some(results);
        self
    }

    /// Gives each hit a snippet: its passage, read back from its file, with the query's words
    /// marked. One search reads at most `max_files` files for snippets, those of its best
    /// hits first, and the hits of other files have none; zero, the default, gives no
    /// snippets and reads nothing. In an index with stored text, a hit whose chunk has text
    /// takes it from the index instead, without counting toward `max_files`.
    #[must_use]
    pub fn snippets(mut self, max_files: usize) -> Self {
        self.snippet_files = max_files;
        self
    }

    /// Keeps passage text read for snippets and reranking in `passages`, which readers of
    /// any index can share, so a passage a search needs again is not read from its file
    /// again.
    #[must_use]
    pub fn passage_cache(mut self, passages: SharedPassages) -> Self {
        self.passages = Some(passages);
        self
    }

    /// Records heat in `heat`: each search counts [`Signal::Search`] for the shards it
    /// covered, those holding a file in its scope that passes its filter or left unread, and
    /// [`Signal::Hit`] for the shards of the passages it returns, and each read or expand
    /// counts [`Signal::Read`] for the shard of its ref. Only calls that succeed count.
    #[must_use]
    pub fn heat(mut self, heat: SharedHeat) -> Self {
        self.heat = Some(heat);
        self
    }

    /// Keeps the file tables and chunk records this reader decodes for hits, scopes and
    /// filters in `tables`, which readers of any index can share under one budget. Unless
    /// set, each reader has its own cache of [`DEFAULT_TABLE_BUDGET`] bytes.
    #[must_use]
    pub fn tables(mut self, tables: TableCache) -> Self {
        self.tables = Some(tables);
        self
    }
}

#[derive(Debug)]
struct State {
    manifest: Arc<Manifest>,
    files: Arc<CachedFiles>,
    refreshed_at: Instant,
    gone: Option<Gone>,
}

/// How the index a reader opened was found gone. It never comes back, so every later call
/// fails the same way.
#[derive(Debug, Clone)]
enum Gone {
    Removed,
    Replaced(IndexId),
}

/// An open index, searched and read through the manifest snapshot it holds until refreshed.
#[derive(Debug)]
pub struct Reader {
    data: DataStore,
    metadata: MetadataStore,
    index: IndexHandle,
    plugins: Plugins,
    options: ReaderOptions,
    state: Mutex<State>,
    tables: Tables,
}

impl Reader {
    /// Opens the index at `metadata` over the files at `data`, reading only `INDEX` and the
    /// manifest.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidFusion`] for a fusion setting that is not finite and above zero,
    /// before reading anything, [`Error::NotIndexed`] or [`Error::Incomplete`] if there is no
    /// usable index, [`Error::MissingPlugins`] naming every plug-in the index needs that
    /// `plugins` lacks, and [`Error::Store`] if the store fails or the index was created
    /// differently.
    pub async fn open(
        data: DataStore,
        metadata: MetadataStore,
        file_source: FileSource,
        plugins: Plugins,
        options: ReaderOptions,
    ) -> Result<Reader> {
        options.fusion.check()?;
        let state = metadata.open(data.location(), file_source).await?;
        if matches!(state, OpenState::NotIndexed) {
            return Err(Error::NotIndexed);
        }
        let OpenState::Ready(ready) = state else {
            return Err(Error::Incomplete);
        };
        let (index, manifest) = ready.into_parts();
        check_plugins(&plugins, &manifest)?;
        let files = cached_files(&options, index.id(), &manifest);
        let shared = options
            .tables
            .clone()
            .unwrap_or_else(|| TableCache::new(DEFAULT_TABLE_BUDGET));
        let tables = Tables::new(index.id().clone(), shared);
        Ok(Reader {
            data,
            metadata,
            index,
            plugins,
            options,
            state: Mutex::new(State {
                manifest: Arc::new(manifest),
                files,
                refreshed_at: Instant::now(),
                gone: None,
            }),
            tables,
        })
    }

    /// About how many bytes of decoded tables this reader holds: what it keeps for every
    /// search, and the tables of its index in its table cache, which readers of the same
    /// index sharing the cache share.
    pub fn resident_bytes(&self) -> u64 {
        self.tables.resident_bytes()
    }

    /// The manifest this reader searches.
    pub fn manifest(&self) -> Arc<Manifest> {
        Arc::clone(&self.state().manifest)
    }

    /// The id of the index this reader searches, under which a cache keeps its files.
    pub fn index_id(&self) -> &IndexId {
        self.index.id()
    }

    /// The plug-ins this reader has.
    pub fn plugins(&self) -> &Plugins {
        &self.plugins
    }

    /// Re-reads `INDEX` and moves to the newest manifest, returning its version.
    ///
    /// # Errors
    ///
    /// Returns [`Error::IndexRemoved`] or [`Error::IndexReplaced`] if another process
    /// destroyed the index, and so does every later call of this reader,
    /// [`Error::Incomplete`] if its manifest no longer resolves,
    /// [`Error::MissingPlugins`] if the newest manifest needs a plug-in this
    /// reader lacks, in which case it keeps its snapshot, and [`Error::Store`] if the store fails.
    pub async fn refresh(&self) -> Result<ManifestVersion> {
        self.check_index().await?;
        let head = self.index.head().await?.ok_or(Error::Incomplete)?;
        check_plugins(&self.plugins, &head)?;
        let files = cached_files(&self.options, self.index.id(), &head);
        let mut state = self.state();
        if head.version() >= state.manifest.version() {
            state.manifest = Arc::new(head);
            state.files = files;
        }
        state.refreshed_at = Instant::now();
        let current = Arc::clone(&state.manifest);
        drop(state);
        self.tables.retain(&current);
        Ok(current.version())
    }

    /// Finds the best passages for `request` among the files in its scope that pass its
    /// filter, up to its limit. Shards whose key ranges hold no key of the scope are not
    /// opened, and an index plug-in is given only segments with chunks the search may return.
    ///
    /// With a reranker, the best fused hits up to the shortlist are read, once per file, and
    /// rescored, then put first in the reranker's order; a hit whose passage cannot be read
    /// follows them unscored. A reranker that fails leaves the hits in fused order, and
    /// [`SearchResults::reranking`] says so.
    ///
    /// A segment found missing triggers one refresh and a retry. If it is still missing, its
    /// in-scope shards are left unread and reported in the results, and the search goes on
    /// without them.
    ///
    /// # Errors
    ///
    /// Returns [`Error::LimitTooLarge`] for a request asking for more than
    /// [`SearchRequest::MAX_LIMIT`] hits, [`Error::ScopeOutsideData`] for a scope prefix no
    /// file of the data location can have, [`Error::IndexRemoved`] or [`Error::IndexReplaced`]
    /// if a refresh finds the index gone, and an error if the store, a stored table or an
    /// index plug-in fails.
    pub async fn search(&self, request: &SearchRequest) -> Result<SearchResults> {
        self.check(request)?;
        let deadline = request
            .budget
            .time()
            .and_then(|limit| time::Instant::now().checked_add(limit));
        let (mut manifest, cache) = self.snapshot_for_call(deadline).await?;
        let identity = self.identity(request);
        let SearchRequest {
            query,
            scope,
            limit,
            budget,
            ..
        } = request;
        let limit = *limit;
        if let Some(found) = self.cached(identity.as_ref(), &manifest) {
            self.record_search(&manifest, &found);
            return Ok(found);
        }
        let mut loader = Loader::new(&self.index, &self.tables, cache);
        let fused = if self.reranker(limit).is_some() {
            limit.max(self.options.shortlist)
        } else {
            limit
        };
        let mut reached = Reached::default();
        reached.select(&manifest, scope, budget.shards());
        let mut refreshed = false;
        let mut index_checked = false;
        let mut spent = Usage::new();
        loop {
            let attempt = deadline::within(
                deadline,
                search::run(
                    &loader,
                    &self.plugins,
                    &manifest,
                    &reached.open,
                    request,
                    fused,
                    &self.options.fusion,
                    &mut spent,
                ),
            )
            .await;
            let Some(attempt) = attempt else {
                return Ok(self.out_of_time(&manifest, &loader, spent, reached));
            };
            let segment = match attempt {
                Ok(searched) => {
                    let mut passages = Passages::new(
                        self.sources(&loader),
                        &manifest,
                        self.index.id(),
                        self.options.passages.as_ref(),
                    );
                    let (hits, reranking, tokens, completed) = self
                        .rank(&mut passages, query, searched.hits, limit, deadline)
                        .await?;
                    let usage = loader
                        .counter()
                        .usage()
                        .saturating_add(spent)
                        .saturating_add(tokens);
                    let version = manifest.version();
                    let searched = Searched { hits, ..searched };
                    let found = SearchResults::new(version, searched, reached, reranking, usage);
                    return Ok(self.finish(&manifest, identity.as_ref(), found, !completed));
                }
                Err(Error::Index(IndexError::MissingSegment { segment })) => segment,
                Err(err) => return Err(err),
            };
            if !refreshed {
                refreshed = true;
                let refresh = deadline::within(deadline, self.refresh()).await;
                if refresh.transpose()?.is_none() {
                    return Ok(self.out_of_time(&manifest, &loader, spent, reached));
                }
                let cache;
                (manifest, cache) = self.snapshot();
                loader = loader.rebind(cache);
                reached.select(&manifest, scope, budget.shards());
                continue;
            }
            if !index_checked {
                index_checked = true;
                let check = deadline::within(deadline, self.check_index()).await;
                if check.transpose()?.is_none() {
                    return Ok(self.out_of_time(&manifest, &loader, spent, reached));
                }
            }
            let holders = holders(&manifest, &segment, &reached.open);
            if holders.is_empty() {
                return Err(Error::Index(IndexError::MissingSegment { segment }));
            }
            warn!(segment = %segment, shards = holders.len(), "a segment is missing, so its shards need a rebuild");
            reached.leave_unread(holders);
        }
    }

    /// The results of a search whose time ran out before it found anything.
    fn out_of_time(
        &self,
        manifest: &Manifest,
        loader: &Loader<'_>,
        spent: Usage,
        mut reached: Reached,
    ) -> SearchResults {
        reached.abandon();
        let usage = loader.counter().usage().saturating_add(spent);
        let nothing = Searched::default();
        let found = SearchResults::new(manifest.version(), nothing, reached, Reranking::Off, usage);
        self.finish(manifest, None, found, true)
    }

    /// The reranker a search for `limit` hits uses, if any.
    fn reranker(&self, limit: usize) -> Option<&SharedReranker> {
        self.plugins
            .shared_reranker()
            .filter(|_| limit > 0 && self.options.shortlist > 0)
    }

    /// Reranks `hits`, places the reranked ones again by their boosts, keeps the best `limit`
    /// and gives them snippets, by `deadline`. Returns the hits, how reranking went, the tokens
    /// it read and whether all of it was done in time.
    async fn rank(
        &self,
        passages: &mut Passages<'_>,
        query: &Query,
        hits: Vec<SearchHit>,
        limit: usize,
        deadline: Option<time::Instant>,
    ) -> Result<(Vec<SearchHit>, Reranking, Usage, bool)> {
        let shortlist = self.options.shortlist;
        let (mut hits, reranking, tokens) = match self.reranker(limit) {
            Some(reranker) => {
                rerank::run(passages, reranker, query, hits, shortlist, deadline).await?
            }
            None => (hits, Reranking::Off, Usage::new()),
        };
        let reranked = hits
            .iter()
            .take_while(|hit| hit.rerank_score().is_some())
            .count();
        if reranked > 1 {
            let rest = hits.split_off(reranked);
            hits = boost::place(hits, SearchHit::boost);
            hits.extend(rest);
        }
        hits.truncate(limit);
        let snippets = self.add_snippets(passages, query, &mut hits);
        let snipped = deadline::within(deadline, snippets).await.is_some();
        let completed = snipped && !matches!(reranking, Reranking::OutOfTime);
        Ok((hits, reranking, tokens, completed))
    }

    /// Records the heat of `found`, marked as cut short by time if `out_of_time`, and keeps it
    /// in the result cache under `identity` if it may be kept.
    fn finish(
        &self,
        manifest: &Manifest,
        identity: Option<&SearchIdentity>,
        found: SearchResults,
        out_of_time: bool,
    ) -> SearchResults {
        let found = if out_of_time {
            found.ran_out_of_time()
        } else {
            found
        };
        self.record_search(manifest, &found);
        self.keep(identity, manifest, &found);
        found
    }

    /// Reads the passage `reference` names, from the file version it was made from.
    ///
    /// # Errors
    ///
    /// As for [`Reader::expand`].
    pub async fn read(&self, reference: &SourceRef) -> Result<Reading> {
        self.expand(reference, 0, 0).await
    }

    /// Reads the passage `reference` names together with up to `before` and `after`
    /// neighbouring chunks of the same file.
    ///
    /// The file is extracted again with the extractor that indexed it, and every byte read is
    /// pinned to the ref's ETag. Any ref to an indexed passage inside the data location is
    /// read, whatever scope or filter the search that returned it had, so a host that takes
    /// refs from users checks their access to the ref's file itself, or calls
    /// [`Reader::expand_within`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutsideData`] for a ref outside the data location,
    /// [`Error::NotInIndex`] for a file the index does not hold, [`Error::StaleRef`] if the
    /// file has changed, [`Error::NoPassage`] if no indexed passage is at the ref's locator,
    /// [`Error::ShardUnread`] if the shard's segments stay missing after a refresh, and an
    /// error if the store or the extractor fails.
    pub async fn expand(
        &self,
        reference: &SourceRef,
        before: usize,
        after: usize,
    ) -> Result<Reading> {
        let reach = Reach { before, after };
        self.read_target(Target::Passage { reference, reach }, None)
            .await
    }

    /// Reads the passage `reference` names, as [`Reader::read`] does, only if its file is in
    /// `scope` and passes `filter`, as a search with them would require.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotInIndex`] for a file outside `scope` or one that does not pass
    /// `filter`, as for a file the index does not hold, so the error tells nothing about such
    /// a file; otherwise as for [`Reader::expand`].
    pub async fn read_within(
        &self,
        reference: &SourceRef,
        scope: &Scope,
        filter: &Filter,
    ) -> Result<Reading> {
        self.expand_within(reference, 0, 0, scope, filter).await
    }

    /// Reads the passage `reference` names with its neighbours, as [`Reader::expand`] does,
    /// only if its file is in `scope` and passes `filter`.
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
        let access = Access { scope, filter };
        let reach = Reach { before, after };
        self.read_target(Target::Passage { reference, reach }, Some(access))
            .await
    }

    /// The file with this key as the index holds it in this snapshot: its status and why it
    /// is not ready, its display name, media type, modified time and attributes. Like
    /// [`Reader::read`], it checks no scope or filter; [`Reader::file_within`] does.
    ///
    /// # Errors
    ///
    /// Returns [`Error::FileOutsideData`] for a key that is not a whole object key inside the
    /// data location, [`Error::ShardUnread`] if the file's shard stays unreadable after a
    /// refresh, and an error if the file table cannot be read.
    pub async fn file(&self, key: &str) -> Result<Option<FileEntry>> {
        self.file_entry(key, None).await
    }

    /// The file with this key, as [`Reader::file`] gives it, only if it is in `scope` and
    /// passes `filter`; `None` otherwise, as for a file the index does not hold.
    ///
    /// # Errors
    ///
    /// As for [`Reader::file`].
    pub async fn file_within(
        &self,
        key: &str,
        scope: &Scope,
        filter: &Filter,
    ) -> Result<Option<FileEntry>> {
        self.file_entry(key, Some(Access { scope, filter })).await
    }

    async fn file_entry(&self, key: &str, access: Option<Access<'_>>) -> Result<Option<FileEntry>> {
        read::file_inside(&self.data, key)?;
        if access.is_some_and(|access| !access.scope.contains(key)) {
            return Ok(None);
        }
        let (mut manifest, cache) = self.snapshot_for_call(None).await?;
        let mut loader = Loader::new(&self.index, &self.tables, cache);
        let mut refreshed = false;
        loop {
            let Some((shard_id, shard)) = manifest.shard_for(key) else {
                return Ok(None);
            };
            match loader.file(shard, key).await {
                Ok(entry) => {
                    let entry = entry.filter(|entry| {
                        access.is_none_or(|access| {
                            access.filter.matches(entry.attributes(), entry.modified())
                        })
                    });
                    return Ok(entry);
                }
                Err(Error::Index(IndexError::MissingSegment { segment })) if !refreshed => {
                    warn!(segment = %segment, "a segment is missing, so the reader refreshes");
                    refreshed = true;
                    self.refresh().await?;
                    let cache;
                    (manifest, cache) = self.snapshot();
                    loader = loader.rebind(cache);
                }
                Err(Error::Index(IndexError::MissingSegment { .. })) => {
                    self.check_index().await?;
                    return Err(Error::ShardUnread {
                        shard: shard_id.clone(),
                    });
                }
                Err(err) => return Err(err),
            }
        }
    }

    /// Reads up to `limit` chunks of the file `key` in reading order, starting at its chunk
    /// at position `first`, counting from 0, so a host can read a whole file page by page
    /// without a search hit. Past the file's last chunk, the reading holds no chunks, and so
    /// does a reading of a file with no chunks indexed yet, which [`Reader::file`] tells apart
    /// by its status. Each call reads the file again: the bytes of its chunks for plain text,
    /// and the whole file for formats that must be parsed, such as PDF, so fewer, larger
    /// pages cost less. Like [`Reader::read`], it checks no scope or filter;
    /// [`Reader::chunks_within`] does.
    ///
    /// # Errors
    ///
    /// Returns [`Error::FileOutsideData`] for a key outside the data location,
    /// [`Error::FileNotInIndex`] for a file the index does not hold, [`Error::StaleRef`] if
    /// the file has changed since it was indexed, [`Error::ShardUnread`] if the shard's
    /// segments stay missing after a refresh, and an error if the store or the extractor
    /// fails.
    pub async fn chunks(&self, key: &str, first: usize, limit: usize) -> Result<Reading> {
        self.read_target(Target::File { key, first, limit }, None)
            .await
    }

    /// Reads chunks of the file `key`, as [`Reader::chunks`] does, only if the file is in
    /// `scope` and passes `filter`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::FileNotInIndex`] for a file outside `scope` or one that does not pass
    /// `filter`, as for a file the index does not hold; otherwise as for [`Reader::chunks`].
    pub async fn chunks_within(
        &self,
        key: &str,
        first: usize,
        limit: usize,
        scope: &Scope,
        filter: &Filter,
    ) -> Result<Reading> {
        let access = Access { scope, filter };
        self.read_target(Target::File { key, first, limit }, Some(access))
            .await
    }

    async fn read_target(&self, target: Target<'_>, access: Option<Access<'_>>) -> Result<Reading> {
        let (mut manifest, cache) = self.snapshot_for_call(None).await?;
        let mut loader = Loader::new(&self.index, &self.tables, cache);
        let mut refreshed = false;
        loop {
            let sources = self.sources(&loader);
            match read::run(sources, &manifest, target, access).await {
                Ok(chunks) => {
                    if let Some((shard, _)) = manifest.shard_for(target.key()) {
                        self.record(Signal::Read, &[shard]);
                    }
                    return Ok(Reading::new(chunks, loader.counter().usage()));
                }
                Err(Error::Index(IndexError::MissingSegment { segment })) if !refreshed => {
                    warn!(segment = %segment, "a segment is missing, so the reader refreshes");
                    refreshed = true;
                    self.refresh().await?;
                    let cache;
                    (manifest, cache) = self.snapshot();
                    loader = loader.rebind(cache);
                }
                Err(Error::Index(IndexError::MissingSegment { .. })) => {
                    self.check_index().await?;
                    let shard = manifest.shard_for(target.key()).map(|(id, _)| id.clone());
                    return Err(match shard {
                        Some(shard) => Error::ShardUnread { shard },
                        None => target.not_in_index(),
                    });
                }
                Err(err) => return Err(err),
            }
        }
    }

    /// Counts a search for the shards it covered or left unread, and a hit for the shards of
    /// the passages it returns, whether it was searched or answered by the result cache.
    fn record_search(&self, manifest: &Manifest, found: &SearchResults) {
        let coverage = found.coverage();
        let mut covered: Vec<&ShardId> = found.covered().iter().chain(coverage.unread()).collect();
        covered.sort_unstable();
        covered.dedup();
        self.record(Signal::Search, &covered);
        if found.status() == Status::Inconclusive {
            let unexplored: Vec<&ShardId> = coverage.unexplored().iter().collect();
            self.record(Signal::Unmet, &unexplored);
        }
        let mut hit: Vec<&ShardId> = found
            .hits()
            .iter()
            .filter_map(|hit| manifest.shard_for(hit.source().location().key()))
            .map(|(shard, _)| shard)
            .collect();
        hit.sort_unstable();
        hit.dedup();
        self.record(Signal::Hit, &hit);
    }

    /// Gives each of `hits` whose passage can be had its snippet, reading at most the set
    /// number of files.
    async fn add_snippets(
        &self,
        passages: &mut Passages<'_>,
        query: &Query,
        hits: &mut [SearchHit],
    ) {
        if self.options.snippet_files == 0 {
            return;
        }
        let views: Vec<&SearchHit> = hits.iter().collect();
        let texts = passages.texts(&views, self.options.snippet_files).await;
        for (hit, text) in hits.iter_mut().zip(texts) {
            if let Some(text) = text {
                let marked = snippet::matches(self.plugins.indexes(), query, &text);
                hit.set_snippet(Snippet::new(text, marked));
            }
        }
    }

    /// The results a result cache holds for `identity` in `manifest`'s version.
    fn cached(
        &self,
        identity: Option<&SearchIdentity>,
        manifest: &Manifest,
    ) -> Option<SearchResults> {
        let key = identity?.key(self.index.id(), manifest.version());
        let found = self.options.results.as_ref()?.get().get(&key)?;
        found.get::<SearchResults>().map(SearchResults::to_cached)
    }

    /// Keeps `found` in the result cache, unless a later search could do better: one with a
    /// shard left unread, a reranker that failed or a time limit that cut it short.
    fn keep(&self, identity: Option<&SearchIdentity>, manifest: &Manifest, found: &SearchResults) {
        let (Some(identity), Some(results)) = (identity, &self.options.results) else {
            return;
        };
        if !found.coverage().unread().is_empty()
            || found.out_of_time()
            || matches!(found.reranking(), Reranking::Failed(_))
        {
            return;
        }
        results.get().put(
            identity.key(self.index.id(), manifest.version()),
            CachedResults::new(found.clone()),
            found.weight(),
        );
    }

    /// Everything that decides a search's results in one snapshot of this reader's index, if
    /// there is a result cache to key them in.
    fn identity(&self, request: &SearchRequest) -> Option<SearchIdentity> {
        self.options.results.as_ref()?;
        let limit = request.limit;
        let mut kinds: Vec<&IndexKind> = self.plugins.index_kinds().collect();
        kinds.sort_unstable();
        let mut identity = SearchIdentity::new()
            .text(request.query.text())
            .number(u64::from(request.query.exact_phrases()))
            .number(u64::from(request.query.every_word()))
            .scope(&request.scope)
            .filter(&request.filter)
            .number(request.budget.shards().map_or(u64::MAX, count))
            .number(count(limit))
            .number(count(self.options.snippet_files))
            .number(self.options.fusion.k().to_bits())
            .number(count(request.boosts.len()));
        for boost in &request.boosts {
            identity = boost.add_to(identity);
        }
        identity = identity.number(count(kinds.len()));
        for kind in kinds {
            identity = identity
                .text(kind.as_str())
                .number(self.options.fusion.weight(kind).to_bits());
            if let Some(index) = self.plugins.index(kind) {
                let index = index.get();
                let settings = index.search_settings();
                identity = identity
                    .text(index.model_id().map_or("", ModelId::as_str))
                    .number(count(settings.len()));
                for setting in settings {
                    identity = identity.number(setting);
                }
            }
        }
        let reranker = self
            .plugins
            .reranker()
            .filter(|_| limit > 0 && self.options.shortlist > 0);
        identity = match reranker {
            Some(reranker) => identity
                .number(1)
                .text(reranker.model())
                .text(reranker.revision())
                .number(u64::from(reranker.max_tokens().get()))
                .number(count(self.options.shortlist)),
            None => identity.number(0),
        };
        if reranker.is_none() && self.options.snippet_files == 0 {
            return Some(identity);
        }
        let limits = self.options.limits;
        Some(
            identity
                .number(count(limits.pages()))
                .number(count(limits.chars()))
                .number(count(limits.chunks()))
                .number(limits.file_bytes())
                .number(count(limits.archive_entries()))
                .number(limits.archive_bytes())
                .number(u64::from(limits.archive_ratio()))
                .number(limits.decoded_bytes())
                .number(count(limits.depth())),
        )
    }

    fn check(&self, request: &SearchRequest) -> Result<()> {
        request.check()?;
        match request.scope.outside(self.data.location().prefix_key()) {
            Some(prefix) => Err(Error::ScopeOutsideData {
                prefix: prefix.to_owned(),
            }),
            None => Ok(()),
        }
    }

    fn record(&self, signal: Signal, shards: &[&ShardId]) {
        if let Some(heat) = &self.options.heat
            && !shards.is_empty()
        {
            heat.get().record(signal, shards);
        }
    }

    fn sources<'a>(&'a self, loader: &'a Loader<'a>) -> Sources<'a> {
        Sources {
            loader,
            plugins: &self.plugins,
            data: &self.data,
            limits: self.options.limits,
        }
    }

    #[expect(
        clippy::expect_used,
        reason = "a poisoned lock means an earlier panic, and its state is not trusted"
    )]
    fn state(&self) -> MutexGuard<'_, State> {
        self.state
            .lock()
            .expect("no thread panicked while holding the lock")
    }

    /// The snapshot for a call, refreshed first if the interval has lapsed. Only a destroyed or
    /// unresolvable index fails the call; on any other failure, or if the refresh is not done
    /// by `deadline`, the call goes on with the snapshot held, and the next attempt waits
    /// another interval.
    async fn snapshot_for_call(
        &self,
        deadline: Option<time::Instant>,
    ) -> Result<(Arc<Manifest>, Option<CacheView>)> {
        self.check_gone()?;
        if self.claim_refresh() {
            match deadline::within(deadline, self.refresh()).await {
                None => {
                    warn!(
                        "a refresh on the interval ran out of time, so the call uses the snapshot held"
                    );
                }
                Some(Ok(_)) => {}
                Some(Err(
                    err @ (Error::IndexRemoved | Error::IndexReplaced { .. } | Error::Incomplete),
                )) => {
                    return Err(err);
                }
                Some(Err(err)) => {
                    warn!(error = %err, "a refresh on the interval failed, so the call uses the snapshot held");
                }
            }
        }
        Ok(self.snapshot())
    }

    fn claim_refresh(&self) -> bool {
        let Some(interval) = self.options.refresh_interval else {
            return false;
        };
        claim(&mut self.state().refreshed_at, Instant::now(), interval)
    }

    /// The manifest, and the cache with that manifest's files if there is a cache.
    fn snapshot(&self) -> (Arc<Manifest>, Option<CacheView>) {
        let state = self.state();
        let cache = self.options.cache.as_ref().map(|cache| CacheView {
            cache: cache.clone(),
            files: Arc::clone(&state.files),
        });
        (Arc::clone(&state.manifest), cache)
    }

    /// Fails if this reader has found its index gone.
    fn check_gone(&self) -> Result<()> {
        match &self.state().gone {
            None => Ok(()),
            Some(gone) => Err(self.gone_error(gone.clone())),
        }
    }

    fn gone_error(&self, gone: Gone) -> Error {
        match gone {
            Gone::Removed => Error::IndexRemoved,
            Gone::Replaced(found) => Error::IndexReplaced {
                opened: self.index.id().clone(),
                found,
            },
        }
    }

    /// Checks that `INDEX` still names this reader's index, and remembers it if not.
    async fn check_index(&self) -> Result<()> {
        let gone = match self.metadata.current_index().await? {
            Some(found) if &found == self.index.id() => return Ok(()),
            Some(found) => Gone::Replaced(found),
            None => Gone::Removed,
        };
        self.state().gone = Some(gone.clone());
        Err(self.gone_error(gone))
    }
}

/// Whether `interval` has passed since `last`, moving `last` to `now` if so, so that one
/// caller in each interval refreshes.
fn claim(last: &mut Instant, now: Instant, interval: Duration) -> bool {
    if now.saturating_duration_since(*last) < interval {
        return false;
    }
    *last = now;
    true
}

fn count(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

/// The files of `manifest` as a cache keys them, or none without a cache.
fn cached_files(options: &ReaderOptions, index: &IndexId, manifest: &Manifest) -> Arc<CachedFiles> {
    match options.cache {
        Some(_) => Arc::new(CachedFiles::of(index, manifest)),
        None => Arc::new(CachedFiles::default()),
    }
}

fn check_plugins(plugins: &Plugins, manifest: &Manifest) -> Result<()> {
    let missing = plugins.missing(manifest);
    if missing.is_empty() {
        Ok(())
    } else {
        Err(Error::MissingPlugins(missing))
    }
}

/// The shards of `manifest` in `open` that reference `segment`.
fn holders(manifest: &Manifest, segment: &SegmentId, open: &BTreeSet<ShardId>) -> Vec<ShardId> {
    manifest
        .shards()
        .iter()
        .filter(|(id, shard)| {
            open.contains(*id) && shard.segments().iter().any(|s| s.id() == segment)
        })
        .map(|(id, _)| id.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_caller_claims_each_lapsed_interval() {
        let interval = Duration::from_secs(60);
        let start = Instant::now();
        let mut last = start;
        assert!(!claim(&mut last, start + interval / 2, interval));
        let lapsed = start + interval;
        let claimed: Vec<bool> = (0..4).map(|_| claim(&mut last, lapsed, interval)).collect();
        assert_eq!(claimed, [true, false, false, false]);
        assert!(!claim(&mut last, lapsed + interval / 2, interval));
        assert!(claim(&mut last, lapsed + interval, interval));
    }
}
