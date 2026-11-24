use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::num::NonZeroU32;
use std::sync::Arc;

use cosmosearch_core::{
    AllowedRows, ExtractorProfile, Filter, Hit, IndexError, IndexKind, Level, Locator, Manifest,
    ManifestVersion, Scope, SegmentEntry, SegmentId, ShardEntry, ShardId, SourceRef, Status, Usage,
};
use cosmosearch_format::{ChunkTable, FileEntry, FileTable};
use futures_util::{StreamExt, TryStreamExt, future, stream};

use crate::coverage::Counts;
use crate::fuse::{self, Fusion};
use crate::tables::{self, Loader};
use crate::{Coverage, Plugins, Reranking, Result, SearchRequest, Snippet};

const LOAD_CONCURRENCY: usize = 8;

/// Hits each index kind gives fusion at the least, so a passage another kind ranks in its
/// top 50 is still found by this one, and a reranker's shortlist is drawn from lists that
/// reach well past it.
const FUSION_DEPTH: usize = 50;

/// What a hit takes in memory besides its ref and headings, and results besides their hits,
/// roughly: enough for a cache's budget to follow what it holds.
const HIT_BYTES: usize = 128;
const RESULTS_BYTES: usize = 256;
const LIST_SCORE_BYTES: usize = 32;
const SHARD_ID_BYTES: usize = 48;

/// One passage a search found.
#[derive(Clone, PartialEq)]
pub struct SearchHit {
    source: SourceRef,
    score: f32,
    list_scores: Vec<(IndexKind, f32)>,
    list_ranks: Vec<(IndexKind, u32)>,
    rerank_score: Option<f32>,
    heading_path: Vec<String>,
    display_name: Option<String>,
    snippet: Option<Snippet>,
}

impl SearchHit {
    /// Where the passage is.
    pub fn source(&self) -> &SourceRef {
        &self.source
    }

    /// The reciprocal rank fusion score over the lists of the index kinds searched, higher
    /// being better. It orders hits but means nothing on its own.
    pub fn score(&self) -> f32 {
        self.score
    }

    /// The score each index kind's list gave the passage, for the kinds whose list held it,
    /// in the order of the kinds. Scores of different kinds are not comparable: BM25 gives an
    /// unbounded score, a vector index a similarity.
    pub fn list_scores(&self) -> &[(IndexKind, f32)] {
        &self.list_scores
    }

    /// The rank each index kind's list gave the passage, counting from 1, for the kinds whose
    /// list held it, in the order of the kinds: with [`SearchHit::list_scores`], why fusion
    /// placed the hit where it is.
    pub fn list_ranks(&self) -> &[(IndexKind, u32)] {
        &self.list_ranks
    }

    /// The name the host gave the passage's file, if any.
    pub fn display_name(&self) -> Option<&str> {
        self.display_name.as_deref()
    }

    /// The reranker's score, for a hit in the shortlist whose passage could be read. Reranked
    /// hits come first, ordered by it, higher being better.
    pub fn rerank_score(&self) -> Option<f32> {
        self.rerank_score
    }

    pub(crate) fn set_rerank_score(&mut self, score: f32) {
        self.rerank_score = Some(score);
    }

    /// The page the passage is on, for formats that have pages.
    pub fn page(&self) -> Option<NonZeroU32> {
        if let Locator::Page(page) = self.source.locator() {
            Some(page)
        } else {
            None
        }
    }

    /// The passage with the query's words marked, if the reader gives snippets and this hit's
    /// file was among those it read for them.
    pub fn snippet(&self) -> Option<&Snippet> {
        self.snippet.as_ref()
    }

    pub(crate) fn set_snippet(&mut self, snippet: Snippet) {
        self.snippet = Some(snippet);
    }

    /// The headings above the passage, outermost first.
    pub fn heading_path(&self) -> &[String] {
        &self.heading_path
    }
}

impl fmt::Debug for SearchHit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SearchHit")
            .field("source", &self.source)
            .field("score", &self.score)
            .field("list_scores", &self.list_scores)
            .field("list_ranks", &self.list_ranks)
            .field("rerank_score", &self.rerank_score)
            .field("headings", &self.heading_path.len())
            .field(
                "display_name_bytes",
                &self.display_name.as_ref().map(String::len),
            )
            .field("snippet", &self.snippet)
            .finish()
    }
}

/// What a search found, in which snapshot, how it ended, what it looked at and how reranking
/// went.
#[derive(Debug, Clone)]
pub struct SearchResults {
    version: ManifestVersion,
    hits: Vec<SearchHit>,
    status: Status,
    covered: Vec<ShardId>,
    coverage: Coverage,
    reranking: Reranking,
    usage: Usage,
    cached: bool,
    out_of_time: bool,
}

impl SearchResults {
    pub(crate) fn new(
        version: ManifestVersion,
        searched: Searched,
        reached: Reached,
        reranking: Reranking,
        usage: Usage,
    ) -> Self {
        let Searched {
            hits,
            covered,
            counts,
        } = searched;
        let status = if hits.is_empty() {
            Status::Inconclusive
        } else {
            Status::Found
        };
        SearchResults {
            version,
            hits,
            status,
            covered,
            coverage: Coverage::new(counts, reached.unread, reached.unexplored),
            reranking,
            usage,
            cached: false,
            out_of_time: false,
        }
    }

    /// These results, marked as cut short by the search's time limit.
    pub(crate) fn ran_out_of_time(self) -> Self {
        SearchResults {
            out_of_time: true,
            ..self
        }
    }

    /// These results as a result cache gives them back: nothing was read for them.
    pub(crate) fn to_cached(&self) -> Self {
        SearchResults {
            usage: Usage::new(),
            cached: true,
            ..self.clone()
        }
    }

    /// About how many bytes the results hold, for a cache's budget.
    pub(crate) fn weight(&self) -> u64 {
        let hits: usize = self
            .hits
            .iter()
            .map(|hit| {
                let headings: usize = hit.heading_path.iter().map(String::len).sum();
                let snippet = hit.snippet.as_ref().map_or(0, Snippet::weight);
                let lists: usize = hit
                    .list_scores
                    .iter()
                    .map(|(kind, _)| kind.as_str().len().saturating_add(LIST_SCORE_BYTES))
                    .sum();
                let ranks: usize = hit
                    .list_ranks
                    .iter()
                    .map(|(kind, _)| kind.as_str().len().saturating_add(LIST_SCORE_BYTES))
                    .sum();
                let name = hit.display_name.as_ref().map_or(0, String::len);
                HIT_BYTES
                    .saturating_add(snippet)
                    .saturating_add(lists)
                    .saturating_add(ranks)
                    .saturating_add(name)
                    .saturating_add(hit.source.to_string().len())
                    .saturating_add(headings)
            })
            .sum();
        let coverage = self.coverage();
        let shards: usize = self
            .covered
            .iter()
            .chain(coverage.unread())
            .chain(coverage.unexplored())
            .map(|id| SHARD_ID_BYTES.saturating_add(id.as_str().len()))
            .sum();
        let total = hits.saturating_add(shards).saturating_add(RESULTS_BYTES);
        u64::try_from(total).unwrap_or(u64::MAX)
    }

    /// The manifest version searched.
    pub fn version(&self) -> ManifestVersion {
        self.version
    }

    /// The passages found, best first.
    pub fn hits(&self) -> &[SearchHit] {
        &self.hits
    }

    /// The shards the search covered: those holding a file in its scope that passes its
    /// filter.
    pub(crate) fn covered(&self) -> &[ShardId] {
        &self.covered
    }

    /// How the search ended: [`Status::Found`] with passages, [`Status::Inconclusive`]
    /// without. It never ends absent; [`SearchResults::coverage`] says what it looked at.
    pub fn status(&self) -> Status {
        self.status
    }

    /// The files the search searched and those it could not, and the shards of its scope it
    /// left unread or unexplored.
    pub fn coverage(&self) -> &Coverage {
        &self.coverage
    }

    /// Whether the search's time limit cut it short: before any hit was found, in which case
    /// it has none, or while reranking or reading snippets.
    pub fn out_of_time(&self) -> bool {
        self.out_of_time
    }

    /// How reranking went. When it failed, the hits are in fused order and may be less
    /// relevant than reranked ones.
    pub fn reranking(&self) -> &Reranking {
        &self.reranking
    }

    /// What the search read from storage, and the tokens a reranker read; nothing for results
    /// that came from a result cache.
    pub fn usage(&self) -> Usage {
        self.usage
    }

    /// Whether a result cache answered, so nothing was searched or read for these results.
    pub fn cached(&self) -> bool {
        self.cached
    }
}

/// A chunk the same whichever segment holds it: its file at one content version, the
/// extractor that read it and its position among that file's chunks. Lists of different index
/// kinds hold the same passage in different segments and rows, so they are fused on this.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ChunkKey {
    key: String,
    content_version: ManifestVersion,
    profile: ExtractorProfile,
    position: usize,
}

/// The hits a search found, the shards it covered and its files by state.
#[derive(Default)]
pub(crate) struct Searched {
    pub(crate) hits: Vec<SearchHit>,
    pub(crate) covered: Vec<ShardId>,
    pub(crate) counts: Counts,
}

/// Where a found chunk's file is: its shard's file table, and its ordinal and content version
/// there.
struct Place {
    table: Arc<FileTable>,
    file_ordinal: u64,
    content_version: ManifestVersion,
}

impl Searchable {
    /// Whether this segment holds chunks of the file at `place`.
    fn holds(&self, place: &Place) -> bool {
        self.holders
            .iter()
            .any(|holder| Arc::ptr_eq(&holder.files, &place.table))
            && !self
                .chunks
                .file_rows(place.file_ordinal, place.content_version)
                .is_empty()
    }
}

/// A segment to search, with its chunk records and the file tables of the shards that hold it.
struct Searchable {
    entry: SegmentEntry,
    chunks: Arc<ChunkTable>,
    holders: Vec<Holder>,
}

/// The file table of a shard holding a segment, the segment it was read from, if any, and
/// whether the search's scope holds the shard's whole key range.
#[derive(Clone)]
struct Holder {
    files_segment: Option<SegmentId>,
    files: Arc<FileTable>,
    whole: bool,
}

/// The shards of a scope a search opens, and those it leaves unread or unexplored.
#[derive(Debug, Default)]
pub(crate) struct Reached {
    pub(crate) open: BTreeSet<ShardId>,
    pub(crate) unread: BTreeSet<ShardId>,
    pub(crate) unexplored: BTreeSet<ShardId>,
}

impl Reached {
    /// Opens at most `limit` shards of `scope` in `manifest`, in the key order of their
    /// ranges, leaving the rest unexplored. Shards already unread stay so.
    pub(crate) fn select(&mut self, manifest: &Manifest, scope: &Scope, limit: Option<usize>) {
        let mut in_scope: Vec<(&ShardId, &ShardEntry)> = manifest
            .shards()
            .iter()
            .filter(|(_, shard)| scope.overlaps(shard.range()))
            .collect();
        in_scope.sort_by(|(_, a), (_, b)| a.range().start().cmp(b.range().start()));
        let limit = limit.unwrap_or(usize::MAX);
        let unread = &self.unread;
        self.open = in_scope
            .iter()
            .take(limit)
            .map(|(id, _)| (*id).clone())
            .filter(|id| !unread.contains(id))
            .collect();
        self.unexplored = in_scope
            .iter()
            .skip(limit)
            .map(|(id, _)| (*id).clone())
            .collect();
    }

    /// Leaves `shards` unread.
    pub(crate) fn leave_unread(&mut self, shards: Vec<ShardId>) {
        for shard in &shards {
            self.open.remove(shard);
        }
        self.unread.extend(shards);
    }

    /// Leaves the shards still open unexplored.
    pub(crate) fn abandon(&mut self) {
        self.unexplored.append(&mut self.open);
    }
}

/// Searches the shards in `open` of `manifest` for the best `limit` fused hits of `request`,
/// and adds what the index plug-ins report spending to `usage`. Also gives the shards the
/// search covered: those holding a file in its scope that passes its filter. Only segments
/// with chunks the search may return reach an index plug-in.
#[expect(
    clippy::too_many_arguments,
    reason = "one search's inputs, passed through from the reader"
)]
pub(crate) async fn run(
    loader: &Loader<'_>,
    plugins: &Plugins,
    manifest: &Manifest,
    open: &BTreeSet<ShardId>,
    request: &SearchRequest,
    limit: usize,
    fusion: &Fusion,
    usage: &mut Usage,
) -> Result<Searched> {
    let SearchRequest {
        query,
        scope,
        filter,
        ..
    } = request;
    let shards: Vec<_> = manifest
        .shards()
        .iter()
        .filter(|(id, _)| open.contains(*id))
        .collect();
    let loading: Vec<_> = shards
        .iter()
        .map(|(id, shard)| load_files(loader, id, shard))
        .collect();
    let files: BTreeMap<&ShardId, Arc<FileTable>> = stream::iter(loading)
        .buffer_unordered(LOAD_CONCURRENCY)
        .try_collect()
        .await?;
    let (covered, counts) = counted(loader, &shards, &files, scope, filter).await;

    let searchable = searchable(loader, &shards, &files, scope).await?;

    let allowed = allowed_rows(loader, &searchable, scope, filter).await;
    let by_kind = by_kind(&searchable, &allowed);

    let by_id: BTreeMap<&SegmentId, &Searchable> = searchable
        .iter()
        .map(|segment| (segment.entry.id(), segment))
        .collect();
    let mut passages: BTreeMap<ChunkKey, SearchHit> = BTreeMap::new();
    let mut places: BTreeMap<ChunkKey, Place> = BTreeMap::new();
    let mut lists: Vec<(f64, Vec<ChunkKey>)> = Vec::with_capacity(by_kind.len());
    let mut kinds_segments: Vec<Vec<&Searchable>> = Vec::with_capacity(by_kind.len());
    let allowed = &allowed;
    let by_id = &by_id;
    let searches = by_kind.into_iter().filter_map(|(kind, segments)| {
        let index = plugins.index(kind)?;
        let weight = fusion.weight(kind);
        Some(async move {
            let found = index
                .get()
                .search(loader.source(), &segments, allowed, query, depth(limit))
                .await?;
            let of_kind: Vec<&Searchable> = segments
                .iter()
                .filter_map(|segment| by_id.get(segment.id()).copied())
                .collect();
            Ok::<_, IndexError>((kind, weight, found, of_kind))
        })
    });
    let mut places_in_lists = ListPlaces::default();
    for (kind, weight, found, of_kind) in future::try_join_all(searches).await? {
        kinds_segments.push(of_kind);
        *usage = usage.saturating_add(found.usage());
        let mut hits = found.into_hits();
        hits.sort_by(|a, b| {
            b.score()
                .total_cmp(&a.score())
                .then_with(|| a.segment().cmp(b.segment()))
                .then_with(|| a.row().cmp(&b.row()))
        });
        let mut list = Vec::with_capacity(hits.len());
        let mut rank = 0;
        for hit in hits {
            let Some(segment) = by_id.get(hit.segment()) else {
                continue;
            };
            if let Some((key, passage, place)) = resolve(segment, &hit, manifest)? {
                places_in_lists.record(&key, kind, hit.score(), &mut rank);
                passages.entry(key.clone()).or_insert(passage);
                places.entry(key.clone()).or_insert(place);
                list.push(key);
            }
        }
        lists.push((weight, list));
    }

    let mut found = Vec::with_capacity(limit.min(passages.len()));
    let covers = |list: usize, key: &ChunkKey| {
        let (Some(segments), Some(place)) = (kinds_segments.get(list), places.get(key)) else {
            return false;
        };
        segments.iter().any(|segment| segment.holds(place))
    };
    for (key, score) in fuse::fuse(&lists, fusion.k(), limit, covers) {
        if let Some(mut passage) = passages.remove(&key) {
            passage.score = score;
            places_in_lists.place(&key, &mut passage);
            found.push(passage);
        }
    }
    Ok(Searched {
        hits: found,
        covered,
        counts,
    })
}

/// Where each passage stood in each index kind's list: its score and its rank among the
/// distinct passages of that list.
#[derive(Default)]
struct ListPlaces {
    scores: BTreeMap<ChunkKey, Vec<(IndexKind, f32)>>,
    ranks: BTreeMap<ChunkKey, Vec<(IndexKind, u32)>>,
}

impl ListPlaces {
    /// Records the passage's place in `kind`'s list unless the list held it already, counting
    /// it in `rank`.
    fn record(&mut self, key: &ChunkKey, kind: &IndexKind, score: f32, rank: &mut u32) {
        let scores = self.scores.entry(key.clone()).or_default();
        if scores.iter().any(|(scored, _)| scored == kind) {
            return;
        }
        scores.push((kind.clone(), score));
        *rank = rank.saturating_add(1);
        self.ranks
            .entry(key.clone())
            .or_default()
            .push((kind.clone(), *rank));
    }

    fn place(&mut self, key: &ChunkKey, hit: &mut SearchHit) {
        hit.list_scores = self.scores.remove(key).unwrap_or_default();
        hit.list_ranks = self.ranks.remove(key).unwrap_or_default();
    }
}

/// The segments of `searchable` with rows in `allowed`, by the index kind that built them.
fn by_kind<'a>(
    searchable: &'a [Searchable],
    allowed: &AllowedRows,
) -> BTreeMap<&'a IndexKind, Vec<SegmentEntry>> {
    let mut by_kind: BTreeMap<&IndexKind, Vec<SegmentEntry>> = BTreeMap::new();
    for segment in searchable {
        if allowed.rows(segment.entry.id()).is_none() {
            continue;
        }
        if let Some(kind) = segment.entry.built_by().index() {
            by_kind.entry(kind).or_default().push(segment.entry.clone());
        }
    }
    by_kind
}

/// The rows of `searchable` a search may score: those live in a holder's file table whose
/// file is in `scope` and passes `filter`. For a holder wholly in scope, without a filter,
/// they are the rows kept for each segment and table.
async fn allowed_rows(
    loader: &Loader<'_>,
    searchable: &[Searchable],
    scope: &Scope,
    filter: &Filter,
) -> AllowedRows {
    let mut allowed = AllowedRows::new();
    for segment in searchable {
        for holder in &segment.holders {
            let rows = match &holder.files_segment {
                Some(files_segment) if holder.whole && filter.is_empty() => {
                    loader
                        .live_rows(
                            segment.entry.id(),
                            &segment.chunks,
                            files_segment,
                            &holder.files,
                        )
                        .await
                }
                Some(_) | None => Arc::new(segment.chunks.rows(&holder.files, scope, filter)),
            };
            allowed.allow(segment.entry.id(), rows);
        }
    }
    allowed
}

/// The segments of `shards` an index plug-in built, each with its chunk records and the file
/// tables of the shards that hold it; one without chunk records is left out.
async fn searchable(
    loader: &Loader<'_>,
    shards: &[(&ShardId, &ShardEntry)],
    files: &BTreeMap<&ShardId, Arc<FileTable>>,
    scope: &Scope,
) -> Result<Vec<Searchable>> {
    let mut holders: BTreeMap<&SegmentId, (&SegmentEntry, Vec<Holder>)> = BTreeMap::new();
    for (id, shard) in shards {
        let Some(table) = files.get(id) else {
            continue;
        };
        let holder = Holder {
            files_segment: tables::files_segment(shard).map(|segment| segment.id().clone()),
            files: Arc::clone(table),
            whole: scope.covers(shard.range()),
        };
        for segment in shard.segments() {
            if segment.level() == Level::L0 || segment.built_by().index().is_none() {
                continue;
            }
            let entry = holders
                .entry(segment.id())
                .or_insert_with(|| (segment, Vec::new()));
            entry.1.push(holder.clone());
        }
    }
    let loading: Vec<_> = holders
        .into_values()
        .map(|(entry, tables)| load_chunks(loader, entry, tables))
        .collect();
    let segments: Vec<Option<Searchable>> = stream::iter(loading)
        .buffered(LOAD_CONCURRENCY)
        .try_collect()
        .await?;
    Ok(segments.into_iter().flatten().collect())
}

/// The shards holding a file in `scope` that passes `filter`, and those files by state. A
/// shard wholly in scope, without a filter, gives the counts kept for its file table.
async fn counted(
    loader: &Loader<'_>,
    shards: &[(&ShardId, &ShardEntry)],
    files: &BTreeMap<&ShardId, Arc<FileTable>>,
    scope: &Scope,
    filter: &Filter,
) -> (Vec<ShardId>, Counts) {
    let mut covered = Vec::new();
    let mut total = Counts::default();
    for (id, shard) in shards {
        let Some(table) = files.get(id) else {
            continue;
        };
        let whole = filter.is_empty() && scope.covers(shard.range());
        let counts = match tables::files_segment(shard) {
            Some(segment) if whole => *loader.counts(segment.id(), table).await,
            Some(_) | None => Counts::of(table, scope, filter),
        };
        if counts.any() {
            covered.push((*id).clone());
        }
        total = total.plus(counts);
    }
    (covered, total)
}

/// How many hits each index kind is asked for: never fewer than `limit`, and at least
/// [`FUSION_DEPTH`] so fusion sees passages one kind ranks far below another.
fn depth(limit: usize) -> usize {
    if limit == 0 {
        0
    } else {
        limit.max(FUSION_DEPTH)
    }
}

async fn load_files<'a>(
    loader: &Loader<'_>,
    id: &'a ShardId,
    shard: &ShardEntry,
) -> Result<(&'a ShardId, Arc<FileTable>)> {
    Ok((id, loader.files(shard).await?))
}

async fn load_chunks(
    loader: &Loader<'_>,
    entry: &SegmentEntry,
    holders: Vec<Holder>,
) -> Result<Option<Searchable>> {
    let chunks = loader.chunks(entry).await?;
    Ok(chunks.map(|chunks| Searchable {
        entry: entry.clone(),
        chunks,
        holders,
    }))
}

fn resolve(
    segment: &Searchable,
    hit: &Hit,
    manifest: &Manifest,
) -> Result<Option<(ChunkKey, SearchHit, Place)>> {
    let Ok(row) = usize::try_from(hit.row()) else {
        return Ok(None);
    };
    let Some(record) = segment.chunks.get(row) else {
        return Ok(None);
    };
    let first = segment.chunks.neighbours(row, row, 0).start;
    let Some(position) = row.checked_sub(first) else {
        return Ok(None);
    };
    for holder in &segment.holders {
        if let Some(source) =
            segment
                .chunks
                .source_ref(row, &holder.files, manifest.data_location())?
        {
            let key = ChunkKey {
                key: source.location().key().to_owned(),
                content_version: record.content_version(),
                profile: record.profile().clone(),
                position,
            };
            let passage = SearchHit {
                source,
                score: hit.score(),
                list_scores: Vec::new(),
                list_ranks: Vec::new(),
                rerank_score: None,
                heading_path: record.heading_path().to_vec(),
                display_name: holder
                    .files
                    .by_ordinal(record.file_ordinal())
                    .and_then(FileEntry::display_name)
                    .map(str::to_owned),
                snippet: None,
            };
            let place = Place {
                table: Arc::clone(&holder.files),
                file_ordinal: record.file_ordinal(),
                content_version: record.content_version(),
            };
            return Ok(Some((key, passage, place)));
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shards(prefix: &str, count: usize) -> BTreeSet<ShardId> {
        (0..count)
            .map(|i| ShardId::new(format!("{prefix}-{i:04}")).unwrap())
            .collect()
    }

    #[test]
    fn results_weigh_every_shard_they_name() {
        let empty = SearchResults::new(
            ManifestVersion::FIRST,
            Searched::default(),
            Reached::default(),
            Reranking::Off,
            Usage::new(),
        );
        let reached = Reached {
            open: BTreeSet::new(),
            unread: shards("unread", 1000),
            unexplored: shards("unexplored", 1000),
        };
        let searched = Searched {
            covered: shards("covered", 1000).into_iter().collect(),
            ..Searched::default()
        };
        let many = SearchResults::new(
            ManifestVersion::FIRST,
            searched,
            reached,
            Reranking::Off,
            Usage::new(),
        );
        assert!(
            many.weight() > empty.weight() + 3000 * 48,
            "{}",
            many.weight()
        );
    }
}
