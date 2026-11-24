use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use cosmosearch_core::{
    AllowedRows, Built, CachedResults, Chunk, ExtractLimits, FileSource, Filter, Hits, Index,
    IndexError, IndexKind, Level, Query, ResultCache, Scope, SearchKey, SegmentEntry, SegmentId,
    SegmentSource, ShardId, SharedHeat, SharedIndex, SharedReranker, SharedResults, Signal,
    Timestamp, Usage,
};
use cosmosearch_lexical::Bm25;
use cosmosearch_query::{
    Boost, Error, Plugins, Reader, ReaderOptions, Reranking, SearchRequest, SearchResults,
};

use cosmosearch_store::CreateOutcome;

use super::rerank::{Broken, Fault, fake, identity};
use super::{Fixture, Tally, extractors, plugins};

#[derive(Debug, Clone, Default)]
struct Kept(Arc<Mutex<BTreeMap<SearchKey, CachedResults>>>);

impl ResultCache for Kept {
    fn get(&self, key: &SearchKey) -> Option<CachedResults> {
        self.0.lock().unwrap().get(key).cloned()
    }

    fn put(&self, key: SearchKey, results: CachedResults, _: u64) {
        self.0.lock().unwrap().insert(key, results);
    }
}

impl Kept {
    fn len(&self) -> usize {
        self.0.lock().unwrap().len()
    }
}

async fn reader(f: &Fixture, kept: &Kept, options: ReaderOptions) -> Reader {
    reader_with(f, kept, options, None).await
}

async fn reader_with(
    f: &Fixture,
    kept: &Kept,
    options: ReaderOptions,
    reranker: Option<SharedReranker>,
) -> Reader {
    let mut plugins = plugins();
    if let Some(reranker) = reranker {
        plugins.set_reranker(reranker);
    }
    Reader::open(
        f.data.clone(),
        f.metadata.clone(),
        FileSource::HostNamed,
        plugins,
        options.result_cache(SharedResults::new(kept.clone())),
    )
    .await
    .unwrap()
}

async fn search(reader: &Reader, query: &str, filter: &Filter, limit: usize) -> SearchResults {
    reader
        .search(
            &SearchRequest::new(Query::new(query))
                .filter(filter.clone())
                .limit(limit),
        )
        .await
        .unwrap()
}

#[tokio::test]
async fn a_limit_above_the_maximum_is_refused_before_any_read() {
    let f = Fixture::new().await;
    f.index(&[("docs/a.txt", b"refund policy text")]).await;
    let reader = reader(&f, &Kept::default(), ReaderOptions::new()).await;
    let at_most = SearchRequest::new(Query::new("refund")).limit(SearchRequest::MAX_LIMIT);
    assert_eq!(reader.search(&at_most).await.unwrap().hits().len(), 1);
    for limit in [SearchRequest::MAX_LIMIT + 1, usize::MAX] {
        let request = SearchRequest::new(Query::new("refund")).limit(limit);
        let refused = reader.search(&request).await;
        assert!(matches!(refused, Err(Error::LimitTooLarge { limit: l }) if l == limit));
    }
}

#[tokio::test]
async fn a_repeated_search_reads_nothing_and_still_records_heat() {
    let f = Fixture::new().await;
    f.index(&[("docs/a.txt", b"refund policy text")]).await;
    let (kept, tally) = (Kept::default(), Tally::default());
    let options = ReaderOptions::new().heat(SharedHeat::new(tally.clone()));
    let reader = reader(&f, &kept, options).await;
    let all = Filter::new();

    let first = search(&reader, "refund", &all, 10).await;
    assert!(!first.cached());
    assert_ne!(first.usage(), Usage::new());
    let again = search(&reader, "refund", &all, 10).await;
    assert!(again.cached());
    assert_eq!(again.usage(), Usage::new());
    assert_eq!(again.hits(), first.hits());
    assert_eq!(again.version(), first.version());
    let shard = ShardId::new("0").unwrap();
    let once = [
        (Signal::Search, vec![shard.clone()]),
        (Signal::Hit, vec![shard]),
    ];
    assert_eq!(tally.take(), [once.clone(), once].concat());

    let legal = Filter::new().equals("owner", "legal");
    for (query, filter, limit) in [
        ("policy", &all, 10),
        ("refund", &legal, 10),
        ("refund", &all, 5),
    ] {
        assert!(!search(&reader, query, filter, limit).await.cached());
    }
    let scoped = SearchRequest::new(Query::new("refund")).scope(Scope::prefix("docs/"));
    assert!(!reader.search(&scoped).await.unwrap().cached());
    assert!(reader.search(&scoped).await.unwrap().cached());
    let exact = SearchRequest::new(Query::new("refund").with_exact_phrases());
    assert!(!reader.search(&exact).await.unwrap().cached());
    let every = SearchRequest::new(Query::new("refund").with_every_word());
    assert!(!reader.search(&every).await.unwrap().cached());
    let boosted = |weight| {
        SearchRequest::new(Query::new("refund")).boost(Boost::matching(legal.clone(), weight))
    };
    assert!(!reader.search(&boosted(1.0)).await.unwrap().cached());
    assert!(!reader.search(&boosted(2.0)).await.unwrap().cached());
    assert!(reader.search(&boosted(1.0)).await.unwrap().cached());
    let now = Timestamp::from_millis(5);
    let recent = SearchRequest::new(Query::new("refund")).boost(Boost::recency(
        now,
        Duration::from_secs(60),
        1.0,
    ));
    assert!(!reader.search(&recent).await.unwrap().cached());
    assert_eq!(kept.len(), 10);
}

#[tokio::test]
async fn a_commit_or_another_ranking_never_hits() {
    let f = Fixture::new().await;
    f.index(&[("docs/a.txt", b"refund policy text")]).await;
    let kept = Kept::default();
    let all = Filter::new();
    let plain = reader(&f, &kept, ReaderOptions::new()).await;
    assert_eq!(search(&plain, "refund", &all, 10).await.hits().len(), 1);

    let weighted = ReaderOptions::new().fusion_k(10.0);
    let weighted = reader(&f, &kept, weighted).await;
    assert!(!search(&weighted, "refund", &all, 10).await.cached());
    let reranked = reader_with(
        &f,
        &kept,
        ReaderOptions::new(),
        Some(SharedReranker::new(fake())),
    )
    .await;
    assert!(!search(&reranked, "refund", &all, 10).await.cached());
    assert!(search(&plain, "refund", &all, 10).await.cached());

    f.index(&[("docs/b.txt", b"refund rules")]).await;
    assert!(search(&plain, "refund", &all, 10).await.cached());
    plain.refresh().await.unwrap();
    let after = search(&plain, "refund", &all, 10).await;
    assert!(!after.cached());
    assert_eq!(after.hits().len(), 2);
}

#[tokio::test]
async fn results_of_a_reranker_that_failed_are_not_kept() {
    let f = Fixture::new().await;
    f.index(&[("docs/a.txt", b"refund policy text")]).await;
    let kept = Kept::default();
    let failing = SharedReranker::new(Broken(identity(), Fault::Fails));
    let reader = reader_with(&f, &kept, ReaderOptions::new(), Some(failing)).await;
    for _ in 0..2 {
        let results = search(&reader, "refund", &Filter::new(), 10).await;
        assert!(matches!(results.reranking(), Reranking::Failed(_)));
        assert!(!results.cached());
    }
    assert_eq!(kept.len(), 0);
}

#[tokio::test]
async fn once_a_reader_finds_its_index_gone_no_call_answers_from_it() {
    let f = Fixture::new().await;
    f.index(&[("docs/a.txt", b"refund policy text")]).await;
    let options = ReaderOptions::new().refresh_interval(Duration::from_secs(3600));
    let reader = reader(&f, &Kept::default(), options).await;
    let all = Filter::new();
    let hit = search(&reader, "refund", &all, 10).await.hits()[0].clone();
    assert!(search(&reader, "refund", &all, 10).await.cached());

    f.metadata.destroy().await.unwrap();
    let err = reader.refresh().await.unwrap_err();
    assert!(matches!(err, Error::IndexRemoved), "{err}");
    let request = SearchRequest::new(Query::new("refund"));
    let err = reader.search(&request).await.unwrap_err();
    assert!(matches!(err, Error::IndexRemoved), "{err}");
    let err = reader.read(hit.source()).await.unwrap_err();
    assert!(matches!(err, Error::IndexRemoved), "{err}");

    let created = f
        .metadata
        .create(f.data.location(), FileSource::HostNamed)
        .await
        .unwrap();
    assert!(matches!(created, CreateOutcome::Created(_)));
    let err = reader.search(&request).await.unwrap_err();
    assert!(matches!(err, Error::IndexRemoved), "{err}");
}

#[tokio::test]
async fn a_reader_that_finds_its_index_replaced_says_so_on_every_call() {
    let f = Fixture::new().await;
    f.index(&[("docs/a.txt", b"refund policy text")]).await;
    let options = ReaderOptions::new().refresh_interval(Duration::from_secs(3600));
    let reader = reader(&f, &Kept::default(), options).await;
    let all = Filter::new();
    search(&reader, "refund", &all, 10).await;

    f.metadata.destroy().await.unwrap();
    f.metadata
        .create(f.data.location(), FileSource::HostNamed)
        .await
        .unwrap();
    let err = reader.refresh().await.unwrap_err();
    assert!(matches!(err, Error::IndexReplaced { .. }), "{err}");
    let err = reader
        .search(&SearchRequest::new(Query::new("refund")))
        .await
        .unwrap_err();
    assert!(matches!(err, Error::IndexReplaced { .. }), "{err}");
}

#[derive(Debug)]
struct Tuned {
    inner: Bm25,
    setting: u64,
}

#[async_trait]
impl Index for Tuned {
    fn kind(&self) -> &IndexKind {
        self.inner.kind()
    }

    fn level(&self) -> Level {
        self.inner.level()
    }

    fn search_settings(&self) -> Vec<u64> {
        vec![self.setting]
    }

    async fn build(&self, segment: &SegmentId, chunks: &[Chunk]) -> Result<Built, IndexError> {
        self.inner.build(segment, chunks).await
    }

    async fn search(
        &self,
        source: &SegmentSource,
        segments: &[SegmentEntry],
        allowed: &AllowedRows,
        query: &Query,
        limit: usize,
    ) -> Result<Hits, IndexError> {
        self.inner
            .search(source, segments, allowed, query, limit)
            .await
    }
}

async fn tuned_reader(f: &Fixture, kept: &Kept, setting: u64) -> Reader {
    let mut plugins = Plugins::new(extractors());
    plugins
        .add_index(SharedIndex::new(Tuned {
            inner: Bm25::new().unwrap(),
            setting,
        }))
        .unwrap();
    Reader::open(
        f.data.clone(),
        f.metadata.clone(),
        FileSource::HostNamed,
        plugins,
        ReaderOptions::new().result_cache(SharedResults::new(kept.clone())),
    )
    .await
    .unwrap()
}

#[tokio::test]
async fn readers_whose_plugins_search_differently_do_not_share_results() {
    let f = Fixture::new().await;
    f.index(&[("docs/a.txt", b"refund policy text")]).await;
    let kept = Kept::default();
    let all = Filter::new();
    let one = tuned_reader(&f, &kept, 1).await;
    let also_one = tuned_reader(&f, &kept, 1).await;
    let three = tuned_reader(&f, &kept, 3).await;
    assert!(!search(&one, "refund", &all, 10).await.cached());
    assert!(search(&also_one, "refund", &all, 10).await.cached());
    assert!(!search(&three, "refund", &all, 10).await.cached());
}

#[tokio::test]
async fn readers_with_snippets_under_other_limits_do_not_share_results() {
    let f = Fixture::new().await;
    f.index(&[("docs/a.txt", b"refund policy text")]).await;
    let kept = Kept::default();
    let all = Filter::new();
    let snippets = || ReaderOptions::new().snippets(1);
    let default = reader(&f, &kept, snippets()).await;
    let fewer_pages = snippets().limits(ExtractLimits::builder().pages(1).build().unwrap());
    let other = reader(&f, &kept, fewer_pages).await;
    assert!(!search(&default, "refund", &all, 10).await.cached());
    assert!(!search(&other, "refund", &all, 10).await.cached());

    let plain = reader(&f, &kept, ReaderOptions::new()).await;
    let plain_fewer = reader(
        &f,
        &kept,
        ReaderOptions::new().limits(ExtractLimits::builder().pages(1).build().unwrap()),
    )
    .await;
    assert!(!search(&plain, "refund", &all, 10).await.cached());
    assert!(search(&plain_fewer, "refund", &all, 10).await.cached());
}
