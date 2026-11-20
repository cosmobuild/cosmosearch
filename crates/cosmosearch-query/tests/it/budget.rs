use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use bytes::Bytes;
use cosmosearch_core::{
    AllowedRows, Budget, Built, ByteRange, Chunk, ETag, FileSource, Hits, Index, IndexError,
    IndexKind, KeyRange, Level, Query, RerankCause, RerankError, RerankScores, Reranker,
    RerankerIdentity, Scope, SegmentEntry, SegmentId, SegmentSource, ShardEntry, ShardId,
    SharedIndex, SharedReranker, Usage,
};
use cosmosearch_query::{
    Reader, ReaderOptions, Reranking, SearchHit, SearchRequest, SearchResults,
};
use cosmosearch_store::{
    MemoryStore, MetadataStore, ObjectInfo, ObjectKey, ReadStore, Result as StoreResult, WriteStore,
};
use futures_util::stream::BoxStream;
use futures_util::{TryStreamExt, future};

use super::rerank::identity;
use super::scopes::{fail_segments_of, faulty_reader, keys};
use super::{Fixture, plugins};

const TIME: Duration = Duration::from_millis(300);

fn shards(ids: &[&str]) -> BTreeSet<ShardId> {
    ids.iter().map(|id| ShardId::new(*id).unwrap()).collect()
}

fn refund() -> SearchRequest {
    SearchRequest::new(Query::new("refund"))
}

async fn split(f: &Fixture, low: &str, key: &str, high: &str) {
    let head = f.handle.head().await.unwrap().unwrap();
    let parts = [
        (low, KeyRange::new("", Some(key.to_owned())).unwrap()),
        (high, KeyRange::new(key, None).unwrap()),
    ];
    let shards = parts
        .into_iter()
        .map(|(id, range)| (ShardId::new(id).unwrap(), ShardEntry::new(range, vec![])))
        .collect();
    let commit = cosmosearch_store::new_commit_id().unwrap();
    let next = head.successor(commit, shards, BTreeSet::new()).unwrap();
    f.handle.commit(&next).await.unwrap();
}

async fn shards_out_of_key_order() -> Fixture {
    let f = Fixture::new().await;
    split(&f, "z", "docs/m", "a").await;
    f.index(&[("docs/a.txt", b"refund policy text")]).await;
    f.index(&[("docs/z.txt", b"refund rules")]).await;
    f
}

#[tokio::test]
async fn a_search_opens_at_most_its_shard_limit_in_key_order() {
    let f = shards_out_of_key_order().await;
    let (reader, faults) = faulty_reader(&f, ReaderOptions::new()).await;
    fail_segments_of(&f, &faults, "a").await;

    let one = refund().budget(Budget::new().max_shards(1));
    let found = reader.search(&one).await.unwrap();
    assert_eq!(keys(&found), ["docs/a.txt"]);
    assert_eq!(found.coverage().unexplored(), &shards(&["a"]));
    assert!(!found.out_of_time());
    reader.search(&refund()).await.unwrap_err();

    let none = reader
        .search(&refund().budget(Budget::new().max_shards(0)))
        .await
        .unwrap();
    assert_eq!(none.hits().to_vec(), Vec::<SearchHit>::new());
    assert_eq!(none.coverage().unexplored(), &shards(&["a", "z"]));
    assert_eq!(none.usage(), Usage::new());

    let high = one.scope(Scope::prefix("docs/z"));
    let found = f.reader.search(&high).await.unwrap();
    assert_eq!(keys(&found), ["docs/z.txt"]);
    assert!(found.coverage().unexplored().is_empty());
    let all = f.reader.search(&refund()).await.unwrap();
    assert_eq!(all.hits().len(), 2);
    assert!(all.coverage().unexplored().is_empty());
}

#[tokio::test]
async fn a_shard_left_unread_is_not_replaced_by_one_beyond_the_limit() {
    let f = shards_out_of_key_order().await;
    let head = f.handle.head().await.unwrap().unwrap();
    let low = head.shards().get(&ShardId::new("z").unwrap()).unwrap();
    let built = low
        .segments()
        .iter()
        .find(|segment| segment.level() == Level::L1)
        .unwrap();
    for file in built.files() {
        let key = format!(
            "index/{}/segments/{}/{}",
            f.handle.id(),
            built.id(),
            file.name()
        );
        f.store.delete(&ObjectKey::new(key).unwrap()).await.unwrap();
    }

    let found = f
        .reader
        .search(&refund().budget(Budget::new().max_shards(1)))
        .await
        .unwrap();
    assert_eq!(found.hits().to_vec(), Vec::<SearchHit>::new());
    assert_eq!(found.coverage().unread(), &shards(&["z"]));
    assert_eq!(found.coverage().unexplored(), &shards(&["a"]));
}

#[derive(Debug)]
struct Stalls(IndexKind);

#[async_trait]
impl Index for Stalls {
    fn kind(&self) -> &IndexKind {
        &self.0
    }

    fn level(&self) -> Level {
        Level::L2
    }

    async fn build(&self, _: &SegmentId, _: &[Chunk]) -> Result<Built, IndexError> {
        Ok(Built::default())
    }

    async fn search(
        &self,
        _: &SegmentSource,
        _: &[SegmentEntry],
        _: &AllowedRows,
        _: &Query,
        _: usize,
    ) -> Result<Hits, IndexError> {
        std::future::pending().await
    }
}

#[tokio::test]
async fn a_search_out_of_time_before_any_hit_returns_none_and_says_so() {
    let f = Fixture::new().await;
    f.index(&[("docs/a.txt", b"refund policy text")]).await;
    let stalls = Stalls(IndexKind::new("stalls").unwrap());
    f.add_twin_segments(&stalls).await;
    let mut plugins = plugins();
    plugins.add_index(SharedIndex::new(stalls)).unwrap();
    let reader = Reader::open(
        f.data.clone(),
        f.metadata.clone(),
        FileSource::HostNamed,
        plugins,
        ReaderOptions::new(),
    )
    .await
    .unwrap();

    let started = Instant::now();
    let found = reader
        .search(&refund().budget(Budget::new().max_time(TIME)))
        .await
        .unwrap();
    assert!(started.elapsed() < TIME * 10, "{:?}", started.elapsed());
    assert!(found.out_of_time());
    assert_eq!(found.hits().to_vec(), Vec::<SearchHit>::new());
    assert_eq!(found.coverage().unexplored(), &shards(&["0"]));
}

#[derive(Debug)]
struct Held(RerankerIdentity, Mutex<Receiver<()>>);

impl Reranker for Held {
    fn identity(&self) -> &RerankerIdentity {
        &self.0
    }

    fn rerank(&self, _: &str, passages: &[&str]) -> Result<RerankScores, RerankError> {
        let Ok(held) = self.1.lock() else {
            return Err(RerankError::Model(RerankCause::new("the lock is poisoned")));
        };
        let released = held.recv_timeout(Duration::from_secs(30));
        assert!(released.is_ok(), "the test released the reranker");
        Ok(RerankScores::new(vec![0.0; passages.len()], Usage::new()))
    }
}

#[tokio::test]
async fn a_search_out_of_time_while_reranking_keeps_its_fused_order() {
    let f = Fixture::new().await;
    f.index(&[
        ("docs/a.txt", b"refund policy text"),
        ("docs/b.txt", b"refund rules"),
    ])
    .await;
    let fused = f.reader.search(&refund()).await.unwrap();
    let (release, held) = mpsc::channel();
    let mut plugins = plugins();
    plugins.set_reranker(SharedReranker::new(Held(identity(), Mutex::new(held))));
    let reader = Reader::open(
        f.data.clone(),
        f.metadata.clone(),
        FileSource::HostNamed,
        plugins,
        ReaderOptions::new(),
    )
    .await
    .unwrap();

    let found: SearchResults = reader
        .search(&refund().budget(Budget::new().max_time(TIME)))
        .await
        .unwrap();
    release.send(()).unwrap();
    assert!(found.out_of_time());
    assert!(matches!(found.reranking(), Reranking::OutOfTime));
    assert_eq!(found.hits(), fused.hits());
    assert!(found.coverage().unexplored().is_empty());
}

#[derive(Debug, Clone)]
struct Stalling {
    inner: MemoryStore,
    stalled: Arc<AtomicBool>,
}

impl Stalling {
    async fn wait_if_stalled(&self, key: &ObjectKey) {
        if self.stalled.load(Ordering::SeqCst) && key.as_str().ends_with("INDEX") {
            std::future::pending::<()>().await;
        }
    }
}

#[async_trait]
impl ReadStore for Stalling {
    async fn read_range(
        &self,
        key: &ObjectKey,
        range: ByteRange,
        expected: &ETag,
    ) -> StoreResult<Bytes> {
        self.inner.read_range(key, range, expected).await
    }

    async fn head(&self, key: &ObjectKey) -> StoreResult<ObjectInfo> {
        self.inner.head(key).await
    }

    fn list<'a>(
        &'a self,
        prefix: Option<&'a ObjectKey>,
        after: Option<&'a ObjectKey>,
    ) -> BoxStream<'a, StoreResult<ObjectInfo>> {
        self.inner.list(prefix, after)
    }
}

#[async_trait]
impl WriteStore for Stalling {
    async fn get(&self, key: &ObjectKey) -> StoreResult<Bytes> {
        self.wait_if_stalled(key).await;
        self.inner.get(key).await
    }

    async fn get_range(&self, key: &ObjectKey, range: ByteRange) -> StoreResult<Bytes> {
        self.wait_if_stalled(key).await;
        self.inner.get_range(key, range).await
    }

    async fn put_create(&self, key: &ObjectKey, body: Bytes) -> StoreResult<()> {
        self.inner.put_create(key, body).await
    }

    async fn put_overwrite(&self, key: &ObjectKey, body: Bytes) -> StoreResult<()> {
        self.inner.put_overwrite(key, body).await
    }

    async fn delete(&self, key: &ObjectKey) -> StoreResult<()> {
        self.inner.delete(key).await
    }
}

async fn stalling_reader(f: &Fixture, options: ReaderOptions) -> (Reader, Arc<AtomicBool>) {
    let stalled = Arc::new(AtomicBool::new(false));
    let store = Stalling {
        inner: f.store.clone(),
        stalled: Arc::clone(&stalled),
    };
    let metadata = MetadataStore::new(store, "s3://example-bucket/index".parse().unwrap()).unwrap();
    let reader = Reader::open(
        f.data.clone(),
        metadata,
        FileSource::HostNamed,
        plugins(),
        options,
    )
    .await
    .unwrap();
    (reader, stalled)
}

#[tokio::test]
async fn a_slow_refresh_on_the_interval_does_not_hold_a_search_past_its_time() {
    let f = Fixture::new().await;
    f.index(&[("docs/a.txt", b"refund policy text")]).await;
    let options = ReaderOptions::new().refresh_interval(Duration::ZERO);
    let (reader, stalled) = stalling_reader(&f, options).await;
    stalled.store(true, Ordering::SeqCst);

    let started = Instant::now();
    reader
        .search(&refund().budget(Budget::new().max_time(TIME)))
        .await
        .unwrap();
    assert!(started.elapsed() < TIME * 10, "{:?}", started.elapsed());
}

#[tokio::test]
async fn a_slow_refresh_after_a_missing_segment_does_not_hold_a_search_past_its_time() {
    let f = Fixture::new().await;
    f.index(&[("docs/a.txt", b"refund policy text")]).await;
    let (reader, stalled) = stalling_reader(&f, ReaderOptions::new()).await;
    let segments: Vec<ObjectKey> = f
        .store
        .list(None, None)
        .map_ok(|info| info.key().clone())
        .try_filter(|key| future::ready(key.as_str().contains("/segments/")))
        .try_collect()
        .await
        .unwrap();
    assert!(!segments.is_empty(), "the index holds segments");
    for key in &segments {
        f.store.delete(key).await.unwrap();
    }
    stalled.store(true, Ordering::SeqCst);

    let started = Instant::now();
    let found = reader
        .search(&refund().budget(Budget::new().max_time(TIME)))
        .await
        .unwrap();
    assert!(started.elapsed() < TIME * 10, "{:?}", started.elapsed());
    assert!(found.out_of_time());
}
