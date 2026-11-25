use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

use bytes::Bytes;
use cosmosearch::{
    CreateOutcome, DataStore, Engine, Error, FileSource, Filter, MetadataStore, Query,
    RerankFailure, Reranking, Scope, SearchHit, SearchRequest, SearchResults, SourceRef,
    core::{RerankError, RerankScores, Reranker, RerankerIdentity},
    index::{
        Extractors, FileChange, FileStatus, FileUpsert, Indexer, PublishOutcome, index_pending,
    },
    store::MemoryStore,
};
use cosmosearch_core::{
    CommitId, ExtractLimits, FakeReranker, KeyRange, Level, RerankCause, SegmentEntry, SegmentId,
    ShardEntry, ShardId, SharedIndex,
};
use cosmosearch_lexical::Bm25;
use cosmosearch_store::{IndexHandle, ObjectKey, ReadStore, WriteStore};
use futures_util::TryStreamExt;

pub(super) struct Fixture {
    data_store: MemoryStore,
    metadata_store: MemoryStore,
    data: DataStore,
    metadata: MetadataStore,
    pub(super) index: IndexHandle,
    pub(super) indexer: Indexer,
}

impl Fixture {
    pub(super) async fn new() -> Fixture {
        let data_store = MemoryStore::new();
        let metadata_store = MemoryStore::new();
        let data = DataStore::new(
            data_store.clone(),
            "s3://example-bucket/docs/".parse().unwrap(),
        )
        .unwrap();
        let metadata = MetadataStore::new(
            metadata_store.clone(),
            "s3://example-index-bucket/index".parse().unwrap(),
        )
        .unwrap();
        let index = create(&metadata, &data).await;
        let indexer = Indexer::new(index.clone(), data.clone()).unwrap();
        Fixture {
            data_store,
            metadata_store,
            data,
            metadata,
            index,
            indexer,
        }
    }

    async fn engine(&self) -> Engine {
        self.builder().open().await.unwrap()
    }

    pub(super) fn builder(&self) -> cosmosearch::EngineBuilder {
        Engine::builder(self.data.clone(), self.metadata.clone())
            .file_source(FileSource::HostNamed)
            .index(Bm25::new().unwrap())
    }

    pub(super) async fn split_at(&self, key: &str) {
        let head = self.index.head().await.unwrap().unwrap();
        let shards = [
            (
                ShardId::new("a").unwrap(),
                ShardEntry::new(KeyRange::new("", Some(key.to_owned())).unwrap(), vec![]),
            ),
            (
                ShardId::new("b").unwrap(),
                ShardEntry::new(KeyRange::new(key, None).unwrap(), vec![]),
            ),
        ];
        let next = head
            .successor(
                CommitId::new("split").unwrap(),
                shards.into_iter().collect(),
                BTreeSet::new(),
            )
            .unwrap();
        self.index.commit(&next).await.unwrap();
    }

    pub(super) async fn index(&self, files: &[(&str, &'static [u8])]) {
        for (key, body) in files {
            self.data_store
                .put_overwrite(&ObjectKey::new(*key).unwrap(), Bytes::from_static(body))
                .await
                .unwrap();
        }
        let changes = files
            .iter()
            .map(|(key, _)| FileChange::Upsert(FileUpsert::new(*key)))
            .collect();
        self.indexer.apply(changes).await.unwrap();
        let plan = self.indexer.plan(1_000).await.unwrap();
        let bm25 = SharedIndex::new(Bm25::new().unwrap());
        for batch in plan.batches() {
            let mut extracted = Vec::new();
            for job in batch.jobs() {
                let file = self
                    .indexer
                    .extract(
                        job.clone(),
                        &Extractors::builtin().unwrap(),
                        &ExtractLimits::default(),
                    )
                    .await
                    .unwrap();
                extracted.push(file);
            }
            let built = self
                .indexer
                .build(&bm25, extracted, vec![])
                .await
                .unwrap()
                .unwrap();
            let outcome = self.indexer.publish(&built).await.unwrap();
            assert!(
                matches!(outcome, PublishOutcome::Published(_)),
                "the segment was not published: {outcome:?}"
            );
        }
    }

    async fn rebuild(&self) {
        let bm25 = SharedIndex::new(Bm25::new().unwrap());
        for rebuild in self.indexer.plan_rebuilds().await.unwrap() {
            let mut extracted = Vec::new();
            for job in rebuild.jobs() {
                extracted.push(
                    self.indexer
                        .extract(
                            job.clone(),
                            &Extractors::builtin().unwrap(),
                            &ExtractLimits::default(),
                        )
                        .await
                        .unwrap(),
                );
            }
            let built = self
                .indexer
                .build(&bm25, extracted, rebuild.replaces().to_vec())
                .await
                .unwrap()
                .unwrap();
            self.indexer.publish(&built).await.unwrap();
        }
    }

    async fn chunk_segments(&self) -> BTreeMap<SegmentId, SegmentEntry> {
        let head = self.index.head().await.unwrap().unwrap();
        head.shards()
            .values()
            .flat_map(ShardEntry::segments)
            .filter(|segment| segment.level() == Level::L1)
            .map(|segment| (segment.id().clone(), segment.clone()))
            .collect()
    }

    async fn sweep(&self, segment: &SegmentEntry) {
        let prefix = ObjectKey::new(format!(
            "index/{}/segments/{}",
            self.index.id(),
            segment.id()
        ))
        .unwrap();
        let keys: Vec<ObjectKey> = self
            .metadata_store
            .list(Some(&prefix), None)
            .map_ok(|info| info.key().clone())
            .try_collect()
            .await
            .unwrap();
        assert_ne!(
            keys,
            Vec::<ObjectKey>::new(),
            "the segment has files to sweep"
        );
        for key in keys {
            self.metadata_store.delete(&key).await.unwrap();
        }
    }
}

async fn create(metadata: &MetadataStore, data: &DataStore) -> IndexHandle {
    let outcome = metadata
        .create(data.location(), FileSource::HostNamed)
        .await
        .unwrap();
    let CreateOutcome::Created(index) = outcome else {
        panic!("expected created, got {outcome:?}");
    };
    index
}

pub(super) async fn search(engine: &Engine, text: &str) -> SearchResults {
    engine
        .search(&SearchRequest::new(Query::new(text)))
        .await
        .unwrap()
}

pub(super) fn keys(results: &SearchResults) -> Vec<&str> {
    results
        .hits()
        .iter()
        .map(|hit| hit.source().location().key())
        .collect()
}

#[tokio::test]
async fn searches_and_reads_with_data_and_metadata_in_separate_stores() {
    let f = Fixture::new().await;
    f.index(&[("docs/a.txt", b"refund policy text")]).await;
    let engine = f.engine().await;
    let results = search(&engine, "refund").await;
    assert_eq!(keys(&results), ["docs/a.txt"]);
    assert!(results.coverage().unread().is_empty());
    assert!(results.usage().object_reads() > 0);
    let reading = engine.read(results.hits()[0].source()).await.unwrap();
    assert_eq!(reading.chunks()[0].text(), "refund policy text");
    assert_eq!(reading.usage().object_reads(), 1);
    let in_metadata = f.metadata_store.list(None, None).try_next().await.unwrap();
    assert!(in_metadata.is_some());
    let in_data: Vec<String> = f
        .data_store
        .list(None, None)
        .map_ok(|info| info.key().to_string())
        .try_collect()
        .await
        .unwrap();
    assert_eq!(in_data, ["docs/a.txt"]);
}

#[tokio::test]
async fn an_engine_with_a_reranker_rescores_its_shortlist() {
    let f = Fixture::new().await;
    f.index(&[
        ("docs/a.txt", b"refund policy text"),
        ("docs/b.txt", b"refund rules"),
        ("docs/c.txt", b"refund"),
    ])
    .await;
    let plain = f.engine().await;
    assert_eq!(plain.reranker(), None);
    assert!(
        search(&plain, "refund")
            .await
            .hits()
            .iter()
            .all(|hit| hit.rerank_score().is_none())
    );

    let identity = RerankerIdentity::new("fake-reranker", "1", 512).unwrap();
    let engine = f
        .builder()
        .reranker(FakeReranker::new(identity.clone()))
        .rerank_shortlist(2)
        .open()
        .await
        .unwrap();
    assert_eq!(engine.reranker(), Some(&identity));
    let results = search(&engine, "refund").await;
    let scores: Vec<Option<f32>> = results.hits().iter().map(SearchHit::rerank_score).collect();
    assert!(scores[..2].iter().all(Option::is_some), "{scores:?}");
    assert_eq!(scores[2], None);
    assert!(results.usage().tokens() > 0);
    assert!(matches!(
        results.reranking(),
        Reranking::Reranked {
            scored: 2,
            unreadable: 0
        }
    ));
}

#[derive(Debug)]
struct Failing(RerankerIdentity);

impl Reranker for Failing {
    fn identity(&self) -> &RerankerIdentity {
        &self.0
    }

    fn rerank(&self, _: &str, _: &[&str]) -> Result<RerankScores, RerankError> {
        Err(RerankError::Model(RerankCause::new("the endpoint is down")))
    }
}

#[tokio::test]
async fn a_failing_reranker_leaves_fused_order_and_the_search_succeeds() {
    let f = Fixture::new().await;
    f.index(&[
        ("docs/a.txt", b"refund policy text"),
        ("docs/b.txt", b"refund rules"),
        ("docs/c.txt", b"refund"),
    ])
    .await;
    let fused = search(&f.engine().await, "refund").await;
    let identity = RerankerIdentity::new("failing-reranker", "1", 512).unwrap();
    let engine = f
        .builder()
        .reranker(Failing(identity))
        .open()
        .await
        .unwrap();
    let results = search(&engine, "refund").await;
    assert_eq!(results.hits(), fused.hits());
    let Reranking::Failed(failure) = results.reranking() else {
        panic!("expected a failure, got {:?}", results.reranking());
    };
    assert!(matches!(failure, RerankFailure::Reranker(_)), "{failure}");
}

#[tokio::test]
async fn the_engine_searches_its_snapshot_until_refreshed() {
    let f = Fixture::new().await;
    let engine = f.engine().await;
    let eager = f
        .builder()
        .refresh_interval(Duration::ZERO)
        .open()
        .await
        .unwrap();
    let lazy = f
        .builder()
        .refresh_interval(Duration::from_secs(3_600))
        .open()
        .await
        .unwrap();
    f.index(&[("docs/a.txt", b"refund policy text")]).await;

    assert_eq!(search(&engine, "refund").await.hits(), []);
    assert_eq!(search(&lazy, "refund").await.hits(), []);
    assert_eq!(keys(&search(&eager, "refund").await), ["docs/a.txt"]);

    let version = engine.refresh().await.unwrap();
    assert_eq!(engine.manifest().version(), version);
    assert_eq!(keys(&search(&engine, "refund").await), ["docs/a.txt"]);
}

#[tokio::test]
async fn a_segment_swept_after_a_newer_commit_costs_one_refresh() {
    let f = Fixture::new().await;
    f.index(&[("docs/a.txt", b"refund policy text")]).await;
    let engine = f.engine().await;
    let opened_at = engine.manifest().version();
    let old = f.chunk_segments().await;

    f.index(&[("docs/a.txt", b"refund rules changed")]).await;
    f.rebuild().await;
    let now = f.chunk_segments().await;
    let gone: Vec<&SegmentEntry> = old
        .iter()
        .filter(|(id, _)| !now.contains_key(*id))
        .map(|(_, segment)| segment)
        .collect();
    assert_ne!(gone, Vec::<&SegmentEntry>::new());
    for segment in gone {
        f.sweep(segment).await;
    }

    let results = search(&engine, "refund").await;
    assert!(results.version() > opened_at);
    assert!(results.coverage().unread().is_empty());
    assert_eq!(keys(&results), ["docs/a.txt"]);
    let reading = engine.read(results.hits()[0].source()).await.unwrap();
    assert_eq!(reading.chunks()[0].text(), "refund rules changed");
}

#[tokio::test]
async fn a_swept_segment_never_crashes_a_search_and_its_shard_is_reported_unread() {
    let f = Fixture::new().await;
    f.split_at("docs/m").await;
    f.index(&[
        ("docs/a.txt", b"refund words"),
        ("docs/z.txt", b"refund rules"),
    ])
    .await;
    let engine = f.engine().await;
    let hit_in_a = search(&engine, "words").await.hits()[0].source().clone();
    let head = f.index.head().await.unwrap().unwrap();
    let in_a = &head.shards()[&ShardId::new("a").unwrap()];
    let swept = in_a
        .segments()
        .iter()
        .find(|segment| segment.level() == Level::L1)
        .unwrap();
    f.sweep(swept).await;

    let fresh = f.engine().await;
    let results = search(&fresh, "refund").await;
    assert_eq!(keys(&results), ["docs/z.txt"]);
    let unread: Vec<&str> = results
        .coverage()
        .unread()
        .iter()
        .map(ShardId::as_str)
        .collect();
    assert_eq!(unread, ["a"]);

    let err = fresh.read(&hit_in_a).await.unwrap_err();
    assert!(matches!(err, Error::ShardUnread { .. }), "{err}");
    assert_eq!(keys(&search(&engine, "rules").await), ["docs/z.txt"]);
}

#[tokio::test]
async fn a_file_whose_table_is_swept_is_reported_unread() {
    let f = Fixture::new().await;
    f.index(&[("docs/a.txt", b"refund words")]).await;
    let head = f.index.head().await.unwrap().unwrap();
    let table = head
        .shards()
        .values()
        .flat_map(ShardEntry::segments)
        .find(|segment| segment.level() == Level::L0)
        .unwrap();
    f.sweep(table).await;

    let fresh = f.engine().await;
    let err = fresh.file("docs/a.txt").await.unwrap_err();
    assert!(matches!(err, Error::ShardUnread { .. }), "{err}");
    let all = Scope::all();
    let err = fresh
        .file_within("docs/a.txt", &all, &Filter::new())
        .await
        .unwrap_err();
    assert!(matches!(err, Error::ShardUnread { .. }), "{err}");
}

#[tokio::test]
async fn a_changed_or_removed_file_loses_its_old_chunks_in_a_warm_engine() {
    let f = Fixture::new().await;
    let old: Vec<String> = (0..60).map(|at| format!("docs/old-{at:02}.txt")).collect();
    let mut files: Vec<(&str, &'static [u8])> = old
        .iter()
        .map(|key| (key.as_str(), b"refund refund refund".as_slice()))
        .collect();
    files.push(("docs/kept.txt", b"refund window and a few other words"));
    f.index(&files).await;
    let engine = f.engine().await;
    assert_eq!(search(&engine, "refund").await.hits().len(), 10);
    let before = f.chunk_segments().await;

    f.index(&[("docs/old-00.txt", b"shipping rates")]).await;
    let removals = old
        .iter()
        .skip(1)
        .map(|key| FileChange::Remove {
            key: key.clone(),
            expected: None,
        })
        .collect();
    f.indexer.apply(removals).await.unwrap();
    let after = f.chunk_segments().await;
    assert!(
        before.keys().all(|id| after.contains_key(id)),
        "the segments holding the old chunks are still searched"
    );

    engine.refresh().await.unwrap();
    assert_eq!(keys(&search(&engine, "refund").await), ["docs/kept.txt"]);
    assert_eq!(
        keys(&search(&engine, "shipping").await),
        ["docs/old-00.txt"]
    );
}

#[tokio::test]
async fn a_reader_learns_when_another_process_ran_destroy_and_create() {
    let f = Fixture::new().await;
    f.index(&[("docs/a.txt", b"refund policy text")]).await;
    let engine = f.engine().await;
    let searching = f.engine().await;
    let opened = f.index.id().clone();

    f.metadata.destroy().await.unwrap();
    let err = engine.refresh().await.unwrap_err();
    assert!(matches!(err, Error::IndexRemoved), "{err}");

    let replacement = create(&f.metadata, &f.data).await;
    let err = engine.refresh().await.unwrap_err();
    let Error::IndexReplaced { opened: was, found } = err else {
        panic!("expected a replaced index, got {err}");
    };
    assert_eq!(was, opened);
    assert_eq!(&found, replacement.id());

    let err = searching
        .search(&SearchRequest::new(Query::new("refund")))
        .await
        .unwrap_err();
    assert!(matches!(err, Error::IndexReplaced { .. }), "{err}");
}

#[tokio::test]
async fn reads_are_refused_for_stale_refs_and_files_outside_the_index() {
    let f = Fixture::new().await;
    f.index(&[("docs/a.txt", b"refund policy text")]).await;
    let engine = f.engine().await;
    let hit = search(&engine, "refund").await.hits()[0].source().clone();
    f.data_store
        .put_overwrite(
            &ObjectKey::new("docs/a.txt").unwrap(),
            Bytes::from_static(b"changed"),
        )
        .await
        .unwrap();
    let err = engine.read(&hit).await.unwrap_err();
    assert!(matches!(err, Error::StaleRef { .. }), "{err}");

    let outside: SourceRef = "s3://example-bucket/private/a.txt#bytes=0-6@etag=e"
        .parse()
        .unwrap();
    let err = engine.read(&outside).await.unwrap_err();
    assert!(matches!(err, Error::OutsideData { .. }), "{err}");
}

#[test]
fn searching_and_reading_are_send() {
    fn assert_send<T: Send>(_: &T) {}
    fn check(engine: &Engine, reference: &SourceRef) {
        assert_send(&engine.search(&SearchRequest::new(Query::new("x")).limit(1)));
        assert_send(&engine.read(reference));
        assert_send(&engine.refresh());
    }
    let _: fn(&Engine, &SourceRef) = check;
}

#[tokio::test]
async fn index_pending_indexes_every_file_in_rounds_and_records_failures() {
    let f = Fixture::new().await;
    let files: [(&str, &[u8]); 3] = [
        ("docs/a.txt", b"refund policy for annual plans"),
        ("docs/b.txt", b"parcels ship on mondays"),
        ("docs/bad.txt", b"\xff\xfe not text"),
    ];
    for (key, body) in files {
        f.data_store
            .put_overwrite(&ObjectKey::new(key).unwrap(), Bytes::from_static(body))
            .await
            .unwrap();
    }
    let changes = files
        .iter()
        .map(|(key, _)| FileChange::Upsert(FileUpsert::new(*key)))
        .collect();
    f.indexer.apply(changes).await.unwrap();

    let bm25 = SharedIndex::new(Bm25::new().unwrap());
    let one = std::num::NonZeroUsize::MIN;
    let indexed = index_pending(
        &f.indexer,
        &bm25,
        &Extractors::builtin().unwrap(),
        &ExtractLimits::default(),
        one,
    )
    .await
    .unwrap();
    assert_eq!(indexed.extracted(), 3);
    assert_eq!(indexed.not_indexed(), 1);
    assert_eq!(
        indexed.segments(),
        2,
        "one round per file at a batch of one"
    );
    let bad = f.indexer.file("docs/bad.txt").await.unwrap().unwrap();
    assert_eq!(bad.entry().status(), FileStatus::Failed);

    let engine = f.builder().open().await.unwrap();
    let results = engine
        .search(&SearchRequest::new(Query::new("refund")))
        .await
        .unwrap();
    assert_eq!(results.hits().len(), 1);
    assert_eq!(results.coverage().searched(), 2);
    assert_eq!(results.coverage().failed(), 1);
}
