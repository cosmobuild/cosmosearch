use cosmosearch_chunk::Extractors;
use cosmosearch_core::{
    AllowedRows, ExtractFailure, ExtractLimits, Filter, Hit, Level, MediaType, Query, RowSet,
    SegmentEntry, SegmentSource, SharedIndex,
};
use cosmosearch_format::ChunkTable;
use cosmosearch_indexer::{ExtractOutcome, ExtractedFile, Plan, PublishOutcome};
use cosmosearch_lexical::Bm25;
use cosmosearch_query::{Plugins, Reader, ReaderOptions, SearchRequest};

use std::num::NonZeroU64;

use super::*;

fn extractors() -> Extractors {
    Extractors::builtin().unwrap()
}

pub(super) fn bm25() -> SharedIndex {
    SharedIndex::new(Bm25::new().unwrap())
}

fn limits() -> ExtractLimits {
    ExtractLimits::default()
}

impl Fixture {
    pub(super) async fn named_file(&self, key: &str, body: &'static [u8]) {
        self.put(key, body).await;
        self.send(vec![FileChange::Upsert(FileUpsert::new(key))])
            .await;
    }

    pub(super) async fn extract_all(&self, plan: &Plan) -> Vec<ExtractedFile> {
        let mut found = Vec::new();
        for batch in plan.batches() {
            for job in batch.jobs() {
                found.push(
                    self.indexer
                        .extract(job.clone(), &extractors(), &limits())
                        .await
                        .unwrap(),
                );
            }
        }
        found
    }

    pub(super) async fn build(
        &self,
        files: Vec<ExtractedFile>,
    ) -> cosmosearch_indexer::BuiltSegment {
        self.indexer
            .build(&bm25(), files, vec![])
            .await
            .unwrap()
            .unwrap()
    }

    pub(super) async fn index_everything(&self) -> PublishOutcome {
        let plan = self.indexer.plan(1_000).await.unwrap();
        let files = self.extract_all(&plan).await;
        self.publish_new(files).await
    }

    pub(super) async fn publish_new(&self, files: Vec<ExtractedFile>) -> PublishOutcome {
        let built = self.build(files).await;
        self.indexer.publish(&built).await.unwrap()
    }

    pub(super) async fn status(&self, key: &str) -> FileStatus {
        self.file(key).await.unwrap().status()
    }

    pub(super) async fn segments(&self) -> Vec<SegmentEntry> {
        let head = self.index.head().await.unwrap().unwrap();
        let mut found: Vec<SegmentEntry> = head
            .shards()
            .values()
            .flat_map(|shard| shard.segments().iter().cloned())
            .filter(|segment| segment.level() == Level::L1)
            .collect();
        found.sort_by(|a, b| a.id().cmp(b.id()));
        found.dedup();
        found
    }

    pub(super) async fn chunks_of(&self, segment: &SegmentEntry) -> ChunkTable {
        let file = segment
            .files()
            .iter()
            .find(|f| f.name().as_str().starts_with("chunks-"))
            .unwrap();
        let bytes = self
            .index
            .read_segment_file(segment.id(), file.name())
            .await
            .unwrap();
        ChunkTable::decode(bytes).unwrap()
    }

    pub(super) async fn search(&self, text: &str, allowed: &AllowedRows, limit: usize) -> Vec<Hit> {
        let source = SegmentSource::new(self.index.clone());
        let segments = self.segments().await;
        let bm25 = Bm25::new().unwrap();
        cosmosearch_core::Index::search(
            &bm25,
            &source,
            &segments,
            allowed,
            &Query::new(text),
            limit,
        )
        .await
        .unwrap()
        .into_hits()
    }

    pub(super) async fn keys_for(&self, text: &str, filter: &Filter, limit: usize) -> Vec<String> {
        let mut plugins = Plugins::new(extractors());
        plugins.add_index(bm25()).unwrap();
        let reader = Reader::open(
            self.data.clone(),
            self.metadata.clone(),
            self.source,
            plugins,
            ReaderOptions::new(),
        )
        .await
        .unwrap();
        let results = reader
            .search(
                &SearchRequest::new(Query::new(text))
                    .filter(filter.clone())
                    .limit(limit),
            )
            .await
            .unwrap();
        let mut keys: Vec<String> = Vec::new();
        for hit in results.hits() {
            let key = hit.source().location().key().to_owned();
            if !keys.contains(&key) {
                keys.push(key);
            }
        }
        keys
    }

    pub(super) async fn raw_hits(&self, text: &str) -> Vec<Hit> {
        let mut every = RowSet::new();
        every.insert(0..u64::MAX);
        let every = Arc::new(every);
        let mut allowed = AllowedRows::new();
        for segment in self.segments().await {
            allowed.allow(segment.id(), Arc::clone(&every));
        }
        self.search(text, &allowed, 50).await
    }

    pub(super) async fn live_keys_for(&self, text: &str) -> Vec<String> {
        let mut keys = self.keys_for(text, &Filter::new(), 50).await;
        keys.sort();
        keys
    }
}

#[tokio::test]
async fn a_pending_file_becomes_searchable_and_rerunning_does_nothing() {
    let f = fixture(FileSource::HostNamed).await;
    f.named_file("docs/a.txt", b"refund policy text").await;
    assert_eq!(f.status("docs/a.txt").await, FileStatus::Pending);

    assert!(matches!(
        f.index_everything().await,
        PublishOutcome::Published(_)
    ));
    assert_eq!(f.status("docs/a.txt").await, FileStatus::Ready);
    assert_eq!(f.live_keys_for("refund").await, ["docs/a.txt"]);

    let version = f.version().await;
    assert!(f.indexer.plan(10).await.unwrap().is_empty());
    assert_eq!(f.version().await, version);
}

#[tokio::test]
async fn a_plan_and_its_results_survive_being_stored_and_resumed() {
    let f = fixture(FileSource::HostNamed).await;
    f.named_file("docs/a.txt", b"refund policy text").await;
    let plan = f.indexer.plan(10).await.unwrap();
    let stored = serde_json::to_string(&plan).unwrap();

    let resumed: Plan = serde_json::from_str(&stored).unwrap();
    assert_eq!(resumed, plan);
    let results = f.extract_all(&resumed).await;
    let stored = serde_json::to_string(&results).unwrap();
    let resumed: Vec<ExtractedFile> = serde_json::from_str(&stored).unwrap();
    assert_eq!(resumed, results);

    let built = f.build(resumed).await;
    let stored = serde_json::to_string(&built).unwrap();
    let built = serde_json::from_str(&stored).unwrap();
    assert!(matches!(
        f.indexer.publish(&built).await.unwrap(),
        PublishOutcome::Published(_)
    ));
}

#[tokio::test]
async fn publishing_the_same_result_twice_does_nothing() {
    let f = fixture(FileSource::HostNamed).await;
    f.named_file("docs/a.txt", b"refund").await;
    let plan = f.indexer.plan(10).await.unwrap();
    let files = f.extract_all(&plan).await;
    let built = f.build(files.clone()).await;
    assert!(matches!(
        f.indexer.publish(&built).await.unwrap(),
        PublishOutcome::Published(_)
    ));
    let version = f.version().await;
    assert_eq!(
        f.indexer.publish(&built).await.unwrap(),
        PublishOutcome::AlreadyDone
    );

    let second_build = f.build(files).await;
    assert_eq!(
        f.indexer.publish(&second_build).await.unwrap(),
        PublishOutcome::AlreadyDone
    );
    assert_eq!(f.version().await, version);
    assert_eq!(f.segments().await.len(), 1);
}

#[tokio::test]
async fn a_job_from_before_destroy_and_create_is_refused() {
    let f = fixture(FileSource::HostNamed).await;
    f.named_file("docs/a.txt", b"refund").await;
    let plan = f.indexer.plan(10).await.unwrap();
    let job = plan.batches()[0].jobs()[0].clone();

    let other = fixture(FileSource::HostNamed).await;
    let err = other
        .indexer
        .extract(job, &extractors(), &limits())
        .await
        .unwrap_err();
    assert!(matches!(err, Error::WrongIndex { .. }), "{err}");

    let files = f.extract_all(&plan).await;
    let built = f.build(files).await;
    let err = other.indexer.publish(&built).await.unwrap_err();
    assert!(matches!(err, Error::WrongIndex { .. }), "{err}");
}

#[tokio::test]
async fn a_result_for_a_changed_file_is_stale_and_its_old_chunks_never_return() {
    let f = fixture(FileSource::HostNamed).await;
    f.named_file("docs/a.txt", b"original words").await;
    assert!(matches!(
        f.index_everything().await,
        PublishOutcome::Published(_)
    ));
    assert_eq!(f.live_keys_for("original").await, ["docs/a.txt"]);

    f.put("docs/a.txt", b"replacement words").await;
    let plan_before = {
        f.send(vec![FileChange::Upsert(FileUpsert::new("docs/a.txt"))])
            .await;
        f.indexer.plan(10).await.unwrap()
    };
    assert_eq!(f.status("docs/a.txt").await, FileStatus::Pending);
    assert_eq!(f.live_keys_for("original").await, Vec::<String>::new());

    let files = f.extract_all(&plan_before).await;
    assert!(matches!(
        f.indexer.publish(&f.build(files).await).await.unwrap(),
        PublishOutcome::Published(_)
    ));
    assert_eq!(f.live_keys_for("replacement").await, ["docs/a.txt"]);
    assert_eq!(f.live_keys_for("original").await, Vec::<String>::new());
}

#[tokio::test]
async fn an_old_event_for_a_replaced_version_indexes_nothing_and_leaves_the_file_pending() {
    let f = fixture(FileSource::HostNamed).await;
    f.named_file("docs/a.txt", b"first").await;
    let plan = f.indexer.plan(10).await.unwrap();
    f.put("docs/a.txt", b"second").await;
    let files = f.extract_all(&plan).await;
    assert_eq!(files[0].outcome(), &ExtractOutcome::SourceMoved);
    let built = f.indexer.build(&bm25(), files, vec![]).await.unwrap();
    assert_eq!(built, None);
    assert_eq!(f.status("docs/a.txt").await, FileStatus::Pending);
}

#[tokio::test]
async fn removal_and_a_job_give_the_right_result_in_both_orders() {
    let f = fixture(FileSource::HostNamed).await;
    f.named_file("docs/a.txt", b"refund").await;
    let plan = f.indexer.plan(10).await.unwrap();
    let files = f.extract_all(&plan).await;
    let built = f.build(files).await;
    f.send(vec![remove("docs/a.txt")]).await;
    assert_eq!(
        f.indexer.publish(&built).await.unwrap(),
        PublishOutcome::Stale(vec!["docs/a.txt".to_owned()])
    );
    assert_eq!(f.segments().await.len(), 0);

    let g = fixture(FileSource::HostNamed).await;
    g.named_file("docs/a.txt", b"refund").await;
    assert!(matches!(
        g.index_everything().await,
        PublishOutcome::Published(_)
    ));
    g.send(vec![remove("docs/a.txt")]).await;
    assert_eq!(g.live_keys_for("refund").await, Vec::<String>::new());
}

#[tokio::test]
async fn remove_then_add_again_never_matches_an_old_job() {
    let f = fixture(FileSource::HostNamed).await;
    f.named_file("docs/a.txt", b"refund").await;
    let plan = f.indexer.plan(10).await.unwrap();
    let files = f.extract_all(&plan).await;
    let built = f.build(files).await;
    f.send(vec![remove("docs/a.txt")]).await;
    f.send(vec![FileChange::Upsert(FileUpsert::new("docs/a.txt"))])
        .await;
    assert!(matches!(
        f.indexer.publish(&built).await.unwrap(),
        PublishOutcome::Stale(_)
    ));
    assert_eq!(f.status("docs/a.txt").await, FileStatus::Pending);
}

#[tokio::test]
async fn a_batch_under_constant_change_still_makes_progress() {
    let f = fixture(FileSource::HostNamed).await;
    f.named_file("docs/steady.txt", b"steady words").await;
    f.named_file("docs/busy.txt", b"busy words").await;
    let plan = f.indexer.plan(10).await.unwrap();
    let mut files = f.extract_all(&plan).await;
    f.put("docs/busy.txt", b"busy words again").await;
    f.send(vec![FileChange::Upsert(FileUpsert::new("docs/busy.txt"))])
        .await;

    let outcome = f
        .indexer
        .publish(&f.build(files.clone()).await)
        .await
        .unwrap();
    let PublishOutcome::Stale(stale) = outcome else {
        panic!("expected stale, got {outcome:?}");
    };
    assert_eq!(stale, ["docs/busy.txt"]);
    assert_eq!(f.status("docs/steady.txt").await, FileStatus::Pending);

    files.retain(|file| !stale.iter().any(|key| key == file.job().key()));
    assert!(matches!(
        f.indexer.publish(&f.build(files).await).await.unwrap(),
        PublishOutcome::Published(_)
    ));
    assert_eq!(f.status("docs/steady.txt").await, FileStatus::Ready);
    assert_eq!(f.status("docs/busy.txt").await, FileStatus::Pending);
}

#[tokio::test]
async fn racing_publishers_and_recorders_all_land() {
    let f = fixture(FileSource::HostNamed).await;
    for i in 0..4 {
        f.named_file(&format!("docs/{i}.txt"), b"shared words")
            .await;
    }
    let plan = f.indexer.plan(10).await.unwrap();
    let files = f.extract_all(&plan).await;
    let built = f.build(files).await;

    let publisher = f.indexer.clone();
    let publish = tokio::spawn(async move { publisher.publish(&built).await.unwrap() });
    let recorder = f.indexer.clone();
    let record = tokio::spawn(async move {
        recorder
            .apply(vec![add("docs/extra.txt", "e1")])
            .await
            .unwrap()
    });
    assert!(matches!(
        publish.await.unwrap(),
        PublishOutcome::Published(_)
    ));
    record.await.unwrap();
    assert_eq!(f.status("docs/0.txt").await, FileStatus::Ready);
    assert_eq!(f.status("docs/extra.txt").await, FileStatus::Pending);
}

#[tokio::test]
async fn a_rebuild_replaces_named_segments_once_and_only_if_they_remain() {
    let f = fixture(FileSource::HostNamed).await;
    f.named_file("docs/a.txt", b"refund").await;
    let plan = f.indexer.plan(10).await.unwrap();
    let files = f.extract_all(&plan).await;
    let first = f.build(files.clone()).await;
    let first_id = first.id().unwrap().clone();
    f.indexer.publish(&first).await.unwrap();

    let rebuild = f
        .indexer
        .build(&bm25(), files.clone(), vec![first_id.clone()])
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(
        f.indexer.publish(&rebuild).await.unwrap(),
        PublishOutcome::Published(_)
    ));
    let ids: Vec<_> = f.segments().await.iter().map(|s| s.id().clone()).collect();
    assert_eq!(ids, [rebuild.id().unwrap().clone()]);
    let version = f.version().await;
    assert_eq!(
        f.indexer.publish(&rebuild).await.unwrap(),
        PublishOutcome::AlreadyDone
    );

    let late = f
        .indexer
        .build(&bm25(), files, vec![first_id])
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        f.indexer.publish(&late).await.unwrap(),
        PublishOutcome::Superseded
    );
    assert_eq!(f.version().await, version);
}

#[tokio::test]
async fn a_failed_file_goes_into_no_segment_and_stays_pending() {
    let f = fixture(FileSource::HostNamed).await;
    f.put("docs/blank.txt", b"   ").await;
    f.send(vec![FileChange::Upsert(FileUpsert::new("docs/blank.txt"))])
        .await;
    let plan = f.indexer.plan(10).await.unwrap();
    let files = f.extract_all(&plan).await;
    assert_eq!(
        files[0].outcome(),
        &ExtractOutcome::Failed(ExtractFailure::NoText)
    );
    assert_eq!(f.indexer.build(&bm25(), files, vec![]).await.unwrap(), None);
    assert_eq!(f.status("docs/blank.txt").await, FileStatus::Pending);
}

#[tokio::test]
async fn a_batch_cannot_mix_shards() {
    let f = fixture(FileSource::HostNamed).await;
    split_into_two_shards(&f).await;
    f.named_file("docs/a.txt", b"refund").await;
    f.named_file("docs/z.txt", b"refund").await;
    let plan = f.indexer.plan(10).await.unwrap();
    assert_eq!(plan.batches().len(), 2);
    let files = f.extract_all(&plan).await;
    let err = f.indexer.build(&bm25(), files, vec![]).await.unwrap_err();
    assert!(matches!(err, Error::MixedShards { .. }), "{err}");
}

async fn split_into_two_shards(f: &Fixture) {
    let head = f.index.head().await.unwrap().unwrap();
    let split = [
        (
            ShardId::new("a").unwrap(),
            ShardEntry::new(
                KeyRange::new("", Some("docs/m".to_owned())).unwrap(),
                vec![],
            ),
        ),
        (
            ShardId::new("b").unwrap(),
            ShardEntry::new(KeyRange::new("docs/m", None).unwrap(), vec![]),
        ),
    ];
    let next = head
        .successor(
            CommitId::new("c2").unwrap(),
            split.into_iter().collect(),
            BTreeSet::new(),
        )
        .unwrap();
    f.index.commit(&next).await.unwrap();
}

fn big(key: &str, tag: &str, megabytes: u64) -> FileChange {
    FileChange::Upsert(
        FileUpsert::new(key).fields(StoreFields::new(megabytes * 1024 * 1024, etag(tag))),
    )
}

fn shard_ranges(head: &cosmosearch_core::Manifest) -> Vec<(String, Option<String>)> {
    let mut ranges: Vec<_> = head
        .shards()
        .values()
        .map(|s| {
            (
                s.range().start().to_owned(),
                s.range().end().map(str::to_owned),
            )
        })
        .collect();
    ranges.sort();
    ranges
}

#[tokio::test]
async fn a_shard_over_the_limit_splits_at_a_folder_near_its_middle() {
    let f = fixture(FileSource::HostNamed).await;
    let changes = ["a/1", "a/2", "a/3", "b/1", "b/2", "b/3"]
        .iter()
        .map(|name| big(&format!("docs/{name}"), "e1", 300))
        .collect();
    f.send(changes).await;
    let head = f.index.head().await.unwrap().unwrap();
    assert_eq!(
        shard_ranges(&head),
        [
            (String::new(), Some("docs/b/1".to_owned())),
            ("docs/b/1".to_owned(), None)
        ]
    );
    assert_eq!(f.indexer.files().await.unwrap().len(), 6);
}

#[tokio::test]
async fn a_host_set_shard_size_splits_sooner() {
    let f = fixture(FileSource::HostNamed).await;
    let small = Indexer::new(f.index.clone(), f.data.clone())
        .unwrap()
        .shard_bytes(NonZeroU64::new(250 * 1024 * 1024).unwrap());
    let changes: Vec<FileChange> = ["a/1", "a/2", "b/1", "b/2", "c/1", "c/2", "d/1", "d/2"]
        .iter()
        .map(|name| big(&format!("docs/{name}"), "e1", 100))
        .collect();
    f.send(changes.clone()).await;
    assert_eq!(
        f.index.head().await.unwrap().unwrap().shards().len(),
        1,
        "800 MB fits one shard of the default size"
    );
    small.apply(vec![big("docs/e/1", "e1", 100)]).await.unwrap();
    let head = f.index.head().await.unwrap().unwrap();
    assert_eq!(
        shard_ranges(&head).len(),
        5,
        "900 MB splits until every shard is within 250 MB"
    );
    assert_eq!(small.files().await.unwrap().len(), 9);
}

#[tokio::test]
async fn a_small_shard_does_not_split() {
    let f = fixture(FileSource::HostNamed).await;
    f.send(vec![big("docs/a", "e1", 900)]).await;
    assert_eq!(f.index.head().await.unwrap().unwrap().shards().len(), 1);
}

#[tokio::test]
async fn one_enormous_file_stays_whole() {
    let f = fixture(FileSource::HostNamed).await;
    f.send(vec![big("docs/a", "e1", 5_000)]).await;
    assert_eq!(f.index.head().await.unwrap().unwrap().shards().len(), 1);
}

#[tokio::test]
async fn children_of_a_split_share_the_segments_and_stay_searchable() {
    let f = fixture(FileSource::HostNamed).await;
    f.put("docs/a/1.txt", b"alpha words").await;
    f.put("docs/b/1.txt", b"beta words").await;
    let upserts = ["docs/a/1.txt", "docs/b/1.txt"]
        .into_iter()
        .map(|key| FileChange::Upsert(FileUpsert::new(key)))
        .collect();
    f.send(upserts).await;
    assert!(matches!(
        f.index_everything().await,
        PublishOutcome::Published(_)
    ));
    let before = f.segments().await;

    let growth = ["docs/a/2", "docs/a/3", "docs/b/2", "docs/b/3"]
        .iter()
        .map(|key| big(key, "e1", 400))
        .collect();
    f.send(growth).await;
    let head = f.index.head().await.unwrap().unwrap();
    assert_eq!(head.shards().len(), 2);
    assert_eq!(f.segments().await, before);
    for shard in head.shards().values() {
        assert!(shard.segments().iter().any(|s| s.id() == before[0].id()));
    }
    assert_eq!(f.status("docs/a/1.txt").await, FileStatus::Ready);
    assert_eq!(f.status("docs/b/1.txt").await, FileStatus::Ready);
    let ordinal = f.file("docs/b/1.txt").await.unwrap().ordinal();
    assert!(ordinal < 2);
}

#[tokio::test]
async fn an_unknown_publish_is_reported_and_publishing_again_is_safe() {
    let f = fixture(FileSource::HostNamed).await;
    f.named_file("docs/a.txt", b"refund").await;
    let plan = f.indexer.plan(10).await.unwrap();
    let built = f.build(f.extract_all(&plan).await).await;
    for _ in 0..3 {
        f.faults.inject(Op::PutCreate, "manifests/", Fault::Timeout);
    }
    let err = f.indexer.publish(&built).await.unwrap_err();
    assert!(matches!(err, Error::CommitUnknown), "{err}");
    assert!(matches!(
        f.indexer.publish(&built).await.unwrap(),
        PublishOutcome::Published(_)
    ));
}

#[tokio::test]
async fn a_named_media_type_chooses_the_extractor_and_changing_it_indexes_the_file_again() {
    let f = fixture(FileSource::HostNamed).await;
    f.put("docs/notes", b"refund policy text").await;
    let text = MediaType::new("text/plain").unwrap();
    let outcomes = f
        .send(vec![FileChange::Upsert(
            FileUpsert::new("docs/notes").media_type(text.clone()),
        )])
        .await;
    assert_eq!(outcomes, [ChangeOutcome::Applied]);
    f.index_everything().await;
    assert_eq!(f.status("docs/notes").await, FileStatus::Ready);
    assert_eq!(
        f.file("docs/notes").await.unwrap().media_type(),
        Some(&text)
    );
    assert_eq!(f.live_keys_for("refund").await, ["docs/notes"]);

    let outcomes = f
        .send(vec![FileChange::Upsert(
            FileUpsert::new("docs/notes").media_type(text),
        )])
        .await;
    assert_eq!(outcomes, [ChangeOutcome::Unchanged]);

    let outcomes = f
        .send(vec![FileChange::Upsert(FileUpsert::new("docs/notes"))])
        .await;
    assert_eq!(outcomes, [ChangeOutcome::Applied]);
    assert_eq!(f.status("docs/notes").await, FileStatus::Pending);
    let plan = f.indexer.plan(10).await.unwrap();
    let extracted = f.extract_all(&plan).await;
    assert!(matches!(
        extracted[0].outcome(),
        ExtractOutcome::Failed(ExtractFailure::NoExtractor)
    ));
}
