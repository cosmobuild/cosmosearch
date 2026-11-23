use cosmosearch_chunk::Extractors;
use cosmosearch_core::{ExtractLimits, Level, SegmentEntry, SegmentId, SharedIndex};
use cosmosearch_indexer::{BuiltSegment, ExtractedFile, FileJob, Promotion, PublishOutcome};

use super::jobs::bm25;
use super::levels::vectors;
use super::*;

fn shard() -> ShardId {
    ShardId::new("0").unwrap()
}

impl Fixture {
    async fn extract_promotion(&self, promotion: &Promotion) -> Vec<ExtractedFile> {
        let extractors = Extractors::builtin().unwrap();
        let mut found = Vec::new();
        for job in promotion.jobs() {
            found.push(
                self.indexer
                    .extract(job.clone(), &extractors, &ExtractLimits::default())
                    .await
                    .unwrap(),
            );
        }
        found
    }

    pub(super) async fn build_promotion(
        &self,
        index: &SharedIndex,
        promotion: &Promotion,
    ) -> BuiltSegment {
        let files = self.extract_promotion(promotion).await;
        self.indexer
            .build_promotion(index, promotion, files)
            .await
            .unwrap()
            .unwrap()
    }

    pub(super) async fn promote(&self, index: &SharedIndex) -> PublishOutcome {
        let promotion = self.indexer.plan_promotion(index, &shard()).await.unwrap();
        let built = self.build_promotion(index, &promotion).await;
        self.indexer.publish(&built).await.unwrap()
    }

    pub(super) async fn ready_files(&self, keys: &[(&str, &'static [u8])]) {
        for (key, body) in keys {
            self.named_file(key, body).await;
        }
        assert!(
            matches!(self.index_everything().await, PublishOutcome::Published(_)),
            "the first segment is published"
        );
    }

    async fn keys_in(&self, segment: &SegmentEntry) -> Vec<u64> {
        let chunks = self.chunks_of(segment).await;
        chunks
            .file_versions()
            .iter()
            .map(|(ordinal, _)| *ordinal)
            .collect()
    }
}

fn job_keys(promotion: &Promotion) -> Vec<&str> {
    promotion.jobs().iter().map(FileJob::key).collect()
}

#[tokio::test]
async fn a_promotion_holds_every_ready_file_and_leaves_l1_and_status_alone() {
    let f = fixture(FileSource::HostNamed).await;
    f.ready_files(&[
        ("docs/a.txt", b"alpha words"),
        ("docs/b.txt", b"beta words"),
    ])
    .await;
    f.named_file("docs/c.txt", b"pending words").await;
    let index = vectors("m1");

    let promotion = f.indexer.plan_promotion(&index, &shard()).await.unwrap();
    assert_eq!(job_keys(&promotion), ["docs/a.txt", "docs/b.txt"]);
    assert_eq!(promotion.replaces(), []);
    assert_eq!(promotion.kind().as_str(), "vec-m1");
    let lexical = f.segments_at(Level::L1).await;

    let built = f.build_promotion(&index, &promotion).await;
    assert_eq!(built.usage().tokens(), 2);
    assert!(matches!(
        f.indexer.publish(&built).await.unwrap(),
        PublishOutcome::Published(_)
    ));
    let promoted = f.segments_at(Level::L2).await;
    assert_eq!(promoted.len(), 1);
    assert_eq!(f.keys_in(&promoted[0]).await.len(), 2);
    assert_eq!(f.segments_at(Level::L1).await, lexical);
    assert_eq!(f.status("docs/a.txt").await, FileStatus::Ready);
    assert_eq!(f.status("docs/c.txt").await, FileStatus::Pending);

    let version = f.version().await;
    assert_eq!(
        f.indexer.publish(&built).await.unwrap(),
        PublishOutcome::AlreadyDone
    );
    assert_eq!(f.version().await, version);
}

#[tokio::test]
async fn promoting_again_replaces_the_segments_of_that_plugin_only() {
    let f = fixture(FileSource::HostNamed).await;
    f.ready_files(&[("docs/a.txt", b"alpha words")]).await;
    f.promote(&vectors("m1")).await;
    f.promote(&vectors("m2")).await;
    let before = f.segments_at(Level::L2).await;
    assert_eq!(before.len(), 2);

    f.named_file("docs/b.txt", b"beta words").await;
    f.index_everything().await;
    let promotion = f
        .indexer
        .plan_promotion(&vectors("m1"), &shard())
        .await
        .unwrap();
    let m1: Vec<&SegmentEntry> = before
        .iter()
        .filter(|segment| segment.model_id().unwrap().as_str() == "m1")
        .collect();
    assert_eq!(promotion.replaces(), [m1[0].id().clone()]);
    assert_eq!(job_keys(&promotion), ["docs/a.txt", "docs/b.txt"]);
    let built = f.build_promotion(&vectors("m1"), &promotion).await;
    assert!(matches!(
        f.indexer.publish(&built).await.unwrap(),
        PublishOutcome::Published(_)
    ));

    let after = f.segments_at(Level::L2).await;
    assert_eq!(after.len(), 2);
    assert!(!after.iter().any(|segment| segment.id() == m1[0].id()));
    assert!(after.contains(built.entry().unwrap()));
    assert_eq!(
        f.indexer.publish(&built).await.unwrap(),
        PublishOutcome::AlreadyDone
    );
}

#[tokio::test]
async fn a_promotion_does_not_commit_once_another_segment_of_its_plugin_appears() {
    let f = fixture(FileSource::HostNamed).await;
    f.ready_files(&[
        ("docs/a.txt", b"alpha words"),
        ("docs/b.txt", b"beta words"),
    ])
    .await;
    let index = vectors("m1");
    let first = f.indexer.plan_promotion(&index, &shard()).await.unwrap();
    let second = f.indexer.plan_promotion(&index, &shard()).await.unwrap();
    let eager = f
        .indexer
        .plan_embedding(&index, &["docs/a.txt".to_owned()])
        .await
        .unwrap();
    let first = f.build_promotion(&index, &first).await;
    let second = f.build_promotion(&index, &second).await;
    let eager = f.build_promotion(&index, &eager[0]).await;

    assert!(matches!(
        f.indexer.publish(&first).await.unwrap(),
        PublishOutcome::Published(_)
    ));
    let version = f.version().await;
    for late in [&second, &eager] {
        assert_eq!(
            f.indexer.publish(late).await.unwrap(),
            PublishOutcome::Superseded
        );
    }
    assert_eq!(f.version().await, version);
    assert_eq!(f.segments_at(Level::L2).await.len(), 1);
}

#[tokio::test]
async fn a_promotion_is_superseded_when_a_segment_it_replaces_is_rebuilt() {
    let f = fixture(FileSource::HostNamed).await;
    f.ready_files(&[
        ("docs/a.txt", b"alpha words"),
        ("docs/b.txt", b"beta words"),
    ])
    .await;
    let index = vectors("m1");
    f.promote(&index).await;
    let promotion = f.indexer.plan_promotion(&index, &shard()).await.unwrap();
    let built = f.build_promotion(&index, &promotion).await;

    f.send(vec![remove("docs/a.txt")]).await;
    for rebuild in f.indexer.plan_rebuilds().await.unwrap() {
        let plugin = if rebuild.kind().as_str() == "bm25" {
            bm25()
        } else {
            index.clone()
        };
        let files = f.extract_jobs(&rebuild).await;
        let rebuilt = f
            .indexer
            .build(&plugin, files, rebuild.replaces().to_vec())
            .await
            .unwrap()
            .unwrap();
        f.indexer.publish(&rebuilt).await.unwrap();
    }
    assert_eq!(
        f.indexer.publish(&built).await.unwrap(),
        PublishOutcome::Superseded
    );
}

#[tokio::test]
async fn eager_embedding_takes_only_named_ready_files_not_held_yet() {
    let f = fixture(FileSource::HostNamed).await;
    f.ready_files(&[
        ("docs/a.txt", b"alpha words"),
        ("docs/b.txt", b"beta words"),
        ("docs/c.txt", b"gamma words"),
    ])
    .await;
    f.named_file("docs/d.txt", b"pending words").await;
    let index = vectors("m1");
    let named: Vec<String> = ["docs/a.txt", "docs/d.txt", "docs/none.txt"]
        .iter()
        .map(|key| (*key).to_owned())
        .collect();

    let planned = f.indexer.plan_embedding(&index, &named).await.unwrap();
    assert_eq!(planned.len(), 1);
    assert_eq!(job_keys(&planned[0]), ["docs/a.txt"]);
    assert_eq!(planned[0].replaces(), []);
    let built = f.build_promotion(&index, &planned[0]).await;
    assert!(matches!(
        f.indexer.publish(&built).await.unwrap(),
        PublishOutcome::Published(_)
    ));
    assert_eq!(
        f.indexer.plan_embedding(&index, &named).await.unwrap(),
        Vec::<Promotion>::new()
    );

    let more = ["docs/a.txt".to_owned(), "docs/b.txt".to_owned()];
    let planned = f.indexer.plan_embedding(&index, &more).await.unwrap();
    assert_eq!(job_keys(&planned[0]), ["docs/b.txt"]);
    let built = f.build_promotion(&index, &planned[0]).await;
    f.indexer.publish(&built).await.unwrap();
    assert_eq!(f.segments_at(Level::L2).await.len(), 2);

    let promotion = f.indexer.plan_promotion(&index, &shard()).await.unwrap();
    assert_eq!(promotion.replaces().len(), 2);
    assert_eq!(
        job_keys(&promotion),
        ["docs/a.txt", "docs/b.txt", "docs/c.txt"]
    );
    f.indexer
        .publish(&f.build_promotion(&index, &promotion).await)
        .await
        .unwrap();
    let promoted = f.segments_at(Level::L2).await;
    assert_eq!(promoted.len(), 1);
    assert_eq!(f.keys_in(&promoted[0]).await.len(), 3);
}

#[tokio::test]
async fn a_promotion_needs_a_plugin_above_l1_and_the_plugin_it_was_planned_with() {
    let f = fixture(FileSource::HostNamed).await;
    f.ready_files(&[("docs/a.txt", b"alpha words")]).await;
    let err = f
        .indexer
        .plan_promotion(&bm25(), &shard())
        .await
        .unwrap_err();
    assert!(
        matches!(err, Error::WrongLevel { level: Level::L1 }),
        "{err}"
    );
    let err = f
        .indexer
        .plan_embedding(&bm25(), &["docs/a.txt".to_owned()])
        .await
        .unwrap_err();
    assert!(matches!(err, Error::WrongLevel { .. }), "{err}");
    let err = f
        .indexer
        .plan_promotion(&vectors("m1"), &ShardId::new("elsewhere").unwrap())
        .await
        .unwrap_err();
    assert!(matches!(err, Error::UnknownShard { .. }), "{err}");

    let promotion = f
        .indexer
        .plan_promotion(&vectors("m1"), &shard())
        .await
        .unwrap();
    let files = f.extract_promotion(&promotion).await;
    let err = f
        .indexer
        .build_promotion(&vectors("m2"), &promotion, files)
        .await
        .unwrap_err();
    assert!(
        matches!(&err, Error::WrongPlugin { kind } if kind.as_str() == "vec-m1"),
        "{err}"
    );
}

#[tokio::test]
async fn a_promotion_survives_being_stored_and_resumed() {
    let f = fixture(FileSource::HostNamed).await;
    f.ready_files(&[("docs/a.txt", b"alpha words")]).await;
    let index = vectors("m1");
    let promotion = f.indexer.plan_promotion(&index, &shard()).await.unwrap();
    let stored = serde_json::to_string(&promotion).unwrap();
    let resumed: Promotion = serde_json::from_str(&stored).unwrap();
    assert_eq!(resumed, promotion);

    let built = f.build_promotion(&index, &resumed).await;
    let stored = serde_json::to_string(&built).unwrap();
    let resumed: BuiltSegment = serde_json::from_str(&stored).unwrap();
    assert!(matches!(
        f.indexer.publish(&resumed).await.unwrap(),
        PublishOutcome::Published(_)
    ));
    assert_eq!(
        f.indexer.publish(&built).await.unwrap(),
        PublishOutcome::AlreadyDone
    );
}

#[tokio::test]
async fn a_promotion_replacing_a_segment_split_shards_share_holds_their_files_too() {
    let f = fixture(FileSource::HostNamed).await;
    f.ready_files(&[
        ("docs/a.txt", b"alpha words"),
        ("docs/b.txt", b"beta words"),
    ])
    .await;
    let index = vectors("m1");
    f.promote(&index).await;
    let promoted = f.segments_at(Level::L2).await.remove(0);
    let large = FileUpsert::new("docs/am.bin").fields(StoreFields::new(2 << 30, etag("e1")));
    f.send(vec![FileChange::Upsert(large)]).await;
    let head = f.index.head().await.unwrap().unwrap();
    let (a_shard, _) = head.shard_for("docs/a.txt").unwrap();
    let (b_shard, _) = head.shard_for("docs/b.txt").unwrap();
    assert_ne!(a_shard, b_shard, "the large file splits the shard");

    let promotion = f.indexer.plan_promotion(&index, a_shard).await.unwrap();
    assert_eq!(promotion.replaces(), [promoted.id().clone()]);
    assert_eq!(job_keys(&promotion), ["docs/a.txt", "docs/b.txt"]);
    let built = f.build_promotion(&index, &promotion).await;
    assert!(matches!(
        f.indexer.publish(&built).await.unwrap(),
        PublishOutcome::Published(_)
    ));
    let head = f.index.head().await.unwrap().unwrap();
    for key in ["docs/a.txt", "docs/b.txt"] {
        let (_, shard) = head.shard_for(key).unwrap();
        let promoted: Vec<&SegmentId> = shard
            .segments()
            .iter()
            .filter(|segment| segment.level() == Level::L2)
            .map(SegmentEntry::id)
            .collect();
        assert_eq!(promoted, [built.id().unwrap()], "{key}");
    }
}

impl Fixture {
    async fn level(&self, key: &str) -> Option<Level> {
        let one = self.indexer.file(key).await.unwrap().unwrap().level();
        let all = self.indexer.files().await.unwrap();
        let listed = all.iter().find(|file| file.entry().key() == key).unwrap();
        assert_eq!(listed.level(), one, "{key}");
        one
    }
}

#[tokio::test]
async fn a_promoted_file_reports_l2_and_a_file_left_out_stays_l1() {
    let f = fixture(FileSource::HostNamed).await;
    f.ready_files(&[("docs/a.txt", b"alpha words")]).await;
    f.named_file("docs/b.txt", b"beta words").await;
    assert_eq!(f.level("docs/a.txt").await, Some(Level::L1));

    f.promote(&vectors("m1")).await;
    f.index_everything().await;
    assert_eq!(f.level("docs/a.txt").await, Some(Level::L2));
    assert_eq!(f.level("docs/b.txt").await, Some(Level::L1));
}

#[tokio::test]
async fn a_file_whose_content_moved_on_after_promotion_does_not_report_l2() {
    let f = fixture(FileSource::HostNamed).await;
    f.ready_files(&[
        ("docs/a.txt", b"alpha words"),
        ("docs/b.txt", b"beta words"),
    ])
    .await;
    f.promote(&vectors("m1")).await;
    let promoted = f.segments_at(Level::L2).await;

    f.named_file("docs/a.txt", b"alpha words changed").await;
    assert_eq!(f.level("docs/a.txt").await, None);
    f.index_everything().await;
    assert_eq!(f.segments_at(Level::L2).await, promoted);
    assert_eq!(f.level("docs/a.txt").await, Some(Level::L1));
    assert_eq!(f.level("docs/b.txt").await, Some(Level::L2));
}

#[tokio::test]
async fn planning_and_file_lookups_refuse_keys_outside_the_data_location() {
    let f = fixture(FileSource::HostNamed).await;
    f.ready_files(&[("docs/a.txt", b"alpha words")]).await;
    let err = f
        .indexer
        .plan_embedding(&vectors("m1"), &["a.txt".to_owned()])
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Store(_)), "{err}");
    let err = f.indexer.file("a.txt").await.unwrap_err();
    assert!(matches!(err, Error::Store(_)), "{err}");
    assert!(f.indexer.file("docs/a.txt").await.unwrap().is_some());
    assert_eq!(
        f.indexer
            .plan_embedding(&vectors("m1"), &["docs/a.txt".to_owned()])
            .await
            .unwrap()
            .len(),
        1
    );
}
