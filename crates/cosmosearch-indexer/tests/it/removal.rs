use cosmosearch_chunk::Extractors;
use cosmosearch_core::ExtractLimits;
use cosmosearch_indexer::{ExtractedFile, FileJob, PublishOutcome, Rebuild};

use super::jobs::bm25;
use super::*;

impl Fixture {
    pub(super) async fn extract_jobs(&self, rebuild: &Rebuild) -> Vec<ExtractedFile> {
        let extractors = Extractors::builtin().unwrap();
        let limits = ExtractLimits::default();
        let mut found = Vec::new();
        for job in rebuild.jobs() {
            found.push(
                self.indexer
                    .extract(job.clone(), &extractors, &limits)
                    .await
                    .unwrap(),
            );
        }
        found
    }

    pub(super) async fn rebuild_all(&self) -> Vec<PublishOutcome> {
        let mut outcomes = Vec::new();
        for rebuild in self.indexer.plan_rebuilds().await.unwrap() {
            let files = self.extract_jobs(&rebuild).await;
            let built = self
                .indexer
                .build(&bm25(), files, rebuild.replaces().to_vec())
                .await
                .unwrap()
                .unwrap();
            outcomes.push(self.indexer.publish(&built).await.unwrap());
        }
        outcomes
    }

    async fn holds_file(&self, ordinal: u64, version: ManifestVersion) -> bool {
        for segment in self.segments().await {
            let chunks = self.chunks_of(&segment).await;
            if chunks.file_versions().contains(&(ordinal, version)) {
                return true;
            }
        }
        false
    }

    async fn rebuild_keys(&self) -> Vec<Vec<String>> {
        self.indexer
            .plan_rebuilds()
            .await
            .unwrap()
            .iter()
            .map(|rebuild| {
                rebuild
                    .jobs()
                    .iter()
                    .map(|job| job.key().to_owned())
                    .collect()
            })
            .collect()
    }
}

#[tokio::test]
async fn a_removed_file_leaves_results_at_the_commit_and_every_segment_after_a_rebuild() {
    let f = fixture(FileSource::HostNamed).await;
    f.named_file("docs/a.txt", b"zebracorn alpha").await;
    f.named_file("docs/b.txt", b"beta words").await;
    f.index_everything().await;
    let removed = f.file("docs/a.txt").await.unwrap();
    assert_eq!(f.rebuild_keys().await, Vec::<Vec<String>>::new());

    f.send(vec![remove("docs/a.txt")]).await;
    assert_eq!(f.live_keys_for("zebracorn").await, Vec::<String>::new());
    assert_eq!(f.raw_hits("zebracorn").await.len(), 1);
    assert_eq!(f.rebuild_keys().await, [["docs/b.txt"]]);

    assert!(matches!(
        f.rebuild_all().await.as_slice(),
        [PublishOutcome::Published(_)]
    ));
    assert_eq!(f.raw_hits("zebracorn").await.len(), 0);
    assert!(
        !f.holds_file(removed.ordinal(), removed.content_version())
            .await
    );
    assert_eq!(f.live_keys_for("beta").await, ["docs/b.txt"]);
    assert_eq!(f.status("docs/b.txt").await, FileStatus::Ready);
    assert_eq!(f.rebuild_keys().await, Vec::<Vec<String>>::new());
}

#[tokio::test]
async fn publishing_the_same_rebuild_twice_does_nothing() {
    let f = fixture(FileSource::HostNamed).await;
    f.named_file("docs/a.txt", b"zebracorn").await;
    f.named_file("docs/b.txt", b"beta").await;
    f.index_everything().await;
    f.send(vec![remove("docs/a.txt")]).await;
    let rebuild = f.indexer.plan_rebuilds().await.unwrap().remove(0);
    let files = f.extract_jobs(&rebuild).await;
    let built = f
        .indexer
        .build(&bm25(), files, rebuild.replaces().to_vec())
        .await
        .unwrap()
        .unwrap();

    assert!(matches!(
        f.indexer.publish(&built).await.unwrap(),
        PublishOutcome::Published(_)
    ));
    let version = f.version().await;
    assert_eq!(
        f.indexer.publish(&built).await.unwrap(),
        PublishOutcome::AlreadyDone
    );
    assert_eq!(f.version().await, version);
}

#[tokio::test]
async fn a_segment_with_no_live_file_left_is_taken_away_once() {
    let f = fixture(FileSource::HostNamed).await;
    f.named_file("docs/a.txt", b"zebracorn").await;
    f.index_everything().await;
    f.send(vec![remove("docs/a.txt")]).await;
    let rebuild = f.indexer.plan_rebuilds().await.unwrap().remove(0);
    assert_eq!(rebuild.jobs(), []);

    let built = f
        .indexer
        .build(&bm25(), vec![], rebuild.replaces().to_vec())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(built.id(), None);
    assert!(matches!(
        f.indexer.publish(&built).await.unwrap(),
        PublishOutcome::Published(_)
    ));
    assert_eq!(f.segments().await, Vec::new());
    let version = f.version().await;
    assert_eq!(
        f.indexer.publish(&built).await.unwrap(),
        PublishOutcome::Superseded
    );
    assert_eq!(f.version().await, version);
}

#[tokio::test]
async fn a_changed_file_has_its_old_text_rebuilt_away_and_stays_pending() {
    let f = fixture(FileSource::HostNamed).await;
    f.named_file("docs/a.txt", b"zebracorn first").await;
    f.named_file("docs/b.txt", b"beta words").await;
    f.index_everything().await;
    let old = f.file("docs/a.txt").await.unwrap();
    f.named_file("docs/a.txt", b"second version").await;

    assert_eq!(f.rebuild_keys().await, [["docs/b.txt"]]);
    f.rebuild_all().await;
    assert_eq!(f.raw_hits("zebracorn").await.len(), 0);
    assert!(!f.holds_file(old.ordinal(), old.content_version()).await);
    assert_eq!(f.status("docs/a.txt").await, FileStatus::Pending);
    assert_eq!(f.live_keys_for("beta").await, ["docs/b.txt"]);
}

#[tokio::test]
async fn a_rebuild_that_leaves_out_a_live_file_changes_nothing() {
    let f = fixture(FileSource::HostNamed).await;
    f.named_file("docs/a.txt", b"zebracorn").await;
    f.named_file("docs/b.txt", b"beta").await;
    f.named_file("docs/c.txt", b"gamma").await;
    f.index_everything().await;
    f.send(vec![remove("docs/a.txt")]).await;
    let rebuild = f.indexer.plan_rebuilds().await.unwrap().remove(0);
    let mut files = f.extract_jobs(&rebuild).await;
    files.retain(|file| file.job().key() != "docs/c.txt");
    let partial = f
        .indexer
        .build(&bm25(), files, rebuild.replaces().to_vec())
        .await
        .unwrap()
        .unwrap();
    let version = f.version().await;

    assert_eq!(
        f.indexer.publish(&partial).await.unwrap(),
        PublishOutcome::Incomplete(vec!["docs/c.txt".to_owned()])
    );
    assert_eq!(f.version().await, version);
    assert_eq!(f.live_keys_for("gamma").await, ["docs/c.txt"]);
}

#[tokio::test]
async fn a_file_removed_while_its_segment_is_rebuilt_makes_the_rebuild_stale() {
    let f = fixture(FileSource::HostNamed).await;
    f.named_file("docs/a.txt", b"zebracorn").await;
    f.named_file("docs/b.txt", b"beta").await;
    f.index_everything().await;
    f.send(vec![remove("docs/a.txt")]).await;
    let rebuild = f.indexer.plan_rebuilds().await.unwrap().remove(0);
    let files = f.extract_jobs(&rebuild).await;
    let built = f
        .indexer
        .build(&bm25(), files, rebuild.replaces().to_vec())
        .await
        .unwrap()
        .unwrap();
    f.send(vec![remove("docs/b.txt")]).await;

    assert_eq!(
        f.indexer.publish(&built).await.unwrap(),
        PublishOutcome::Stale(vec!["docs/b.txt".to_owned()])
    );
    f.rebuild_all().await;
    assert_eq!(f.segments().await, Vec::new());
    assert_eq!(f.raw_hits("beta").await.len(), 0);
}

#[tokio::test]
async fn a_segment_shared_by_split_shards_is_rebuilt_for_all_of_them_at_once() {
    let f = fixture(FileSource::HostNamed).await;
    let texts: [(&str, &'static [u8]); 3] = [
        ("docs/a/0.txt", b"gamma words"),
        ("docs/a/1.txt", b"zebracorn words"),
        ("docs/b/1.txt", b"beta words"),
    ];
    for (key, body) in texts {
        f.put(key, body).await;
    }
    let upserts = texts
        .iter()
        .map(|(key, _)| FileChange::Upsert(FileUpsert::new(*key)))
        .collect();
    f.send(upserts).await;
    f.index_everything().await;
    let growth = ["docs/a/2", "docs/a/3", "docs/b/2", "docs/b/3"]
        .iter()
        .map(|key| {
            FileChange::Upsert(
                FileUpsert::new(*key).fields(StoreFields::new(400 * 1024 * 1024, etag("e1"))),
            )
        })
        .collect();
    f.send(growth).await;
    assert_eq!(f.index.head().await.unwrap().unwrap().shards().len(), 2);

    f.send(vec![remove("docs/a/1.txt")]).await;
    assert_eq!(f.rebuild_keys().await, [["docs/a/0.txt", "docs/b/1.txt"]]);
    assert!(matches!(
        f.rebuild_all().await.as_slice(),
        [PublishOutcome::Published(_)]
    ));
    assert_eq!(f.raw_hits("zebracorn").await.len(), 0);
    assert_eq!(f.live_keys_for("gamma").await, ["docs/a/0.txt"]);
    assert_eq!(f.live_keys_for("beta").await, ["docs/b/1.txt"]);
    let head = f.index.head().await.unwrap().unwrap();
    let rebuilt = f.segments().await;
    assert_eq!(rebuilt.len(), 1);
    for shard in head.shards().values() {
        assert!(shard.segments().iter().any(|s| s.id() == rebuilt[0].id()));
    }
}

#[tokio::test]
async fn a_shard_with_too_many_segments_is_merged_into_one() {
    let f = fixture(FileSource::HostNamed).await;
    for (key, body) in [
        ("docs/a.txt", b"alpha words" as &'static [u8]),
        ("docs/b.txt", b"beta words"),
        ("docs/c.txt", b"gamma words"),
    ] {
        f.named_file(key, body).await;
        f.index_everything().await;
    }
    f.send(vec![remove("docs/a.txt")]).await;
    assert_eq!(f.segments().await.len(), 3);
    assert_eq!(f.indexer.plan_merges(3).await.unwrap(), []);

    let merges = f.indexer.plan_merges(2).await.unwrap();
    let [merge] = merges.as_slice() else {
        panic!("one merge for the one shard: {merges:?}");
    };
    assert_eq!(merge.replaces().len(), 3);
    let keys: Vec<&str> = merge.jobs().iter().map(FileJob::key).collect();
    assert_eq!(keys, ["docs/b.txt", "docs/c.txt"], "only live files");

    let files = f.extract_jobs(merge).await;
    let built = f
        .indexer
        .build(&bm25(), files, merge.replaces().to_vec())
        .await
        .unwrap()
        .unwrap();
    let outcome = f.indexer.publish(&built).await.unwrap();
    assert!(
        matches!(outcome, PublishOutcome::Published(_)),
        "{outcome:?}"
    );
    assert_eq!(f.segments().await.len(), 1);
    assert_eq!(f.live_keys_for("words").await, ["docs/b.txt", "docs/c.txt"]);
    assert_eq!(f.raw_hits("alpha").await.len(), 0);
    assert_eq!(f.indexer.plan_merges(1).await.unwrap(), []);
    assert_eq!(
        f.indexer.publish(&built).await.unwrap(),
        PublishOutcome::AlreadyDone
    );
}
