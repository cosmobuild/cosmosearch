use std::time::{Duration, SystemTime};

use cosmosearch_core::{ChunkHash, Level, ModelId, Query};
use cosmosearch_indexer::PublishOutcome;
use cosmosearch_query::{Plugins, Reader, ReaderOptions, SearchRequest};
use cosmosearch_store::CollectionSettings;
use futures_util::TryStreamExt;

use super::jobs::bm25;
use super::levels::vectors;
use super::*;

const RETENTION: Duration = Duration::from_secs(100);
const GRACE: Duration = Duration::from_secs(50);

fn at(seconds: u64) -> SystemTime {
    SystemTime::UNIX_EPOCH + Duration::from_secs(1_000_000 + seconds)
}

fn settings() -> CollectionSettings {
    CollectionSettings::new(RETENTION, GRACE)
}

impl Fixture {
    async fn holding(&self, needle: &[u8]) -> Vec<String> {
        let prefix = ObjectKey::new("index").unwrap();
        let listed: Vec<ObjectInfo> = self
            .store
            .list(Some(&prefix), None)
            .try_collect()
            .await
            .unwrap();
        let mut found = Vec::new();
        for info in listed {
            let body = self.store.get(info.key()).await.unwrap();
            if body.windows(needle.len()).any(|window| window == needle) {
                found.push(info.key().to_string());
            }
        }
        found
    }

    async fn collect(&self, now: SystemTime) {
        self.metadata
            .collect_garbage(&settings(), now)
            .await
            .unwrap();
    }

    async fn hashes_of(&self, level: Level) -> BTreeSet<ChunkHash> {
        let mut hashes = BTreeSet::new();
        for segment in self.segments_at(level).await {
            let chunks = self.chunks_of(&segment).await;
            hashes.extend((0..chunks.len()).map(|row| chunks.get(row).unwrap().hash()));
        }
        hashes
    }

    async fn reader(&self) -> Reader {
        let mut plugins = Plugins::new(cosmosearch_chunk::Extractors::builtin().unwrap());
        plugins.add_index(bm25()).unwrap();
        Reader::open(
            self.data.clone(),
            self.metadata.clone(),
            self.source,
            plugins,
            ReaderOptions::new(),
        )
        .await
        .unwrap()
    }
}

async fn removed_and_rebuilt() -> Fixture {
    let f = fixture(FileSource::HostNamed).await;
    f.store.set_time(at(0));
    f.named_file("docs/a.txt", b"zebracorn alpha").await;
    f.named_file("docs/b.txt", b"beta words").await;
    f.index_everything().await;
    f.store.set_time(at(10));
    f.send(vec![remove("docs/a.txt")]).await;
    f.store.set_time(at(20));
    assert!(
        matches!(
            f.rebuild_all().await.as_slice(),
            [PublishOutcome::Published(_)]
        ),
        "the shard is rebuilt without the removed file"
    );
    f
}

#[tokio::test]
async fn a_removed_files_text_is_gone_after_rebuild_and_collection_and_not_before() {
    let f = removed_and_rebuilt().await;
    assert_ne!(f.holding(b"zebracorn").await, Vec::<String>::new());

    f.collect(at(60)).await;
    assert_ne!(f.holding(b"zebracorn").await, Vec::<String>::new());
    assert_ne!(f.holding(b"docs/a.txt").await, Vec::<String>::new());

    f.collect(at(20 + 100 + 1)).await;
    assert_eq!(f.holding(b"zebracorn").await, Vec::<String>::new());
    assert_eq!(f.holding(b"docs/a.txt").await, Vec::<String>::new());
    assert_eq!(f.live_keys_for("beta").await, ["docs/b.txt"]);
}

#[tokio::test]
async fn a_reader_open_on_an_old_version_keeps_answering() {
    let f = fixture(FileSource::HostNamed).await;
    f.store.set_time(at(0));
    f.named_file("docs/a.txt", b"zebracorn alpha").await;
    f.named_file("docs/b.txt", b"beta words").await;
    f.index_everything().await;
    let reader = f.reader().await;
    let opened = reader.manifest().version();
    f.store.set_time(at(10));
    f.send(vec![remove("docs/a.txt")]).await;
    f.store.set_time(at(20));
    f.rebuild_all().await;
    f.collect(at(1_000)).await;

    let results = reader
        .search(&SearchRequest::new(Query::new("beta")))
        .await
        .unwrap();
    assert!(results.version() > opened);
    let keys: Vec<&str> = results
        .hits()
        .iter()
        .map(|hit| hit.source().location().key())
        .collect();
    assert_eq!(keys, ["docs/b.txt"]);
    assert_eq!(results.coverage().unread().len(), 0);
}

#[tokio::test]
async fn live_vectors_follow_the_retained_manifests() {
    let f = fixture(FileSource::HostNamed).await;
    f.store.set_time(at(0));
    f.ready_files(&[
        ("docs/a.txt", b"zebracorn alpha"),
        ("docs/b.txt", b"beta words"),
    ])
    .await;
    let model = vectors("m");
    assert!(matches!(
        f.promote(&model).await,
        PublishOutcome::Published(_)
    ));
    let before = f.hashes_of(Level::L1).await;
    let live = f.indexer.live_vectors(RETENTION, at(0)).await.unwrap();
    assert_eq!(
        live.keys().collect::<Vec<_>>(),
        [&ModelId::new("m").unwrap()]
    );
    assert_eq!(live.values().next().unwrap(), &before);

    f.store.set_time(at(10));
    f.send(vec![remove("docs/a.txt")]).await;
    assert!(matches!(
        f.promote(&model).await,
        PublishOutcome::Published(_)
    ));
    f.rebuild_all().await;
    let after = f.hashes_of(Level::L1).await;
    assert!(after.len() < before.len());

    let live = f.indexer.live_vectors(RETENTION, at(50)).await.unwrap();
    assert_eq!(live.values().next().unwrap(), &before);
    let live = f.indexer.live_vectors(RETENTION, at(1_000)).await.unwrap();
    assert_eq!(live.values().next().unwrap(), &after);
}

#[tokio::test]
async fn a_segment_published_after_collection_took_its_files_changes_nothing() {
    let f = fixture(FileSource::HostNamed).await;
    f.store.set_time(at(0));
    f.named_file("docs/a.txt", b"zebracorn alpha").await;
    let plan = f.indexer.plan(1_000).await.unwrap();
    let files = f.extract_all(&plan).await;
    let built = f.build(files).await;
    let version = f.version().await;

    f.collect(at(100)).await;

    assert_eq!(
        f.indexer.publish(&built).await.unwrap(),
        PublishOutcome::Expired
    );
    assert_eq!(f.version().await, version);
    assert_eq!(f.status("docs/a.txt").await, FileStatus::Pending);
}
