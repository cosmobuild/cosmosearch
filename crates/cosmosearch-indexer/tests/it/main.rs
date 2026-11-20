//! Contract tests for `cosmosearch-indexer`.

#![expect(
    clippy::panic,
    clippy::unwrap_used,
    reason = "helpers in a test binary fail the test by panicking"
)]

use std::collections::BTreeSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use async_trait::async_trait;
use bytes::Bytes;
use cosmosearch_core::{
    AttributeValue, ByteRange, CommitId, ETag, FileSource, KeyRange, Location, ManifestVersion,
    ShardEntry, ShardId,
};
use cosmosearch_format::{FileEntry, FileStatus};
use cosmosearch_indexer::{ChangeOutcome, Error, FileChange, FileUpsert, Indexer, StoreFields};
use cosmosearch_store::{
    BackendError, CommitOutcome, CreateOutcome, DataStore, Fault, FaultStore, IndexHandle,
    MemoryStore, MetadataStore, ObjectInfo, ObjectKey, Op, ReadStore, Result as StoreResult,
    WriteStore,
};

use futures_util::stream::{self, BoxStream};
use futures_util::{StreamExt, future};

#[derive(Debug, Clone)]
struct Spy {
    inner: MemoryStore,
    heads: Arc<AtomicUsize>,
    lists: Arc<AtomicUsize>,
    reads: Arc<AtomicUsize>,
    reads_fail: Arc<AtomicBool>,
}

#[async_trait]
impl ReadStore for Spy {
    async fn read_range(
        &self,
        key: &ObjectKey,
        range: ByteRange,
        expected: &ETag,
    ) -> StoreResult<Bytes> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        if self.reads_fail.load(Ordering::SeqCst) {
            return Err(cosmosearch_store::Error::Backend {
                key: key.to_string(),
                source: BackendError::new("the network is down"),
            });
        }
        self.inner.read_range(key, range, expected).await
    }

    async fn head(&self, key: &ObjectKey) -> StoreResult<ObjectInfo> {
        self.heads.fetch_add(1, Ordering::SeqCst);
        self.inner.head(key).await
    }

    fn list<'a>(
        &'a self,
        prefix: Option<&'a ObjectKey>,
        after: Option<&'a ObjectKey>,
    ) -> BoxStream<'a, StoreResult<ObjectInfo>> {
        self.lists.fetch_add(1, Ordering::SeqCst);
        self.inner.list(prefix, after)
    }
}

mod attributes;
mod collection;
mod jobs;
mod levels;
mod promotion;
mod removal;
mod s3;
mod status;

struct Fixture {
    store: MemoryStore,
    data: DataStore,
    metadata: MetadataStore,
    source: FileSource,
    faults: FaultStore,
    heads: Arc<AtomicUsize>,
    lists: Arc<AtomicUsize>,
    reads: Arc<AtomicUsize>,
    reads_fail: Arc<AtomicBool>,
    index: IndexHandle,
    indexer: Indexer,
}

fn data_location() -> Location {
    "s3://example-bucket/docs/".parse().unwrap()
}

async fn fixture(source: FileSource) -> Fixture {
    let store = MemoryStore::new();
    let faults = FaultStore::new(store.clone());
    let heads = Arc::new(AtomicUsize::new(0));
    let lists = Arc::new(AtomicUsize::new(0));
    let reads = Arc::new(AtomicUsize::new(0));
    let reads_fail = Arc::new(AtomicBool::new(false));
    let spy = Spy {
        inner: store.clone(),
        heads: Arc::clone(&heads),
        lists: Arc::clone(&lists),
        reads: Arc::clone(&reads),
        reads_fail: Arc::clone(&reads_fail),
    };
    let data = DataStore::new(spy, data_location()).unwrap();
    let metadata =
        MetadataStore::new(faults.clone(), "s3://example-bucket/index".parse().unwrap()).unwrap();
    let outcome = metadata.create(&data_location(), source).await.unwrap();
    let CreateOutcome::Created(index) = outcome else {
        panic!("expected created, got {outcome:?}");
    };
    let indexer = Indexer::new(index.clone(), data.clone()).unwrap();
    Fixture {
        store,
        data,
        metadata,
        source,
        faults,
        heads,
        lists,
        reads,
        reads_fail,
        index,
        indexer,
    }
}

impl Fixture {
    async fn version(&self) -> u64 {
        self.index.head().await.unwrap().unwrap().version().get()
    }

    async fn put(&self, key: &str, body: &'static [u8]) {
        let key = ObjectKey::new(key).unwrap();
        self.store
            .put_overwrite(&key, Bytes::from_static(body))
            .await
            .unwrap();
    }

    async fn file(&self, key: &str) -> Option<FileEntry> {
        let files = self.indexer.files().await.unwrap();
        files
            .into_iter()
            .map(|file| file.entry().clone())
            .find(|entry| entry.key() == key)
    }

    async fn send(&self, changes: Vec<FileChange>) -> Vec<ChangeOutcome> {
        self.indexer
            .apply(changes)
            .await
            .unwrap()
            .outcomes()
            .to_vec()
    }
}

fn etag(value: &str) -> ETag {
    ETag::new(value).unwrap()
}

fn upsert(key: &str, tag: &str) -> FileUpsert {
    FileUpsert::new(key).fields(StoreFields::new(5, etag(tag)))
}

fn add(key: &str, tag: &str) -> FileChange {
    FileChange::Upsert(upsert(key, tag))
}

fn remove(key: &str) -> FileChange {
    FileChange::Remove {
        key: key.to_owned(),
        expected: None,
    }
}

#[tokio::test]
async fn a_bulk_call_costs_one_version_and_never_touches_the_data() {
    let f = fixture(FileSource::HostNamed).await;
    let changes = (0..1_000)
        .map(|i| add(&format!("docs/{i:04}.txt"), "e1"))
        .collect();
    let applied = f.indexer.apply(changes).await.unwrap();
    assert_eq!(applied.version().map(ManifestVersion::get), Some(2));
    assert!(
        applied
            .outcomes()
            .iter()
            .all(|o| *o == ChangeOutcome::Applied)
    );
    assert_eq!(f.version().await, 2);
    assert_eq!(f.indexer.files().await.unwrap().len(), 1_000);
    assert_eq!(f.lists.load(Ordering::SeqCst), 0);
    assert_eq!(f.heads.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn reads_store_fields_with_one_head_only_when_the_host_has_none() {
    let f = fixture(FileSource::HostNamed).await;
    f.put("docs/real.txt", b"hello").await;
    let outcomes = f
        .send(vec![
            FileChange::Upsert(FileUpsert::new("docs/real.txt")),
            FileChange::Upsert(FileUpsert::new("docs/gone.txt")),
            add("docs/given.txt", "e1"),
        ])
        .await;
    assert_eq!(
        outcomes,
        [
            ChangeOutcome::Applied,
            ChangeOutcome::NotFound,
            ChangeOutcome::Applied
        ]
    );
    assert_eq!(f.heads.load(Ordering::SeqCst), 2);
    assert_eq!(f.file("docs/real.txt").await.unwrap().size(), 5);
    assert_eq!(f.lists.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn the_same_etag_changes_nothing() {
    let f = fixture(FileSource::HostNamed).await;
    f.send(vec![add("docs/a.txt", "e1")]).await;
    let applied = f
        .indexer
        .apply(vec![add("docs/a.txt", "e1")])
        .await
        .unwrap();
    assert_eq!(applied.version(), None);
    assert_eq!(applied.outcomes(), [ChangeOutcome::Unchanged]);
    assert_eq!(f.version().await, 2);
}

#[tokio::test]
async fn a_new_etag_moves_the_content_version_and_makes_the_file_pending() {
    let f = fixture(FileSource::HostNamed).await;
    f.send(vec![add("docs/a.txt", "e1")]).await;
    f.send(vec![add("docs/a.txt", "e2")]).await;
    let entry = f.file("docs/a.txt").await.unwrap();
    assert_eq!(entry.etag(), &etag("e2"));
    assert_eq!(entry.content_version().get(), 3);
    assert_eq!(entry.status(), FileStatus::Pending);
}

#[tokio::test]
async fn an_expected_version_fails_if_the_file_has_moved_on() {
    let f = fixture(FileSource::HostNamed).await;
    f.send(vec![add("docs/a.txt", "e1"), add("docs/b.txt", "e1")])
        .await;
    let at_two = ManifestVersion::new(2).unwrap();
    let wrong = ManifestVersion::new(9).unwrap();
    let outcomes = f
        .send(vec![
            FileChange::Upsert(upsert("docs/a.txt", "e2").expected(wrong)),
            FileChange::Upsert(upsert("docs/b.txt", "e2").expected(at_two)),
            FileChange::Upsert(upsert("docs/new.txt", "e1").expected(at_two)),
            FileChange::Remove {
                key: "docs/a.txt".to_owned(),
                expected: Some(wrong),
            },
        ])
        .await;
    assert_eq!(
        outcomes,
        [
            ChangeOutcome::Stale,
            ChangeOutcome::Applied,
            ChangeOutcome::Stale,
            ChangeOutcome::Stale
        ]
    );
    assert_eq!(f.file("docs/a.txt").await.unwrap().etag(), &etag("e1"));
    assert_eq!(f.file("docs/b.txt").await.unwrap().etag(), &etag("e2"));
}

#[tokio::test]
async fn removing_takes_the_file_out_and_a_new_add_gets_a_new_version() {
    let f = fixture(FileSource::HostNamed).await;
    f.send(vec![add("docs/a.txt", "e1")]).await;
    assert_eq!(
        f.send(vec![remove("docs/a.txt")]).await,
        [ChangeOutcome::Applied]
    );
    assert_eq!(f.file("docs/a.txt").await, None);
    assert_eq!(
        f.send(vec![remove("docs/a.txt")]).await,
        [ChangeOutcome::Unchanged]
    );
    f.send(vec![add("docs/a.txt", "e1")]).await;
    assert_eq!(
        f.file("docs/a.txt").await.unwrap().content_version().get(),
        4
    );
}

#[tokio::test]
async fn keeps_what_the_host_says_about_a_file() {
    let f = fixture(FileSource::HostNamed).await;
    let change = upsert("docs/a.txt", "e1")
        .display_name("Contract A")
        .attribute("owner", AttributeValue::One("legal".to_owned()));
    f.send(vec![FileChange::Upsert(change)]).await;
    let entry = f.file("docs/a.txt").await.unwrap();
    assert_eq!(entry.display_name(), Some("Contract A"));
    assert_eq!(
        entry.attributes().get("owner"),
        Some(&AttributeValue::One("legal".to_owned()))
    );
}

#[tokio::test]
async fn refuses_a_key_outside_the_data_location_and_records_nothing() {
    let f = fixture(FileSource::HostNamed).await;
    let err = f
        .indexer
        .apply(vec![add("docs/a.txt", "e1"), add("other/b.txt", "e1")])
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Store(_)), "{err}");
    assert_eq!(f.version().await, 1);
}

#[tokio::test]
async fn each_call_works_only_with_its_own_file_source() {
    let listing = fixture(FileSource::Listing).await;
    let err = listing
        .indexer
        .apply(vec![add("docs/a.txt", "v1")])
        .await
        .unwrap_err();
    assert!(matches!(err, Error::WrongFileSource { .. }), "{err}");
    let named = fixture(FileSource::HostNamed).await;
    let err = named.indexer.sync_listing().await.unwrap_err();
    assert!(matches!(err, Error::WrongFileSource { .. }), "{err}");
}

#[tokio::test]
async fn a_listing_records_additions_changes_and_removals_in_one_commit() {
    let f = fixture(FileSource::Listing).await;
    f.put("docs/a.txt", b"one").await;
    f.put("docs/b.txt", b"two").await;
    let first = f.indexer.sync_listing().await.unwrap();
    assert_eq!(first.version().map(ManifestVersion::get), Some(2));
    assert_eq!(first.outcomes().len(), 2);

    let unchanged = f.indexer.sync_listing().await.unwrap();
    assert_eq!(unchanged.version(), None);

    let old_etag = f.file("docs/a.txt").await.unwrap().etag().clone();
    f.put("docs/a.txt", b"changed").await;
    f.put("docs/c.txt", b"new").await;
    f.store
        .delete(&ObjectKey::new("docs/b.txt").unwrap())
        .await
        .unwrap();
    let second = f.indexer.sync_listing().await.unwrap();
    assert_eq!(second.version().map(ManifestVersion::get), Some(3));

    let keys: Vec<String> = f
        .indexer
        .files()
        .await
        .unwrap()
        .iter()
        .map(|file| file.entry().key().to_owned())
        .collect();
    assert_eq!(keys, ["docs/a.txt", "docs/c.txt"]);
    assert_ne!(f.file("docs/a.txt").await.unwrap().etag(), &old_etag);
}

#[derive(Debug, Clone)]
struct Paused {
    inner: MemoryStore,
    armed: Arc<AtomicBool>,
    listed: Arc<tokio::sync::Notify>,
    release: Arc<tokio::sync::Notify>,
}

#[async_trait]
impl ReadStore for Paused {
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
        let entries = self.inner.list(prefix, after);
        if !self.armed.swap(false, Ordering::SeqCst) {
            return entries;
        }
        let pause = stream::once(async move {
            self.listed.notify_one();
            self.release.notified().await;
            None
        });
        entries
            .map(Some)
            .chain(pause)
            .filter_map(future::ready)
            .boxed()
    }
}

async fn sync_overtaken<F: Future<Output = ()>>(f: &Fixture, meanwhile: F) {
    let paused = Paused {
        inner: f.store.clone(),
        armed: Arc::new(AtomicBool::new(true)),
        listed: Arc::default(),
        release: Arc::default(),
    };
    let data = DataStore::new(paused.clone(), data_location()).unwrap();
    let slow = Indexer::new(f.index.clone(), data).unwrap();
    let sync = tokio::spawn(async move { slow.sync_listing().await.unwrap() });
    paused.listed.notified().await;
    meanwhile.await;
    paused.release.notify_one();
    sync.await.unwrap();
}

#[tokio::test]
async fn an_older_listing_does_not_remove_a_file_a_newer_sync_added() {
    let f = fixture(FileSource::Listing).await;
    sync_overtaken(&f, async {
        f.put("docs/a.txt", b"one").await;
        f.indexer.sync_listing().await.unwrap();
    })
    .await;
    assert!(f.file("docs/a.txt").await.is_some());
}

#[tokio::test]
async fn an_older_listing_does_not_bring_back_an_old_version() {
    let f = fixture(FileSource::Listing).await;
    f.put("docs/a.txt", b"one").await;
    f.put("docs/b.txt", b"two").await;
    f.indexer.sync_listing().await.unwrap();
    f.put("docs/a.txt", b"changed").await;
    f.put("docs/new.txt", b"first").await;
    sync_overtaken(&f, async {
        f.put("docs/a.txt", b"changed again").await;
        f.put("docs/new.txt", b"second").await;
        f.indexer.sync_listing().await.unwrap();
    })
    .await;
    for key in ["docs/a.txt", "docs/new.txt"] {
        let head = f.data.head(&ObjectKey::new(key).unwrap()).await.unwrap();
        assert_eq!(f.file(key).await.unwrap().etag(), head.etag(), "{key}");
    }
    assert!(f.file("docs/b.txt").await.is_some());
}

#[tokio::test]
async fn an_older_listing_does_not_remove_a_file_added_again() {
    let f = fixture(FileSource::Listing).await;
    f.put("docs/a.txt", b"one").await;
    f.indexer.sync_listing().await.unwrap();
    let key = ObjectKey::new("docs/a.txt").unwrap();
    f.store.delete(&key).await.unwrap();
    sync_overtaken(&f, async {
        f.indexer.sync_listing().await.unwrap();
        f.put("docs/a.txt", b"back").await;
        f.indexer.sync_listing().await.unwrap();
    })
    .await;
    let head = f.data.head(&key).await.unwrap();
    assert_eq!(f.file("docs/a.txt").await.unwrap().etag(), head.etag());
}

#[tokio::test]
async fn a_metadata_location_inside_the_data_is_never_indexed() {
    let store = MemoryStore::new();
    let data = DataStore::new(store.clone(), data_location()).unwrap();
    let inside: Location = "s3://example-bucket/docs/.index/".parse().unwrap();
    let metadata = MetadataStore::new(store.clone(), inside).unwrap();
    let CreateOutcome::Created(index) = metadata
        .create(&data_location(), FileSource::Listing)
        .await
        .unwrap()
    else {
        panic!("the location is empty");
    };
    let indexer = Indexer::new(index, data).unwrap();
    let key = ObjectKey::new("docs/a.txt").unwrap();
    store
        .put_overwrite(&key, Bytes::from_static(b"alpha"))
        .await
        .unwrap();
    for _ in 0..3 {
        indexer.sync_listing().await.unwrap();
    }
    let keys: Vec<String> = indexer
        .files()
        .await
        .unwrap()
        .iter()
        .map(|file| file.entry().key().to_owned())
        .collect();
    assert_eq!(keys, ["docs/a.txt"]);
}

#[tokio::test]
async fn an_indexer_given_another_data_location_refuses_to_work() {
    let f = fixture(FileSource::Listing).await;
    let other = DataStore::new(
        f.store.clone(),
        "s3://example-bucket/other/".parse().unwrap(),
    )
    .unwrap();
    let indexer = Indexer::new(f.index.clone(), other).unwrap();
    let err = indexer.sync_listing().await.unwrap_err();
    assert!(
        matches!(
            err,
            Error::Store(cosmosearch_store::Error::WrongDataLocation { .. })
        ),
        "{err}"
    );
}

#[tokio::test]
async fn racing_writers_lose_no_changes() {
    let f = fixture(FileSource::HostNamed).await;
    let calls: Vec<_> = (0..6)
        .map(|i| {
            let indexer = f.indexer.clone();
            tokio::spawn(async move {
                indexer
                    .apply(vec![add(&format!("docs/{i}.txt"), "e1")])
                    .await
                    .unwrap()
                    .version()
                    .unwrap()
                    .get()
            })
        })
        .collect();
    let mut versions = BTreeSet::new();
    for call in calls {
        versions.insert(call.await.unwrap());
    }
    assert_eq!(versions.len(), 6, "{versions:?}");
    assert_eq!(f.indexer.files().await.unwrap().len(), 6);
}

#[tokio::test]
async fn an_unknown_commit_is_reported_and_sending_again_is_safe() {
    let f = fixture(FileSource::HostNamed).await;
    for _ in 0..3 {
        f.faults.inject(Op::PutCreate, "manifests/", Fault::Timeout);
    }
    let err = f
        .indexer
        .apply(vec![add("docs/a.txt", "e1")])
        .await
        .unwrap_err();
    assert!(matches!(err, Error::CommitUnknown), "{err}");
    assert_eq!(
        f.send(vec![add("docs/a.txt", "e1")]).await,
        [ChangeOutcome::Applied]
    );
    assert_eq!(f.indexer.files().await.unwrap().len(), 1);
}

#[tokio::test]
async fn only_the_shards_a_call_touches_are_rewritten() {
    let f = fixture(FileSource::HostNamed).await;
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
    assert_eq!(
        f.index.commit(&next).await.unwrap(),
        CommitOutcome::Committed(next.version())
    );

    f.send(vec![add("docs/a.txt", "e1"), add("docs/z.txt", "e1")])
        .await;
    let segments = |manifest: &cosmosearch_core::Manifest, shard: &str| {
        manifest.shards()[&ShardId::new(shard).unwrap()]
            .segments()
            .to_vec()
    };
    let before = f.index.head().await.unwrap().unwrap();
    f.send(vec![add("docs/a.txt", "e2")]).await;
    let after = f.index.head().await.unwrap().unwrap();
    assert_ne!(segments(&before, "a"), segments(&after, "a"));
    assert_eq!(segments(&before, "b"), segments(&after, "b"));
    assert_eq!(f.indexer.files().await.unwrap().len(), 2);
}

#[test]
fn recording_changes_is_send() {
    fn assert_send<T: Send>(_: &T) {}
    fn check(indexer: &Indexer) {
        assert_send(&indexer.apply(vec![]));
        assert_send(&indexer.sync_listing());
    }
    let _: fn(&Indexer) = check;
}
