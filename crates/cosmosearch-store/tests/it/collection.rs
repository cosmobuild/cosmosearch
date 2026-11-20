use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use async_trait::async_trait;
use bytes::Bytes;
use cosmosearch_core::{
    BuiltBy, ByteRange, ETag, IndexId, IndexRecord, Level, Manifest, ManifestObject,
    ManifestVersion, SegmentEntry, SegmentFile, SegmentFileName, SegmentId, ShardEntry,
};
use cosmosearch_store::{
    BackendError, CollectionReport, CollectionSettings, CommitOutcome, Error, FaultStore,
    IndexHandle, MemoryStore, MetadataStore, ObjectInfo, ObjectKey, ReadStore, Result, WriteStore,
    new_commit_id,
};
use futures_util::future::{self, BoxFuture};
use futures_util::stream::{self, BoxStream};
use futures_util::{FutureExt, StreamExt, TryStreamExt};
use tokio::task::JoinSet;

use crate::support::{create, fixture, head, metadata_location};

const RETENTION: Duration = Duration::from_secs(100);
const GRACE: Duration = Duration::from_secs(50);
const OTHER_INDEX: &str = "0123456789abcdef0123456789abcdef";

fn at(seconds: u64) -> SystemTime {
    SystemTime::UNIX_EPOCH + Duration::from_secs(1_000_000 + seconds)
}

fn settings() -> CollectionSettings {
    CollectionSettings::new(RETENTION, GRACE)
}

fn key(name: &str) -> ObjectKey {
    ObjectKey::new(name).unwrap()
}

async fn write_segment(index: &IndexHandle, name: &str) -> SegmentEntry {
    let id = SegmentId::new(name).unwrap();
    let file = SegmentFileName::new(format!("meta-{name}.parquet")).unwrap();
    index
        .write_segment_file(&id, &file, Bytes::from_static(b"table"))
        .await
        .unwrap();
    SegmentEntry::new(
        id,
        Level::L0,
        BuiltBy::new(None, BTreeSet::new()),
        None,
        vec![SegmentFile::new(file, 5)],
    )
}

fn successor(parent: &Manifest, segments: Vec<SegmentEntry>) -> Manifest {
    let (shard, entry) = parent.shards().iter().next().unwrap();
    let mut shards = parent.shards().clone();
    shards.insert(
        shard.clone(),
        ShardEntry::new(entry.range().clone(), segments),
    );
    parent
        .successor(new_commit_id().unwrap(), shards, parent.model_ids().clone())
        .unwrap()
}

async fn commit_with(index: &IndexHandle, segments: Vec<SegmentEntry>) -> Manifest {
    let manifest = successor(&head(index).await, segments);
    let outcome = index.commit(&manifest).await.unwrap();
    assert_eq!(
        outcome,
        CommitOutcome::Committed(manifest.version()),
        "an uncontended commit must succeed"
    );
    manifest
}

fn slot_key(index: &IndexHandle, version: u64) -> ObjectKey {
    let version = ManifestVersion::new(version).unwrap();
    key(&format!(
        "meta/{}/manifests/{}",
        index.id(),
        version.file_name()
    ))
}

async fn is_stub(store: &MemoryStore, index: &IndexHandle, version: u64) -> bool {
    let bytes = store.get(&slot_key(index, version)).await.unwrap();
    ManifestObject::decode(&bytes)
        .unwrap()
        .into_manifest()
        .is_none()
}

async fn segment_exists(store: &MemoryStore, index: &IndexHandle, name: &str) -> bool {
    let prefix = key(&format!("meta/{}/segments/{name}", index.id()));
    let found: Vec<ObjectInfo> = store.list(Some(&prefix), None).try_collect().await.unwrap();
    !found.is_empty()
}

async fn keys_under(store: &MemoryStore, prefix: &str) -> Vec<String> {
    let prefix = key(prefix);
    let found: Vec<ObjectInfo> = store.list(Some(&prefix), None).try_collect().await.unwrap();
    found.iter().map(|info| info.key().to_string()).collect()
}

async fn snapshot(store: &MemoryStore) -> Vec<(String, Bytes)> {
    let listed: Vec<ObjectInfo> = store.list(None, None).try_collect().await.unwrap();
    let mut objects = Vec::new();
    for info in listed {
        objects.push((info.key().to_string(), store.get(info.key()).await.unwrap()));
    }
    objects.sort();
    objects
}

async fn write_other_index(store: &MemoryStore, time: SystemTime) {
    store.set_time(time);
    for name in [
        "MANIFEST",
        "manifests/v000000000001.json",
        "segments/s/lex-s.bin",
    ] {
        store
            .put_overwrite(
                &key(&format!("meta/{OTHER_INDEX}/{name}")),
                Bytes::from_static(b"old"),
            )
            .await
            .unwrap();
    }
}

async fn history(store: &MemoryStore, metadata: &MetadataStore) -> IndexHandle {
    store.set_time(at(0));
    let index = create(metadata).await;
    let a = write_segment(&index, "a").await;
    let b = write_segment(&index, "b").await;
    write_segment(&index, "d").await;
    store.set_time(at(100));
    commit_with(&index, vec![a, b.clone()]).await;
    store.set_time(at(200));
    commit_with(&index, vec![b]).await;
    store.set_time(at(300));
    let c = write_segment(&index, "c").await;
    commit_with(&index, vec![c]).await;
    index
}

async fn collect(metadata: &MetadataStore, now: SystemTime) -> CollectionReport {
    metadata.collect_garbage(&settings(), now).await.unwrap()
}

async fn assert_collected(store: &MemoryStore, index: &IndexHandle) {
    for version in 1..=3 {
        assert!(is_stub(store, index, version).await, "version {version}");
    }
    assert!(
        !is_stub(store, index, 4).await,
        "the newest is never a stub"
    );
    for (name, kept) in [("a", false), ("b", false), ("c", true), ("d", false)] {
        assert_eq!(segment_exists(store, index, name).await, kept, "{name}");
    }
    assert_eq!(
        keys_under(store, &format!("meta/{OTHER_INDEX}")).await,
        Vec::<String>::new(),
        "the old index is gone"
    );
}

#[tokio::test]
async fn a_manifest_inside_the_retention_keeps_its_segments() {
    let (metadata, _, store) = fixture();
    let index = history(&store, &metadata).await;

    let report = collect(&metadata, at(250)).await;
    assert_eq!(report.manifests_stubbed(), 1);
    assert_eq!(report.manifests_retained(), 3);
    assert!(is_stub(&store, &index, 1).await);
    assert!(!is_stub(&store, &index, 2).await);
    for name in ["a", "b", "c"] {
        assert!(segment_exists(&store, &index, name).await, "{name}");
    }
    assert!(!segment_exists(&store, &index, "d").await);
    assert_eq!(report.segments_deleted(), 1);
    assert_eq!(report.bytes_deleted(), 5);

    let report = collect(&metadata, at(1_000)).await;
    assert_eq!(report.manifests_stubbed(), 2);
    assert_eq!(report.manifests_retained(), 1);
    assert_eq!(report.segments_deleted(), 2);
    assert_collected(&store, &index).await;
}

#[tokio::test]
async fn the_newest_manifest_is_never_a_stub() {
    let (metadata, _, store) = fixture();
    let index = create(&metadata).await;

    let report = collect(&metadata, at(1_000_000)).await;
    assert_eq!(report.manifests_stubbed(), 0);
    assert_eq!(report.manifests_retained(), 1);
    assert!(!is_stub(&store, &index, 1).await);
    assert_eq!(head(&index).await.version(), ManifestVersion::FIRST);
}

#[tokio::test]
async fn a_stubbed_slot_refuses_a_late_create_and_its_writer_still_knows_its_commit() {
    let (metadata, _, store) = fixture();
    store.set_time(at(0));
    let index = create(&metadata).await;
    let first = head(&index).await;
    let mine = successor(&first, Vec::new());
    assert_eq!(
        index.commit(&mine).await.unwrap(),
        CommitOutcome::Committed(mine.version())
    );
    store.set_time(at(10));
    commit_with(&index, Vec::new()).await;
    collect(&metadata, at(1_000)).await;
    assert!(is_stub(&store, &index, 2).await);

    assert_eq!(
        index.commit(&mine).await.unwrap(),
        CommitOutcome::Committed(mine.version())
    );
    let late = successor(&first, Vec::new());
    assert_eq!(index.commit(&late).await.unwrap(), CommitOutcome::Conflict);
    let err = store
        .put_create(&slot_key(&index, 2), Bytes::from(late.encode().unwrap()))
        .await
        .unwrap_err();
    assert!(matches!(err, Error::AlreadyExists { .. }), "{err}");
    assert!(is_stub(&store, &index, 2).await);
}

#[tokio::test]
async fn a_young_orphan_survives_and_an_old_one_goes() {
    let (metadata, _, store) = fixture();
    store.set_time(at(0));
    let index = create(&metadata).await;
    write_segment(&index, "old").await;
    store.set_time(at(60));
    write_segment(&index, "young").await;

    let report = collect(&metadata, at(100)).await;
    assert_eq!(report.segments_deleted(), 1);
    assert_eq!(report.segments_young(), 1);
    assert!(!segment_exists(&store, &index, "old").await);
    assert!(segment_exists(&store, &index, "young").await);
}

#[tokio::test]
async fn collecting_again_changes_nothing_more() {
    let (metadata, _, store) = fixture();
    let index = history(&store, &metadata).await;
    write_other_index(&store, at(0)).await;
    store.set_time(at(900));

    collect(&metadata, at(1_000)).await;
    let after = snapshot(&store).await;
    let again = collect(&metadata, at(1_000)).await;
    assert_eq!(snapshot(&store).await, after);
    assert_eq!(again.manifests_stubbed(), 0);
    assert_eq!(again.segments_deleted(), 0);
    assert_eq!(again.indexes_removed(), 0);
    assert_collected(&store, &index).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn collections_running_at_once_end_as_one_does() {
    let (metadata, _, store) = fixture();
    let index = history(&store, &metadata).await;
    write_other_index(&store, at(0)).await;
    store.set_time(at(900));

    let mut runs = JoinSet::new();
    for _ in 0..4 {
        let metadata = metadata.clone();
        runs.spawn(async move { metadata.collect_garbage(&settings(), at(1_000)).await });
    }
    while let Some(run) = runs.join_next().await {
        run.unwrap().unwrap();
    }
    assert_collected(&store, &index).await;
}

#[tokio::test]
async fn a_crash_at_any_step_leaves_what_the_next_run_finishes() {
    let mut writes = 0;
    loop {
        let (metadata, faults, store) = fixture();
        let index = history(&store, &metadata).await;
        write_other_index(&store, at(0)).await;
        store.set_time(at(900));
        faults.stop_after_writes(writes);
        let crashed = metadata.collect_garbage(&settings(), at(1_000)).await;
        faults.resume_writes();
        if crashed.is_ok() {
            assert!(writes > 5, "a full run writes stubs and deletes");
            break;
        }
        assert!(!is_stub(&store, &index, 4).await);
        assert!(segment_exists(&store, &index, "c").await);
        collect(&metadata, at(1_000)).await;
        assert_collected(&store, &index).await;
        writes += 1;
    }
}

#[tokio::test]
async fn another_index_is_removed_only_when_old() {
    let (metadata, _, store) = fixture();
    store.set_time(at(0));
    let index = create(&metadata).await;
    write_other_index(&store, at(0)).await;

    let report = collect(&metadata, at(40)).await;
    assert_eq!(report.indexes_young(), 1);
    assert_eq!(
        keys_under(&store, &format!("meta/{OTHER_INDEX}"))
            .await
            .len(),
        3
    );

    let report = collect(&metadata, at(100)).await;
    assert_eq!(report.indexes_removed(), 1);
    assert_eq!(
        keys_under(&store, &format!("meta/{OTHER_INDEX}")).await,
        Vec::<String>::new()
    );
    assert!(!is_stub(&store, &index, 1).await);
    store.get(&key("meta/INDEX")).await.unwrap();
}

#[tokio::test]
async fn another_index_with_one_young_object_keeps_it() {
    let (metadata, _, store) = fixture();
    store.set_time(at(0));
    create(&metadata).await;
    write_other_index(&store, at(0)).await;
    store.set_time(at(90));
    store
        .put_overwrite(
            &key(&format!("meta/{OTHER_INDEX}/MANIFEST")),
            Bytes::from_static(b"late"),
        )
        .await
        .unwrap();

    let report = collect(&metadata, at(100)).await;
    assert_eq!(report.indexes_young(), 1);
    assert_eq!(report.indexes_removed(), 0);
    assert_eq!(
        keys_under(&store, &format!("meta/{OTHER_INDEX}"))
            .await
            .len(),
        3
    );
}

#[tokio::test]
async fn without_index_nothing_is_removed() {
    let (metadata, _, store) = fixture();
    store.set_time(at(0));
    create(&metadata).await;
    write_other_index(&store, at(0)).await;
    store.delete(&key("meta/INDEX")).await.unwrap();
    let before = snapshot(&store).await;

    let report = collect(&metadata, at(1_000)).await;
    assert_eq!(report.indexes_removed(), 0);
    assert_eq!(snapshot(&store).await, before);
}

#[tokio::test]
async fn an_index_removed_during_the_run_stops_the_removal() {
    let store = MemoryStore::new();
    let tamper = Tamper::new(&store);
    let metadata = tampered(&tamper);
    store.set_time(at(0));
    create(&metadata).await;
    write_other_index(&store, at(0)).await;
    let index_key = key("meta/INDEX");
    let remover = store.clone();
    tamper.on_scan(async move {
        remover.delete(&index_key).await.unwrap();
    });

    let report = collect(&metadata, at(1_000)).await;
    assert_eq!(report.indexes_removed(), 0);
    assert_eq!(
        keys_under(&store, &format!("meta/{OTHER_INDEX}"))
            .await
            .len(),
        3
    );
}

#[derive(Debug, Clone)]
struct Tamper {
    inner: MemoryStore,
    undated: bool,
    listing_fails_after: Option<usize>,
    on_scan: Arc<Mutex<Option<Hook>>>,
}

struct Hook(BoxFuture<'static, ()>);

impl std::fmt::Debug for Hook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Hook")
    }
}

impl Tamper {
    fn new(inner: &MemoryStore) -> Self {
        Tamper {
            inner: inner.clone(),
            undated: false,
            listing_fails_after: None,
            on_scan: Arc::default(),
        }
    }

    fn on_scan(&self, hook: impl Future<Output = ()> + Send + 'static) {
        *self.on_scan.lock().unwrap() = Some(Hook(hook.boxed()));
    }

    fn strip(&self, info: ObjectInfo) -> ObjectInfo {
        if self.undated {
            ObjectInfo::new(info.key().clone(), info.size(), info.etag().clone())
        } else {
            info
        }
    }
}

#[async_trait]
impl ReadStore for Tamper {
    async fn read_range(
        &self,
        key: &ObjectKey,
        range: ByteRange,
        expected: &ETag,
    ) -> Result<Bytes> {
        self.inner.read_range(key, range, expected).await
    }

    async fn head(&self, key: &ObjectKey) -> Result<ObjectInfo> {
        self.inner.head(key).await.map(|info| self.strip(info))
    }

    fn list<'a>(
        &'a self,
        prefix: Option<&'a ObjectKey>,
        after: Option<&'a ObjectKey>,
    ) -> BoxStream<'a, Result<ObjectInfo>> {
        let scanning = prefix.is_some_and(|prefix| prefix.as_str() == "meta");
        let hook = scanning
            .then(|| self.on_scan.lock().unwrap().take())
            .flatten();
        let listed = self
            .inner
            .list(prefix, after)
            .map_ok(|info| self.strip(info));
        let listed = match self.listing_fails_after {
            Some(count) if scanning => listed
                .take(count)
                .chain(stream::once(async {
                    Err(Error::Backend {
                        key: "meta".to_owned(),
                        source: BackendError::new("the listing broke off"),
                    })
                }))
                .boxed(),
            Some(_) | None => listed.boxed(),
        };
        match hook {
            Some(Hook(hook)) => stream::once(hook)
                .filter_map(|()| future::ready(None))
                .chain(listed)
                .boxed(),
            None => listed,
        }
    }
}

#[async_trait]
impl WriteStore for Tamper {
    async fn get(&self, key: &ObjectKey) -> Result<Bytes> {
        self.inner.get(key).await
    }

    async fn get_range(&self, key: &ObjectKey, range: ByteRange) -> Result<Bytes> {
        self.inner.get_range(key, range).await
    }

    async fn put_create(&self, key: &ObjectKey, body: Bytes) -> Result<()> {
        self.inner.put_create(key, body).await
    }

    async fn put_overwrite(&self, key: &ObjectKey, body: Bytes) -> Result<()> {
        self.inner.put_overwrite(key, body).await
    }

    async fn delete(&self, key: &ObjectKey) -> Result<()> {
        self.inner.delete(key).await
    }
}

fn tampered(tamper: &Tamper) -> MetadataStore {
    MetadataStore::new(tamper.clone(), metadata_location()).unwrap()
}

#[tokio::test]
async fn a_commit_landing_during_the_run_keeps_its_segments() {
    let store = MemoryStore::new();
    let tamper = Tamper::new(&store);
    let metadata = tampered(&tamper);
    store.set_time(at(0));
    let index = create(&metadata).await;
    let late = write_segment(&index, "late").await;
    let writer = index.clone();
    tamper.on_scan(async move {
        commit_with(&writer, vec![late]).await;
    });

    let report = collect(&metadata, at(1_000)).await;
    assert_eq!(report.segments_deleted(), 0);
    assert!(segment_exists(&store, &index, "late").await);
    assert_eq!(head(&index).await.version().get(), 2);
}

#[tokio::test]
async fn a_live_index_is_never_removed_after_a_stale_read() {
    let store = MemoryStore::new();
    let tamper = Tamper::new(&store);
    let metadata = tampered(&tamper);
    store.set_time(at(0));
    create(&metadata).await;
    write_other_index(&store, at(0)).await;
    let index_key = key("meta/INDEX");
    let other = IndexRecord::new(IndexId::new(OTHER_INDEX).unwrap());
    let swapper = store.clone();
    tamper.on_scan(async move {
        swapper
            .put_overwrite(&index_key, Bytes::from(other.encode().unwrap()))
            .await
            .unwrap();
    });

    let report = collect(&metadata, at(1_000)).await;
    assert_eq!(report.indexes_removed(), 0);
    assert_eq!(
        keys_under(&store, &format!("meta/{OTHER_INDEX}"))
            .await
            .len(),
        3
    );
}

#[tokio::test]
async fn without_written_times_nothing_is_removed() {
    let store = MemoryStore::new();
    let mut tamper = Tamper::new(&store);
    tamper.undated = true;
    let metadata = tampered(&tamper);
    let index = history(&store, &metadata).await;
    write_other_index(&store, at(0)).await;
    let before = snapshot(&store).await;

    let report = collect(&metadata, at(1_000_000)).await;
    assert_eq!(snapshot(&store).await, before);
    assert_eq!(report.manifests_stubbed(), 0);
    assert_eq!(report.segments_young(), 1);
    assert_eq!(report.manifests_retained(), 4);
    assert_eq!(report.indexes_young(), 1);
    assert!(segment_exists(&store, &index, "d").await);
}

#[tokio::test]
async fn a_listing_that_breaks_off_deletes_no_segment() {
    let store = MemoryStore::new();
    let index = history(&store, &tampered(&Tamper::new(&store))).await;
    let mut tamper = Tamper::new(&store);
    tamper.listing_fails_after = Some(3);
    let metadata = tampered(&tamper);

    let err = metadata
        .collect_garbage(&settings(), at(1_000))
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Backend { .. }), "{err}");
    for name in ["a", "b", "c", "d"] {
        assert!(segment_exists(&store, &index, name).await, "{name}");
    }
}

#[tokio::test]
async fn an_unreadable_retained_manifest_stops_the_sweep() {
    let (metadata, _, store) = fixture();
    let index = history(&store, &metadata).await;
    store.set_time(at(310));
    store
        .put_overwrite(&slot_key(&index, 3), Bytes::from_static(b"{"))
        .await
        .unwrap();

    let report = collect(&metadata, at(350)).await;
    assert!(report.sweep_skipped());
    assert_eq!(report.manifests_unreadable(), 1);
    assert!(segment_exists(&store, &index, "d").await);
    let err = index.retained(RETENTION, at(350)).await.unwrap_err();
    assert!(matches!(err, Error::ManifestUnreadable { .. }), "{err}");
}

#[tokio::test]
async fn retained_names_what_the_kept_manifests_reference() {
    let (metadata, _, store) = fixture();
    let index = history(&store, &metadata).await;

    let names = |at_time| {
        let index = index.clone();
        async move {
            let retained = index.retained(RETENTION, at_time).await.unwrap().unwrap();
            let names: Vec<String> = retained
                .segments()
                .keys()
                .map(ToString::to_string)
                .collect();
            names
        }
    };
    assert_eq!(names(at(250)).await, ["a", "b", "c"]);
    assert_eq!(names(at(350)).await, ["b", "c"]);
    assert_eq!(names(at(1_000)).await, ["c"]);
}

#[tokio::test]
async fn a_stub_write_with_a_lost_response_is_finished_by_the_next_run() {
    let (metadata, faults, store): (MetadataStore, FaultStore, MemoryStore) = fixture();
    let index = history(&store, &metadata).await;
    faults.inject(
        cosmosearch_store::Op::PutOverwrite,
        "manifests",
        cosmosearch_store::Fault::LostResponse,
    );
    let err = metadata
        .collect_garbage(&settings(), at(1_000))
        .await
        .unwrap_err();
    assert!(matches!(err, Error::OutcomeUnknown { .. }), "{err}");
    collect(&metadata, at(1_000)).await;
    for version in 1..=3 {
        assert!(is_stub(&store, &index, version).await, "version {version}");
    }
    assert!(!is_stub(&store, &index, 4).await);
}
