use std::assert_matches;

use std::sync::Arc;

use bytes::Bytes;
use cosmosearch_core::{ByteRange, ETag, FileSource, Location, Scheme};
use cosmosearch_store::{
    CommitOutcome, CreateOutcome, Error, MemoryStore, MetadataStore, ObjectKey, ReadStore,
    WriteStore,
};
#[cfg(feature = "s3")]
use cosmosearch_store::{ObjectStoreBackend, S3Options};
use futures_util::TryStreamExt;
use tokio::sync::Barrier;
use tokio::task::JoinSet;
use uuid::Uuid;

use crate::support::{create, head, next};

const RACERS: usize = 32;
const COMMIT_RACERS: usize = 16;
const ROUNDS: usize = 5;
fn key(root: &Location, name: &str) -> ObjectKey {
    let prefix = root.key().trim_end_matches('/');
    if prefix.is_empty() {
        ObjectKey::new(name).unwrap()
    } else {
        ObjectKey::new(format!("{prefix}/{name}")).unwrap()
    }
}

fn child(root: &Location, name: &str) -> Location {
    let prefix = root.key().trim_end_matches('/');
    Location::new(root.scheme(), root.authority(), format!("{prefix}/{name}")).unwrap()
}

fn range(start: u64, end: u64) -> ByteRange {
    ByteRange::new(start, end).unwrap()
}

async fn etag_of<S: ReadStore>(store: &S, key: &ObjectKey) -> ETag {
    store.head(key).await.unwrap().etag().clone()
}

async fn create_only_refuses_an_occupied_key<S: WriteStore>(store: &S, root: &Location) {
    let k = key(root, "create/a");
    store
        .put_create(&k, Bytes::from_static(b"first"))
        .await
        .unwrap();
    let err = store
        .put_create(&k, Bytes::from_static(b"second"))
        .await
        .unwrap_err();
    assert_matches!(err, Error::AlreadyExists { .. });
    assert_eq!(
        &*store.get(&k).await.unwrap(),
        b"first",
        "a refused create must leave the first body in place"
    );
}

async fn overwrite_replaces_and_changes_the_etag<S: WriteStore>(store: &S, root: &Location) {
    let k = key(root, "overwrite/a");
    store
        .put_overwrite(&k, Bytes::from_static(b"one"))
        .await
        .unwrap();
    let before = etag_of(store, &k).await;
    store
        .put_overwrite(&k, Bytes::from_static(b"three"))
        .await
        .unwrap();
    assert_eq!(
        &*store.get(&k).await.unwrap(),
        b"three",
        "an overwrite replaces the body"
    );
    assert_eq!(
        store.head(&k).await.unwrap().size(),
        5,
        "head reports the size of the new body"
    );
    assert_ne!(
        etag_of(store, &k).await,
        before,
        "an overwrite changes the ETag"
    );
}

async fn reads_are_pinned_to_their_etag<S: WriteStore>(store: &S, root: &Location) {
    let k = key(root, "range/a");
    store
        .put_overwrite(&k, Bytes::from_static(b"0123456789"))
        .await
        .unwrap();
    let etag = etag_of(store, &k).await;
    assert_eq!(
        &*store.read_range(&k, range(2, 5), &etag).await.unwrap(),
        b"234",
        "a pinned range read returns exactly the requested bytes"
    );

    for past_the_end in [range(5, 50), range(10, 20), range(20, 30)] {
        let err = store.read_range(&k, past_the_end, &etag).await.unwrap_err();
        assert_matches!(err, Error::ShortRead { .. });
    }

    store
        .put_overwrite(&k, Bytes::from_static(b"abcdefghij"))
        .await
        .unwrap();
    let err = store.read_range(&k, range(2, 5), &etag).await.unwrap_err();
    assert_matches!(err, Error::Stale { .. });
}

async fn unpinned_range_reads_return_the_exact_bytes<S: WriteStore>(store: &S, root: &Location) {
    let k = key(root, "unpinned/a");
    store
        .put_create(&k, Bytes::from_static(b"0123456789"))
        .await
        .unwrap();
    assert_eq!(
        &*store.get_range(&k, range(2, 5)).await.unwrap(),
        b"234",
        "a range read returns exactly the requested bytes"
    );
    assert_eq!(
        &*store.get_range(&k, range(9, 10)).await.unwrap(),
        b"9",
        "a range read can end at the last byte"
    );
    store.get_range(&k, range(5, 11)).await.unwrap_err();
    let missing = key(root, "unpinned/missing");
    let err = store.get_range(&missing, range(0, 1)).await.unwrap_err();
    assert_matches!(err, Error::NotFound { .. });
}

async fn missing_objects_report_not_found<S: WriteStore>(store: &S, root: &Location) {
    let k = key(root, "missing/a");
    let etag = ETag::new("e").unwrap();
    for err in [
        store.get(&k).await.unwrap_err(),
        store.head(&k).await.unwrap_err(),
        store.read_range(&k, range(0, 1), &etag).await.unwrap_err(),
    ] {
        assert_matches!(err, Error::NotFound { .. });
    }
}

async fn delete_is_idempotent<S: WriteStore>(store: &S, root: &Location) {
    let k = key(root, "delete/a");
    store.delete(&k).await.unwrap();
    store.put_create(&k, Bytes::new()).await.unwrap();
    store.delete(&k).await.unwrap();
    store.head(&k).await.unwrap_err();
    store.put_create(&k, Bytes::new()).await.unwrap();
}

async fn listing_treats_a_prefix_as_a_directory<S: WriteStore>(store: &S, root: &Location) {
    for name in ["list/a/1", "list/a/2", "list/a/3", "list/ab/4"] {
        store
            .put_create(&key(root, name), Bytes::new())
            .await
            .unwrap();
    }
    let dir = key(root, "list/a");
    let after = key(root, "list/a/1");
    let collect = |after: Option<ObjectKey>| {
        let dir = dir.clone();
        async move {
            let mut keys: Vec<String> = store
                .list(Some(&dir), after.as_ref())
                .map_ok(|info| info.key().file_name().to_owned())
                .try_collect()
                .await
                .unwrap();
            keys.sort();
            keys
        }
    };
    assert_eq!(
        collect(None).await,
        ["1", "2", "3"],
        "listing returns every key under the prefix"
    );
    assert_eq!(
        collect(Some(after)).await,
        ["2", "3"],
        "listing after a key excludes that key"
    );
}
async fn exactly_one_racing_create_wins<S>(store: &S, root: &Location)
where
    S: WriteStore + Clone + 'static,
{
    for round in 0..ROUNDS {
        let k = key(root, &format!("race/{round}"));
        let barrier = Arc::new(Barrier::new(RACERS));
        let mut tasks = JoinSet::new();
        for writer in 0..RACERS {
            let (store, k, barrier) = (store.clone(), k.clone(), Arc::clone(&barrier));
            tasks.spawn(async move {
                barrier.wait().await;
                store.put_create(&k, Bytes::from(writer.to_string())).await
            });
        }
        let mut winners = 0;
        while let Some(outcome) = tasks.join_next().await {
            match outcome.unwrap() {
                Ok(()) => winners += 1,
                Err(Error::AlreadyExists { .. }) => {}
                Err(err) => panic!("unexpected: {err}"),
            }
        }
        assert_eq!(winners, 1, "round {round}");
    }
}

async fn exactly_one_racing_commit_wins<S>(store: &S, root: &Location)
where
    S: WriteStore + Clone + 'static,
{
    let metadata = MetadataStore::new(store.clone(), child(root, "commit")).unwrap();
    let index = create(&metadata).await;
    for round in 0..ROUNDS {
        let parent = head(&index).await;
        let barrier = Arc::new(Barrier::new(COMMIT_RACERS));
        let mut tasks = JoinSet::new();
        for _ in 0..COMMIT_RACERS {
            let manifest = next(&parent);
            let (index, barrier) = (index.clone(), Arc::clone(&barrier));
            tasks.spawn(async move {
                barrier.wait().await;
                let outcome = index.commit(&manifest).await.unwrap();
                (outcome, manifest)
            });
        }
        let mut winners = Vec::new();
        while let Some(result) = tasks.join_next().await {
            let (outcome, manifest) = result.unwrap();
            if matches!(outcome, CommitOutcome::Committed(_)) {
                winners.push(manifest);
            } else {
                assert_eq!(
                    outcome,
                    CommitOutcome::Conflict,
                    "a commit that does not win must report a conflict"
                );
            }
        }
        assert_eq!(winners.len(), 1, "round {round}");
        assert_eq!(Some(head(&index).await), winners.pop(), "round {round}");
    }
}

async fn exactly_one_racing_index_create_wins<S>(store: &S, root: &Location)
where
    S: WriteStore + Clone + 'static,
{
    let data: Location = "s3://example-bucket/docs/".parse().unwrap();
    for round in 0..ROUNDS {
        let location = child(root, &format!("create-race/{round}"));
        let barrier = Arc::new(Barrier::new(COMMIT_RACERS));
        let mut tasks = JoinSet::new();
        for _ in 0..COMMIT_RACERS {
            let metadata = MetadataStore::new(store.clone(), location.clone()).unwrap();
            let (data, barrier) = (data.clone(), Arc::clone(&barrier));
            tasks.spawn(async move {
                barrier.wait().await;
                metadata.create(&data, FileSource::Listing).await.unwrap()
            });
        }
        let mut winners = 0;
        while let Some(result) = tasks.join_next().await {
            let outcome = result.unwrap();
            if matches!(outcome, CreateOutcome::Created(_)) {
                winners += 1;
            } else {
                assert_matches!(outcome, CreateOutcome::Conflict);
            }
        }
        assert_eq!(winners, 1, "round {round}");
    }
}

async fn objects_report_when_they_were_written<S: WriteStore>(store: &S, root: &Location) {
    let k = key(root, "written/a");
    store
        .put_overwrite(&k, Bytes::from_static(b"one"))
        .await
        .unwrap();
    let first = store.head(&k).await.unwrap().last_modified().unwrap();
    let prefix = key(root, "written");
    let listed: Vec<_> = store.list(Some(&prefix), None).try_collect().await.unwrap();
    assert_eq!(listed.len(), 1, "one object was written");
    assert!(
        listed[0].last_modified().is_some(),
        "a listing reports the written time"
    );
    store
        .put_overwrite(&k, Bytes::from_static(b"two"))
        .await
        .unwrap();
    let second = store.head(&k).await.unwrap().last_modified().unwrap();
    assert!(second >= first, "an overwrite never moves the time back");
}

async fn an_overwritten_key_still_refuses_a_create<S: WriteStore>(store: &S, root: &Location) {
    let k = key(root, "stub/a");
    store
        .put_create(&k, Bytes::from_static(b"manifest"))
        .await
        .unwrap();
    store
        .put_overwrite(&k, Bytes::from_static(b"stub"))
        .await
        .unwrap();
    let err = store
        .put_create(&k, Bytes::from_static(b"late"))
        .await
        .unwrap_err();
    assert_matches!(err, Error::AlreadyExists { .. });
    assert_eq!(
        &*store.get(&k).await.unwrap(),
        b"stub",
        "a refused create leaves the overwritten body"
    );
}

async fn suite<S>(store: S, root: Location)
where
    S: WriteStore + Clone + 'static,
{
    create_only_refuses_an_occupied_key(&store, &root).await;
    overwrite_replaces_and_changes_the_etag(&store, &root).await;
    reads_are_pinned_to_their_etag(&store, &root).await;
    unpinned_range_reads_return_the_exact_bytes(&store, &root).await;
    missing_objects_report_not_found(&store, &root).await;
    delete_is_idempotent(&store, &root).await;
    listing_treats_a_prefix_as_a_directory(&store, &root).await;
    objects_report_when_they_were_written(&store, &root).await;
    an_overwritten_key_still_refuses_a_create(&store, &root).await;
    exactly_one_racing_create_wins(&store, &root).await;
    exactly_one_racing_commit_wins(&store, &root).await;
    exactly_one_racing_index_create_wins(&store, &root).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn memory_backend_conforms() {
    let root = Location::new(Scheme::S3, "example-bucket", "root").unwrap();
    suite(MemoryStore::new(), root).await;
}

#[cfg(feature = "fs")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn local_filesystem_conforms() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_str().unwrap().trim_start_matches('/');
    let root = Location::new(Scheme::File, "", path).unwrap();
    suite(ObjectStoreBackend::local(), root).await;
}
#[cfg(feature = "s3")]
fn s3_endpoint() -> Option<String> {
    #[expect(
        clippy::disallowed_methods,
        reason = "a test, not the library, chooses its store"
    )]
    std::env::var("COSMOSEARCH_TEST_S3_ENDPOINT").ok()
}

#[cfg(feature = "s3")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn s3_compatible_store_conforms() {
    let Some(endpoint) = s3_endpoint() else {
        return;
    };
    let options = S3Options::builder("cosmosearch-test")
        .endpoint(endpoint)
        .credentials("cosmosearch-local", "cosmosearch-local-only")
        .allow_http(true)
        .build()
        .unwrap();
    let store = ObjectStoreBackend::s3(&options).unwrap();
    let prefix = format!("conformance/{}", Uuid::new_v4().simple());
    let root = Location::new(Scheme::S3, "cosmosearch-test", prefix).unwrap();
    suite(store, root).await;
}
