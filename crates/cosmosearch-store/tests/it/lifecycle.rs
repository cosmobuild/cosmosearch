use std::assert_matches;

use async_trait::async_trait;
use bytes::Bytes;
use cosmosearch_core::{ByteRange, ETag, FileSource, IndexId, IndexRecord, Location};
use cosmosearch_store::{
    CreateOutcome, DataStore, Error, Fault, MemoryStore, MetadataStore, ObjectInfo, ObjectKey,
    OpenState, ReadStore, Result, WriteStore,
};
use futures_util::TryStreamExt;
use futures_util::stream::BoxStream;

use crate::support::{create, data_location, fail_every_attempt, fixture, metadata_location, open};

#[tokio::test]
async fn an_empty_location_is_not_indexed_and_create_makes_it_ready() {
    let (metadata, _, _) = fixture();
    assert_matches!(open(&metadata).await, OpenState::NotIndexed);

    let handle = create(&metadata).await;

    let OpenState::Ready(ready) = open(&metadata).await else {
        panic!("expected ready");
    };
    assert_eq!(ready.handle().id(), handle.id());
    assert_eq!(ready.head().version().get(), 1);
    assert_eq!(ready.head().data_location(), &data_location());
}

#[tokio::test]
async fn the_current_index_follows_destroy_and_create() {
    let (metadata, _, _) = fixture();
    assert_eq!(metadata.current_index().await.unwrap(), None);
    let first = create(&metadata).await;
    assert_eq!(
        metadata.current_index().await.unwrap().as_ref(),
        Some(first.id())
    );
    metadata.destroy().await.unwrap();
    assert_eq!(metadata.current_index().await.unwrap(), None);
    let second = create(&metadata).await;
    assert_ne!(first.id(), second.id());
    assert_eq!(
        metadata.current_index().await.unwrap().as_ref(),
        Some(second.id())
    );
}

#[tokio::test]
async fn create_is_refused_on_a_location_in_use() {
    let (metadata, _, _) = fixture();
    create(&metadata).await;
    let outcome = metadata
        .create(&data_location(), FileSource::Listing)
        .await
        .unwrap();
    assert_matches!(outcome, CreateOutcome::Conflict);
}

#[tokio::test]
async fn open_refuses_a_different_data_location_or_file_source() {
    let (metadata, _, _) = fixture();
    create(&metadata).await;
    let other: Location = "s3://example-bucket/other/".parse().unwrap();

    let err = metadata
        .open(&other, FileSource::Listing)
        .await
        .unwrap_err();
    assert_matches!(err, Error::WrongDataLocation { .. });
    let err = metadata
        .open(&data_location(), FileSource::HostNamed)
        .await
        .unwrap_err();
    assert_matches!(err, Error::WrongFileSource { .. });
}
#[tokio::test]
async fn a_create_that_loses_the_index_write_reports_conflict() {
    let (metadata, faults, inner) = fixture();
    fail_every_attempt(&faults, "INDEX", Fault::Timeout);
    let CreateOutcome::Unknown(id) = metadata
        .create(&data_location(), FileSource::Listing)
        .await
        .unwrap()
    else {
        panic!("expected unknown");
    };
    let winner = IndexRecord::new(IndexId::new("winner").unwrap());
    inner
        .put_create(
            &ObjectKey::new("meta/INDEX").unwrap(),
            winner.encode().unwrap().into(),
        )
        .await
        .unwrap();

    let outcome = metadata
        .retry_create(&id, &data_location(), FileSource::Listing)
        .await
        .unwrap();
    assert_matches!(outcome, CreateOutcome::Conflict);
}

#[tokio::test]
async fn a_crash_at_any_step_of_create_gives_a_clear_state_and_destroy_recovers() {
    for writes in 0..6 {
        let (metadata, faults, _) = fixture();
        faults.stop_after_writes(writes);
        let outcome = metadata.create(&data_location(), FileSource::Listing).await;
        faults.resume_writes();

        match (outcome, open(&metadata).await) {
            (Ok(CreateOutcome::Created(_)), OpenState::Ready(_))
            | (Err(_), OpenState::NotIndexed | OpenState::Incomplete) => {}
            (outcome, state) => panic!("{writes} writes: {outcome:?} left {state:?}"),
        }

        metadata.destroy().await.unwrap();
        assert_matches!(open(&metadata).await, OpenState::NotIndexed);
        create(&metadata).await;
    }
}

#[tokio::test]
async fn a_crash_at_any_step_of_destroy_reads_incomplete_and_destroy_finishes() {
    for writes in 0..3 {
        let (metadata, faults, _) = fixture();
        create(&metadata).await;
        faults.stop_after_writes(writes);
        metadata.destroy().await.unwrap_err();
        faults.resume_writes();

        let state = open(&metadata).await;
        if writes == 0 {
            assert_matches!(state, OpenState::Ready(_));
        } else {
            assert_matches!(state, OpenState::Incomplete);
        }

        metadata.destroy().await.unwrap();
        metadata.destroy().await.unwrap();
        assert_matches!(open(&metadata).await, OpenState::NotIndexed);
        create(&metadata).await;
    }
}

#[tokio::test]
async fn an_unknown_create_is_finished_by_retrying_with_the_same_id() {
    let (metadata, faults, _) = fixture();
    fail_every_attempt(&faults, "INDEX", Fault::Timeout);
    let outcome = metadata
        .create(&data_location(), FileSource::Listing)
        .await
        .unwrap();
    let CreateOutcome::Unknown(id) = outcome else {
        panic!("expected unknown, got {outcome:?}");
    };
    assert_matches!(open(&metadata).await, OpenState::Incomplete);

    let outcome = metadata
        .retry_create(&id, &data_location(), FileSource::Listing)
        .await
        .unwrap();
    let CreateOutcome::Created(handle) = outcome else {
        panic!("expected created, got {outcome:?}");
    };
    assert_eq!(handle.id(), &id);
    assert_matches!(open(&metadata).await, OpenState::Ready(_));
}

#[tokio::test]
async fn a_retry_finds_its_own_create_that_landed_after_all() {
    let (metadata, faults, _) = fixture();
    fail_every_attempt(&faults, "INDEX", Fault::Delayed);
    let CreateOutcome::Unknown(id) = metadata
        .create(&data_location(), FileSource::Listing)
        .await
        .unwrap()
    else {
        panic!("expected unknown");
    };
    faults.release_delayed().await.unwrap();

    let outcome = metadata
        .retry_create(&id, &data_location(), FileSource::Listing)
        .await
        .unwrap();
    assert_matches!(outcome, CreateOutcome::Created(_));
}
#[tokio::test]
async fn a_late_index_write_leaves_only_incomplete_or_an_empty_index() {
    let (metadata, faults, _) = fixture();
    fail_every_attempt(&faults, "INDEX", Fault::Delayed);
    metadata
        .create(&data_location(), FileSource::Listing)
        .await
        .unwrap();
    metadata.destroy().await.unwrap();
    faults.release_delayed().await.unwrap();
    assert_matches!(open(&metadata).await, OpenState::Incomplete);

    metadata.destroy().await.unwrap();
    assert_matches!(open(&metadata).await, OpenState::NotIndexed);
    let (metadata, faults, _) = fixture();
    fail_every_attempt(&faults, "INDEX", Fault::Delayed);
    metadata
        .create(&data_location(), FileSource::Listing)
        .await
        .unwrap();
    faults.release_delayed().await.unwrap();
    let OpenState::Ready(ready) = open(&metadata).await else {
        panic!("expected ready");
    };
    for shard in ready.head().shards().values() {
        assert_eq!(shard.segments().len(), 0);
    }
}
#[derive(Debug, Clone, Default)]
struct IgnoresCreateOnly(MemoryStore);

#[async_trait]
impl ReadStore for IgnoresCreateOnly {
    async fn read_range(
        &self,
        key: &ObjectKey,
        range: ByteRange,
        expected: &ETag,
    ) -> Result<Bytes> {
        self.0.read_range(key, range, expected).await
    }

    async fn head(&self, key: &ObjectKey) -> Result<ObjectInfo> {
        self.0.head(key).await
    }

    fn list<'a>(
        &'a self,
        prefix: Option<&'a ObjectKey>,
        after: Option<&'a ObjectKey>,
    ) -> BoxStream<'a, Result<ObjectInfo>> {
        self.0.list(prefix, after)
    }
}

#[async_trait]
impl WriteStore for IgnoresCreateOnly {
    async fn get(&self, key: &ObjectKey) -> Result<Bytes> {
        self.0.get(key).await
    }

    async fn get_range(&self, key: &ObjectKey, range: ByteRange) -> Result<Bytes> {
        self.0.get_range(key, range).await
    }

    async fn put_create(&self, key: &ObjectKey, body: Bytes) -> Result<()> {
        self.0.put_overwrite(key, body).await
    }

    async fn put_overwrite(&self, key: &ObjectKey, body: Bytes) -> Result<()> {
        self.0.put_overwrite(key, body).await
    }

    async fn delete(&self, key: &ObjectKey) -> Result<()> {
        self.0.delete(key).await
    }
}

#[tokio::test]
async fn a_store_that_ignores_create_only_is_refused() {
    let metadata = MetadataStore::new(IgnoresCreateOnly::default(), metadata_location()).unwrap();
    let err = metadata
        .create(&data_location(), FileSource::Listing)
        .await
        .unwrap_err();
    assert_matches!(err, Error::CreateOnlyIgnored { .. });
}

#[tokio::test]
async fn listings_skip_a_metadata_location_inside_the_data_location() {
    let store = MemoryStore::new();
    for name in [
        "docs/a.txt",
        "docs/index/INDEX",
        "docs/index/x/y",
        "docs/indexed.txt",
        "other/b",
    ] {
        store
            .put_create(&ObjectKey::new(name).unwrap(), Bytes::new())
            .await
            .unwrap();
    }
    let inside: Location = "s3://example-bucket/docs/index/".parse().unwrap();
    let elsewhere: Location = "s3://another-bucket/docs/index/".parse().unwrap();

    for (metadata, expected) in [
        (&inside, vec!["docs/a.txt", "docs/indexed.txt"]),
        (
            &elsewhere,
            vec![
                "docs/a.txt",
                "docs/index/INDEX",
                "docs/index/x/y",
                "docs/indexed.txt",
            ],
        ),
    ] {
        let data = DataStore::new(store.clone(), data_location())
            .unwrap()
            .skipping_metadata(metadata)
            .unwrap();
        let mut keys: Vec<String> = data
            .list()
            .map_ok(|info| info.key().to_string())
            .try_collect()
            .await
            .unwrap();
        keys.sort();
        assert_eq!(keys, expected);
    }
}

#[tokio::test]
async fn a_trailing_slash_does_not_change_the_data_location() {
    let (metadata, _, _) = fixture();
    create(&metadata).await;
    let without_slash: Location = "s3://example-bucket/docs".parse().unwrap();
    assert_matches!(
        metadata
            .open(&without_slash, FileSource::Listing)
            .await
            .unwrap(),
        OpenState::Ready(_)
    );
}

#[tokio::test]
async fn create_refuses_a_metadata_location_holding_the_data_location() {
    for (metadata, data) in [
        ("s3://example-bucket/", "s3://example-bucket/docs/"),
        ("s3://example-bucket/x", "s3://example-bucket/x/docs"),
        ("s3://example-bucket/docs/", "s3://example-bucket/docs"),
    ] {
        let store = MetadataStore::new(MemoryStore::new(), metadata.parse().unwrap()).unwrap();
        let err = store
            .create(&data.parse().unwrap(), FileSource::Listing)
            .await
            .unwrap_err();
        assert_matches!(err, Error::DataInsideMetadata { .. });
    }
}

#[tokio::test]
async fn a_location_of_only_slashes_is_refused_rather_than_taken_as_the_root() {
    let location: Location = "s3://example-bucket//".parse().unwrap();
    assert_matches!(
        MetadataStore::new(MemoryStore::new(), location),
        Err(Error::InvalidKey { .. })
    );
}

#[tokio::test]
async fn open_reports_a_newer_format_instead_of_calling_it_incomplete() {
    for (name, body) in [
        ("meta/INDEX", r#"{"format": 2, "index_id": "new"}"#),
        ("meta/INDEX", r#"{"format": 2, "shape": "changed"}"#),
    ] {
        let (metadata, _, inner) = fixture();
        inner
            .put_create(
                &ObjectKey::new(name).unwrap(),
                Bytes::from_static(body.as_bytes()),
            )
            .await
            .unwrap();
        let err = metadata
            .open(&data_location(), FileSource::Listing)
            .await
            .unwrap_err();
        assert_matches!(
            err,
            Error::Core(cosmosearch_core::Error::UnsupportedFormat { format: 2 })
        );
    }
}

#[tokio::test]
async fn open_reports_corrupt_metadata_as_incomplete() {
    let (metadata, _, inner) = fixture();
    inner
        .put_create(
            &ObjectKey::new("meta/INDEX").unwrap(),
            Bytes::from_static(b"garbage"),
        )
        .await
        .unwrap();
    assert_matches!(open(&metadata).await, OpenState::Incomplete);
}

#[tokio::test]
async fn data_reads_stay_inside_the_data_location() {
    let store = MemoryStore::new();
    let etag = ETag::new("e").unwrap();
    let range = ByteRange::new(0, 1).unwrap();
    let metadata: Location = "s3://example-bucket/docs/index".parse().unwrap();
    let data = DataStore::new(store, data_location())
        .unwrap()
        .skipping_metadata(&metadata)
        .unwrap();
    for outside in ["secret/x", "docs2/x", "docs/index/INDEX"] {
        let key = ObjectKey::new(outside).unwrap();
        assert_matches!(
            data.read_range(&key, range, &etag).await,
            Err(Error::InvalidKey { .. })
        );
        assert_matches!(data.head(&key).await, Err(Error::InvalidKey { .. }));
    }
    let inside = ObjectKey::new("docs/a").unwrap();
    assert_matches!(data.head(&inside).await, Err(Error::NotFound { .. }));
}

#[tokio::test]
async fn skipping_refuses_a_metadata_location_holding_the_data_location() {
    let root: Location = "s3://example-bucket/".parse().unwrap();
    let data = DataStore::new(MemoryStore::new(), root.clone()).unwrap();
    assert_matches!(
        data.skipping_metadata(&root),
        Err(Error::DataInsideMetadata { .. })
    );
}
