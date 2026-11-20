use std::assert_matches;

use bytes::Bytes;
use cosmosearch_core::{ByteRange, ETag};
use cosmosearch_store::{
    Error, Fault, FaultStore, MemoryStore, ObjectKey, Op, ReadStore, Result, WriteStore,
};
use futures_util::TryStreamExt;

fn key(name: &str) -> ObjectKey {
    ObjectKey::new(name).unwrap()
}

fn range(start: u64, end: u64) -> ByteRange {
    ByteRange::new(start, end).unwrap()
}
fn assert_send<T: Send>(value: T) -> T {
    value
}

#[tokio::test]
async fn backends_work_as_trait_objects_with_send_futures() {
    let store: Box<dyn WriteStore> = Box::new(MemoryStore::new());
    assert_send(store.put_create(&key("a"), Bytes::from_static(b"1")))
        .await
        .unwrap();
    assert_send(store.get(&key("a"))).await.unwrap();
}

#[tokio::test]
async fn a_read_of_a_changed_object_reports_stale() {
    let store = MemoryStore::new();
    store
        .put_overwrite(&key("a"), Bytes::from_static(b"one"))
        .await
        .unwrap();
    let old: ETag = store.head(&key("a")).await.unwrap().etag().clone();
    assert_eq!(
        &*store
            .read_range(&key("a"), range(0, 2), &old)
            .await
            .unwrap(),
        b"on"
    );

    store
        .put_overwrite(&key("a"), Bytes::from_static(b"two"))
        .await
        .unwrap();
    let err = store
        .read_range(&key("a"), range(0, 2), &old)
        .await
        .unwrap_err();
    assert_matches!(err, Error::Stale { .. });
}

#[tokio::test]
async fn lost_response_applies_and_timeout_does_not() {
    let inner = MemoryStore::new();
    let store = FaultStore::new(inner.clone());
    store.inject(Op::PutCreate, "lost", Fault::LostResponse);
    store.inject(Op::PutCreate, "timeout", Fault::Timeout);

    for (name, landed) in [("lost", true), ("timeout", false)] {
        let err = store
            .put_create(&key(name), Bytes::new())
            .await
            .unwrap_err();
        assert_matches!(err, Error::OutcomeUnknown { .. });
        assert_eq!(inner.head(&key(name)).await.is_ok(), landed, "{name}");
    }
}

#[tokio::test]
async fn a_delayed_create_lands_only_if_the_key_is_still_free() {
    let inner = MemoryStore::new();
    let store = FaultStore::new(inner.clone());
    store.inject(Op::PutCreate, "free", Fault::Delayed);
    store.inject(Op::PutCreate, "taken", Fault::Delayed);
    store
        .put_create(&key("free"), Bytes::from_static(b"late"))
        .await
        .unwrap_err();
    store
        .put_create(&key("taken"), Bytes::from_static(b"late"))
        .await
        .unwrap_err();
    inner
        .put_create(&key("taken"), Bytes::from_static(b"first"))
        .await
        .unwrap();

    store.release_delayed().await.unwrap();

    assert_eq!(&*inner.get(&key("free")).await.unwrap(), b"late");
    assert_eq!(&*inner.get(&key("taken")).await.unwrap(), b"first");
}

#[tokio::test]
async fn an_invisible_conflict_reports_exists_with_nothing_there() {
    let inner = MemoryStore::new();
    let store = FaultStore::new(inner.clone());
    store.inject(Op::PutCreate, "a", Fault::InvisibleConflict);
    let err = store.put_create(&key("a"), Bytes::new()).await.unwrap_err();
    assert_matches!(err, Error::AlreadyExists { .. });
    inner.head(&key("a")).await.unwrap_err();
}

#[tokio::test]
async fn can_stop_a_writer_between_any_two_writes() {
    let inner = MemoryStore::new();
    let store = FaultStore::new(inner.clone());
    store.stop_after_writes(1);
    store.put_create(&key("first"), Bytes::new()).await.unwrap();
    store
        .put_create(&key("second"), Bytes::new())
        .await
        .unwrap_err();
    inner.head(&key("first")).await.unwrap();
    inner.head(&key("second")).await.unwrap_err();
    store.resume_writes();
    store
        .put_create(&key("second"), Bytes::new())
        .await
        .unwrap();
}

#[tokio::test]
async fn listing_honours_prefix_and_offset() -> Result<()> {
    let store = MemoryStore::new();
    for name in ["a/1", "a/2", "ab/3", "b"] {
        store.put_create(&key(name), Bytes::new()).await?;
    }
    let a = key("a");
    let after = key("a/1");
    let keys: Vec<String> = store
        .list(Some(&a), Some(&after))
        .map_ok(|info| info.key().to_string())
        .try_collect()
        .await?;
    assert_eq!(keys, ["a/2"]);
    Ok(())
}
