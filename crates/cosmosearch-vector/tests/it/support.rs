use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use cosmosearch_core::{ChunkHash, EmbeddingIdentity, FileSource, Location, ModelId};
use cosmosearch_store::{
    CreateOutcome, IndexHandle, MemoryStore, MetadataStore, ObjectKey, WriteStore,
};
use cosmosearch_vector::VectorStore;
use futures_util::TryStreamExt;

pub(crate) const DIMENSIONS: u32 = 4;

pub(crate) async fn create(backend: impl WriteStore + 'static, metadata: Location) -> IndexHandle {
    let metadata = MetadataStore::new(backend, metadata).unwrap();
    let data = "s3://example-bucket/docs/".parse().unwrap();
    let outcome = metadata.create(&data, FileSource::Listing).await.unwrap();
    let CreateOutcome::Created(handle) = outcome else {
        panic!("expected created, got {outcome:?}");
    };
    handle
}

pub(crate) async fn fixture() -> (VectorStore, IndexHandle, MemoryStore) {
    let backend = MemoryStore::new();
    let index = create(backend.clone(), "s3://example-bucket/meta".parse().unwrap()).await;
    (VectorStore::new(&index).unwrap(), index, backend)
}

pub(crate) fn identity(model: &str) -> EmbeddingIdentity {
    EmbeddingIdentity::builder(model, "r1", DIMENSIONS, 512)
        .build()
        .unwrap()
}

pub(crate) fn hash(text: &str) -> ChunkHash {
    ChunkHash::of(text)
}

pub(crate) fn vector(text: &str) -> Vec<f32> {
    let bytes = hash(text);
    bytes.as_bytes()[..4]
        .iter()
        .map(|byte| f32::from(*byte) / 255.0 - 0.5)
        .collect()
}

pub(crate) fn at(ms: u64) -> SystemTime {
    UNIX_EPOCH + Duration::from_millis(ms)
}

pub(crate) async fn put(
    store: &VectorStore,
    identity: &EmbeddingIdentity,
    texts: &[&str],
    ms: u64,
) {
    let vectors: Vec<(ChunkHash, Vec<f32>)> = texts
        .iter()
        .map(|text| (hash(text), vector(text)))
        .collect();
    let borrowed: Vec<(ChunkHash, &[f32])> = vectors
        .iter()
        .map(|(hash, vector)| (*hash, vector.as_slice()))
        .collect();
    store.put(identity, &borrowed, at(ms)).await.unwrap();
}

pub(crate) fn assert_close(found: &[f32], expected: &[f32]) {
    let scale = expected
        .iter()
        .fold(0.0_f32, |max, value| max.max(value.abs()));
    let tolerance = scale / 127.0 * 0.5 + 1e-6;
    assert_eq!(found.len(), expected.len(), "a vector keeps its dimensions");
    for (a, b) in found.iter().zip(expected) {
        assert!(
            (a - b).abs() <= tolerance,
            "{a} is not within {tolerance} of {b}"
        );
    }
}

pub(crate) async fn check_found(
    store: &VectorStore,
    identity: &EmbeddingIdentity,
    present: &[&str],
    absent: &[&str],
) {
    let asked: Vec<ChunkHash> = present
        .iter()
        .chain(absent)
        .map(|text| hash(text))
        .collect();
    let lookup = store.get(identity, &asked).await;
    for text in present {
        let found = lookup
            .found()
            .get(&hash(text))
            .unwrap_or_else(|| panic!("{text} has no vector"));
        assert_close(found, &vector(text));
    }
    let missing: Vec<ChunkHash> = absent.iter().map(|text| hash(text)).collect();
    assert_eq!(
        lookup.missing(),
        missing,
        "only the absent chunks are missing"
    );
    assert_eq!(
        lookup.found().len(),
        present.len(),
        "only the present chunks are found"
    );
}

pub(crate) fn live(
    entries: &[(&EmbeddingIdentity, &[&str])],
) -> BTreeMap<ModelId, BTreeSet<ChunkHash>> {
    entries
        .iter()
        .map(|(identity, texts)| {
            (
                identity.model_id().unwrap(),
                texts.iter().map(|text| hash(text)).collect(),
            )
        })
        .collect()
}

pub(crate) async fn packs(backend: &impl WriteStore, index: &IndexHandle) -> Vec<ObjectKey> {
    let root = ObjectKey::new(format!("meta/{}/vectors", index.id())).unwrap();
    let objects: Vec<_> = backend.list(Some(&root), None).try_collect().await.unwrap();
    let mut keys: Vec<ObjectKey> = objects.iter().map(|object| object.key().clone()).collect();
    keys.sort();
    keys
}
