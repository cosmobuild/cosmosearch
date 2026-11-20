use cosmosearch_core::{FileSource, Location, Manifest};
use cosmosearch_store::{
    CommitOutcome, CreateOutcome, Fault, FaultStore, IndexHandle, MemoryStore, MetadataStore, Op,
    OpenState, new_commit_id,
};

pub(crate) fn metadata_location() -> Location {
    "s3://example-bucket/meta".parse().unwrap()
}

pub(crate) fn data_location() -> Location {
    "s3://example-bucket/docs/".parse().unwrap()
}
pub(crate) fn fixture() -> (MetadataStore, FaultStore, MemoryStore) {
    let inner = MemoryStore::new();
    let faults = FaultStore::new(inner.clone());
    let metadata = MetadataStore::new(faults.clone(), metadata_location()).unwrap();
    (metadata, faults, inner)
}

pub(crate) async fn create(metadata: &MetadataStore) -> IndexHandle {
    let outcome = metadata
        .create(&data_location(), FileSource::Listing)
        .await
        .unwrap();
    let CreateOutcome::Created(handle) = outcome else {
        panic!("expected created, got {outcome:?}");
    };
    handle
}

pub(crate) async fn open(metadata: &MetadataStore) -> OpenState {
    metadata
        .open(&data_location(), FileSource::Listing)
        .await
        .unwrap()
}

pub(crate) async fn head(index: &IndexHandle) -> Manifest {
    index.head().await.unwrap().unwrap()
}
pub(crate) fn next(parent: &Manifest) -> Manifest {
    parent
        .successor(
            new_commit_id().unwrap(),
            parent.shards().clone(),
            parent.model_ids().clone(),
        )
        .unwrap()
}

pub(crate) async fn commit_next(index: &IndexHandle) -> Manifest {
    let manifest = next(&head(index).await);
    let outcome = index.commit(&manifest).await.unwrap();
    assert_eq!(
        outcome,
        CommitOutcome::Committed(manifest.version()),
        "an uncontended commit must succeed"
    );
    manifest
}

pub(crate) fn fail_every_attempt(faults: &FaultStore, key_contains: &str, fault: Fault) {
    for _ in 0..3 {
        faults.inject(Op::PutCreate, key_contains, fault);
    }
}
