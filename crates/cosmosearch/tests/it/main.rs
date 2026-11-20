//! Contract tests for `cosmosearch`: plug-ins defined here, outside the kernel.

#![expect(
    clippy::panic,
    clippy::unwrap_used,
    reason = "helpers in a test binary fail the test by panicking"
)]

use std::collections::BTreeSet;

use async_trait::async_trait;
use bytes::Bytes;
use cosmosearch::{
    CreateOutcome, DataStore, Engine, EngineBuilder, Error, FileSource, Location, MetadataStore,
    core::{
        ExtractFailure, ExtractLimits, Extraction, Extractor, ExtractorProfile, Index, IndexKind,
        Level,
    },
    store::MemoryStore,
};
use cosmosearch_core::{
    AllowedRows, Built, BuiltBy, Chunk, CommitId, Hits, IndexError, KeyRange, Query, SegmentEntry,
    SegmentId, SegmentSource, ShardEntry, ShardId,
};
use cosmosearch_store::{ObjectKey, WriteStore};

mod heat;
mod heat_promotion;
mod promotion;
mod reading;

#[derive(Debug)]
struct FakeExtractor(ExtractorProfile);

impl Extractor for FakeExtractor {
    fn profile(&self) -> &ExtractorProfile {
        &self.0
    }

    fn extensions(&self) -> &[&str] {
        &[]
    }

    fn extract(&self, _: &[u8], _: &ExtractLimits) -> Result<Extraction, ExtractFailure> {
        Ok(Extraction::new(vec![]))
    }
}

#[derive(Debug)]
struct FakeIndex(IndexKind);

#[async_trait]
impl Index for FakeIndex {
    fn kind(&self) -> &IndexKind {
        &self.0
    }

    fn level(&self) -> Level {
        Level::L1
    }

    async fn build(&self, _: &SegmentId, _: &[Chunk]) -> Result<Built, IndexError> {
        Ok(Built::default())
    }

    async fn search(
        &self,
        _: &SegmentSource,
        _: &[SegmentEntry],
        _: &AllowedRows,
        _: &Query,
        _: usize,
    ) -> Result<Hits, IndexError> {
        Ok(Hits::default())
    }
}

fn extractor(name: &str) -> FakeExtractor {
    FakeExtractor(ExtractorProfile::new(name).unwrap())
}

fn index(name: &str) -> FakeIndex {
    FakeIndex(IndexKind::new(name).unwrap())
}

fn data_location() -> Location {
    "s3://example-bucket/docs/".parse().unwrap()
}

fn stores(store: &MemoryStore) -> (DataStore, MetadataStore) {
    let data = DataStore::new(store.clone(), data_location()).unwrap();
    let metadata =
        MetadataStore::new(store.clone(), "s3://example-bucket/index".parse().unwrap()).unwrap();
    (data, metadata)
}

fn segment(id: &str, level: Level, kind: Option<&str>, profiles: &[&str]) -> SegmentEntry {
    let extractors = profiles
        .iter()
        .map(|p| ExtractorProfile::new(*p).unwrap())
        .collect();
    let kind = kind.map(|k| IndexKind::new(k).unwrap());
    SegmentEntry::new(
        SegmentId::new(id).unwrap(),
        level,
        BuiltBy::new(kind, extractors),
        None,
        vec![],
    )
}

async fn index_holding(segments: Vec<SegmentEntry>) -> (DataStore, MetadataStore) {
    let store = MemoryStore::new();
    let (data, metadata) = stores(&store);
    let outcome = metadata
        .create(data.location(), FileSource::Listing)
        .await
        .unwrap();
    let CreateOutcome::Created(handle) = outcome else {
        panic!("expected created, got {outcome:?}");
    };
    let head = handle.head().await.unwrap().unwrap();
    let shards = [(
        ShardId::new("0").unwrap(),
        ShardEntry::new(KeyRange::all(), segments),
    )];
    let next = head
        .successor(
            CommitId::new("c2").unwrap(),
            shards.into_iter().collect(),
            BTreeSet::new(),
        )
        .unwrap();
    handle.commit(&next).await.unwrap();
    (data, metadata)
}

fn with_plugins(
    mut builder: EngineBuilder,
    extractors: &[&str],
    indexes: &[&str],
) -> EngineBuilder {
    for name in extractors {
        builder = builder.extractor(extractor(name));
    }
    for name in indexes {
        builder = builder.index(index(name));
    }
    builder
}

async fn check(
    segments: Vec<SegmentEntry>,
    extractors: &[&str],
    indexes: &[&str],
) -> Option<String> {
    let (data, metadata) = index_holding(segments).await;
    with_plugins(Engine::builder(data, metadata), extractors, indexes)
        .open()
        .await
        .err()
        .map(|err| err.to_string())
}

#[tokio::test]
async fn opens_when_the_host_added_every_plugin_the_index_needs() {
    let segments = vec![segment("s1", Level::L1, Some("bm25"), &["pdf-1"])];
    assert_eq!(check(segments, &["pdf-1"], &["bm25"]).await, None);
}

#[tokio::test]
async fn an_engine_reports_the_plugins_it_was_given() {
    let (data, metadata) = index_holding(vec![]).await;
    let engine = with_plugins(
        Engine::builder(data, metadata),
        &["pdf-1", "docx-1"],
        &["bm25"],
    )
    .open()
    .await
    .unwrap();
    let extractors: Vec<_> = engine.extractors().map(ExtractorProfile::as_str).collect();
    let indexes: Vec<_> = engine.indexes().map(IndexKind::as_str).collect();
    assert_eq!(extractors, ["docx-1", "markdown-2", "pdf-1", "text-2"]);
    assert_eq!(indexes, ["bm25"]);
}

#[tokio::test]
async fn names_every_plugin_the_index_needs_and_the_host_left_out() {
    let pdf_bm25 = || vec![segment("s1", Level::L1, Some("bm25"), &["pdf-1"])];
    let needs = "this index needs plug-ins that were not added:";
    let cases: [(&[&str], &[&str], String); 4] = [
        (&[], &[], format!("{needs} extractor `pdf-1`, index `bm25`")),
        (&["pdf-1"], &[], format!("{needs} index `bm25`")),
        (&[], &["bm25"], format!("{needs} extractor `pdf-1`")),
        (&["docx-1"], &["bm25"], format!("{needs} extractor `pdf-1`")),
    ];
    for (extractors, indexes, expected) in cases {
        assert_eq!(check(pdf_bm25(), extractors, indexes).await, Some(expected));
    }
}

#[tokio::test]
async fn a_missing_plugin_is_named_once_across_segments() {
    let segments = vec![
        segment("s1", Level::L1, Some("bm25"), &[]),
        segment("s2", Level::L1, Some("bm25"), &[]),
    ];
    assert_eq!(
        check(segments, &[], &[]).await.unwrap(),
        "this index needs plug-ins that were not added: index `bm25`"
    );
}

#[tokio::test]
async fn needs_no_plugins_for_an_empty_index_or_kernel_segments() {
    assert_eq!(check(vec![], &[], &[]).await, None);
    let kernel = vec![segment("s1", Level::L0, None, &[])];
    assert_eq!(check(kernel, &[], &[]).await, None);
}

#[tokio::test]
async fn refuses_a_plugin_added_twice() {
    let cases: [(&[&str], &[&str], &str); 3] = [
        (
            &["pdf-1", "pdf-1"],
            &[],
            "extractor `pdf-1` was added twice",
        ),
        (&[], &["bm25", "bm25"], "index `bm25` was added twice"),
        (&["text-2"], &[], "extractor `text-2` was added twice"),
    ];
    for (extractors, indexes, expected) in cases {
        assert_eq!(check(vec![], extractors, indexes).await.unwrap(), expected);
    }
}

#[tokio::test]
async fn refuses_a_fusion_setting_that_is_not_finite_and_positive() {
    let (data, metadata) = index_holding(vec![]).await;
    let builder = || Engine::builder(data.clone(), metadata.clone());
    let err = builder().fusion_k(0.0).open().await.unwrap_err();
    assert_eq!(
        err.to_string(),
        "the fusion rank constant must be finite and above zero, not 0"
    );
    let bm25 = IndexKind::new("bm25").unwrap();
    let err = builder()
        .fusion_weight(bm25.clone(), f64::NAN)
        .open()
        .await
        .unwrap_err();
    assert_eq!(
        err.to_string(),
        "the fusion weight of index `bm25` must be finite and above zero, not NaN"
    );
    builder()
        .fusion_k(20.0)
        .fusion_weight(bm25, 2.0)
        .open()
        .await
        .unwrap();
}

#[tokio::test]
async fn reports_a_location_without_a_usable_index() {
    let store = MemoryStore::new();
    let (data, metadata) = stores(&store);
    let err = Engine::builder(data, metadata).open().await.unwrap_err();
    assert!(matches!(err, Error::NotIndexed), "{err}");

    let stray = ObjectKey::new("index/stray").unwrap();
    store
        .put_create(&stray, Bytes::from_static(b"x"))
        .await
        .unwrap();
    let (data, metadata) = stores(&store);
    let err = Engine::builder(data, metadata).open().await.unwrap_err();
    assert!(matches!(err, Error::Incomplete), "{err}");
}

#[tokio::test]
async fn passes_on_a_refusal_from_the_store() {
    let (_, metadata) = index_holding(vec![]).await;
    let other = DataStore::new(
        MemoryStore::new(),
        "s3://example-bucket/other/".parse().unwrap(),
    )
    .unwrap();
    let err = Engine::builder(other, metadata).open().await.unwrap_err();
    assert!(matches!(err, Error::Store(_)), "{err}");
}

#[test]
fn opening_is_send() {
    fn assert_send<T: Send>(_: &T) {}
    let store = MemoryStore::new();
    let (data, metadata) = stores(&store);
    assert_send(&Engine::builder(data, metadata).open());
}
