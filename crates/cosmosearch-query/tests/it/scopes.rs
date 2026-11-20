use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use cosmosearch_core::{
    AllowedRows, AttributeValue, Built, Chunk, FileSource, Filter, Hits, Index, IndexError,
    IndexKind, Level, Query, Scope, SegmentEntry, SegmentId, SegmentSource, ShardEntry, ShardId,
    SharedHeat, SharedIndex, Signal,
};
use cosmosearch_indexer::{AttributeChange, FileChange};
use cosmosearch_query::{Reader, ReaderOptions, SearchRequest, SearchResults};
use cosmosearch_store::{Fault, FaultStore, MetadataStore, Op};

use super::{Fixture, Tally, plugins};

pub(super) fn keys(results: &SearchResults) -> Vec<&str> {
    results
        .hits()
        .iter()
        .map(|hit| hit.source().location().key())
        .collect()
}

async fn search(reader: &Reader, text: &str, scope: Scope) -> SearchResults {
    reader
        .search(&SearchRequest::new(Query::new(text)).scope(scope))
        .await
        .unwrap()
}

pub(super) async fn faulty_reader(f: &Fixture, options: ReaderOptions) -> (Reader, FaultStore) {
    let faults = FaultStore::new(f.store.clone());
    let metadata =
        MetadataStore::new(faults.clone(), "s3://example-bucket/index".parse().unwrap()).unwrap();
    let reader = Reader::open(
        f.data.clone(),
        metadata,
        FileSource::HostNamed,
        plugins(),
        options,
    )
    .await
    .unwrap();
    (reader, faults)
}

pub(super) async fn fail_segments_of(f: &Fixture, faults: &FaultStore, shard: &str) {
    let head = f.handle.head().await.unwrap().unwrap();
    let shard = head.shards().get(&ShardId::new(shard).unwrap()).unwrap();
    assert!(!shard.segments().is_empty(), "the shard holds segments");
    for segment in shard.segments() {
        faults.inject(Op::Get, segment.id().as_str(), Fault::Crash);
    }
}

#[tokio::test]
async fn a_scoped_search_opens_only_the_shards_of_its_scope() {
    let f = Fixture::new().await;
    f.split_at("docs/m").await;
    f.index(&[("docs/a.txt", b"refund policy text")]).await;
    f.index(&[("docs/z.txt", b"refund rules")]).await;
    let (reader, faults) = faulty_reader(&f, ReaderOptions::new()).await;
    fail_segments_of(&f, &faults, "z").await;

    let found = search(&reader, "refund", Scope::prefix("docs/a")).await;
    assert_eq!(keys(&found), ["docs/a.txt"]);
    assert!(found.coverage().unread().is_empty());
    reader
        .search(&SearchRequest::new(Query::new("refund")))
        .await
        .unwrap_err();
}

#[tokio::test]
async fn a_scope_keeps_only_its_files_within_a_shard() {
    let f = Fixture::new().await;
    f.index(&[
        ("docs/a/1.txt", b"refund policy"),
        ("docs/b/1.txt", b"refund rules"),
        ("docs/b2.txt", b"refund terms"),
    ])
    .await;
    let cases: [(Scope, &[&str]); 5] = [
        (Scope::prefix("docs/a/"), &["docs/a/1.txt"]),
        (Scope::prefix("docs/b"), &["docs/b/1.txt", "docs/b2.txt"]),
        (
            Scope::prefixes(&["docs/b/", "docs/a/"]),
            &["docs/a/1.txt", "docs/b/1.txt"],
        ),
        (Scope::prefix("logs/"), &[]),
        (Scope::prefixes(&[]), &[]),
    ];
    for (scope, expected) in cases {
        let found = search(&f.reader, "refund", scope.clone()).await;
        let mut found = keys(&found);
        found.sort_unstable();
        assert_eq!(found, expected, "{scope:?}");
    }
}

#[derive(Debug, Clone)]
struct Seen(IndexKind, Arc<Mutex<Vec<Vec<SegmentId>>>>);

#[async_trait]
impl Index for Seen {
    fn kind(&self) -> &IndexKind {
        &self.0
    }

    fn level(&self) -> Level {
        Level::L2
    }

    async fn build(&self, _: &SegmentId, _: &[Chunk]) -> Result<Built, IndexError> {
        Ok(Built::default())
    }

    async fn search(
        &self,
        _: &SegmentSource,
        segments: &[SegmentEntry],
        _: &AllowedRows,
        _: &Query,
        _: usize,
    ) -> Result<Hits, IndexError> {
        let ids = segments
            .iter()
            .map(|segment| segment.id().clone())
            .collect();
        self.1.lock().unwrap().push(ids);
        Ok(Hits::default())
    }
}

#[tokio::test]
async fn an_index_is_given_only_segments_with_chunks_the_search_may_return() {
    let f = Fixture::new().await;
    f.index(&[("docs/a.txt", b"refund policy")]).await;
    f.index(&[("docs/b.txt", b"refund rules")]).await;
    let change = AttributeChange::new("docs/b.txt")
        .attribute("owner", AttributeValue::One("legal".to_owned()));
    f.indexer
        .apply(vec![FileChange::Attributes(change)])
        .await
        .unwrap();
    let seen = Seen(IndexKind::new("seen").unwrap(), Arc::default());
    f.add_twin_segments(&seen).await;
    let mut plugins = plugins();
    plugins.add_index(SharedIndex::new(seen.clone())).unwrap();
    let reader = Reader::open(
        f.data.clone(),
        f.metadata.clone(),
        FileSource::HostNamed,
        plugins,
        ReaderOptions::new(),
    )
    .await
    .unwrap();
    let twins: Vec<SegmentId> = f
        .handle
        .head()
        .await
        .unwrap()
        .unwrap()
        .shards()
        .values()
        .flat_map(ShardEntry::segments)
        .filter(|segment| segment.built_by().index() == Some(&seen.0))
        .map(|segment| segment.id().clone())
        .collect();
    assert_eq!(twins.len(), 2);

    let request = SearchRequest::new(Query::new("refund"));
    let legal = request
        .clone()
        .filter(Filter::new().equals("owner", "legal"));
    for request in [
        request.clone(),
        legal,
        request.clone().scope(Scope::prefix("docs/a")),
        request.scope(Scope::prefix("docs/c")),
    ] {
        reader.search(&request).await.unwrap();
    }
    let given: Vec<BTreeSet<SegmentId>> = std::mem::take(&mut *seen.1.lock().unwrap())
        .into_iter()
        .map(BTreeSet::from_iter)
        .collect();
    let (a, b) = (twins[0].clone(), twins[1].clone());
    assert_eq!(
        given,
        [
            BTreeSet::from([a.clone(), b.clone()]),
            BTreeSet::from([b]),
            BTreeSet::from([a]),
        ]
    );
}

#[tokio::test]
async fn a_scoped_search_records_heat_only_for_its_shards() {
    let f = Fixture::new().await;
    f.split_at("docs/m").await;
    f.index(&[("docs/a.txt", b"refund policy text")]).await;
    f.index(&[("docs/z.txt", b"refund rules")]).await;
    let tally = Tally::default();
    let (reader, _) = faulty_reader(
        &f,
        ReaderOptions::new().heat(SharedHeat::new(tally.clone())),
    )
    .await;
    let (a, z) = (ShardId::new("a").unwrap(), ShardId::new("z").unwrap());

    search(&reader, "refund", Scope::prefix("docs/z")).await;
    search(&reader, "refund", Scope::prefix("docs/")).await;
    search(&reader, "refund", Scope::prefix("docs/q")).await;
    assert_eq!(
        tally.take(),
        [
            (Signal::Search, vec![z.clone()]),
            (Signal::Hit, vec![z.clone()]),
            (Signal::Search, vec![a.clone(), z.clone()]),
            (Signal::Hit, vec![a, z]),
        ]
    );
}
