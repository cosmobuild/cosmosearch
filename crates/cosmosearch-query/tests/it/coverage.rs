use cosmosearch_core::{
    AttributeValue, Budget, FileSource, Filter, Level, Query, Scope, ShardId, SharedHeat, Signal,
    Status,
};
use cosmosearch_indexer::{AttributeChange, FileChange};
use cosmosearch_query::{Coverage, Error, Reader, ReaderOptions, SearchRequest, SearchResults};
use cosmosearch_store::{ObjectKey, WriteStore};

use super::{Fixture, Tally, plugins};

fn counts(coverage: &Coverage) -> [u64; 5] {
    [
        coverage.searched(),
        coverage.pending(),
        coverage.stale(),
        coverage.failed(),
        coverage.unsupported(),
    ]
}

async fn search(reader: &Reader, request: SearchRequest) -> SearchResults {
    reader.search(&request).await.unwrap()
}

fn refund() -> SearchRequest {
    SearchRequest::new(Query::new("refund"))
}

async fn every_state() -> Fixture {
    let f = Fixture::new().await;
    f.add(&[("docs/moved.txt", b"refund before")]).await;
    f.put("docs/moved.txt", b"refund after").await;
    f.index(&[
        ("docs/a.txt", b"refund policy"),
        ("docs/bad.pages", b"refund \xff\xfe"),
        ("docs/scan.xyz", b"refund"),
    ])
    .await;
    f.add(&[("docs/new.txt", b"refund later")]).await;
    let change = AttributeChange::new("docs/a.txt")
        .attribute("owner", AttributeValue::One("legal".to_owned()));
    f.indexer
        .apply(vec![FileChange::Attributes(change)])
        .await
        .unwrap();
    f.reader.refresh().await.unwrap();
    f
}

#[tokio::test]
async fn a_scope_prefix_outside_the_data_location_is_refused() {
    let f = Fixture::new().await;
    f.index(&[("docs/a.txt", b"refund policy")]).await;
    for prefix in ["a.txt", "teama/", "docsx/"] {
        let err = f
            .reader
            .search(&refund().scope(Scope::prefixes(&["docs/", prefix])))
            .await
            .unwrap_err();
        assert!(
            matches!(&err, Error::ScopeOutsideData { prefix: found } if found == prefix),
            "{err}"
        );
    }
    for prefix in ["docs/a", "docs", "d", ""] {
        let found = search(&f.reader, refund().scope(Scope::prefix(prefix))).await;
        assert_eq!(found.hits().len(), 1, "{prefix}");
    }
}

#[tokio::test]
async fn coverage_counts_each_state_of_the_files_in_scope() {
    let f = every_state().await;
    let found = search(&f.reader, refund()).await;
    assert_eq!(found.status(), Status::Found);
    assert_eq!(found.hits().len(), 1);
    assert_eq!(counts(found.coverage()), [1, 1, 1, 1, 1]);
    assert!(!found.coverage().is_complete());

    let cases: [(SearchRequest, [u64; 5], bool); 4] = [
        (
            refund().scope(Scope::prefix("docs/a")),
            [1, 0, 0, 0, 0],
            true,
        ),
        (
            refund().scope(Scope::prefix("docs/s")),
            [0, 0, 0, 0, 1],
            true,
        ),
        (
            refund().scope(Scope::prefix("docs/n")),
            [0, 1, 0, 0, 0],
            false,
        ),
        (
            refund().filter(Filter::new().equals("owner", "legal")),
            [1, 0, 0, 0, 0],
            true,
        ),
    ];
    for (request, expected, complete) in cases {
        let found = search(&f.reader, request).await;
        assert_eq!(counts(found.coverage()), expected, "{expected:?}");
        assert_eq!(found.coverage().is_complete(), complete, "{expected:?}");
    }
}

#[tokio::test]
async fn a_search_without_hits_is_inconclusive_whatever_it_covered() {
    let f = every_state().await;
    let none = search(&f.reader, SearchRequest::new(Query::new("zebracorn"))).await;
    assert_eq!(none.status(), Status::Inconclusive);
    assert_eq!(counts(none.coverage()), [1, 1, 1, 1, 1]);

    let legal = Filter::new().equals("owner", "legal");
    let complete = search(
        &f.reader,
        SearchRequest::new(Query::new("zebracorn")).filter(legal),
    )
    .await;
    assert_eq!(complete.status(), Status::Inconclusive);
    assert!(complete.coverage().is_complete());

    let nobody = Filter::new().equals("owner", "nobody");
    let empty = search(&f.reader, refund().filter(nobody)).await;
    assert_eq!(empty.status(), Status::Inconclusive);
    assert_eq!(counts(empty.coverage()), [0; 5]);
}

#[tokio::test]
async fn a_shard_left_unread_makes_coverage_incomplete() {
    let f = Fixture::new().await;
    f.split_at("docs/m").await;
    f.index(&[("docs/a.txt", b"refund words")]).await;
    f.index(&[("docs/z.txt", b"refund rules")]).await;
    let head = f.handle.head().await.unwrap().unwrap();
    let high = head.shards().get(&ShardId::new("z").unwrap()).unwrap();
    let built = high
        .segments()
        .iter()
        .find(|segment| segment.level() == Level::L1)
        .unwrap();
    for file in built.files() {
        let key = format!(
            "index/{}/segments/{}/{}",
            f.handle.id(),
            built.id(),
            file.name()
        );
        f.store.delete(&ObjectKey::new(key).unwrap()).await.unwrap();
    }

    let found = search(&f.reader, refund()).await;
    assert_eq!(found.status(), Status::Found);
    let coverage = found.coverage();
    assert_eq!(counts(coverage), [1, 0, 0, 0, 0]);
    let unread: Vec<&str> = coverage.unread().iter().map(ShardId::as_str).collect();
    assert_eq!(unread, ["z"]);
    assert!(!coverage.is_complete());
}

#[tokio::test]
async fn an_inconclusive_search_records_unmet_heat_for_the_shards_it_left_unexplored() {
    let f = Fixture::new().await;
    f.split_at("docs/m").await;
    f.index(&[("docs/a.txt", b"refund words")]).await;
    f.index(&[("docs/z.txt", b"refund rules")]).await;
    let tally = Tally::default();
    let reader = Reader::open(
        f.data.clone(),
        f.metadata.clone(),
        FileSource::HostNamed,
        plugins(),
        ReaderOptions::new().heat(SharedHeat::new(tally.clone())),
    )
    .await
    .unwrap();
    let (a, z) = (ShardId::new("a").unwrap(), ShardId::new("z").unwrap());
    let one = Budget::new().max_shards(1);

    let none = search(&reader, SearchRequest::new(Query::new("rules")).budget(one)).await;
    assert_eq!(none.status(), Status::Inconclusive);
    assert!(!none.coverage().is_complete());
    search(&reader, refund().budget(one)).await;
    assert_eq!(
        tally.take(),
        [
            (Signal::Search, vec![a.clone()]),
            (Signal::Unmet, vec![z]),
            (Signal::Search, vec![a.clone()]),
            (Signal::Hit, vec![a]),
        ]
    );
}

#[tokio::test]
async fn a_file_read_only_in_part_is_searched_and_leaves_coverage_incomplete() {
    let f = Fixture::new().await;
    f.index(&[
        ("docs/a.txt", b"refund policy"),
        ("docs/report.pages", b"refund schedule\x0c[image]\x0cterms"),
    ])
    .await;
    f.reader.refresh().await.unwrap();
    let found = search(&f.reader, refund()).await;
    assert_eq!(found.hits().len(), 2);
    assert_eq!(found.coverage().searched(), 2);
    assert_eq!(found.coverage().partly_searched(), 1);
    assert!(!found.coverage().is_complete());
}

#[tokio::test]
async fn a_file_with_no_page_read_fails_and_leaves_coverage_incomplete() {
    let f = Fixture::new().await;
    f.index(&[
        ("docs/a.txt", b"refund policy"),
        ("docs/scan.pages", b"[image]\x0c[image]"),
        ("docs/empty.pages", b"   "),
    ])
    .await;
    f.reader.refresh().await.unwrap();
    let found = search(&f.reader, refund()).await;
    assert_eq!(found.hits().len(), 1);
    assert_eq!(found.coverage().searched(), 1);
    assert_eq!(found.coverage().failed(), 1);
    assert_eq!(found.coverage().unsupported(), 1);
    assert!(!found.coverage().is_complete());
}
