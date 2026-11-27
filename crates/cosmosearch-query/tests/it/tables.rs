use cosmosearch_core::{FileSource, Query, Scope};
use cosmosearch_query::{Reader, ReaderOptions, SearchRequest, TableCache};

use super::{Fixture, plugins};

async fn reader(f: &Fixture, tables: &TableCache) -> Reader {
    Reader::open(
        f.data.clone(),
        f.metadata.clone(),
        FileSource::HostNamed,
        plugins(),
        ReaderOptions::new().tables(tables.clone()),
    )
    .await
    .unwrap()
}

async fn many_files(f: &Fixture) {
    let texts: Vec<(String, String)> = (0..40)
        .map(|at| {
            let folder = if at % 2 == 0 { "even" } else { "odd" };
            let words = (0..30)
                .map(|line| format!("refund {at} line {line} policy number {}.", at * line))
                .collect::<Vec<_>>()
                .join("\n\n");
            (format!("docs/{folder}/{at}.txt"), words)
        })
        .collect();
    let files: Vec<(&str, &[u8])> = texts
        .iter()
        .map(|(key, text)| (key.as_str(), text.as_bytes()))
        .collect();
    f.index(&files).await;
}

fn requests() -> Vec<SearchRequest> {
    vec![
        SearchRequest::new(Query::new("refund 7 policy")).limit(20),
        SearchRequest::new(Query::new("line 12")).scope(Scope::prefix("docs/odd/")),
        SearchRequest::new(Query::new("number 90")).limit(5),
    ]
}

#[tokio::test]
async fn a_table_cache_too_small_for_any_table_gives_the_same_results() {
    let f = Fixture::new().await;
    many_files(&f).await;
    let roomy = TableCache::new(u64::MAX);
    let tiny = TableCache::new(1);
    let (with_room, starved) = (reader(&f, &roomy).await, reader(&f, &tiny).await);
    for _ in 0..2 {
        for request in requests() {
            let expected = with_room.search(&request).await.unwrap();
            let found = starved.search(&request).await.unwrap();
            assert_eq!(found.hits(), expected.hits());
            let hit = found.hits().first().unwrap();
            let read = starved.expand(hit.source(), 1, 1).await.unwrap();
            let expected = with_room.expand(hit.source(), 1, 1).await.unwrap();
            assert_eq!(read.into_chunks(), expected.into_chunks());
        }
    }
    assert_eq!(tiny.bytes(), 0);
    assert!(roomy.bytes() > 0);
    assert!(starved.resident_bytes() > 0, "chunk indexes are kept");
    assert!(starved.resident_bytes() < with_room.resident_bytes());
}

#[tokio::test]
async fn readers_sharing_a_table_cache_stay_under_its_budget() {
    let f = Fixture::new().await;
    many_files(&f).await;
    let probe = TableCache::new(u64::MAX);
    let alone = reader(&f, &probe).await;
    for request in requests() {
        alone.search(&request).await.unwrap();
    }
    let budget = probe.bytes() / 2;
    let shared = TableCache::new(budget);
    let readers = [reader(&f, &shared).await, reader(&f, &shared).await];
    for reader in &readers {
        for request in requests() {
            reader.search(&request).await.unwrap();
            assert!(shared.bytes() <= budget);
        }
    }
    assert!(shared.bytes() > 0);
}

fn long_file() -> String {
    (0..1100)
        .map(|at| format!("marker{at} {}", "word ".repeat(300)))
        .collect::<Vec<_>>()
        .join("\n\n")
}

#[tokio::test]
async fn reading_a_passage_of_a_long_file_reads_only_the_groups_before_it() {
    let f = Fixture::new().await;
    let text = long_file();
    f.index(&[("docs/long.txt", text.as_bytes())]).await;
    let roomy = reader(&f, &TableCache::new(u64::MAX)).await;
    let starved = reader(&f, &TableCache::new(1)).await;
    for marker in ["marker3", "marker700", "marker1090"] {
        let request = SearchRequest::new(Query::new(marker)).limit(1);
        let expected = roomy.search(&request).await.unwrap();
        let found = starved.search(&request).await.unwrap();
        assert_eq!(found.hits(), expected.hits());
        let hit = found.hits().first().unwrap();
        let read = starved.expand(hit.source(), 2, 2).await.unwrap();
        let wanted = roomy.expand(hit.source(), 2, 2).await.unwrap();
        assert_eq!(read.into_chunks(), wanted.into_chunks());
    }
    let request = SearchRequest::new(Query::new("marker3")).limit(1);
    let first = starved.search(&request).await.unwrap();
    let hit = first.hits().first().unwrap().source().clone();
    let read = starved.read(&hit).await.unwrap();
    assert_eq!(
        read.usage().object_reads(),
        4,
        "a file group, the first record group twice with no room to keep it, and the file"
    );
    let page = starved.chunks("docs/long.txt", 0, 3).await.unwrap();
    assert_eq!(
        page.usage().object_reads(),
        3,
        "a file group, the first record group and the file"
    );
    let late = starved.chunks("docs/long.txt", 1050, 3).await.unwrap();
    let expected = roomy.chunks("docs/long.txt", 1050, 3).await.unwrap();
    assert_eq!(late.into_chunks(), expected.into_chunks());
}

#[tokio::test]
async fn a_search_reads_each_group_of_a_file_table_once() {
    let f = Fixture::new().await;
    many_files(&f).await;
    let request = SearchRequest::new(Query::new("refund")).limit(40);
    let roomy = reader(&f, &TableCache::new(u64::MAX)).await;
    let cold = roomy.search(&request).await.unwrap();
    let starved = reader(&f, &TableCache::new(1)).await;
    starved.search(&request).await.unwrap();
    let again = starved.search(&request).await.unwrap();
    assert_eq!(again.hits(), cold.hits());
    assert!(
        again.usage().object_reads() <= cold.usage().object_reads(),
        "{:?} against {:?}",
        again.usage(),
        cold.usage()
    );
}
