use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use cosmosearch_core::{
    FileSource, IndexId, PassageCache, Query, SharedPassages, SharedReranker, SourceRef,
};
use cosmosearch_query::{Reader, ReaderOptions, SearchRequest, SearchResults};

use super::rerank::fake;
use super::{Fixture, plugins};

type Texts = BTreeMap<(IndexId, String), Arc<str>>;

#[derive(Debug, Clone, Default)]
struct Kept(Arc<Mutex<Texts>>);

impl PassageCache for Kept {
    fn get(&self, index: &IndexId, reference: &SourceRef) -> Option<Arc<str>> {
        let key = (index.clone(), reference.to_string());
        self.0.lock().unwrap().get(&key).cloned()
    }

    fn put(&self, index: &IndexId, reference: &SourceRef, text: Arc<str>) {
        let key = (index.clone(), reference.to_string());
        self.0.lock().unwrap().insert(key, text);
    }
}

async fn reader(f: &Fixture, options: ReaderOptions, reranker: bool) -> Reader {
    let mut plugins = plugins();
    if reranker {
        plugins.set_reranker(SharedReranker::new(fake()));
    }
    Reader::open(
        f.data.clone(),
        f.metadata.clone(),
        FileSource::HostNamed,
        plugins,
        options,
    )
    .await
    .unwrap()
}

async fn search(reader: &Reader, query: &str) -> SearchResults {
    reader
        .search(&SearchRequest::new(Query::new(query)))
        .await
        .unwrap()
}

async fn three_files(f: &Fixture) {
    f.index(&[
        ("docs/a.txt", b"Refund policy: a refund within 30 days."),
        ("docs/b.txt", b"refund rules"),
        ("docs/c.txt", b"no refund on sale items"),
    ])
    .await;
}

#[tokio::test]
async fn a_hit_carries_its_passage_with_the_querys_words_marked() {
    let f = Fixture::new().await;
    three_files(&f).await;
    let plain = search(&f.reader, "refund days").await;
    assert!(plain.hits().iter().all(|hit| hit.snippet().is_none()));

    let reader = reader(&f, ReaderOptions::new().snippets(10), false).await;
    let results = search(&reader, "refund days").await;
    assert_eq!(results.hits().len(), 3);
    for hit in results.hits() {
        let snippet = hit.snippet().unwrap();
        let passage = f.read(hit.source()).await;
        assert_eq!(snippet.text(), passage.text());
        let marked: Vec<String> = snippet
            .matches()
            .iter()
            .map(|range| snippet.text().get(range.clone()).unwrap().to_lowercase())
            .collect();
        assert!(!marked.is_empty(), "every hit holds a word of the query");
        assert!(marked.iter().all(|word| word == "refund" || word == "days"));
    }
    assert_eq!(
        results.usage().object_reads(),
        plain.usage().object_reads() + 3,
        "one read for each file"
    );
}

#[tokio::test]
async fn other_forms_of_a_word_are_marked_as_the_keyword_index_matched_them() {
    let f = Fixture::new().await;
    f.index(&[(
        "docs/a.txt",
        b"Refunds are paid weekly. The refunded orders ship back.",
    )])
    .await;
    let reader = reader(&f, ReaderOptions::new().snippets(10), false).await;
    let results = search(&reader, "refund").await;
    let snippet = results.hits().first().unwrap().snippet().unwrap();
    let marked: Vec<&str> = snippet
        .matches()
        .iter()
        .map(|range| snippet.text().get(range.clone()).unwrap())
        .collect();
    assert_eq!(marked, ["Refunds", "refunded"]);
}

#[tokio::test]
async fn at_most_the_set_number_of_files_is_read_for_snippets() {
    let f = Fixture::new().await;
    three_files(&f).await;
    let plain = search(&f.reader, "refund").await;
    let reader = reader(&f, ReaderOptions::new().snippets(2), false).await;
    let results = search(&reader, "refund").await;
    let with: Vec<bool> = results
        .hits()
        .iter()
        .map(|hit| hit.snippet().is_some())
        .collect();
    assert_eq!(with, [true, true, false], "the best hits' files come first");
    assert_eq!(
        results.usage().object_reads(),
        plain.usage().object_reads() + 2
    );
}

#[tokio::test]
async fn a_warm_repeat_and_a_reranked_search_read_no_file_twice() {
    let f = Fixture::new().await;
    three_files(&f).await;
    let plain = search(&f.reader, "refund").await.usage().object_reads();
    let kept = Kept::default();
    let options = || {
        ReaderOptions::new()
            .snippets(10)
            .passage_cache(SharedPassages::new(kept.clone()))
    };

    let reranked = reader(&f, ReaderOptions::new().snippets(10), true).await;
    let once = search(&reranked, "refund").await;
    assert!(once.hits().iter().all(|hit| hit.snippet().is_some()));
    assert_eq!(
        once.usage().object_reads(),
        plain + 3,
        "the reranker and the snippets share one read of each file"
    );

    let cached = reader(&f, options(), false).await;
    assert_eq!(
        search(&cached, "refund").await.usage().object_reads(),
        plain + 3
    );
    let warm = search(&cached, "refund rules").await;
    assert!(warm.hits().iter().all(|hit| hit.snippet().is_some()));
    let other = search(&f.reader, "refund rules")
        .await
        .usage()
        .object_reads();
    assert_eq!(warm.usage().object_reads(), other, "no file is read again");

    let limited = ReaderOptions::new()
        .snippets(1)
        .passage_cache(SharedPassages::new(kept.clone()));
    let limited = reader(&f, limited, false).await;
    let all = search(&limited, "refund").await;
    assert!(
        all.hits().iter().all(|hit| hit.snippet().is_some()),
        "cached passages do not count against the limit"
    );
}
