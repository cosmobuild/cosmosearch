use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

use cosmosearch_core::{
    AttributeValue, FakeReranker, FileSource, Filter, Query, RerankCause, RerankError,
    RerankScores, Reranker, RerankerIdentity, SharedReranker, Usage,
};
use cosmosearch_indexer::{AttributeChange, FileChange};
use cosmosearch_query::{
    Boost, Reader, ReaderOptions, RerankFailure, Reranking, SearchHit, SearchRequest, SearchResults,
};

use super::{Fixture, long_text, plugins};

const QUERY: &str = "refund";

pub(crate) fn identity() -> RerankerIdentity {
    RerankerIdentity::new("fake-reranker", "1", 512).unwrap()
}

pub(crate) fn fake() -> FakeReranker {
    FakeReranker::new(identity())
}

async fn reader(f: &Fixture, reranker: Option<SharedReranker>, shortlist: usize) -> Reader {
    let mut plugins = plugins();
    if let Some(reranker) = reranker {
        plugins.set_reranker(reranker);
    }
    let options = ReaderOptions::new().rerank_shortlist(shortlist);
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

async fn search(reader: &Reader, query: &str, limit: usize) -> SearchResults {
    reader
        .search(&SearchRequest::new(Query::new(query)).limit(limit))
        .await
        .unwrap()
}

fn keys(hits: &[SearchHit]) -> Vec<String> {
    hits.iter()
        .map(|hit| hit.source().location().key().to_owned())
        .collect()
}

async fn eight_files(f: &Fixture) {
    let bodies: Vec<(String, String)> = (0..8)
        .map(|i| {
            let body = format!("refund case {i} {}", "filler ".repeat(i));
            (format!("docs/f{i}.txt"), body)
        })
        .collect();
    let files: Vec<(&str, &[u8])> = bodies
        .iter()
        .map(|(key, body)| (key.as_str(), body.as_bytes()))
        .collect();
    f.index(&files).await;
}

async fn reranked(f: &Fixture, hits: &[SearchHit]) -> Vec<(String, f32)> {
    let mut scored = Vec::new();
    for hit in hits {
        let text = f.read(hit.source()).await.text().to_owned();
        let found = fake().rerank(QUERY, &[&text]).unwrap();
        let score = found.scores().first().copied().unwrap();
        scored.push((hit.source().location().key().to_owned(), score));
    }
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored
}

#[tokio::test]
async fn the_shortlist_is_reordered_and_the_hits_after_it_are_untouched() {
    let f = Fixture::new().await;
    eight_files(&f).await;
    let fused = search(&f.reader, QUERY, 8).await.hits().to_vec();
    assert_eq!(fused.len(), 8);
    let expected = reranked(&f, &fused[..5]).await;
    let expected_keys: Vec<&str> = expected.iter().map(|(key, _)| key.as_str()).collect();
    assert_ne!(expected_keys, keys(&fused[..5]), "the fixture must reorder");

    let reader = reader(&f, Some(SharedReranker::new(fake())), 5).await;
    let hits = search(&reader, QUERY, 8).await.hits().to_vec();
    assert_eq!(keys(&hits[..5]), expected_keys);
    for (hit, (_, score)) in hits.iter().zip(&expected) {
        assert_eq!(hit.rerank_score(), Some(*score));
        let before = fused.iter().find(|b| b.source() == hit.source()).unwrap();
        assert!((hit.score() - before.score()).abs() < 1e-9, "{hit:?}");
    }
    assert_eq!(&hits[5..], &fused[5..]);
    assert!(hits[5..].iter().all(|hit| hit.rerank_score().is_none()));
}

#[tokio::test]
async fn a_shortlist_longer_than_the_limit_is_reranked_before_the_cut() {
    let f = Fixture::new().await;
    eight_files(&f).await;
    let fused = search(&f.reader, QUERY, 8).await.hits().to_vec();
    let expected = reranked(&f, &fused[..6]).await;
    let reranking = reader(&f, Some(SharedReranker::new(fake())), 6).await;
    let hits = search(&reranking, QUERY, 2).await.hits().to_vec();
    let expected_keys: Vec<&str> = expected[..2].iter().map(|(key, _)| key.as_str()).collect();
    assert_eq!(keys(&hits), expected_keys);

    let off = reader(&f, Some(SharedReranker::new(fake())), 0).await;
    assert_eq!(search(&off, QUERY, 8).await.hits(), fused);
}

#[tokio::test]
async fn a_passage_that_cannot_be_read_keeps_its_fused_order_after_the_reranked() {
    let f = Fixture::new().await;
    eight_files(&f).await;
    let fused = search(&f.reader, QUERY, 8).await.hits().to_vec();
    let changed = [fused[1].source().clone(), fused[3].source().clone()];
    for source in &changed {
        f.put(source.location().key(), b"rewritten since indexing")
            .await;
    }
    let readable: Vec<SearchHit> = fused[..5]
        .iter()
        .filter(|hit| !changed.contains(hit.source()))
        .cloned()
        .collect();
    let expected = reranked(&f, &readable).await;

    let reader = reader(&f, Some(SharedReranker::new(fake())), 5).await;
    let hits = search(&reader, QUERY, 8).await.hits().to_vec();
    let mut expected_keys: Vec<String> = expected.into_iter().map(|(key, _)| key).collect();
    expected_keys.extend(keys(&fused[1..2]));
    expected_keys.extend(keys(&fused[3..4]));
    expected_keys.extend(keys(&fused[5..]));
    assert_eq!(keys(&hits), expected_keys);
    assert!(hits[..3].iter().all(|hit| hit.rerank_score().is_some()));
    assert!(hits[3..].iter().all(|hit| hit.rerank_score().is_none()));
}

#[tokio::test]
async fn reads_once_per_file_and_counts_reads_and_tokens_in_usage() {
    let f = Fixture::new().await;
    let text = long_text();
    f.index(&[
        ("docs/a.txt", text.as_bytes()),
        ("docs/b.txt", b"words words words"),
        ("docs/c.txt", b"words and more"),
    ])
    .await;
    let plain = reader(&f, None, 6).await;
    let reranking = reader(&f, Some(SharedReranker::new(fake())), 6).await;
    search(&plain, "words", 10).await;
    search(&reranking, "words", 10).await;

    let without = search(&plain, "words", 10).await;
    let with = search(&reranking, "words", 10).await;
    let shortlisted = &without.hits()[..6];
    let files: BTreeSet<String> = keys(shortlisted).into_iter().collect();
    assert!(files.len() < shortlisted.len(), "{files:?}");
    assert_eq!(
        with.usage().object_reads(),
        without.usage().object_reads() + u64::try_from(files.len()).unwrap(),
    );
    assert!(with.usage().bytes() > without.usage().bytes());
    assert_eq!(without.usage().tokens(), 0);

    let mut texts = Vec::new();
    for hit in shortlisted {
        texts.push(f.read(hit.source()).await.text().to_owned());
    }
    let texts: Vec<&str> = texts.iter().map(String::as_str).collect();
    let tokens = fake().rerank("words", &texts).unwrap().usage().tokens();
    assert_eq!(with.usage().tokens(), tokens);
}

#[derive(Debug)]
pub(crate) enum Fault {
    Fails,
    WrongCount,
    NotFinite,
}

#[derive(Debug)]
pub(crate) struct Broken(pub(crate) RerankerIdentity, pub(crate) Fault);

impl Reranker for Broken {
    fn identity(&self) -> &RerankerIdentity {
        &self.0
    }

    fn rerank(&self, _: &str, passages: &[&str]) -> Result<RerankScores, RerankError> {
        match self.1 {
            Fault::Fails => Err(RerankError::Model(RerankCause::new("the model is gone"))),
            Fault::WrongCount => Ok(RerankScores::new(vec![1.0], Usage::new())),
            Fault::NotFinite => Ok(RerankScores::new(
                vec![f32::NAN; passages.len()],
                Usage::new(),
            )),
        }
    }
}

type Expected = fn(&RerankFailure) -> bool;

#[tokio::test]
async fn a_reranker_that_fails_or_misfits_leaves_fused_order_and_says_so() {
    let f = Fixture::new().await;
    eight_files(&f).await;
    let fused = search(&f.reader, QUERY, 8).await;
    assert!(matches!(fused.reranking(), Reranking::Off));
    let cases: [(Fault, Expected); 3] = [
        (
            Fault::Fails,
            |failure| matches!(failure, RerankFailure::Reranker(cause) if cause.to_string().contains("model")),
        ),
        (Fault::WrongCount, |failure| {
            matches!(
                failure,
                RerankFailure::WrongCount {
                    expected: 5,
                    found: 1
                }
            )
        }),
        (Fault::NotFinite, |failure| {
            matches!(failure, RerankFailure::NotFinite)
        }),
    ];
    for (fault, expected) in cases {
        let broken = SharedReranker::new(Broken(identity(), fault));
        let reader = reader(&f, Some(broken), 5).await;
        let results = search(&reader, QUERY, 8).await;
        assert_eq!(results.hits(), fused.hits());
        assert!(
            results
                .hits()
                .iter()
                .all(|hit| hit.rerank_score().is_none())
        );
        assert_eq!(results.usage().tokens(), 0);
        let Reranking::Failed(failure) = results.reranking() else {
            panic!("expected a failure, got {:?}", results.reranking());
        };
        assert!(expected(failure), "{failure:?}");
        assert!(!format!("{failure:?}").contains(QUERY), "{failure:?}");
    }
}

#[tokio::test]
async fn the_results_say_how_many_passages_were_scored_and_unreadable() {
    let f = Fixture::new().await;
    eight_files(&f).await;
    let fused = search(&f.reader, QUERY, 8).await.hits().to_vec();
    f.put(
        fused[2].source().location().key(),
        b"rewritten since indexing",
    )
    .await;
    let reranking = reader(&f, Some(SharedReranker::new(fake())), 5).await;
    let results = search(&reranking, QUERY, 8).await;
    assert!(
        matches!(
            results.reranking(),
            Reranking::Reranked {
                scored: 4,
                unreadable: 1
            }
        ),
        "{:?}",
        results.reranking()
    );
    let off = reader(&f, Some(SharedReranker::new(fake())), 0).await;
    assert!(matches!(
        search(&off, QUERY, 8).await.reranking(),
        Reranking::Off
    ));
}

#[tokio::test]
async fn the_reader_names_its_reranker() {
    let f = Fixture::new().await;
    let plain = reader(&f, None, 5).await;
    assert_eq!(plain.plugins().reranker(), None);
    let reranking = reader(&f, Some(SharedReranker::new(fake())), 5).await;
    assert_eq!(reranking.plugins().reranker(), Some(&identity()));
}

#[derive(Debug, Clone)]
struct Reads(RerankerIdentity, Arc<Mutex<Vec<String>>>);

impl Reranker for Reads {
    fn identity(&self) -> &RerankerIdentity {
        &self.0
    }

    fn rerank(&self, _: &str, passages: &[&str]) -> Result<RerankScores, RerankError> {
        let Ok(mut read) = self.1.lock() else {
            return Err(RerankError::Model(RerankCause::new("the lock is poisoned")));
        };
        read.extend(passages.iter().map(|passage| (*passage).to_owned()));
        Ok(RerankScores::new(vec![0.0; passages.len()], Usage::new()))
    }
}

#[tokio::test]
async fn the_reranker_reads_each_passage_under_its_headings() {
    let f = Fixture::new().await;
    f.index(&[("docs/d.pages", "refund terms\u{c}refund dates".as_bytes())])
        .await;
    let reads = Reads(identity(), Arc::default());
    let reranking = reader(&f, Some(SharedReranker::new(reads.clone())), 5).await;
    search(&reranking, QUERY, 5).await;
    let mut read = reads.1.lock().unwrap().clone();
    read.sort();
    assert_eq!(read, ["Part 1\nrefund terms", "Part 2\nrefund dates"]);
}

async fn owned_by_x(f: &Fixture, key: &str) {
    let change = AttributeChange::new(key).attribute("owner", AttributeValue::One("x".to_owned()));
    f.indexer
        .apply(vec![FileChange::Attributes(change)])
        .await
        .unwrap();
}

fn boosted(limit: usize, weight: f64) -> SearchRequest {
    SearchRequest::new(Query::new(QUERY))
        .limit(limit)
        .boost(Boost::matching(Filter::new().equals("owner", "x"), weight))
}

#[tokio::test]
async fn a_boost_places_reranked_hits_by_their_rerank_ranks() {
    let f = Fixture::new().await;
    eight_files(&f).await;
    let plain = reader(&f, Some(SharedReranker::new(fake())), 8).await;
    let order = keys(search(&plain, QUERY, 8).await.hits());
    let last = order.last().unwrap().clone();
    owned_by_x(&f, &last).await;
    let reranking = reader(&f, Some(SharedReranker::new(fake())), 8).await;

    let found = reranking.search(&boosted(8, 15.0)).await.unwrap();
    let mut expected = vec![last.clone()];
    expected.extend(order.iter().take(7).cloned());
    assert_eq!(keys(found.hits()), expected);
    assert!(found.hits().iter().all(|hit| hit.rerank_score().is_some()));
    assert!((found.hits()[0].boost() - 16.0).abs() < 1e-12);
    assert!(
        found.hits()[1..]
            .iter()
            .all(|hit| (hit.boost() - 1.0).abs() < 1e-12)
    );

    let level = reranking.search(&boosted(8, 7.0)).await.unwrap();
    assert_eq!(keys(level.hits())[..2], [order[0].clone(), last]);
}

#[tokio::test]
async fn a_boost_chooses_the_shortlist_from_twice_as_many_candidates() {
    let f = Fixture::new().await;
    eight_files(&f).await;
    let fused = keys(search(&reader(&f, None, 0).await, QUERY, 8).await.hits());
    owned_by_x(&f, &fused[3]).await;
    owned_by_x(&f, &fused[5]).await;

    let plain = reader(&f, None, 0).await;
    let found = plain.search(&boosted(2, 9.0)).await.unwrap();
    assert_eq!(keys(found.hits()), [fused[3].clone(), fused[0].clone()]);

    let reads = Reads(identity(), Arc::default());
    let reranking = reader(&f, Some(SharedReranker::new(reads.clone())), 2).await;
    let found = reranking.search(&boosted(2, 9.0)).await.unwrap();
    assert_eq!(keys(found.hits())[0], fused[3]);
    assert_eq!(reads.1.lock().unwrap().len(), 2);
    assert!(!keys(found.hits()).contains(&fused[5]));
}
