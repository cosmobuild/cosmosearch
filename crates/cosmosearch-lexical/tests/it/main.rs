//! Contract tests for `cosmosearch-lexical`, written and read through the store.

#![expect(
    clippy::panic,
    clippy::unwrap_used,
    reason = "helpers in a test binary fail the test by panicking"
)]

use std::collections::BTreeSet;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use cosmosearch_core::{
    AllowedRows, BuiltBy, ByteRange, Chunk, ETag, ExtractorProfile, FileSource, Hit, Index,
    IndexError, IndexKind, Level, Location, Locator, Passage, Query, RowSet, SegmentEntry,
    SegmentFile, SegmentFileName, SegmentFiles, SegmentId, SegmentSource,
};
use cosmosearch_lexical::{Bm25, Language};
use cosmosearch_store::{CreateOutcome, IndexHandle, MemoryStore, MetadataStore};

fn chunks(texts: &[&str]) -> Vec<Chunk> {
    chunks_of("docs/a.txt", texts)
}

fn chunks_of(key: &str, texts: &[&str]) -> Vec<Chunk> {
    let location: Location = format!("s3://example-bucket/{key}").parse().unwrap();
    let etag = ETag::new("9f3a").unwrap();
    texts
        .iter()
        .map(|text| {
            let range = ByteRange::new(0, 1).unwrap();
            Passage::new(Locator::Bytes(range), vec![], *text)
                .into_chunk(&location, &etag)
                .unwrap()
        })
        .collect()
}

async fn index() -> IndexHandle {
    let metadata = MetadataStore::new(
        MemoryStore::new(),
        "s3://example-bucket/index".parse().unwrap(),
    )
    .unwrap();
    let data = "s3://example-bucket/docs/".parse().unwrap();
    let outcome = metadata.create(&data, FileSource::Listing).await.unwrap();
    let CreateOutcome::Created(handle) = outcome else {
        panic!("expected created, got {outcome:?}");
    };
    handle
}

fn plugin() -> Bm25 {
    Bm25::new().unwrap()
}

async fn store_segment(handle: &IndexHandle, id: &str, texts: &[&str]) -> SegmentEntry {
    store_segment_by(&plugin(), handle, id, texts).await
}

async fn store_segment_by(
    bm25: &Bm25,
    handle: &IndexHandle,
    id: &str,
    texts: &[&str],
) -> SegmentEntry {
    store_chunks(bm25, handle, id, &chunks(texts)).await
}

async fn store_chunks(
    bm25: &Bm25,
    handle: &IndexHandle,
    id: &str,
    chunks: &[Chunk],
) -> SegmentEntry {
    let id = SegmentId::new(id).unwrap();
    let mut files = Vec::new();
    for file in bm25.build(&id, chunks).await.unwrap().into_files() {
        let size = u64::try_from(file.body().len()).unwrap();
        let name = file.name().clone();
        handle
            .write_segment_file(&id, &name, file.into_body().into())
            .await
            .unwrap();
        files.push(SegmentFile::new(name, size));
    }
    SegmentEntry::new(
        id,
        Level::L1,
        BuiltBy::new(Some(bm25.kind().clone()), BTreeSet::new()),
        None,
        files,
    )
}

fn every_row(segments: &[SegmentEntry]) -> AllowedRows {
    let mut all = RowSet::new();
    all.insert(0..u64::MAX);
    let all = Arc::new(all);
    let mut allowed = AllowedRows::new();
    for segment in segments {
        allowed.allow(segment.id(), Arc::clone(&all));
    }
    allowed
}

async fn search_rows(
    handle: &IndexHandle,
    segments: &[SegmentEntry],
    allowed: &AllowedRows,
    text: &str,
    limit: usize,
) -> Vec<Hit> {
    plugin()
        .search(
            &SegmentSource::new(handle.clone()),
            segments,
            allowed,
            &Query::new(text),
            limit,
        )
        .await
        .unwrap()
        .into_hits()
}

async fn search(
    handle: &IndexHandle,
    segments: &[SegmentEntry],
    text: &str,
    limit: usize,
) -> Vec<Hit> {
    search_rows(handle, segments, &every_row(segments), text, limit).await
}

async fn search_query(
    bm25: &Bm25,
    handle: &IndexHandle,
    segments: &[SegmentEntry],
    query: Query,
) -> Vec<Hit> {
    bm25.search(
        &SegmentSource::new(handle.clone()),
        segments,
        &every_row(segments),
        &query,
        10,
    )
    .await
    .unwrap()
    .into_hits()
}

async fn search_exact(
    handle: &IndexHandle,
    segments: &[SegmentEntry],
    text: &str,
    limit: usize,
) -> Vec<Hit> {
    plugin()
        .search(
            &SegmentSource::new(handle.clone()),
            segments,
            &every_row(segments),
            &Query::new(text).with_exact_phrases(),
            limit,
        )
        .await
        .unwrap()
        .into_hits()
}

fn rows(hits: &[Hit]) -> Vec<u64> {
    hits.iter().map(Hit::row).collect()
}

#[tokio::test]
async fn a_chunk_matches_the_words_of_the_headings_above_it() {
    let location: Location = "s3://example-bucket/docs/a.txt".parse().unwrap();
    let etag = ETag::new("9f3a").unwrap();
    let range = ByteRange::new(0, 1).unwrap();
    let under = |headings: &[&str], text: &str| {
        let headings = headings.iter().map(|h| (*h).to_owned()).collect();
        Passage::new(Locator::Bytes(range), headings, text)
            .into_chunk(&location, &etag)
            .unwrap()
    };
    let chunks = [
        under(&["Consolidated Balance Sheets"], "Total assets 352,583"),
        under(&["Risk Factors"], "Total assets may fall"),
    ];
    let handle = index().await;
    let bm25 = plugin();
    let id = SegmentId::new("s1").unwrap();
    let mut files = Vec::new();
    for file in bm25.build(&id, &chunks).await.unwrap().into_files() {
        let size = u64::try_from(file.body().len()).unwrap();
        let name = file.name().clone();
        handle
            .write_segment_file(&id, &name, file.into_body().into())
            .await
            .unwrap();
        files.push(SegmentFile::new(name, size));
    }
    let built_by = BuiltBy::new(Some(bm25.kind().clone()), BTreeSet::new());
    let segment = SegmentEntry::new(id, Level::L1, built_by, None, files);
    let segments = [segment];
    assert_eq!(
        rows(&search(&handle, &segments, "balance sheets", 5).await),
        [0]
    );
    assert_eq!(rows(&search(&handle, &segments, "risk", 5).await), [1]);
}

#[tokio::test]
async fn finds_known_terms_through_the_store() {
    let handle = index().await;
    let segment = store_segment(
        &handle,
        "s1",
        &[
            "refund policy for customers",
            "shipping rates",
            "the refund window",
        ],
    )
    .await;
    let hits = search(&handle, std::slice::from_ref(&segment), "refund", 10).await;
    assert_eq!(BTreeSet::from_iter(rows(&hits)), BTreeSet::from([0, 2]));
    assert!(hits.iter().all(|hit| hit.segment() == segment.id()));
}

#[tokio::test]
async fn ranks_a_stronger_match_first() {
    let handle = index().await;
    let segment = store_segment(
        &handle,
        "s1",
        &[
            "refund refund refund policy",
            "a long passage that mentions refund once among many other words here",
        ],
    )
    .await;
    assert_eq!(
        rows(&search(&handle, &[segment], "refund", 10).await),
        [0, 1]
    );
}

#[tokio::test]
async fn matches_any_query_word_ignoring_case() {
    let handle = index().await;
    let segment = store_segment(&handle, "s1", &["Alpha document", "beta document", "gamma"]).await;
    let hits = search(&handle, &[segment], "ALPHA beta", 10).await;
    assert_eq!(BTreeSet::from_iter(rows(&hits)), BTreeSet::from([0, 1]));
}

#[tokio::test]
async fn query_operators_other_than_paired_quotes_are_plain_words() {
    let handle = index().await;
    let segment = store_segment(&handle, "s1", &["refund policy"]).await;
    for text in [
        "refund AND (",
        "text:refund\"",
        "*refund* ~2 ^3",
        "+refund -policy",
    ] {
        let hits = search(&handle, std::slice::from_ref(&segment), text, 10).await;
        assert_eq!(rows(&hits), [0], "{text:?}");
    }
}

#[tokio::test]
async fn a_phrase_matches_only_its_words_next_to_each_other_in_order() {
    let handle = index().await;
    let segment = store_segment(
        &handle,
        "s1",
        &[
            "the refund policy applies",
            "policy on refund",
            "refund and then policy",
        ],
    )
    .await;
    let one = std::slice::from_ref(&segment);
    assert_eq!(
        rows(&search_exact(&handle, one, "\"refund policy\"", 10).await),
        [0]
    );
    assert_eq!(
        rows(&search_exact(&handle, one, "\"policy refund\"", 10).await),
        [0; 0]
    );
}

#[tokio::test]
async fn a_phrase_ignores_case_and_punctuation() {
    let handle = index().await;
    let segment = store_segment(
        &handle,
        "s1",
        &[
            "ERR_CONN_42 raised by the client",
            "conn err 42",
            "Refund. Policy!",
        ],
    )
    .await;
    let one = std::slice::from_ref(&segment);
    assert_eq!(
        rows(&search_exact(&handle, one, "\"err_conn_42\"", 10).await),
        [0]
    );
    assert_eq!(
        rows(&search_exact(&handle, one, "\"refund policy\"", 10).await),
        [2]
    );
}

#[tokio::test]
async fn every_phrase_must_match_and_the_other_words_rank() {
    let handle = index().await;
    let segment = store_segment(
        &handle,
        "s1",
        &[
            "refund policy for monthly plans",
            "refund policy for annual plans",
            "annual report",
            "annual refund under a new policy",
        ],
    )
    .await;
    let one = std::slice::from_ref(&segment);
    assert_eq!(
        rows(&search_exact(&handle, one, "\"refund policy\" annual", 10).await),
        [1, 0]
    );
    assert_eq!(
        rows(&search_exact(&handle, one, "\"refund policy\" \"annual plans\"", 10).await),
        [1]
    );
}

#[tokio::test]
async fn a_one_word_phrase_is_required_and_an_empty_one_is_ignored() {
    let handle = index().await;
    let segment = store_segment(&handle, "s1", &["alpha beta", "beta", "gamma"]).await;
    let one = std::slice::from_ref(&segment);
    assert_eq!(
        rows(&search_exact(&handle, one, "\"alpha\" beta", 10).await),
        [0]
    );
    let hits = search_exact(&handle, one, "\"\" beta", 10).await;
    assert_eq!(BTreeSet::from_iter(rows(&hits)), BTreeSet::from([0, 1]));
}

#[tokio::test]
async fn quotes_are_plain_text_unless_the_query_asks_for_exact_phrases() {
    let handle = index().await;
    let segment = store_segment(
        &handle,
        "s1",
        &["the refund policy applies", "policy on refund"],
    )
    .await;
    let one = std::slice::from_ref(&segment);
    let hits = search(&handle, one, "\"refund policy\"", 10).await;
    assert_eq!(BTreeSet::from_iter(rows(&hits)), BTreeSet::from([0, 1]));
    assert_eq!(
        rows(&search_exact(&handle, one, "\"refund policy\"", 10).await),
        [0]
    );
}

#[tokio::test]
async fn finds_nothing_where_nothing_can_match() {
    let handle = index().await;
    let segment = store_segment(&handle, "s1", &["refund policy"]).await;
    let one = std::slice::from_ref(&segment);
    assert_eq!(search(&handle, one, "absent", 10).await.len(), 0);
    assert_eq!(search(&handle, one, "  ... ", 10).await.len(), 0);
    assert_eq!(search(&handle, one, "refund", 0).await.len(), 0);
    assert_eq!(search(&handle, &[], "refund", 10).await.len(), 0);
}

#[tokio::test]
async fn an_empty_segment_can_be_searched() {
    let handle = index().await;
    let segment = store_segment(&handle, "s1", &[]).await;
    assert_eq!(search(&handle, &[segment], "refund", 10).await.len(), 0);
}

#[tokio::test]
async fn honours_the_limit_across_segments() {
    let handle = index().await;
    let a = store_segment(&handle, "a", &["refund one", "refund two"]).await;
    let b = store_segment(&handle, "b", &["refund three", "refund four"]).await;
    assert_eq!(
        search(&handle, &[a.clone(), b.clone()], "refund", 3)
            .await
            .len(),
        3
    );
    let all = search(&handle, &[a.clone(), b.clone()], "refund", 10).await;
    let found: BTreeSet<(String, u64)> = all
        .iter()
        .map(|hit| (hit.segment().to_string(), hit.row()))
        .collect();
    assert_eq!(found.len(), 4);
    assert_eq!(search(&handle, &[a, b], "refund", 10).await, all);
}

#[tokio::test]
async fn scores_match_a_single_segment_thanks_to_global_statistics() {
    let texts = [
        "rare common common",
        "common words here",
        "common again",
        "another common one",
        "rare appears here",
        "nothing relevant at all",
    ];
    let handle = index().await;
    let whole = store_segment(&handle, "whole", &texts).await;
    let (left, right) = texts.split_at(3);
    let a = store_segment(&handle, "a", left).await;
    let b = store_segment(&handle, "b", right).await;

    let whole_hits = search(&handle, &[whole], "rare common", 10).await;
    let split_hits = search(&handle, &[a.clone(), b], "rare common", 10).await;
    let absolute = |hit: &Hit| {
        if hit.segment() == a.id() {
            hit.row()
        } else {
            hit.row() + 3
        }
    };
    assert_eq!(whole_hits.len(), split_hits.len());
    for (one, other) in whole_hits.iter().zip(&split_hits) {
        assert_eq!(one.row(), absolute(other));
        assert!(
            (one.score() - other.score()).abs() < 1e-4,
            "{one:?} {other:?}"
        );
    }
}

#[tokio::test]
async fn ignores_segments_another_plugin_built() {
    let handle = index().await;
    let mine = store_segment(&handle, "s1", &["refund"]).await;
    let other = SegmentEntry::new(
        SegmentId::new("s2").unwrap(),
        Level::L1,
        BuiltBy::new(
            Some(IndexKind::new("other").unwrap()),
            BTreeSet::from([ExtractorProfile::new("x").unwrap()]),
        ),
        None,
        vec![],
    );
    assert_eq!(search(&handle, &[other, mine], "refund", 10).await.len(), 1);
}

fn rows_of(segment: &SegmentEntry, range: Range<u64>) -> AllowedRows {
    let mut rows = RowSet::new();
    rows.insert(range);
    let mut allowed = AllowedRows::new();
    allowed.allow(segment.id(), Arc::new(rows));
    allowed
}

#[tokio::test]
async fn rows_left_out_never_score_and_never_take_a_place() {
    let handle = index().await;
    let segment = store_segment(
        &handle,
        "s1",
        &[
            "refund refund refund refund",
            "refund policy",
            "shipping",
            "refund window and other words",
        ],
    )
    .await;
    let one = std::slice::from_ref(&segment);
    assert_eq!(rows(&search(&handle, one, "refund", 1).await), [0]);

    let without_first = rows_of(&segment, 1..4);
    assert_eq!(
        rows(&search_rows(&handle, one, &without_first, "refund", 1).await),
        [1]
    );
    assert_eq!(
        rows(&search_rows(&handle, one, &without_first, "refund", 10).await),
        [1, 3]
    );
    let only_shipping = rows_of(&segment, 2..3);
    assert_eq!(
        search_rows(&handle, one, &only_shipping, "refund", 10)
            .await
            .len(),
        0
    );
}

#[tokio::test]
async fn filtering_leaves_the_scores_of_allowed_rows_alone() {
    let handle = index().await;
    let segment = store_segment(&handle, "s1", &["refund policy", "refund refund"]).await;
    let one = std::slice::from_ref(&segment);
    let all = search(&handle, one, "refund", 10).await;
    let some = search_rows(&handle, one, &rows_of(&segment, 0..1), "refund", 10).await;
    let first = all.iter().find(|hit| hit.row() == 0).unwrap();
    assert_eq!(some.len(), 1);
    assert!(
        (some[0].score() - first.score()).abs() < 1e-6,
        "{some:?} {all:?}"
    );
}

const WORDS: [&str; 6] = ["refund", "policy", "window", "shipping", "rates", "terms"];

fn texts_with_ties(seed: u64, count: usize) -> Vec<String> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            let mut words = Vec::new();
            for _ in 0..4 {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                let at = usize::try_from(state >> 61).unwrap() % WORDS.len();
                words.push(*WORDS.get(at).unwrap());
            }
            words.join(" ")
        })
        .collect()
}

async fn tie_heavy_segments(handle: &IndexHandle) -> Vec<SegmentEntry> {
    let mut segments = Vec::new();
    for (at, count) in [12, 30, 7, 25, 18, 3].into_iter().enumerate() {
        let mut texts = texts_with_ties(at as u64 + 1, count);
        texts.insert(count / 2, "shipping rates".to_owned());
        let texts: Vec<&str> = texts.iter().map(String::as_str).collect();
        segments.push(store_segment(handle, &format!("s{at}"), &texts).await);
    }
    segments
}

async fn search_by(
    bm25: &Bm25,
    handle: &IndexHandle,
    segments: &[SegmentEntry],
    allowed: &AllowedRows,
    limit: usize,
) -> Vec<(SegmentId, u64, u32)> {
    bm25.search(
        &SegmentSource::new(handle.clone()),
        segments,
        allowed,
        &Query::new("refund policy window"),
        limit,
    )
    .await
    .unwrap()
    .into_hits()
    .iter()
    .map(|hit| (hit.segment().clone(), hit.row(), hit.score().to_bits()))
    .collect()
}

#[tokio::test]
async fn every_row_allowed_gives_what_the_row_predicate_gives() {
    let handle = index().await;
    let segments = tie_heavy_segments(&handle).await;
    let mut allowed = AllowedRows::new();
    for segment in &segments {
        let mut rows = RowSet::new();
        rows.insert(0..u64::MAX);
        allowed.allow(segment.id(), Arc::new(rows));
    }
    let bm25 = plugin();
    let unrestricted = search_by(&bm25, &handle, &segments, &allowed, 1_000).await;

    let mut nearly = AllowedRows::new();
    let mut matching = 0;
    for segment in &segments {
        let matched: BTreeSet<u64> = unrestricted
            .iter()
            .filter(|(id, _, _)| id == segment.id())
            .map(|(_, row, _)| *row)
            .collect();
        matching += matched.len();
        let left_out = (0..u64::MAX).find(|row| !matched.contains(row)).unwrap();
        let mut rows = RowSet::new();
        rows.insert(0..left_out);
        rows.insert(left_out + 1..u64::MAX);
        nearly.allow(segment.id(), Arc::new(rows));
    }
    assert!(matching > 20, "the texts give many matches");
    for limit in [1, 5, 17, 1_000] {
        let restricted = search_by(&bm25, &handle, &segments, &nearly, limit).await;
        let all = search_by(&bm25, &handle, &segments, &allowed, limit).await;
        assert_eq!(restricted, all, "limit {limit}");
    }
}

#[tokio::test]
async fn any_number_of_scoring_tasks_gives_the_same_hits_in_the_same_order() {
    let handle = index().await;
    let segments = tie_heavy_segments(&handle).await;
    let every = every_row(&segments);
    let mut some = RowSet::new();
    some.insert(0..4);
    some.insert(9..20);
    let some = Arc::new(some);
    let mut allowed = AllowedRows::new();
    for segment in &segments {
        let rows = if segment.id().as_str() == "s1" {
            Arc::clone(&some)
        } else {
            Arc::clone(every.rows(segment.id()).unwrap())
        };
        allowed.allow(segment.id(), rows);
    }
    for limit in [1, 5, 17, 1_000] {
        let one = search_by(&plugin(), &handle, &segments, &allowed, limit).await;
        assert_ne!(one.len(), 0);
        assert!(
            one.iter()
                .filter(|(id, _, _)| id.as_str() == "s1")
                .all(|(_, row, _)| some.contains(*row)),
            "a disallowed row was returned"
        );
        for tasks in [2, 3, 8] {
            let bm25 = plugin().scoring_tasks(NonZeroUsize::new(tasks).unwrap());
            let many = search_by(&bm25, &handle, &segments, &allowed, limit).await;
            assert_eq!(many, one, "{tasks} tasks, limit {limit}");
        }
    }
}

#[derive(Debug, Clone)]
struct Counting {
    inner: IndexHandle,
    reads: Arc<AtomicU64>,
    bytes: Arc<AtomicU64>,
}

#[async_trait]
impl SegmentFiles for Counting {
    async fn read_range(
        &self,
        segment: &SegmentId,
        name: &SegmentFileName,
        range: ByteRange,
    ) -> Result<Vec<u8>, IndexError> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        self.bytes.fetch_add(range.len(), Ordering::SeqCst);
        self.inner.read_range(segment, name, range).await
    }
}

#[tokio::test]
async fn a_search_reads_only_a_small_part_of_the_segment() {
    let handle = index().await;
    let owned: Vec<String> = (0..20_000)
        .map(|i| {
            format!(
                "passage number {i} about subject{} with filler words",
                i % 50
            )
        })
        .collect();
    let texts: Vec<&str> = owned.iter().map(String::as_str).collect();
    let segment = store_segment(&handle, "s1", &texts).await;
    let size = segment.files()[0].size();
    let (reads, bytes) = (Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)));
    let source = SegmentSource::new(Counting {
        inner: handle.clone(),
        reads: Arc::clone(&reads),
        bytes: Arc::clone(&bytes),
    });
    let one = std::slice::from_ref(&segment);
    let hits = plugin()
        .search(&source, one, &every_row(one), &Query::new("subject7"), 5)
        .await
        .unwrap()
        .into_hits();
    assert_eq!(hits.len(), 5);
    let read = bytes.load(Ordering::SeqCst);
    assert!(read < size / 2, "read {read} of {size}");
    assert!(reads.load(Ordering::SeqCst) <= 12, "{reads:?}");
    let phrase = Query::new("\"about subject7 with\"").with_exact_phrases();
    let hits = plugin()
        .search(&source, one, &every_row(one), &phrase, 5)
        .await
        .unwrap()
        .into_hits();
    assert_eq!(hits.len(), 5);
    let read = bytes.load(Ordering::SeqCst);
    assert!(read < size / 2, "read {read} of {size} with a phrase");
}

#[tokio::test]
async fn a_segment_with_no_allowed_rows_is_not_read() {
    let handle = index().await;
    let segment = store_segment(&handle, "s1", &["refund policy"]).await;
    let reads = Arc::new(AtomicU64::new(0));
    let source = SegmentSource::new(Counting {
        inner: handle.clone(),
        reads: Arc::clone(&reads),
        bytes: Arc::new(AtomicU64::new(0)),
    });
    let hits = plugin()
        .search(
            &source,
            std::slice::from_ref(&segment),
            &AllowedRows::new(),
            &Query::new("refund"),
            5,
        )
        .await
        .unwrap();
    assert_eq!(hits.hits().len(), 0);
    assert_eq!(reads.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn a_damaged_or_missing_segment_is_an_error() {
    let handle = index().await;
    let good = store_segment(&handle, "s1", &["refund policy"]).await;
    let file = good.files()[0].clone();

    let wrong_size = SegmentEntry::new(
        good.id().clone(),
        Level::L1,
        good.built_by().clone(),
        None,
        vec![SegmentFile::new(file.name().clone(), file.size() - 1)],
    );
    let missing = SegmentEntry::new(
        SegmentId::new("gone").unwrap(),
        Level::L1,
        good.built_by().clone(),
        None,
        vec![file.clone()],
    );
    for segment in [wrong_size, missing] {
        let one = std::slice::from_ref(&segment);
        plugin()
            .search(
                &SegmentSource::new(handle.clone()),
                one,
                &every_row(one),
                &Query::new("refund"),
                5,
            )
            .await
            .unwrap_err();
    }
}

#[test]
fn searching_is_send() {
    fn assert_send<T: Send>(_: &T) {}
    fn check(bm25: &Bm25, source: &SegmentSource, query: &Query) {
        assert_send(&bm25.search(source, &[], &AllowedRows::new(), query, 1));
    }
    let _: fn(&Bm25, &SegmentSource, &Query) = check;
}

#[derive(Debug, Clone)]
struct Damaged {
    bundle: Arc<[u8]>,
}

#[async_trait]
impl SegmentFiles for Damaged {
    async fn read_range(
        &self,
        _: &SegmentId,
        _: &SegmentFileName,
        range: ByteRange,
    ) -> Result<Vec<u8>, IndexError> {
        let start = usize::try_from(range.start()).unwrap();
        let end = usize::try_from(range.end()).unwrap();
        Ok(self.bundle[start..end].to_vec())
    }
}

#[tokio::test]
async fn a_damaged_segment_gives_an_error_or_hits_and_never_a_panic() {
    let handle = index().await;
    let segment = store_segment(
        &handle,
        "s1",
        &["refund policy", "late fees", "refund window"],
    )
    .await;
    let file = &segment.files()[0];
    let whole = ByteRange::new(0, file.size()).unwrap();
    let good = handle
        .read_range(segment.id(), file.name(), whole)
        .await
        .unwrap();
    let one = std::slice::from_ref(&segment);
    for at in 0..good.len() {
        let mut bad = good.clone();
        bad[at] ^= 0xff;
        let source = SegmentSource::new(Damaged {
            bundle: Arc::from(bad),
        });
        let _outcome = plugin()
            .search(&source, one, &every_row(one), &Query::new("refund fees"), 5)
            .await;
    }
}

#[tokio::test]
async fn a_stemmed_index_matches_other_forms_of_a_word_and_plain_does_not() {
    let handle = index().await;
    let texts = ["refunds are issued", "the refunded order", "a policy"];
    let english = Bm25::new().unwrap();
    let plain = Bm25::plain().unwrap();
    assert_eq!(english.kind().as_str(), "bm25-stem-en");
    assert_eq!(
        Bm25::stemmed(Language::English).unwrap().kind(),
        english.kind()
    );
    assert_eq!(plain.kind().as_str(), "bm25");
    let stemmed = store_segment_by(&english, &handle, "s1", &texts).await;
    let unstemmed = store_segment_by(&plain, &handle, "s2", &texts).await;
    let found = search_query(
        &english,
        &handle,
        std::slice::from_ref(&stemmed),
        Query::new("Refund"),
    )
    .await;
    assert_eq!(rows(&found).len(), 2);
    let phrase = Query::new("\"refund order\"").with_exact_phrases();
    let found = search_query(&english, &handle, std::slice::from_ref(&stemmed), phrase).await;
    assert_eq!(rows(&found), [1]);
    let plain_found = search_query(
        &plain,
        &handle,
        std::slice::from_ref(&unstemmed),
        Query::new("refund"),
    )
    .await;
    assert_eq!(rows(&plain_found), Vec::<u64>::new());
    let mixed = search_query(
        &english,
        &handle,
        std::slice::from_ref(&unstemmed),
        Query::new("refunds"),
    )
    .await;
    assert_eq!(rows(&mixed), Vec::<u64>::new());
    let both = [stemmed, unstemmed];
    assert_eq!(
        search_query(&plain, &handle, &both, Query::new("refunds"))
            .await
            .len(),
        1
    );
}

#[test]
fn words_are_marked_as_the_index_reads_them() {
    let text = "Refunds: the REFUNDED order, refund-policy and a policy.";
    let marked = |index: &Bm25, query: &str| -> Vec<&str> {
        index
            .word_matches(&Query::new(query), text)
            .unwrap()
            .into_iter()
            .map(|range| text.get(range).unwrap())
            .collect()
    };
    let english = Bm25::new().unwrap();
    let plain = Bm25::plain().unwrap();
    assert_eq!(
        marked(&english, "refund"),
        ["Refunds", "REFUNDED", "refund"]
    );
    assert_eq!(marked(&plain, "refund"), ["refund"]);
    assert_eq!(marked(&plain, "\"Policy\""), ["policy", "policy"]);
    assert_eq!(marked(&english, ""), Vec::<&str>::new());
    assert_eq!(marked(&english, "?!"), Vec::<&str>::new());
}

#[tokio::test]
async fn a_file_name_matches_on_its_first_chunk_only_with_its_own_weight() {
    let handle = index().await;
    let mut chunks = chunks_of("docs/travel-policy.pdf", &["flights and hotels", "meals"]);
    chunks.extend(
        chunks_of("docs/b.txt", &["intro", "more"])
            .into_iter()
            .map(|chunk| chunk.with_name(Some("Staff Handbook".to_owned()))),
    );
    chunks.extend(chunks_of("docs/c.txt", &["the handbook says"]));
    let segment = store_chunks(&plugin(), &handle, "s1", &chunks).await;
    let segments = std::slice::from_ref(&segment);
    assert_eq!(rows(&search(&handle, segments, "travel", 10).await), [0]);
    assert_eq!(rows(&search(&handle, segments, "policy", 10).await), [0]);
    assert_eq!(
        rows(&search(&handle, segments, "pdf", 10).await),
        Vec::<u64>::new()
    );
    assert_eq!(
        rows(&search(&handle, segments, "handbook", 10).await),
        [4, 2]
    );
    let exact = search_exact(&handle, segments, "\"staff handbook\"", 10).await;
    assert_eq!(rows(&exact), Vec::<u64>::new());
    let off = plugin().name_weight(0.0);
    let found = search_query(&off, &handle, segments, Query::new("handbook")).await;
    assert_eq!(rows(&found), [4]);
    let heavy = plugin().name_weight(4.0);
    let found = search_query(&heavy, &handle, segments, Query::new("handbook")).await;
    assert_eq!(rows(&found), [2, 4]);
    assert_ne!(off.search_settings(), plugin().search_settings());
}

#[tokio::test]
async fn every_word_mode_needs_each_word_in_the_text_or_the_name() {
    let handle = index().await;
    let mut chunks = chunks_of(
        "docs/refund-policy.txt",
        &["annual plans only", "monthly plans"],
    );
    chunks.extend(chunks_of(
        "docs/b.txt",
        &["refund for annual plans", "refund for monthly plans"],
    ));
    let segment = store_chunks(&plugin(), &handle, "s1", &chunks).await;
    let segments = std::slice::from_ref(&segment);
    let bm25 = plugin();
    let found = |query: Query| {
        let bm25 = bm25.clone();
        let handle = handle.clone();
        async move {
            let mut found = rows(&search_query(&bm25, &handle, segments, query).await);
            found.sort_unstable();
            found
        }
    };
    assert_eq!(found(Query::new("refund annual")).await, [0, 2, 3]);
    assert_eq!(
        found(Query::new("refund annual").with_every_word()).await,
        [0, 2]
    );
    assert_eq!(
        found(Query::new("refund annual unknown").with_every_word()).await,
        Vec::<u64>::new()
    );
    let phrase = Query::new("\"monthly plans\" refund")
        .with_exact_phrases()
        .with_every_word();
    assert_eq!(found(phrase).await, [3]);
    let phrase_only = Query::new("\"annual plans\"")
        .with_exact_phrases()
        .with_every_word();
    assert_eq!(found(phrase_only).await, [0, 2]);
}
