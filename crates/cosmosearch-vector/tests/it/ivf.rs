use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::Arc;

use cosmosearch_core::{
    AllowedRows, BuiltBy, EmbeddingIdentity, FakeEmbedder, Hit, Index, IndexError, IndexKind,
    Level, Metric, Query, RowSet, SegmentEntry, SegmentFile, SegmentId, SegmentSource,
    SharedEmbedder,
};
use cosmosearch_store::{IndexHandle, MemoryStore};
use cosmosearch_vector::Ivf;
use cosmosearch_vector::format::IvfHead;

use crate::segments::{
    ParseEmbedder, Read, Recorder, identity, rounds, search, store_segment, text,
};
use crate::support::create;

async fn index() -> IndexHandle {
    create(
        MemoryStore::new(),
        "s3://example-bucket/meta".parse().unwrap(),
    )
    .await
}

fn size(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).unwrap()
}

fn parsing(dimensions: u32) -> Ivf {
    Ivf::new(ParseEmbedder::shared(identity("parsed", dimensions))).unwrap()
}

fn unit(values: &[f32]) -> Vec<f32> {
    let length = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    values.iter().map(|value| value / length).collect()
}

fn near(direction: &[f32], count: usize) -> Vec<String> {
    (0..count)
        .map(|index| {
            let nudge = f32::from(u16::try_from(index).unwrap()) * 0.01;
            let values: Vec<f32> = direction
                .iter()
                .enumerate()
                .map(|(axis, value)| {
                    let sign = if (index + axis) % 2 == 0 { 1.0 } else { -1.0 };
                    value + sign * nudge
                })
                .collect();
            text(&values)
        })
        .collect()
}

fn rows(hits: &[Hit]) -> Vec<u64> {
    hits.iter().map(Hit::row).collect()
}

fn source(handle: &IndexHandle) -> SegmentSource {
    SegmentSource::new(handle.clone())
}

#[tokio::test]
async fn finds_a_stored_chunk_by_its_own_text() {
    let handle = index().await;
    let embedder = SharedEmbedder::new(FakeEmbedder::new(identity("fake", 64)));
    let ivf = Ivf::new(embedder).unwrap().cluster_size(size(16));
    let texts: Vec<String> = (0..200).map(|i| format!("passage number {i}")).collect();
    let segment = store_segment(&handle, &ivf, "s1", &texts).await;
    let one = std::slice::from_ref(&segment);
    for wanted in [0, 17, 199] {
        let hits = search(&ivf, &source(&handle), one, &texts[wanted], 5).await;
        assert_eq!(hits.len(), 5);
        assert_eq!(hits[0].row(), u64::try_from(wanted).unwrap());
        assert_eq!(hits[0].segment(), segment.id());
        assert!((hits[0].score() - 1.0).abs() < 0.02, "{hits:?}");
        assert!(
            hits.windows(2)
                .all(|pair| pair[0].score() >= pair[1].score())
        );
    }
}

#[tokio::test]
async fn the_score_is_the_similarity_of_the_identitys_metric() {
    let handle = index().await;
    let document = [3.0_f32, 4.0, 0.0, 0.0];
    let query = [1.0_f32, 1.0, 1.0, 0.0];
    let cosine = parsing(4);
    let dot = Ivf::new(ParseEmbedder::shared(
        EmbeddingIdentity::builder("parsed-dot", "r1", 4, 512)
            .normalized(false)
            .metric(Metric::Dot)
            .build()
            .unwrap(),
    ))
    .unwrap();
    let expected = [(cosine, 7.0 / (5.0 * 3.0_f32.sqrt())), (dot, 7.0)];
    for (ivf, score) in expected {
        let segment = store_segment(&handle, &ivf, ivf.kind().as_str(), &[text(&document)]).await;
        let hits = search(&ivf, &source(&handle), &[segment], &text(&query), 1).await;
        assert_eq!(hits.len(), 1);
        assert!(
            (hits[0].score() - score).abs() < 0.02 * score,
            "{hits:?} {score}"
        );
    }
}

#[tokio::test]
async fn searches_only_segments_this_plugin_built() {
    let handle = index().await;
    let mine = parsing(4);
    let other = Ivf::new(ParseEmbedder::shared(identity("other", 4))).unwrap();
    let texts = near(&[1.0, 0.0, 0.0, 0.0], 4);
    let own = store_segment(&handle, &mine, "own", &texts).await;
    let theirs = store_segment(&handle, &other, "theirs", &texts).await;
    let relabelled = SegmentEntry::new(
        SegmentId::new("relabelled").unwrap(),
        Level::L2,
        own.built_by().clone(),
        other.model_id().cloned(),
        own.files().to_vec(),
    );
    let lexical = SegmentEntry::new(
        SegmentId::new("lexical").unwrap(),
        Level::L1,
        BuiltBy::new(Some(IndexKind::new("bm25").unwrap()), BTreeSet::new()),
        None,
        vec![],
    );
    assert_ne!(mine.kind(), other.kind());
    let segments = [theirs, relabelled, lexical, own.clone()];
    let recorder = Recorder::new(handle.clone());
    let hits = search(&mine, &recorder.source(), &segments, &texts[0], 10).await;
    assert_eq!(hits.len(), 4);
    assert!(hits.iter().all(|hit| hit.segment() == own.id()));
    assert!(recorder.take().iter().all(|read| &read.segment == own.id()));
}

fn allowing(segment: &SegmentEntry, rows: Range<u64>) -> AllowedRows {
    let mut set = RowSet::new();
    set.insert(rows);
    let mut allowed = AllowedRows::new();
    allowed.allow(segment.id(), Arc::new(set));
    allowed
}

#[tokio::test]
async fn rows_left_out_are_never_returned_and_take_no_place() {
    let handle = index().await;
    let ivf = parsing(4).cluster_size(size(4));
    let texts = near(&[1.0, 0.0, 0.0, 0.0], 12);
    let segment = store_segment(&handle, &ivf, "s1", &texts).await;
    let one = std::slice::from_ref(&segment);
    let query = Query::new(texts[0].as_str());
    let source = source(&handle);
    let search_with = async |ivf: &Ivf, allowed: AllowedRows, limit: usize| {
        ivf.search(&source, one, &allowed, &query, limit)
            .await
            .unwrap()
            .into_hits()
    };
    assert_eq!(
        rows(&search_with(&ivf, allowing(&segment, 0..12), 1).await),
        [0]
    );
    assert_eq!(
        rows(&search_with(&ivf, allowing(&segment, 1..12), 1).await),
        [1]
    );
    let many = search_with(&ivf, allowing(&segment, 1..12), 20).await;
    assert_eq!(many.len(), 11);
    assert!(!rows(&many).contains(&0));
    assert_eq!(search_with(&ivf, AllowedRows::new(), 5).await.len(), 0);

    let one_candidate = parsing(4).cluster_size(size(4)).candidates_per_hit(size(1));
    let kept = search_with(&one_candidate, allowing(&segment, 1..12), 1).await;
    assert_eq!(kept.len(), 1);
    assert_ne!(kept[0].row(), 0);
}

#[tokio::test]
async fn the_probe_budget_goes_to_the_best_clusters_of_all_segments() {
    let handle = index().await;
    let ivf = || parsing(4).cluster_size(size(8));
    let mut first = near(&[1.0, 0.0, 0.0, 0.0], 8);
    first.extend(near(&[0.0, 1.0, 0.0, 0.0], 8));
    let mut second = near(&[-1.0, 0.0, 0.0, 0.0], 8);
    second.extend(near(&[0.0, 0.0, 1.0, 0.0], 8));
    let builder = ivf();
    let a = store_segment(&handle, &builder, "a", &first).await;
    let b = store_segment(&handle, &builder, "b", &second).await;
    let segments = [a.clone(), b.clone()];
    let query = text(&unit(&[0.9, 0.3, 0.0, 0.0]));

    let touched = |reads: &[Read]| -> BTreeMap<String, usize> {
        let mut touched = BTreeMap::new();
        for read in reads.iter().filter(|read| read.round > 1) {
            *touched.entry(read.segment.to_string()).or_default() += 1;
        }
        touched
    };
    let recorder = Recorder::new(handle.clone());
    let two = ivf().probes(size(2));
    let hits = search(&two, &recorder.source(), &segments, &query, 16).await;
    assert_eq!(hits.len(), 16);
    assert!(hits.iter().all(|hit| hit.segment() == a.id()));
    assert_eq!(touched(&recorder.take()).keys().collect::<Vec<_>>(), ["a"]);

    let three = ivf().probes(size(3));
    let hits = search(&three, &recorder.source(), &segments, &query, 32).await;
    assert_eq!(hits.len(), 24);
    let from_b: BTreeSet<u64> = hits
        .iter()
        .filter(|hit| hit.segment() == b.id())
        .map(Hit::row)
        .collect();
    assert_eq!(from_b, (8..16).collect());
    assert_eq!(
        touched(&recorder.take()).keys().collect::<Vec<_>>(),
        ["a", "b"]
    );
}

#[tokio::test]
async fn an_empty_or_tiny_segment_can_be_searched() {
    let handle = index().await;
    let ivf = parsing(4);
    let empty = store_segment(&handle, &ivf, "empty", &[]).await;
    assert_eq!(empty.files()[0].size(), IvfHead::HEADER_LEN);
    let tiny = store_segment(&handle, &ivf, "tiny", &[text(&[0.5, 0.5, 0.0, 0.0])]).await;
    let query = text(&[1.0, 0.0, 0.0, 0.0]);
    let source = source(&handle);
    assert_eq!(
        search(&ivf, &source, std::slice::from_ref(&empty), &query, 5)
            .await
            .len(),
        0
    );
    let hits = search(&ivf, &source, &[empty, tiny.clone()], &query, 5).await;
    assert_eq!(rows(&hits), [0]);
    assert_eq!(hits[0].segment(), tiny.id());
    assert_eq!(
        search(&ivf, &source, std::slice::from_ref(&tiny), &query, 0)
            .await
            .len(),
        0
    );
    assert_eq!(search(&ivf, &source, &[], &query, 5).await.len(), 0);
}

#[tokio::test]
async fn a_damaged_or_missing_file_is_an_error() {
    let handle = index().await;
    let ivf = parsing(4);
    let wide = Ivf::new(ParseEmbedder::shared(identity("parsed", 8))).unwrap();
    let texts = near(&[1.0, 0.0, 0.0, 0.0], 6);
    let good = store_segment(&handle, &ivf, "good", &texts).await;
    let file = good.files()[0].clone();
    let entry = |id: &str, files: Vec<SegmentFile>| {
        SegmentEntry::new(
            SegmentId::new(id).unwrap(),
            Level::L2,
            good.built_by().clone(),
            good.model_id().cloned(),
            files,
        )
    };
    let garbage = SegmentId::new("garbage").unwrap();
    handle
        .write_segment_file(&garbage, file.name(), vec![7_u8; 400].into())
        .await
        .unwrap();
    let truncated = SegmentId::new("truncated").unwrap();
    let mut bytes = handle
        .read_segment_file(good.id(), file.name())
        .await
        .unwrap()
        .to_vec();
    bytes.truncate(bytes.len() - 3);
    handle
        .write_segment_file(&truncated, file.name(), bytes.into())
        .await
        .unwrap();
    let other_width = store_segment(&handle, &wide, "wide", &[text(&[1.0; 8])]).await;

    let cases = [
        entry(
            "good",
            vec![SegmentFile::new(file.name().clone(), file.size() - 1)],
        ),
        entry(
            "good",
            vec![SegmentFile::new(file.name().clone(), file.size() + 1)],
        ),
        entry("missing", vec![file.clone()]),
        entry("garbage", vec![SegmentFile::new(file.name().clone(), 400)]),
        entry(
            "truncated",
            vec![SegmentFile::new(file.name().clone(), file.size() - 3)],
        ),
        entry("wide", other_width.files().to_vec()),
    ];
    for segment in cases {
        let one = std::slice::from_ref(&segment);
        let err = ivf
            .search(
                &source(&handle),
                one,
                &crate::segments::every_row(one),
                &Query::new(texts[0].as_str()),
                5,
            )
            .await
            .unwrap_err();
        assert!(
            matches!(
                err,
                IndexError::Malformed { .. }
                    | IndexError::MissingSegment { .. }
                    | IndexError::Read(_)
            ),
            "{}: {err}",
            segment.id()
        );
    }
}

#[tokio::test]
async fn a_query_the_embedder_cannot_handle_is_an_error() {
    let handle = index().await;
    let ivf = parsing(4);
    let segment = store_segment(&handle, &ivf, "s1", &near(&[1.0, 0.0, 0.0, 0.0], 3)).await;
    let one = std::slice::from_ref(&segment);
    for query in ["not a vector", "1,0"] {
        let err = ivf
            .search(
                &source(&handle),
                one,
                &crate::segments::every_row(one),
                &Query::new(query),
                5,
            )
            .await
            .unwrap_err();
        assert!(matches!(err, IndexError::Engine(_)), "{err}");
    }
}

#[test]
fn one_identity_has_one_kind() {
    let ivf = parsing(4);
    assert!(ivf.kind().as_str().starts_with("ivf-emb-"));
    assert!(
        ivf.kind()
            .as_str()
            .ends_with(ivf.model_id().unwrap().as_str())
    );
    assert_eq!(ivf.level(), Level::L2);
    assert_eq!(parsing(4).kind(), ivf.kind());
    assert_ne!(parsing(8).kind(), ivf.kind());
}

fn sections(handle_bytes: &[u8]) -> (Range<u64>, Range<u64>) {
    let size = u64::try_from(handle_bytes.len()).unwrap();
    let head_len = IvfHead::peek_len(handle_bytes).unwrap();
    let head = IvfHead::decode(
        handle_bytes
            .get(..usize::try_from(head_len).unwrap())
            .unwrap(),
        size,
    )
    .unwrap();
    let int8_start = head.int8_range(0..1).unwrap().start();
    (head_len..int8_start, int8_start..size)
}

fn within(read: &Read, section: &Range<u64>) -> bool {
    section.start <= read.range.start() && read.range.end() <= section.end
}

#[tokio::test]
async fn a_warm_search_is_exactly_two_rounds_of_parallel_reads() {
    let handle = index().await;
    let ivf = parsing(8).cluster_size(size(16));
    let directions: [[f32; 8]; 3] = [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    ];
    let mut segments = Vec::new();
    let mut layout = BTreeMap::new();
    for (index, direction) in directions.iter().enumerate() {
        let mut texts = near(direction, 40);
        texts.extend(near(&unit(&[1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]), 40));
        let segment = store_segment(&handle, &ivf, &format!("s{index}"), &texts).await;
        let name = segment.files()[0].name().clone();
        let bytes = handle.read_segment_file(segment.id(), &name).await.unwrap();
        layout.insert(segment.id().clone(), sections(&bytes));
        segments.push(segment);
    }
    let query = text(&unit(&[1.0, 0.2, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]));
    let recorder = Recorder::new(handle.clone());

    let cold = search(&ivf, &recorder.source(), &segments, &query, 10).await;
    let cold_reads = recorder.take();
    assert_eq!(rounds(&cold_reads), 3, "{cold_reads:#?}");
    let heads: Vec<&Read> = cold_reads.iter().filter(|read| read.round == 1).collect();
    assert_eq!(heads.len(), 3);
    assert!(heads.iter().all(|read| read.range.start() == 0));

    let warm = search(&ivf, &recorder.source(), &segments, &query, 10).await;
    assert_eq!(warm, cold);
    let reads = recorder.take();
    assert_eq!(rounds(&reads), 2, "{reads:#?}");
    for read in &reads {
        let (binary, int8) = &layout[&read.segment];
        let section = if read.round == 1 { binary } else { int8 };
        assert!(within(read, section), "{read:?} outside {section:?}");
    }
    let first: BTreeSet<&SegmentId> = reads
        .iter()
        .filter(|read| read.round == 1)
        .map(|read| &read.segment)
        .collect();
    assert!(first.len() > 1, "round one read only {first:?}");
}

#[tokio::test]
async fn a_head_is_dropped_once_its_segment_is_no_longer_searched() {
    let handle = index().await;
    let ivf = parsing(4);
    let texts = near(&[1.0, 0.0, 0.0, 0.0], 4);
    let old = store_segment(&handle, &ivf, "old", &texts).await;
    let new = store_segment(&handle, &ivf, "new", &texts).await;
    let recorder = Recorder::new(handle.clone());
    let head_reads = |segment: &SegmentEntry| {
        recorder
            .take()
            .iter()
            .filter(|read| &read.segment == segment.id() && read.range.start() == 0)
            .count()
    };
    let query = &texts[0];

    search(
        &ivf,
        &recorder.source(),
        std::slice::from_ref(&old),
        query,
        1,
    )
    .await;
    assert_eq!(head_reads(&old), 1);
    search(
        &ivf,
        &recorder.source(),
        std::slice::from_ref(&old),
        query,
        1,
    )
    .await;
    assert_eq!(head_reads(&old), 0);

    let mut new_reads = 0;
    for _ in 0..100 {
        search(
            &ivf,
            &recorder.source(),
            std::slice::from_ref(&new),
            query,
            1,
        )
        .await;
        new_reads += head_reads(&new);
    }
    assert_eq!(new_reads, 1);
    search(
        &ivf,
        &recorder.source(),
        std::slice::from_ref(&old),
        query,
        1,
    )
    .await;
    assert_eq!(head_reads(&old), 1);
}

#[test]
fn searching_is_send() {
    fn assert_send<T: Send>(_: &T) {}
    fn check(ivf: &Ivf, source: &SegmentSource, query: &Query) {
        assert_send(&ivf.search(source, &[], &AllowedRows::new(), query, 1));
    }
    let _: fn(&Ivf, &SegmentSource, &Query) = check;
}

#[test]
fn the_probes_and_candidates_are_its_search_settings() {
    assert_eq!(parsing(2).search_settings(), [10, 10]);
    let tuned = parsing(2).probes(size(3)).candidates_per_hit(size(7));
    assert_eq!(tuned.search_settings(), [3, 7]);
    assert_eq!(
        parsing(2).cluster_size(size(16)).search_settings(),
        [10, 10]
    );
}
