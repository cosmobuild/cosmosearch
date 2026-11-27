//! Contract tests for `cosmosearch-query`.

#![expect(
    clippy::panic,
    clippy::unwrap_used,
    reason = "helpers in a test binary fail the test by panicking"
)]

use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroU32;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use cosmosearch_chunk::Extractors;
use cosmosearch_core::{
    AllowedRows, AttributeValue, Built, BuiltBy, ByteRange, Chunk, ETag, ExtractFailure,
    ExtractLimits, Extraction, Extractor, ExtractorProfile, FileSource, Filter, HeatSink, Hit,
    Hits, Index, IndexError, IndexKind, IndexOptions, KeyRange, Level, Locator, Number, Passage,
    Query, Scope, SegmentEntry, SegmentFile, SegmentFileName, SegmentId, SegmentSource, ShardEntry,
    ShardId, SharedExtractor, SharedHeat, SharedIndex, Signal, SourceRef, Timestamp, Usage,
};
use cosmosearch_extract_html::Html;
use cosmosearch_format::FileStatus;
use cosmosearch_indexer::{
    AttributeChange, ExtractOutcome, FileChange, FileUpsert, Indexer, PublishOutcome,
};
use cosmosearch_lexical::Bm25;
use cosmosearch_query::{
    Boost, Error, Plugins, Reader, ReaderOptions, Reading, SearchHit, SearchRequest, SearchResults,
};
use cosmosearch_store::{
    CreateOutcome, DataStore, Fault, FaultStore, IndexHandle, MemoryStore, MetadataStore,
    ObjectKey, Op, WriteStore,
};
use tokio::sync::Barrier;

mod budget;
mod coverage;
mod rerank;
mod results;
mod scopes;
mod snippets;
mod stored;
mod tables;

#[derive(Debug)]
struct Pages(ExtractorProfile);

impl Extractor for Pages {
    fn profile(&self) -> &ExtractorProfile {
        &self.0
    }

    fn extensions(&self) -> &[&str] {
        &["pages"]
    }

    fn media_types(&self) -> &[&str] {
        &[]
    }

    fn extract(&self, file: &[u8], _: &ExtractLimits) -> Result<Extraction, ExtractFailure> {
        let text = std::str::from_utf8(file).map_err(|_cause| ExtractFailure::Malformed)?;
        let mut found = Vec::new();
        let mut unread = 0u32;
        for (index, page) in text.split('\u{c}').enumerate() {
            if page.trim() == "[image]" {
                unread = unread.saturating_add(1);
                continue;
            }
            let number = u32::try_from(index)
                .ok()
                .and_then(|index| index.checked_add(1))
                .and_then(NonZeroU32::new)
                .ok_or(ExtractFailure::Malformed)?;
            for paragraph in page.split("\n\n").filter(|p| !p.trim().is_empty()) {
                let heading = vec![format!("Part {number}")];
                found.push(Passage::new(
                    Locator::Page(number),
                    heading,
                    paragraph.trim(),
                ));
            }
        }
        Ok(Extraction::new(found).with_unread_parts(unread))
    }
}

fn extractors() -> Extractors {
    let mut extractors = Extractors::builtin().unwrap();
    extractors.insert(SharedExtractor::new(Html::new().unwrap()));
    extractors.insert(SharedExtractor::new(Pages(
        ExtractorProfile::new("pages-1").unwrap(),
    )));
    extractors
}

fn plugins() -> Plugins {
    let mut plugins = Plugins::new(extractors());
    plugins
        .add_index(SharedIndex::new(Bm25::new().unwrap()))
        .unwrap();
    plugins
}

#[derive(Debug)]
struct Backwards(IndexKind);

const BACKWARDS_TOKENS: u64 = 7;

#[async_trait]
impl Index for Backwards {
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
        allowed: &AllowedRows,
        _: &Query,
        limit: usize,
    ) -> Result<Hits, IndexError> {
        let mut hits = Vec::new();
        for segment in segments {
            let Some(rows) = allowed.rows(segment.id()) else {
                continue;
            };
            for row in (0..100u16).filter(|row| rows.contains(u64::from(*row))) {
                let score = 1_000.0 + f32::from(row);
                hits.push(Hit::new(segment.id().clone(), u64::from(row), score));
            }
        }
        hits.sort_by(|a, b| b.score().total_cmp(&a.score()));
        hits.truncate(limit);
        Ok(Hits::new(hits, Usage::of_tokens(BACKWARDS_TOKENS)))
    }
}

struct Fixture {
    store: MemoryStore,
    data: DataStore,
    metadata: MetadataStore,
    handle: IndexHandle,
    indexer: Indexer,
    reader: Reader,
}

impl Fixture {
    async fn new() -> Fixture {
        Fixture::with_options(IndexOptions::new()).await
    }

    async fn with_options(options: IndexOptions) -> Fixture {
        let store = MemoryStore::new();
        let data =
            DataStore::new(store.clone(), "s3://example-bucket/docs/".parse().unwrap()).unwrap();
        let metadata =
            MetadataStore::new(store.clone(), "s3://example-bucket/index".parse().unwrap())
                .unwrap();
        let outcome = metadata
            .create_with(data.location(), FileSource::HostNamed, options)
            .await
            .unwrap();
        let CreateOutcome::Created(index) = outcome else {
            panic!("expected created, got {outcome:?}");
        };
        let indexer = Indexer::new(index.clone(), data.clone()).unwrap();
        let reader = Reader::open(
            data.clone(),
            metadata.clone(),
            FileSource::HostNamed,
            plugins(),
            ReaderOptions::new(),
        )
        .await
        .unwrap();
        Fixture {
            store,
            data,
            metadata,
            handle: index,
            indexer,
            reader,
        }
    }

    async fn add_twin_segments(&self, index: &dyn Index) {
        self.add_twins_of_first(index, usize::MAX).await;
    }

    async fn add_twins_of_first(&self, index: &dyn Index, count: usize) {
        let head = self.handle.head().await.unwrap().unwrap();
        let mut shards = BTreeMap::new();
        for (id, shard) in head.shards() {
            let mut segments = shard.segments().to_vec();
            let built = shard
                .segments()
                .iter()
                .filter(|segment| segment.level() == Level::L1);
            for segment in built.take(count) {
                let chunks = segment
                    .files()
                    .iter()
                    .find(|file| file.name().as_str().starts_with("chunks-"))
                    .unwrap();
                let body = self
                    .handle
                    .read_segment_file(segment.id(), chunks.name())
                    .await
                    .unwrap();
                let twin = SegmentId::new(format!("{}-{}", segment.id(), index.kind())).unwrap();
                let name = SegmentFileName::new(format!("chunks-{twin}.bin")).unwrap();
                self.handle
                    .write_segment_file(&twin, &name, body)
                    .await
                    .unwrap();
                let built_by = BuiltBy::new(
                    Some(index.kind().clone()),
                    segment.built_by().extractors().clone(),
                );
                let files = vec![SegmentFile::new(name, chunks.size())];
                segments.push(SegmentEntry::new(twin, Level::L2, built_by, None, files));
            }
            shards.insert(id.clone(), ShardEntry::new(shard.range().clone(), segments));
        }
        let commit = cosmosearch_store::new_commit_id().unwrap();
        let next = head
            .successor(commit, shards, head.model_ids().clone())
            .unwrap();
        self.handle.commit(&next).await.unwrap();
    }

    async fn split_at(&self, key: &str) {
        let head = self.handle.head().await.unwrap().unwrap();
        let shards = [
            (
                ShardId::new("a").unwrap(),
                ShardEntry::new(KeyRange::new("", Some(key.to_owned())).unwrap(), vec![]),
            ),
            (
                ShardId::new("z").unwrap(),
                ShardEntry::new(KeyRange::new(key, None).unwrap(), vec![]),
            ),
        ];
        let commit = cosmosearch_store::new_commit_id().unwrap();
        let next = head
            .successor(commit, shards.into_iter().collect(), BTreeSet::new())
            .unwrap();
        self.handle.commit(&next).await.unwrap();
    }

    async fn put(&self, key: &str, body: &[u8]) {
        let key = ObjectKey::new(key).unwrap();
        self.store
            .put_overwrite(&key, Bytes::copy_from_slice(body))
            .await
            .unwrap();
    }

    async fn add(&self, files: &[(&str, &[u8])]) {
        for (key, body) in files {
            self.put(key, body).await;
        }
        let changes = files
            .iter()
            .map(|(key, _)| FileChange::Upsert(FileUpsert::new(*key)))
            .collect();
        self.indexer.apply(changes).await.unwrap();
    }

    async fn index(&self, files: &[(&str, &[u8])]) -> Vec<Chunk> {
        self.add(files).await;
        let plan = self.indexer.plan(1_000).await.unwrap();
        let mut extracted = Vec::new();
        let mut chunks = Vec::new();
        for batch in plan.batches() {
            for job in batch.jobs() {
                let file = self
                    .indexer
                    .extract(job.clone(), &extractors(), &ExtractLimits::default())
                    .await
                    .unwrap();
                if let ExtractOutcome::Chunks { chunks: found, .. } = file.outcome() {
                    chunks.extend(found.iter().cloned());
                }
                extracted.push(file);
            }
        }
        self.indexer.record(&extracted).await.unwrap();
        let bm25 = SharedIndex::new(Bm25::new().unwrap());
        let built = self
            .indexer
            .build(&bm25, extracted, vec![])
            .await
            .unwrap()
            .unwrap();
        let outcome = self.indexer.publish(&built).await.unwrap();
        assert!(
            matches!(outcome, PublishOutcome::Published(_)),
            "the segment was not published: {outcome:?}"
        );
        self.reader.refresh().await.unwrap();
        chunks
    }

    async fn search(&self, text: &str) -> Vec<SearchHit> {
        self.search_with(text, &Filter::new()).await
    }

    async fn search_with(&self, text: &str, filter: &Filter) -> Vec<SearchHit> {
        let results = self
            .reader
            .search(&SearchRequest::new(Query::new(text)).filter(filter.clone()))
            .await
            .unwrap();
        results.hits().to_vec()
    }

    async fn read(&self, reference: &SourceRef) -> Chunk {
        let mut chunks = self.reader.read(reference).await.unwrap().into_chunks();
        assert_eq!(chunks.len(), 1, "a read returns one chunk");
        chunks.remove(0)
    }

    async fn read_err(&self, reference: &str) -> Error {
        let reference: SourceRef = reference.parse().unwrap();
        self.reader.read(&reference).await.unwrap_err()
    }

    async fn etag(&self, key: &str) -> ETag {
        let key = ObjectKey::new(key).unwrap();
        self.data.head(&key).await.unwrap().etag().clone()
    }
}

fn long_text() -> String {
    (0..20)
        .map(|i| {
            format!(
                "Paragraph {i} talks about topic{i} at some length. {}",
                "words ".repeat(60)
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn long_markdown() -> String {
    let body = "Sentence about the section. ".repeat(40);
    format!(
        "---\ntitle: notes\n---\n# Terms\n\nIntro text.\n\n## Refunds\n\n{body}\n\n{body}\n\nRefunds take thirty days.\n\n## Delivery\n\nShipping notes\n========\n\nParcels {body}\n"
    )
}

const HTML: &[u8] = b"<html><head><title>Policy</title></head><body>\
<h1>Terms</h1><p>General terms &amp; conditions.</p>\
<h2>Refunds</h2><p>Refunds take <b>thirty</b> days.</p><script>var hidden = 1;</script>\
<p>Contact support.</p></body></html>";

#[tokio::test]
async fn every_chunk_reads_back_as_it_was_indexed() {
    let f = Fixture::new().await;
    let text = long_text();
    let markdown = long_markdown();
    let pages = "First page intro.\n\nFirst page more.\u{c}Second page only.";
    let chunks = f
        .index(&[
            ("docs/a.txt", text.as_bytes()),
            ("docs/b.md", markdown.as_bytes()),
            ("docs/c.html", HTML),
            ("docs/d.pages", pages.as_bytes()),
        ])
        .await;
    assert!(chunks.len() > 8, "{}", chunks.len());
    for chunk in chunks
        .iter()
        .filter(|c| matches!(c.source().locator(), Locator::Bytes(_)))
    {
        let read = f.read(chunk.source()).await;
        assert_eq!(read.text(), chunk.text(), "{}", chunk.source());
        assert_eq!(read.heading_path(), chunk.heading_path());
        assert_eq!(read.source(), chunk.source());
    }

    let first_page: SourceRef = format!(
        "s3://example-bucket/docs/d.pages#page=1@etag={}",
        f.etag("docs/d.pages").await
    )
    .parse()
    .unwrap();
    let read = f.read(&first_page).await;
    assert_eq!(read.text(), "First page intro.\n\nFirst page more.");
    assert_eq!(read.heading_path(), ["Part 1"]);
}

#[tokio::test]
async fn hits_carry_the_ref_page_and_heading_path() {
    let f = Fixture::new().await;
    let markdown = long_markdown();
    f.index(&[
        ("docs/b.md", markdown.as_bytes()),
        ("docs/d.pages", b"Nothing here.\x0cA zebracorn appears."),
    ])
    .await;

    let hits = f.search("thirty").await;
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].heading_path(), ["Terms", "Refunds"]);
    assert_eq!(hits[0].page(), None);
    assert!(hits[0].score() > 0.0);
    let read = f.read(hits[0].source()).await;
    assert!(read.text().contains("thirty days"));

    let hits = f.search("zebracorn").await;
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].page(), NonZeroU32::new(2));
    assert_eq!(hits[0].heading_path(), ["Part 2"]);
    assert_eq!(
        hits[0].source().to_string(),
        format!(
            "s3://example-bucket/docs/d.pages#page=2@etag={}",
            f.etag("docs/d.pages").await
        )
    );
}

#[tokio::test]
async fn a_changed_file_reports_stale() {
    let f = Fixture::new().await;
    f.index(&[("docs/a.txt", b"refund policy text")]).await;
    let hit = f.search("refund").await.remove(0);

    f.put("docs/a.txt", b"other words entirely").await;
    let err = f.reader.read(hit.source()).await.unwrap_err();
    assert!(matches!(err, Error::StaleRef { .. }), "{err}");

    f.add(&[("docs/a.txt", b"other words entirely")]).await;
    f.reader.refresh().await.unwrap();
    let err = f.reader.read(hit.source()).await.unwrap_err();
    assert!(matches!(err, Error::StaleRef { .. }), "{err}");
    assert_eq!(f.search("refund").await, Vec::<SearchHit>::new());
}

#[tokio::test]
async fn a_store_failure_on_the_interval_refresh_leaves_the_snapshot_answering() {
    let f = Fixture::new().await;
    f.index(&[("docs/a.txt", b"refund policy text")]).await;
    let faults = FaultStore::new(f.store.clone());
    let metadata =
        MetadataStore::new(faults.clone(), "s3://example-bucket/index".parse().unwrap()).unwrap();
    let reader = Reader::open(
        f.data.clone(),
        metadata,
        FileSource::HostNamed,
        plugins(),
        ReaderOptions::new().refresh_interval(Duration::ZERO),
    )
    .await
    .unwrap();
    let request = SearchRequest::new(Query::new("refund"));

    faults.inject(Op::Get, "INDEX", Fault::Crash);
    let results = reader.search(&request).await.unwrap();
    assert_eq!(results.hits().len(), 1);
    reader.refresh().await.unwrap();

    faults.inject(Op::Get, "INDEX", Fault::Crash);
    reader.read(results.hits()[0].source()).await.unwrap();
    reader.refresh().await.unwrap();

    faults.inject(Op::Get, "INDEX", Fault::Crash);
    let err = reader.refresh().await.unwrap_err();
    assert!(matches!(err, Error::Store(_)), "{err}");

    f.metadata.destroy().await.unwrap();
    let err = reader.search(&request).await.unwrap_err();
    assert!(matches!(err, Error::IndexRemoved), "{err}");
}

#[tokio::test]
async fn refuses_refs_outside_the_data_location_or_the_index() {
    let f = Fixture::new().await;
    f.index(&[("docs/a.txt", b"refund policy text")]).await;
    f.put("docs/secret.txt", b"never indexed").await;
    f.put("other/a.txt", b"refund policy text").await;
    let secret = f.etag("docs/secret.txt").await;
    let etag = f.etag("docs/a.txt").await;

    let outside = [
        format!("s3://example-bucket/other/a.txt#bytes=0-6@etag={etag}"),
        format!("s3://other-bucket/docs/a.txt#bytes=0-6@etag={etag}"),
        format!("file:///docs/a.txt#bytes=0-6@etag={etag}"),
        format!("s3://example-bucket/docsx/a.txt#bytes=0-6@etag={etag}"),
    ];
    for reference in outside {
        let err = f.read_err(&reference).await;
        assert!(
            matches!(err, Error::OutsideData { .. }),
            "{reference}: {err}"
        );
    }
    let err = f
        .read_err(&format!(
            "s3://example-bucket/docs/secret.txt#bytes=0-5@etag={secret}"
        ))
        .await;
    assert!(matches!(err, Error::NotInIndex { .. }), "{err}");
    let err = f
        .read_err(&format!(
            "s3://example-bucket/docs/a.txt#bytes=0-3@etag={etag}"
        ))
        .await;
    assert!(matches!(err, Error::NoPassage { .. }), "{err}");
}

#[tokio::test]
async fn reads_within_a_scope_and_filter_refuse_other_files_as_not_indexed() {
    let f = Fixture::new().await;
    let chunks = f
        .index(&[
            ("docs/a/plan.txt", b"refund policy text"),
            ("docs/b/plan.txt", b"refund launch budget"),
        ])
        .await;
    let change = AttributeChange::new("docs/a/plan.txt")
        .attribute("team", AttributeValue::One("a".to_owned()));
    f.indexer
        .apply(vec![FileChange::Attributes(change)])
        .await
        .unwrap();
    f.reader.refresh().await.unwrap();
    let source = |key: &str| {
        chunks
            .iter()
            .find(|chunk| chunk.source().location().key() == key)
            .unwrap()
            .source()
            .clone()
    };
    let (own, other) = (source("docs/a/plan.txt"), source("docs/b/plan.txt"));
    let team_a = Filter::new().equals("team", "a");
    let cases = [
        (Scope::prefix("docs/a/"), Filter::new()),
        (Scope::all(), team_a.clone()),
        (Scope::prefix("docs/"), team_a),
    ];
    for (scope, filter) in &cases {
        let read = f.reader.read_within(&own, scope, filter).await.unwrap();
        assert_eq!(read.chunks().len(), 1);
        f.reader
            .expand_within(&own, 1, 1, scope, filter)
            .await
            .unwrap();
        let err = f
            .reader
            .read_within(&other, scope, filter)
            .await
            .unwrap_err();
        assert!(matches!(err, Error::NotInIndex { .. }), "{err}");
        let err = f
            .reader
            .expand_within(&other, 1, 1, scope, filter)
            .await
            .unwrap_err();
        assert!(matches!(err, Error::NotInIndex { .. }), "{err}");
    }
    assert_eq!(f.reader.read(&other).await.unwrap().chunks().len(), 1);
}

#[tokio::test]
async fn a_pending_file_has_no_passage_to_read() {
    let f = Fixture::new().await;
    f.add(&[("docs/a.txt", b"refund policy text")]).await;
    f.reader.refresh().await.unwrap();
    let etag = f.etag("docs/a.txt").await;
    let err = f
        .read_err(&format!(
            "s3://example-bucket/docs/a.txt#bytes=0-18@etag={etag}"
        ))
        .await;
    assert!(matches!(err, Error::NoPassage { .. }), "{err}");
}

#[tokio::test]
async fn expand_returns_neighbours_from_the_same_file_only() {
    let f = Fixture::new().await;
    let text = long_text();
    let chunks = f
        .index(&[
            ("docs/a.txt", text.as_bytes()),
            ("docs/b.txt", text.as_bytes()),
        ])
        .await;
    let of_a: Vec<&Chunk> = chunks
        .iter()
        .filter(|c| c.source().location().key() == "docs/a.txt")
        .collect();
    assert!(of_a.len() >= 4, "{}", of_a.len());
    let refs =
        |found: &[Chunk]| -> Vec<String> { found.iter().map(|c| c.source().to_string()).collect() };
    let expected = |from: usize, to: usize| -> Vec<String> {
        of_a[from..to]
            .iter()
            .map(|c| c.source().to_string())
            .collect()
    };

    let middle = f.reader.expand(of_a[1].source(), 1, 2).await.unwrap();
    assert_eq!(refs(middle.chunks()), expected(0, 4));
    for (read, indexed) in middle.chunks().iter().zip(&of_a) {
        assert_eq!(read.text(), indexed.text());
    }
    let first = f.reader.expand(of_a[0].source(), 5, 0).await.unwrap();
    assert_eq!(refs(first.chunks()), expected(0, 1));
    let last = of_a.len() - 1;
    let end = f.reader.expand(of_a[last].source(), 0, 5).await.unwrap();
    assert_eq!(refs(end.chunks()), expected(last, last + 1));
}

#[tokio::test]
async fn a_search_and_a_read_report_what_they_read() {
    let f = Fixture::new().await;
    f.index(&[("docs/a.txt", b"refund policy text")]).await;
    let first = f
        .reader
        .search(&SearchRequest::new(Query::new("refund")))
        .await
        .unwrap();
    assert!(first.usage().object_reads() >= 3, "{:?}", first.usage());
    assert!(first.usage().bytes() > 0);
    let again = f
        .reader
        .search(&SearchRequest::new(Query::new("refund")))
        .await
        .unwrap();
    assert!(again.usage().object_reads() < first.usage().object_reads());

    let reading = f.reader.read(first.hits()[0].source()).await.unwrap();
    assert_eq!(reading.usage().object_reads(), 1);
    assert_eq!(reading.usage().bytes(), 18);
}

#[tokio::test]
async fn filters_apply_before_ranking() {
    let f = Fixture::new().await;
    f.index(&[
        ("docs/a.txt", b"refund policy"),
        ("docs/b.txt", b"refund rules"),
    ])
    .await;
    let change = AttributeChange::new("docs/b.txt")
        .attribute("owner", AttributeValue::One("legal".to_owned()));
    f.indexer
        .apply(vec![FileChange::Attributes(change)])
        .await
        .unwrap();
    f.reader.refresh().await.unwrap();
    let hits = f
        .search_with("refund", &Filter::new().equals("owner", "legal"))
        .await;
    let keys: Vec<&str> = hits.iter().map(|h| h.source().location().key()).collect();
    assert_eq!(keys, ["docs/b.txt"]);
}

#[tokio::test]
async fn ranges_filter_on_numbers_times_and_the_modified_time() {
    let f = Fixture::new().await;
    let old = Timestamp::from_millis(1_000);
    let amount = |value| AttributeValue::Number(Number::new(value).unwrap());
    let files: [(&str, &[u8]); 3] = [
        ("docs/a.txt", b"refund policy"),
        ("docs/b.txt", b"refund rules"),
        ("docs/c.txt", b"refund terms"),
    ];
    for (key, body) in files {
        f.put(key, body).await;
    }
    f.indexer
        .apply(vec![
            FileChange::Upsert(FileUpsert::new("docs/a.txt").attribute("amount", amount(150.0))),
            FileChange::Upsert(
                FileUpsert::new("docs/b.txt")
                    .attribute("amount", amount(900.0))
                    .attribute("signed", AttributeValue::Time(old)),
            ),
            FileChange::Upsert(
                FileUpsert::new("docs/c.txt").modified(Timestamp::from_millis(5_000)),
            ),
        ])
        .await
        .unwrap();
    f.index(&[]).await;
    let keys = |hits: Vec<SearchHit>| -> Vec<String> {
        let mut keys: Vec<String> = hits
            .iter()
            .map(|h| h.source().location().key().to_owned())
            .collect();
        keys.sort();
        keys
    };
    let cases = [
        (
            Filter::new().numbers("amount", 100.0..=200.0),
            vec!["docs/a.txt"],
        ),
        (
            Filter::new().numbers("amount", 100.0..),
            vec!["docs/a.txt", "docs/b.txt"],
        ),
        (Filter::new().times("signed", ..=old), vec!["docs/b.txt"]),
        (
            Filter::new().modified(Timestamp::from_millis(2_000)..),
            vec!["docs/c.txt"],
        ),
        (
            Filter::new().modified(..Timestamp::from_millis(2_000)),
            vec!["docs/a.txt", "docs/b.txt"],
        ),
    ];
    for (filter, expected) in cases {
        assert_eq!(
            keys(f.search_with("refund", &filter).await),
            expected,
            "{filter:?}"
        );
    }
}

#[tokio::test]
async fn an_exact_phrase_finds_only_the_files_that_hold_it() {
    let f = Fixture::new().await;
    f.index(&[
        ("docs/a.txt", b"the refund policy for annual plans"),
        ("docs/b.txt", b"a policy on refund requests"),
    ])
    .await;
    let keys = |hits: &[SearchHit]| -> Vec<String> {
        hits.iter()
            .map(|h| h.source().location().key().to_owned())
            .collect()
    };
    let exact = SearchRequest::new(Query::new("\"refund policy\"").with_exact_phrases());
    let found = f.reader.search(&exact).await.unwrap();
    assert_eq!(keys(found.hits()), ["docs/a.txt"]);
    assert_eq!(keys(&f.search("\"refund policy\"").await).len(), 2);
}

type Recorded = Vec<(Signal, Vec<ShardId>)>;

#[derive(Debug, Clone, Default)]
struct Tally(Arc<Mutex<Recorded>>);

impl HeatSink for Tally {
    fn record(&self, signal: Signal, shards: &[&ShardId]) {
        let shards = shards.iter().map(|shard| (*shard).clone()).collect();
        self.0.lock().unwrap().push((signal, shards));
    }
}

impl Tally {
    fn take(&self) -> Recorded {
        std::mem::take(&mut *self.0.lock().unwrap())
    }
}

#[tokio::test]
async fn searches_and_reads_record_heat_for_the_shards_they_cover() {
    let f = Fixture::new().await;
    f.split_at("docs/m").await;
    f.index(&[("docs/a.txt", b"refund policy text")]).await;
    f.index(&[("docs/z.txt", b"shipping rules")]).await;
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

    let results = reader
        .search(&SearchRequest::new(Query::new("refund")))
        .await
        .unwrap();
    reader
        .search(&SearchRequest::new(Query::new("warranty")))
        .await
        .unwrap();
    reader
        .search(
            &SearchRequest::new(Query::new("refund"))
                .filter(Filter::new().equals("owner", "legal")),
        )
        .await
        .unwrap();
    assert_eq!(
        tally.take(),
        [
            (Signal::Search, vec![a.clone(), z.clone()]),
            (Signal::Hit, vec![a.clone()]),
            (Signal::Search, vec![a.clone(), z]),
        ]
    );

    let source = results.hits()[0].source();
    reader.read(source).await.unwrap();
    reader.expand(source, 1, 1).await.unwrap();
    let elsewhere: SourceRef = "s3://example-bucket/docs/missing.txt#bytes=0-4@etag=9f3a"
        .parse()
        .unwrap();
    reader.read(&elsewhere).await.unwrap_err();
    assert_eq!(
        tally.take(),
        [(Signal::Read, vec![a.clone()]), (Signal::Read, vec![a])]
    );
}

#[tokio::test]
async fn debug_output_never_shows_document_text() {
    let f = Fixture::new().await;
    let markdown = "# Secret heading\n\nconfidential words";
    f.index(&[("docs/a.md", markdown.as_bytes())]).await;
    let hits = f.search("confidential").await;
    let reading = f.reader.read(hits[0].source()).await.unwrap();
    for output in [format!("{hits:?}"), format!("{reading:?}")] {
        assert!(!output.contains("confidential"), "{output}");
        assert!(!output.contains("Secret"), "{output}");
    }
}

#[tokio::test]
async fn a_ref_reads_only_its_own_byte_range() {
    let f = Fixture::new().await;
    let text = long_text();
    let chunks = f.index(&[("docs/a.txt", text.as_bytes())]).await;
    let Locator::Bytes(range) = chunks[1].source().locator() else {
        panic!("expected a byte locator");
    };
    f.reader.read(chunks[0].source()).await.unwrap();
    let reading = f.reader.read(chunks[1].source()).await.unwrap();
    assert_eq!(reading.usage().object_reads(), 1);
    assert_eq!(reading.usage().bytes(), range.len());
    let whole = ByteRange::new(0, u64::try_from(text.len()).unwrap()).unwrap();
    assert!(reading.usage().bytes() < whole.len());
}

#[tokio::test]
async fn opening_and_reading_need_the_plugins_that_built_the_index() {
    let f = Fixture::new().await;
    f.index(&[("docs/a.txt", b"refund policy text")]).await;
    let err = Reader::open(
        f.data.clone(),
        f.metadata.clone(),
        FileSource::HostNamed,
        Plugins::new(Extractors::new()),
        ReaderOptions::new(),
    )
    .await
    .unwrap_err();
    assert_eq!(
        err.to_string(),
        "this index needs plug-ins that were not added: extractor `text-2`, index `bm25-stem-en`"
    );
}

#[tokio::test]
async fn lists_of_two_index_kinds_fuse_on_the_passage() {
    let f = Fixture::new().await;
    f.index(&[
        ("docs/a.txt", b"alpha alpha alpha beta"),
        ("docs/b.txt", b"alpha beta gamma delta"),
        ("docs/c.txt", b"gamma delta"),
    ])
    .await;
    let backwards = Backwards(IndexKind::new("backwards").unwrap());
    f.add_twin_segments(&backwards).await;
    let mut plugins = plugins();
    plugins.add_index(SharedIndex::new(backwards)).unwrap();
    let reader = Reader::open(
        f.data.clone(),
        f.metadata.clone(),
        FileSource::HostNamed,
        plugins,
        ReaderOptions::new(),
    )
    .await
    .unwrap();

    let results = reader
        .search(&SearchRequest::new(Query::new("alpha")))
        .await
        .unwrap();
    assert_eq!(results.usage().tokens(), BACKWARDS_TOKENS);
    assert!(results.usage().object_reads() > 0);
    let hits = results.hits();
    let keys: Vec<&str> = hits.iter().map(|h| h.source().location().key()).collect();
    assert_eq!(keys.len(), 3, "each passage is one hit: {keys:?}");
    assert_eq!(keys[2], "docs/c.txt", "found by one kind only: {keys:?}");
    let mut both = keys[..2].to_vec();
    both.sort_unstable();
    assert_eq!(both, ["docs/a.txt", "docs/b.txt"]);

    let top_of_one_list = 1.0 / 61.0;
    assert!(hits[1].score() > top_of_one_list, "{hits:?}");
    assert!(hits[2].score() <= top_of_one_list, "{hits:?}");
    assert!(
        hits.windows(2)
            .all(|pair| pair[0].score() >= pair[1].score())
    );
    let kinds = |hit: &SearchHit| -> Vec<String> {
        hit.list_scores()
            .iter()
            .map(|(kind, _)| kind.to_string())
            .collect()
    };
    assert_eq!(kinds(&hits[0]), ["backwards", "bm25-stem-en"], "{hits:?}");
    assert_eq!(kinds(&hits[1]), ["backwards", "bm25-stem-en"], "{hits:?}");
    assert_eq!(kinds(&hits[2]), ["backwards"], "{hits:?}");
    let bm25 = |hit: &SearchHit| hit.list_scores()[1].1;
    let a = hits
        .iter()
        .position(|hit| hit.source().location().key() == "docs/a.txt");
    let b = hits
        .iter()
        .position(|hit| hit.source().location().key() == "docs/b.txt");
    assert!(
        bm25(&hits[a.unwrap()]) > bm25(&hits[b.unwrap()]),
        "{hits:?}"
    );

    let alone = f.search("alpha").await;
    let keys: Vec<&str> = alone.iter().map(|h| h.source().location().key()).collect();
    assert_eq!(keys, ["docs/a.txt", "docs/b.txt"]);
    assert!((alone[0].score() - 1.0 / 61.0).abs() < 1e-6, "{alone:?}");
    assert!((alone[1].score() - 1.0 / 62.0).abs() < 1e-6, "{alone:?}");
    let ranks = |hit: &SearchHit| -> Vec<(String, u32)> {
        hit.list_ranks()
            .iter()
            .map(|(kind, rank)| (kind.to_string(), *rank))
            .collect()
    };
    assert_eq!(ranks(&alone[0]), [("bm25-stem-en".to_owned(), 1)]);
    assert_eq!(ranks(&alone[1]), [("bm25-stem-en".to_owned(), 2)]);
    let mut backwards_ranks: Vec<u32> = hits
        .iter()
        .map(|hit| {
            assert_eq!(
                ranks(hit)
                    .iter()
                    .map(|(kind, _)| kind.clone())
                    .collect::<Vec<_>>(),
                kinds(hit)
            );
            ranks(hit)[0].1
        })
        .collect();
    backwards_ranks.sort_unstable();
    assert_eq!(backwards_ranks, [1, 2, 3]);
    let mut bm25_ranks: Vec<u32> = hits[..2].iter().map(|hit| ranks(hit)[1].1).collect();
    bm25_ranks.sort_unstable();
    assert_eq!(bm25_ranks, [1, 2]);
}

#[tokio::test]
async fn a_hit_carries_the_display_name_of_its_file() {
    let f = Fixture::new().await;
    f.index(&[
        ("docs/a.txt", b"refund policy"),
        ("docs/b.txt", b"refund rules"),
    ])
    .await;
    let change = AttributeChange::new("docs/b.txt").display_name("Refund rules 2024");
    f.indexer
        .apply(vec![FileChange::Attributes(change)])
        .await
        .unwrap();
    f.reader.refresh().await.unwrap();
    let hits = f.search("refund").await;
    let names: BTreeMap<&str, Option<&str>> = hits
        .iter()
        .map(|hit| (hit.source().location().key(), hit.display_name()))
        .collect();
    assert_eq!(
        names,
        BTreeMap::from([
            ("docs/a.txt", None),
            ("docs/b.txt", Some("Refund rules 2024"))
        ])
    );
}

#[tokio::test]
async fn a_kind_without_a_files_chunks_does_not_count_against_that_file() {
    let f = Fixture::new().await;
    f.index(&[
        ("docs/a.txt", b"alpha alpha alpha beta"),
        ("docs/b.txt", b"alpha beta gamma delta"),
    ])
    .await;
    f.index(&[("docs/z.txt", b"alpha alpha alpha alpha")]).await;
    let backwards = Backwards(IndexKind::new("backwards").unwrap());
    f.add_twins_of_first(&backwards, 1).await;
    let mut plugins = plugins();
    plugins.add_index(SharedIndex::new(backwards)).unwrap();
    let reader = Reader::open(
        f.data.clone(),
        f.metadata.clone(),
        FileSource::HostNamed,
        plugins,
        ReaderOptions::new(),
    )
    .await
    .unwrap();

    let results = reader
        .search(&SearchRequest::new(Query::new("alpha")))
        .await
        .unwrap();
    let hits = results.hits();
    let keys: Vec<&str> = hits.iter().map(|h| h.source().location().key()).collect();
    assert_eq!(keys, ["docs/z.txt", "docs/b.txt", "docs/a.txt"]);
    let first_in_every_list = 2.0 / 61.0;
    assert!(
        (hits[0].score() - first_in_every_list).abs() < 1e-6,
        "BM25's best hit, which the other kind could not hold: {hits:?}"
    );
}

#[derive(Debug)]
struct Rendezvous {
    kind: IndexKind,
    barrier: Arc<Barrier>,
}

#[async_trait]
impl Index for Rendezvous {
    fn kind(&self) -> &IndexKind {
        &self.kind
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
        _: &[SegmentEntry],
        _: &AllowedRows,
        _: &Query,
        _: usize,
    ) -> Result<Hits, IndexError> {
        match tokio::time::timeout(Duration::from_secs(5), self.barrier.wait()).await {
            Ok(_) => Ok(Hits::new(Vec::new(), Usage::of_tokens(1))),
            Err(_) => Ok(Hits::default()),
        }
    }
}

#[tokio::test]
async fn index_kinds_are_searched_at_the_same_time() {
    let f = Fixture::new().await;
    f.index(&[("docs/a.txt", b"alpha beta")]).await;
    let barrier = Arc::new(Barrier::new(2));
    let mut plugins = plugins();
    for name in ["first", "second"] {
        let index = Rendezvous {
            kind: IndexKind::new(name).unwrap(),
            barrier: Arc::clone(&barrier),
        };
        f.add_twin_segments(&index).await;
        plugins.add_index(SharedIndex::new(index)).unwrap();
    }
    let reader = Reader::open(
        f.data.clone(),
        f.metadata.clone(),
        FileSource::HostNamed,
        plugins,
        ReaderOptions::new(),
    )
    .await
    .unwrap();

    let results = reader
        .search(&SearchRequest::new(Query::new("alpha")))
        .await
        .unwrap();
    assert_eq!(
        results.usage().tokens(),
        2,
        "each kind waits for the other, so both report only when searched together"
    );
}

#[tokio::test]
async fn a_file_reads_page_by_page_in_reading_order_without_a_hit() {
    let f = Fixture::new().await;
    let indexed = f
        .index(&[
            (
                "docs/guide.md",
                b"# Refunds\n\nRefunds take five days.\n\n# Returns\n\nReturns need a label.\n\n# Fees\n\nNo fees apply.\n",
            ),
            ("docs/b/notes.txt", b"other notes"),
        ])
        .await;
    let expected: Vec<String> = indexed
        .iter()
        .filter(|chunk| chunk.source().location().key() == "docs/guide.md")
        .map(|chunk| chunk.text().to_owned())
        .collect();
    assert!(expected.len() >= 3, "{expected:?}");
    let texts = |reading: Reading| -> Vec<String> {
        reading
            .chunks()
            .iter()
            .map(|chunk| chunk.text().to_owned())
            .collect()
    };

    let mut paged = Vec::new();
    let mut first = 0;
    loop {
        let page = texts(f.reader.chunks("docs/guide.md", first, 2).await.unwrap());
        if page.is_empty() {
            break;
        }
        first += page.len();
        paged.extend(page);
    }
    assert_eq!(paged, expected);
    let all = f
        .reader
        .chunks("docs/guide.md", 0, usize::MAX)
        .await
        .unwrap();
    assert_eq!(all.chunks()[1].source().location().key(), "docs/guide.md");
    assert_eq!(texts(all), expected);

    let entry = f.reader.file("docs/guide.md").await.unwrap().unwrap();
    assert_eq!(entry.status(), FileStatus::Ready);
    assert_eq!(f.reader.file("docs/none.txt").await.unwrap(), None);
    let err = f.reader.file("elsewhere/x.txt").await.unwrap_err();
    assert!(matches!(err, Error::FileOutsideData { .. }), "{err}");
    let err = f.reader.chunks("docs/none.txt", 0, 1).await.unwrap_err();
    assert!(matches!(err, Error::FileNotInIndex { .. }), "{err}");

    let own = Scope::prefix("docs/guide");
    let read = f
        .reader
        .chunks_within("docs/guide.md", 0, 1, &own, &Filter::new())
        .await
        .unwrap();
    assert_eq!(read.chunks().len(), 1);
    let other = Scope::prefix("docs/b/");
    let err = f
        .reader
        .chunks_within("docs/guide.md", 0, 1, &other, &Filter::new())
        .await
        .unwrap_err();
    assert!(matches!(err, Error::FileNotInIndex { .. }), "{err}");

    let any = Filter::new();
    let within = f.reader.file_within("docs/guide.md", &own, &any).await;
    assert_eq!(within.unwrap().unwrap().status(), FileStatus::Ready);
    let outside = f.reader.file_within("docs/guide.md", &other, &any).await;
    assert_eq!(outside.unwrap(), None);
    let owned = Filter::new().equals("owner", "x");
    let unmatched = f.reader.file_within("docs/guide.md", &own, &owned).await;
    assert_eq!(unmatched.unwrap(), None);
    let missing = f.reader.file_within("docs/none.txt", &own, &any).await;
    assert_eq!(missing.unwrap(), None);
    let err = f
        .reader
        .file_within("elsewhere/x.txt", &own, &any)
        .await
        .unwrap_err();
    assert!(matches!(err, Error::FileOutsideData { .. }), "{err}");

    f.put("docs/guide.md", b"# Changed\n\nNew text.\n").await;
    let err = f.reader.chunks("docs/guide.md", 0, 1).await.unwrap_err();
    assert!(matches!(err, Error::StaleRef { .. }), "{err}");
}

#[tokio::test]
async fn a_boost_reorders_the_candidates_and_never_adds_one() {
    let f = Fixture::new().await;
    f.index(&[
        ("docs/a.txt", b"refund refund policy"),
        ("docs/b.txt", b"refund rules"),
        ("docs/c.txt", b"travel rules"),
    ])
    .await;
    let change = AttributeChange::new("docs/b.txt")
        .attribute("owner", AttributeValue::One("legal".to_owned()));
    f.indexer
        .apply(vec![FileChange::Attributes(change)])
        .await
        .unwrap();
    f.reader.refresh().await.unwrap();
    let keys = |results: SearchResults| -> Vec<String> {
        results
            .hits()
            .iter()
            .map(|hit| hit.source().location().key().to_owned())
            .collect()
    };
    let plain = SearchRequest::new(Query::new("refund"));
    assert_eq!(
        keys(f.reader.search(&plain).await.unwrap()),
        ["docs/a.txt", "docs/b.txt"]
    );
    let legal = Filter::new().equals("owner", "legal");
    let level = plain.clone().boost(Boost::matching(legal.clone(), 1.0));
    assert_eq!(
        keys(f.reader.search(&level).await.unwrap()),
        ["docs/a.txt", "docs/b.txt"]
    );
    let boosted = plain.clone().boost(Boost::matching(legal.clone(), 2.0));
    assert_eq!(
        keys(f.reader.search(&boosted).await.unwrap()),
        ["docs/b.txt", "docs/a.txt"]
    );
    let none = plain.clone().boost(Boost::matching(legal, 0.0));
    assert_eq!(
        keys(f.reader.search(&none).await.unwrap()),
        ["docs/a.txt", "docs/b.txt"]
    );
    let travel = plain.boost(Boost::matching(Filter::new().equals("owner", "x"), 5.0));
    assert_eq!(keys(f.reader.search(&travel).await.unwrap()).len(), 2);

    let bad = SearchRequest::new(Query::new("refund")).boost(Boost::matching(Filter::new(), -1.0));
    let err = f.reader.search(&bad).await.unwrap_err();
    assert!(matches!(err, Error::InvalidBoost { .. }), "{err}");
}
