use std::collections::BTreeSet;
use std::fmt::Write;
use std::sync::{Arc, Mutex, MutexGuard};

use async_trait::async_trait;
use cosmosearch_core::{
    AllowedRows, BuiltBy, ByteRange, Chunk, ETag, EmbedCause, EmbedError, Embedder,
    EmbeddingIdentity, Embeddings, Hit, Index, IndexError, Level, Location, Locator, Passage,
    Query, RowSet, SegmentEntry, SegmentFile, SegmentFileName, SegmentFiles, SegmentId,
    SegmentSource, SharedEmbedder, Usage,
};
use cosmosearch_store::IndexHandle;
use cosmosearch_vector::Ivf;

#[derive(Debug)]
pub(crate) struct ParseEmbedder(EmbeddingIdentity);

impl ParseEmbedder {
    pub(crate) fn shared(identity: EmbeddingIdentity) -> SharedEmbedder {
        SharedEmbedder::new(ParseEmbedder(identity))
    }
}

fn parse(text: &str) -> Result<Vec<f32>, EmbedError> {
    text.split(',')
        .map(|value| {
            value
                .trim()
                .parse::<f32>()
                .map_err(|err| EmbedError::Model(EmbedCause::new(err)))
        })
        .collect()
}

impl Embedder for ParseEmbedder {
    fn identity(&self) -> &EmbeddingIdentity {
        &self.0
    }

    fn embed_documents(&self, texts: &[&str]) -> Result<Embeddings, EmbedError> {
        let vectors = texts
            .iter()
            .map(|text| parse(text))
            .collect::<Result<_, _>>()?;
        Ok(Embeddings::new(vectors, Usage::new()))
    }

    fn embed_query(&self, text: &str) -> Result<Embeddings, EmbedError> {
        Ok(Embeddings::new(vec![parse(text)?], Usage::new()))
    }
}

pub(crate) fn text(vector: &[f32]) -> String {
    let mut text = String::new();
    for (index, value) in vector.iter().enumerate() {
        if index > 0 {
            text.push(',');
        }
        write!(text, "{value}").unwrap();
    }
    text
}

pub(crate) fn identity(model: &str, dimensions: u32) -> EmbeddingIdentity {
    EmbeddingIdentity::builder(model, "r1", dimensions, 512)
        .build()
        .unwrap()
}

pub(crate) fn chunks(texts: &[String]) -> Vec<Chunk> {
    let location: Location = "s3://example-bucket/docs/a.txt".parse().unwrap();
    let etag = ETag::new("9f3a").unwrap();
    texts
        .iter()
        .map(|text| {
            let range = ByteRange::new(0, 1).unwrap();
            Passage::new(Locator::Bytes(range), vec![], text.as_str())
                .into_chunk(&location, &etag)
                .unwrap()
        })
        .collect()
}

pub(crate) async fn store_segment(
    handle: &IndexHandle,
    ivf: &Ivf,
    id: &str,
    texts: &[String],
) -> SegmentEntry {
    let id = SegmentId::new(id).unwrap();
    let mut files = Vec::new();
    for file in ivf.build(&id, &chunks(texts)).await.unwrap().into_files() {
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
        Level::L2,
        BuiltBy::new(Some(ivf.kind().clone()), BTreeSet::new()),
        ivf.model_id().cloned(),
        files,
    )
}

pub(crate) fn every_row(segments: &[SegmentEntry]) -> AllowedRows {
    let mut all = RowSet::new();
    all.insert(0..u64::MAX);
    let all = Arc::new(all);
    let mut allowed = AllowedRows::new();
    for segment in segments {
        allowed.allow(segment.id(), Arc::clone(&all));
    }
    allowed
}

pub(crate) async fn search(
    ivf: &Ivf,
    source: &SegmentSource,
    segments: &[SegmentEntry],
    query: &str,
    limit: usize,
) -> Vec<Hit> {
    ivf.search(
        source,
        segments,
        &every_row(segments),
        &Query::new(query),
        limit,
    )
    .await
    .unwrap()
    .into_hits()
}

#[derive(Debug, Clone)]
pub(crate) struct Read {
    pub(crate) round: usize,
    pub(crate) segment: SegmentId,
    pub(crate) range: ByteRange,
}

#[derive(Debug, Default)]
struct Log {
    in_flight: usize,
    rounds: usize,
    reads: Vec<Read>,
}

/// Records every read and the round it belongs to: a round starts whenever a read begins
/// while no other is in flight. Each read yields once before reading, so reads started
/// together are all in flight before any finishes.
#[derive(Debug, Clone)]
pub(crate) struct Recorder {
    inner: IndexHandle,
    log: Arc<Mutex<Log>>,
}

impl Recorder {
    pub(crate) fn new(inner: IndexHandle) -> Self {
        Recorder {
            inner,
            log: Arc::new(Mutex::new(Log::default())),
        }
    }

    fn log(&self) -> MutexGuard<'_, Log> {
        self.log.lock().unwrap()
    }

    pub(crate) fn source(&self) -> SegmentSource {
        SegmentSource::new(self.clone())
    }

    pub(crate) fn take(&self) -> Vec<Read> {
        let mut log = self.log();
        log.rounds = 0;
        std::mem::take(&mut log.reads)
    }
}

pub(crate) fn rounds(reads: &[Read]) -> usize {
    reads.iter().map(|read| read.round).max().unwrap_or(0)
}

#[async_trait]
impl SegmentFiles for Recorder {
    async fn read_range(
        &self,
        segment: &SegmentId,
        name: &SegmentFileName,
        range: ByteRange,
    ) -> Result<Vec<u8>, IndexError> {
        let round = {
            let mut log = self.log();
            if log.in_flight == 0 {
                log.rounds += 1;
            }
            log.in_flight += 1;
            log.rounds
        };
        tokio::task::yield_now().await;
        let read = self.inner.read_range(segment, name, range).await;
        let mut log = self.log();
        log.in_flight -= 1;
        log.reads.push(Read {
            round,
            segment: segment.clone(),
            range,
        });
        read
    }
}
