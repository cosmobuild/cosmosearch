use std::collections::{BTreeMap, BTreeSet};
use std::error::Error as StdError;
use std::fmt;
use std::mem;
use std::num::{NonZeroU32, NonZeroUsize};
use std::ops::Range;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::SystemTime;

use crate::format::{
    BinaryVector, Error as FormatError, IvfFile, IvfHead, IvfVector, QuantizedVector,
};
use async_trait::async_trait;
use cosmosearch_core::{
    AllowedRows, Built, BuiltFile, ByteRange, Chunk, ChunkHash, EmbeddingIdentity, Hit, Hits,
    Index, IndexCause, IndexError, IndexKind, Level, Metric, ModelId, Query, RowSet, SegmentEntry,
    SegmentFile, SegmentFileName, SegmentId, SegmentSource, SharedEmbedder, Usage,
};
use futures_util::{StreamExt, TryStreamExt, future, stream};
use tracing::warn;

use crate::kmeans::normalize;
use crate::probe::{self, Block, Candidate, Probe, Target};
use crate::{Error, KMeans, Result, VectorStore};

const KIND_PREFIX: &str = "ivf-";
const FILE_PREFIX: &str = "ivf-";

/// Clusters a search probes, over all the segments it touches together. At 10, a search
/// takes two rounds of reads.
const DEFAULT_PROBES: NonZeroUsize = non_zero(10);

/// Candidates the binary scan keeps for each hit asked for: keeping 100 for the best 10
/// leaves the int8 rescoring enough to choose from.
const DEFAULT_CANDIDATES_PER_HIT: NonZeroUsize = non_zero(10);

/// Vectors a cluster holds on average. With about 512, a cluster's binary vectors are one
/// read of about 48 KB at 768 dimensions.
const DEFAULT_CLUSTER_SIZE: NonZeroUsize = non_zero(512);

/// Chunk texts handed to the embedder at once: a multiple of the backends' own batches (4
/// locally, 64 over HTTP), so each call keeps them full.
const EMBED_BATCH: usize = 256;

/// Texts embedded before their vectors are kept for reuse, so a failure later in a build
/// loses at most this many.
const KEEP_EVERY: usize = 4_096;

/// Reads in flight at once. Enough that the default budget's reads of one round all start
/// together, while a host with a large budget does not open hundreds of connections.
const READ_CONCURRENCY: usize = 32;

/// Unwanted bytes between two wanted ranges that are still read in one request. On object
/// storage 64 KiB take about a millisecond to transfer, far less than another request.
const MERGE_GAP_BYTES: usize = 64 * 1024;

/// Searches a segment's head stays in memory without being searched again. Searches on an
/// older manifest snapshot during a refresh, or on another index sharing the plug-in, keep
/// their heads, while the head of a segment that left the manifest goes soon after.
const KEEP_HEAD_FOR_SEARCHES: u64 = 64;

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Ivf>();
};

const fn non_zero(value: usize) -> NonZeroUsize {
    match NonZeroUsize::new(value) {
        Some(value) => value,
        None => NonZeroUsize::MIN,
    }
}

/// An IVF vector index of one embedding identity, one file per segment.
///
/// A build takes the vectors it can from the vector store, if it has one, embeds only the
/// other chunks and stores their vectors there, clusters them all with spherical k-means and
/// writes binary and int8 vectors cluster by cluster. A search embeds the query, chooses the
/// nearest clusters over all segments it touches, scans their binary vectors in one round of
/// parallel reads and rescores the best candidates from their int8 vectors in a second. Each
/// segment's centroids are read once and kept in memory until the segment has gone unsearched
/// for a while.
pub struct Ivf {
    kind: IndexKind,
    model_id: ModelId,
    embedder: SharedEmbedder,
    store: Option<VectorStore>,
    probes: NonZeroUsize,
    candidates_per_hit: NonZeroUsize,
    cluster_size: NonZeroUsize,
    heads: Mutex<Heads>,
}

#[derive(Debug, Default)]
struct Heads {
    searches: u64,
    by_segment: BTreeMap<SegmentId, CachedHead>,
}

#[derive(Debug, Clone)]
struct CachedHead {
    size: u64,
    head: Arc<IvfHead>,
    searched: u64,
}

#[derive(Debug, Clone, Copy)]
enum Section {
    Binary,
    Int8,
}

impl Ivf {
    /// The plug-in for vectors from `embedder`. Its kind is `ivf-` and the identity's model
    /// id, so plug-ins of several identities can serve one index.
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnsupportedMetric`] for a metric it cannot score, and [`Error::Core`]
    /// only if the id cannot be formed, which is a bug.
    pub fn new(embedder: SharedEmbedder) -> Result<Self> {
        let identity = embedder.get().identity();
        let metric = identity.metric();
        if !matches!(metric, Metric::Cosine | Metric::Dot) {
            return Err(Error::UnsupportedMetric { metric });
        }
        let model_id = identity.model_id()?;
        let kind = IndexKind::new(format!("{KIND_PREFIX}{}", model_id.as_str()))?;
        Ok(Ivf {
            kind,
            model_id,
            embedder,
            store: None,
            probes: DEFAULT_PROBES,
            candidates_per_hit: DEFAULT_CANDIDATES_PER_HIT,
            cluster_size: DEFAULT_CLUSTER_SIZE,
            heads: Mutex::new(Heads::default()),
        })
    }

    /// Reuses vectors from `store` when building, and keeps the vectors it embeds there.
    /// Without a store every chunk is embedded.
    #[must_use]
    pub fn vector_store(mut self, store: VectorStore) -> Self {
        self.store = Some(store);
        self
    }

    /// Sets how many clusters a search probes over all the segments it touches; 10 unless
    /// set.
    #[must_use]
    pub fn probes(mut self, probes: NonZeroUsize) -> Self {
        self.probes = probes;
        self
    }

    /// Sets how many candidates the binary scan keeps for rescoring, per hit asked for; 10
    /// unless set.
    #[must_use]
    pub fn candidates_per_hit(mut self, candidates: NonZeroUsize) -> Self {
        self.candidates_per_hit = candidates;
        self
    }

    /// Sets how many vectors a build puts in a cluster on average; 512 unless set.
    #[must_use]
    pub fn cluster_size(mut self, vectors: NonZeroUsize) -> Self {
        self.cluster_size = vectors;
        self
    }

    /// The identity of the vectors this plug-in builds and searches.
    pub fn identity(&self) -> &EmbeddingIdentity {
        self.embedder.get().identity()
    }

    fn dimensions(&self) -> NonZeroU32 {
        self.identity().dimensions()
    }

    fn file_of<'a>(&self, segment: &'a SegmentEntry) -> Option<&'a SegmentFile> {
        if segment.built_by().index() != Some(&self.kind)
            || segment.model_id() != Some(&self.model_id)
        {
            return None;
        }
        segment
            .files()
            .iter()
            .find(|file| file.name().as_str().starts_with(FILE_PREFIX))
    }

    fn prepare(&self, mut vector: Vec<f32>) -> Result<Vec<f32>, IndexError> {
        if !self.fits(&vector) {
            return Err(engine(
                "the embedder gave a vector that does not fit its identity",
            ));
        }
        if self.identity().metric() == Metric::Cosine {
            normalize(&mut vector);
        }
        Ok(vector)
    }

    fn fits(&self, vector: &[f32]) -> bool {
        u32::try_from(vector.len()).is_ok_and(|len| len == self.dimensions().get())
            && vector.iter().all(|value| value.is_finite())
    }

    /// A vector as the vector store gives it back, normalized again for a cosine identity,
    /// so a segment holds the same vectors whether they were reused or embedded.
    fn settle(&self, mut vector: Vec<f32>) -> Vec<f32> {
        if self.identity().metric() == Metric::Cosine {
            normalize(&mut vector);
        }
        vector
    }

    async fn reuse(&self, hashes: &[ChunkHash]) -> (BTreeMap<ChunkHash, Vec<f32>>, Usage) {
        let Some(store) = &self.store else {
            return (BTreeMap::new(), Usage::new());
        };
        let lookup = store.get(self.identity(), hashes).await;
        let usage = lookup.usage();
        let found = lookup
            .into_found()
            .into_iter()
            .filter(|(_, vector)| self.fits(vector))
            .map(|(hash, vector)| (hash, self.settle(vector)))
            .collect();
        (found, usage)
    }

    async fn embed_missing(
        &self,
        chunks: &[Chunk],
        known: &BTreeMap<ChunkHash, Vec<f32>>,
    ) -> Result<(Vec<(ChunkHash, Vec<f32>)>, Usage), IndexError> {
        let mut wanted = BTreeSet::new();
        let mut hashes = Vec::new();
        let mut texts = Vec::new();
        for chunk in chunks {
            if !known.contains_key(&chunk.hash()) && wanted.insert(chunk.hash()) {
                hashes.push(chunk.hash());
                texts.push(chunk.indexed_text());
            }
        }
        let mut vectors = Vec::with_capacity(texts.len());
        let mut usage = Usage::new();
        for (hashes, texts) in hashes.chunks(KEEP_EVERY).zip(texts.chunks(KEEP_EVERY)) {
            let shared = self.embedder.clone();
            let texts = texts.to_vec();
            let (embedded, spent) = tokio::task::spawn_blocking(move || embed_all(&shared, &texts))
                .await
                .map_err(engine)??;
            usage = usage.saturating_add(spent);
            let mut group = Vec::with_capacity(embedded.len());
            for (hash, vector) in hashes.iter().zip(embedded) {
                group.push((*hash, self.prepare(vector)?));
            }
            self.keep(&group).await;
            vectors.extend(group);
        }
        Ok((vectors, usage))
    }

    async fn keep(&self, vectors: &[(ChunkHash, Vec<f32>)]) {
        let Some(store) = &self.store else {
            return;
        };
        let entries: Vec<(ChunkHash, &[f32])> = vectors
            .iter()
            .map(|(hash, vector)| (*hash, vector.as_slice()))
            .collect();
        if let Err(err) = store
            .put(self.identity(), &entries, SystemTime::now())
            .await
        {
            warn!(model = %self.model_id, vectors = entries.len(), error = %err, "embedded vectors could not be kept for reuse");
        }
    }

    async fn embed_query(&self, query: &Query) -> Result<(Vec<f32>, Usage), IndexError> {
        let shared = self.embedder.clone();
        let text = query.text().to_owned();
        let found = tokio::task::spawn_blocking(move || shared.get().embed_query(&text))
            .await
            .map_err(engine)?
            .map_err(engine)?;
        let usage = found.usage();
        let vector = found
            .into_vectors()
            .into_iter()
            .next()
            .ok_or_else(|| engine("the embedder gave no vector for the query"))?;
        Ok((self.prepare(vector)?, usage))
    }

    #[expect(
        clippy::expect_used,
        reason = "a poisoned lock means an earlier panic, and its state is not trusted"
    )]
    fn heads(&self) -> MutexGuard<'_, Heads> {
        self.heads
            .lock()
            .expect("no thread panicked while holding the lock")
    }

    /// Counts a search of `segments`, keeps their heads and drops those not searched for
    /// [`KEEP_HEAD_FOR_SEARCHES`] searches.
    fn searching(&self, segments: &[SegmentEntry]) {
        let mut heads = self.heads();
        heads.searches = heads.searches.saturating_add(1);
        let now = heads.searches;
        for segment in segments {
            if let Some(cached) = heads.by_segment.get_mut(segment.id()) {
                cached.searched = now;
            }
        }
        heads
            .by_segment
            .retain(|_, cached| now.saturating_sub(cached.searched) < KEEP_HEAD_FOR_SEARCHES);
    }

    fn cached(&self, segment: &SegmentId, size: u64) -> Option<Arc<IvfHead>> {
        self.heads()
            .by_segment
            .get(segment)
            .filter(|cached| cached.size == size)
            .map(|cached| Arc::clone(&cached.head))
    }

    fn remember(&self, segment: &SegmentId, size: u64, head: &Arc<IvfHead>) {
        let head = Arc::clone(head);
        let mut heads = self.heads();
        let searched = heads.searches;
        heads.by_segment.insert(
            segment.clone(),
            CachedHead {
                size,
                head,
                searched,
            },
        );
    }

    async fn open(
        &self,
        source: &SegmentSource,
        wanted: Vec<(SegmentId, SegmentFile, Arc<RowSet>)>,
    ) -> Result<Vec<Target>, IndexError> {
        stream::iter(wanted)
            .map(|(id, file, rows)| async move {
                let head = if let Some(head) = self.cached(&id, file.size()) {
                    head
                } else {
                    let head = Arc::new(self.read_head(source, &id, &file).await?);
                    self.remember(&id, file.size(), &head);
                    head
                };
                let name = file.name().clone();
                Ok::<_, IndexError>(Target {
                    id,
                    name,
                    head,
                    rows,
                })
            })
            .buffered(READ_CONCURRENCY)
            .try_collect()
            .await
    }

    async fn read_head(
        &self,
        source: &SegmentSource,
        segment: &SegmentId,
        file: &SegmentFile,
    ) -> Result<IvfHead, IndexError> {
        let size = file.size();
        let guess = IvfHead::likely_len(self.dimensions(), size, self.cluster_size);
        let mut bytes = read(source, segment, file.name(), 0..guess).await?;
        let head_len = IvfHead::peek_len(&bytes).map_err(format_error)?;
        let have = u64::try_from(bytes.len()).map_err(engine)?;
        if head_len > size {
            return Err(malformed("an IVF file is shorter than its head"));
        }
        if head_len > have {
            bytes.extend(read(source, segment, file.name(), have..head_len).await?);
        }
        let head_len = usize::try_from(head_len).map_err(engine)?;
        let head_bytes = bytes
            .get(..head_len)
            .ok_or_else(|| malformed("an IVF file is shorter than its head"))?;
        let head = IvfHead::decode(head_bytes, size).map_err(format_error)?;
        if head.dimensions() != self.dimensions() {
            return Err(malformed("an IVF file holds vectors of other dimensions"));
        }
        Ok(head)
    }
}

impl fmt::Debug for Ivf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Ivf")
            .field("kind", &self.kind)
            .field("vector_store", &self.store.is_some())
            .field("probes", &self.probes)
            .field("candidates_per_hit", &self.candidates_per_hit)
            .field("cluster_size", &self.cluster_size)
            .field("cached_segments", &self.heads().by_segment.len())
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl Index for Ivf {
    fn kind(&self) -> &IndexKind {
        &self.kind
    }

    fn level(&self) -> Level {
        Level::L2
    }

    fn model_id(&self) -> Option<&ModelId> {
        Some(&self.model_id)
    }

    fn search_settings(&self) -> Vec<u64> {
        [self.probes, self.candidates_per_hit]
            .map(|setting| u64::try_from(setting.get()).unwrap_or(u64::MAX))
            .to_vec()
    }

    async fn build(&self, segment: &SegmentId, chunks: &[Chunk]) -> Result<Built, IndexError> {
        let name = SegmentFileName::new(format!("{FILE_PREFIX}{segment}.bin")).map_err(engine)?;
        let hashes: Vec<ChunkHash> = chunks.iter().map(Chunk::hash).collect();
        let (mut known, reads) = self.reuse(&hashes).await;
        let (embedded, tokens) = self.embed_missing(chunks, &known).await?;
        for (hash, vector) in embedded {
            let stored = QuantizedVector::quantize(&vector).map_err(format_error)?;
            known.insert(hash, self.settle(stored.dequantize()));
        }
        let vectors = hashes
            .iter()
            .map(|hash| known.get(hash).cloned())
            .collect::<Option<Vec<Vec<f32>>>>()
            .ok_or_else(|| engine("a chunk was left without a vector"))?;
        let dimensions = self.dimensions();
        let cluster_size = self.cluster_size;
        let body = tokio::task::spawn_blocking(move || {
            cluster(&vectors, dimensions, cluster_size)?
                .encode()
                .map_err(format_error)
        })
        .await
        .map_err(engine)??;
        Ok(Built::new(
            vec![BuiltFile::new(name, body)],
            reads.saturating_add(tokens),
        ))
    }

    async fn search(
        &self,
        source: &SegmentSource,
        segments: &[SegmentEntry],
        allowed: &AllowedRows,
        query: &Query,
        limit: usize,
    ) -> Result<Hits, IndexError> {
        if limit == 0 {
            return Ok(Hits::default());
        }
        self.searching(segments);
        let wanted: Vec<(SegmentId, SegmentFile, Arc<RowSet>)> = segments
            .iter()
            .filter_map(|segment| {
                let rows = allowed.rows(segment.id())?;
                let file = self.file_of(segment)?;
                Some((segment.id().clone(), file.clone(), Arc::clone(rows)))
            })
            .collect();
        if wanted.is_empty() {
            return Ok(Hits::default());
        }
        let ((query, usage), targets) =
            future::try_join(self.embed_query(query), self.open(source, wanted)).await?;
        let probes = probe::choose(&query, &targets, self.probes.get());
        if probes.is_empty() {
            return Ok(Hits::new(Vec::new(), usage));
        }
        let blocks = read_binary(source, &targets, &probes).await?;
        let keep = limit.saturating_mul(self.candidates_per_hit.get());
        let query_bits = BinaryVector::quantize(&query).map_err(format_error)?;
        let scanned = targets.clone();
        let candidates = tokio::task::spawn_blocking(move || {
            probe::scan(query_bits.as_bytes(), &scanned, &probes, &blocks, keep)
        })
        .await
        .map_err(engine)??;
        let hits = rescore(source, &targets, &candidates, &query).await?;
        Ok(Hits::new(probe::best(hits, limit), usage))
    }
}

fn embed_all(
    shared: &SharedEmbedder,
    texts: &[String],
) -> Result<(Vec<Vec<f32>>, Usage), IndexError> {
    let mut vectors = Vec::with_capacity(texts.len());
    let mut usage = Usage::new();
    for batch in texts.chunks(EMBED_BATCH) {
        let asked: Vec<&str> = batch.iter().map(String::as_str).collect();
        let answer = shared.get().embed_documents(&asked).map_err(engine)?;
        usage = usage.saturating_add(answer.usage());
        let found = answer.into_vectors();
        if found.len() != asked.len() {
            return Err(engine("the embedder gave a vector count other than asked"));
        }
        vectors.extend(found);
    }
    Ok((vectors, usage))
}

fn cluster(
    vectors: &[Vec<f32>],
    dimensions: NonZeroU32,
    cluster_size: NonZeroUsize,
) -> Result<IvfFile, IndexError> {
    let Some(width) = vectors.first().map(Vec::len) else {
        return IvfFile::new(dimensions, Vec::new()).map_err(format_error);
    };
    let clusters =
        NonZeroUsize::new(vectors.len().div_ceil(cluster_size.get())).unwrap_or(NonZeroUsize::MIN);
    let centroids = KMeans::new(clusters)
        .train(&vectors.concat(), width)
        .map_err(engine)?;
    let mut members: Vec<Vec<IvfVector>> = vec![Vec::new(); centroids.len()];
    for (row, vector) in vectors.iter().enumerate() {
        let cluster = centroids.assign(vector).map_err(engine)?;
        let row = u32::try_from(row).map_err(engine)?;
        let quantized = IvfVector::quantize(row, vector).map_err(format_error)?;
        if let Some(cluster) = members.get_mut(cluster) {
            cluster.push(quantized);
        }
    }
    let ordered = centroids
        .adjacent_order()
        .into_iter()
        .filter_map(|cluster| {
            let centroid = centroids.get(cluster)?.to_vec();
            Some((centroid, mem::take(members.get_mut(cluster)?)))
        })
        .collect();
    IvfFile::new(dimensions, ordered).map_err(format_error)
}

async fn read_binary(
    source: &SegmentSource,
    targets: &[Target],
    probes: &[Probe],
) -> Result<Vec<Block>, IndexError> {
    let mut plan = Vec::new();
    for (index, target) in targets.iter().enumerate() {
        let ranges: Vec<Range<usize>> = probes
            .iter()
            .filter(|probe| probe.target == index)
            .map(|probe| probe.positions.clone())
            .collect();
        let gap = MERGE_GAP_BYTES
            .checked_div(target.head.binary_stride())
            .unwrap_or(0);
        for positions in probe::merge(ranges, gap) {
            plan.push((index, positions));
        }
    }
    read_blocks(source, targets, plan, Section::Binary).await
}

async fn rescore(
    source: &SegmentSource,
    targets: &[Target],
    candidates: &[Candidate],
    query: &[f32],
) -> Result<Vec<Hit>, IndexError> {
    let grouped = probe::by_target(candidates);
    let mut plan = Vec::new();
    for (index, positions) in &grouped {
        let Some(target) = targets.get(*index) else {
            continue;
        };
        let gap = MERGE_GAP_BYTES
            .checked_div(target.head.int8_stride())
            .unwrap_or(0);
        let singles = positions
            .iter()
            .map(|position| *position..position.saturating_add(1))
            .collect();
        for run in probe::merge(singles, gap) {
            plan.push((*index, run));
        }
    }
    let blocks = read_blocks(source, targets, plan, Section::Int8).await?;
    let mut hits = Vec::with_capacity(candidates.len());
    for block in blocks {
        let (Some(target), Some(positions)) =
            (targets.get(block.target), grouped.get(&block.target))
        else {
            continue;
        };
        let vectors = target
            .head
            .int8_vectors(&block.bytes)
            .map_err(format_error)?;
        hits.extend(probe::rescore(
            query,
            target,
            &block.positions,
            &vectors,
            positions,
        )?);
    }
    Ok(hits)
}

async fn read_blocks(
    source: &SegmentSource,
    targets: &[Target],
    plan: Vec<(usize, Range<usize>)>,
    section: Section,
) -> Result<Vec<Block>, IndexError> {
    stream::iter(plan)
        .map(|(index, positions)| async move {
            let target = targets
                .get(index)
                .ok_or_else(|| malformed("an IVF read names no segment"))?;
            let range = match section {
                Section::Binary => target.head.binary_range(positions.clone()),
                Section::Int8 => target.head.int8_range(positions.clone()),
            }
            .map_err(format_error)?;
            let bytes = read(source, &target.id, &target.name, range.start()..range.end()).await?;
            Ok::<_, IndexError>(Block {
                target: index,
                positions,
                bytes,
            })
        })
        .buffer_unordered(READ_CONCURRENCY)
        .try_collect()
        .await
}

async fn read(
    source: &SegmentSource,
    segment: &SegmentId,
    name: &SegmentFileName,
    range: Range<u64>,
) -> Result<Vec<u8>, IndexError> {
    let range = ByteRange::new(range.start, range.end)
        .map_err(|_empty| malformed("an IVF file is empty"))?;
    let bytes = source.read_range(segment, name, range).await?;
    if u64::try_from(bytes.len()).ok() != Some(range.len()) {
        return Err(malformed("an IVF file is shorter than recorded"));
    }
    Ok(bytes)
}

pub(crate) fn format_error(err: FormatError) -> IndexError {
    if let FormatError::Malformed { reason } | FormatError::InvalidVector { reason } = err {
        return IndexError::Malformed { reason };
    }
    if matches!(err, FormatError::UnsupportedFormat { .. }) {
        return malformed("an IVF file is in a format this version does not read");
    }
    engine(err)
}

fn malformed(reason: &'static str) -> IndexError {
    IndexError::Malformed { reason }
}

fn engine(source: impl Into<Box<dyn StdError + Send + Sync>>) -> IndexError {
    IndexError::Engine(IndexCause::new(source))
}
