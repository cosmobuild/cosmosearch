use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroUsize;

use cosmosearch_core::{
    BuiltBy, BuiltFile, EmbedCause, EmbedError, Embedder, EmbeddingIdentity, Embeddings, Index,
    Level, Metric, SegmentEntry, SegmentFile, SegmentId, SharedEmbedder, Usage,
};
use cosmosearch_store::{IndexHandle, MemoryStore};
use cosmosearch_vector::Ivf;
use futures_util::{StreamExt, stream};

use crate::segments::{Recorder, chunks, rounds, search};
use crate::support::create;

const SEGMENTS: usize = 4;
const PER_SEGMENT: usize = 1024;
const DIMENSIONS: usize = 128;
const TOPICS: usize = 16;
const SPREAD: f32 = 1.0;
const QUERIES: usize = 40;
const TOP: usize = 10;
const CLUSTER_SIZE: usize = 64;
const UNIFORM_RMS: f32 = 0.577_350_3;

struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f32 {
        let high = u16::try_from(self.next() >> 48).unwrap();
        f32::from(high) / 32768.0 - 1.0
    }

    fn below(&mut self, bound: usize) -> usize {
        usize::try_from(self.next() % u64::try_from(bound).unwrap()).unwrap()
    }
}

fn normalized(values: Vec<f32>) -> Vec<f32> {
    let length = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    values.into_iter().map(|value| value / length).collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

struct Corpus {
    centers: Vec<Vec<f32>>,
    vectors: Vec<Vec<f32>>,
    rng: SplitMix64,
}

impl Corpus {
    fn new(seed: u64, lengths: bool) -> Self {
        let mut rng = SplitMix64(seed);
        let centers: Vec<Vec<f32>> = (0..TOPICS)
            .map(|_| normalized((0..DIMENSIONS).map(|_| rng.unit()).collect()))
            .collect();
        let mut corpus = Corpus {
            centers,
            vectors: Vec::new(),
            rng,
        };
        for index in 0..SEGMENTS * PER_SEGMENT {
            let mut vector = corpus.noisy(index % TOPICS);
            if lengths {
                let length = 1.25 + 0.75 * corpus.rng.unit();
                for value in &mut vector {
                    *value *= length;
                }
            }
            corpus.vectors.push(vector);
        }
        corpus
    }

    fn noisy(&mut self, topic: usize) -> Vec<f32> {
        let per_value = SPREAD / UNIFORM_RMS / f32::from(u16::try_from(DIMENSIONS).unwrap()).sqrt();
        let center = self.centers.get(topic).unwrap().clone();
        normalized(
            center
                .iter()
                .map(|value| value + per_value * self.rng.unit())
                .collect(),
        )
    }

    fn query(&mut self) -> Vec<f32> {
        let topic = self.rng.below(TOPICS);
        self.noisy(topic)
    }

    fn exact(&self, query: &[f32]) -> BTreeSet<usize> {
        let mut scored: Vec<(f32, usize)> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(index, vector)| (dot(vector, query), index))
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        scored
            .into_iter()
            .take(TOP)
            .map(|(_, index)| index)
            .collect()
    }
}

#[derive(Debug)]
struct TableEmbedder {
    identity: EmbeddingIdentity,
    vectors: BTreeMap<String, Vec<f32>>,
}

impl TableEmbedder {
    fn look_up(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        self.vectors
            .get(text)
            .cloned()
            .ok_or_else(|| EmbedError::Model(EmbedCause::new("unknown text")))
    }
}

impl Embedder for TableEmbedder {
    fn identity(&self) -> &EmbeddingIdentity {
        &self.identity
    }

    fn embed_documents(&self, texts: &[&str]) -> Result<Embeddings, EmbedError> {
        let vectors = texts
            .iter()
            .map(|text| self.look_up(text))
            .collect::<Result<_, _>>()?;
        Ok(Embeddings::new(vectors, Usage::new()))
    }

    fn embed_query(&self, text: &str) -> Result<Embeddings, EmbedError> {
        Ok(Embeddings::new(vec![self.look_up(text)?], Usage::new()))
    }
}

struct Fixture {
    handle: IndexHandle,
    embedder: SharedEmbedder,
    segments: Vec<SegmentEntry>,
    truths: Vec<BTreeSet<usize>>,
}

fn size(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).unwrap()
}

impl Fixture {
    async fn new(seed: u64, metric: Metric) -> Self {
        let lengths = metric == Metric::Dot;
        let mut corpus = Corpus::new(seed, lengths);
        let queries: Vec<Vec<f32>> = (0..QUERIES).map(|_| corpus.query()).collect();
        let truths = queries.iter().map(|query| corpus.exact(query)).collect();
        let mut vectors: BTreeMap<String, Vec<f32>> = corpus
            .vectors
            .iter()
            .enumerate()
            .map(|(index, vector)| (format!("d{index}"), vector.clone()))
            .collect();
        vectors.extend(
            queries
                .into_iter()
                .enumerate()
                .map(|(index, query)| (format!("q{index}"), query)),
        );
        let identity = EmbeddingIdentity::builder("table", "r1", 128, 512)
            .normalized(!lengths)
            .metric(metric)
            .build()
            .unwrap();
        let embedder = SharedEmbedder::new(TableEmbedder { identity, vectors });
        let handle = create(
            MemoryStore::new(),
            "s3://example-bucket/meta".parse().unwrap(),
        )
        .await;
        let mut fixture = Fixture {
            handle,
            embedder,
            segments: Vec::new(),
            truths,
        };
        fixture.build().await;
        fixture
    }

    fn ivf(&self, probes: usize, candidates: usize) -> Ivf {
        Ivf::new(self.embedder.clone())
            .unwrap()
            .cluster_size(size(CLUSTER_SIZE))
            .probes(size(probes))
            .candidates_per_hit(size(candidates))
    }

    async fn build(&mut self) {
        let ivf = self.ivf(1, 1);
        let ivf = &ivf;
        let built: Vec<(SegmentId, Vec<BuiltFile>)> = stream::iter(0..SEGMENTS)
            .map(|segment| async move {
                let texts: Vec<String> = (0..PER_SEGMENT)
                    .map(|row| format!("d{}", segment * PER_SEGMENT + row))
                    .collect();
                let id = SegmentId::new(format!("s{segment}")).unwrap();
                let files = ivf.build(&id, &chunks(&texts)).await.unwrap().into_files();
                (id, files)
            })
            .buffered(SEGMENTS)
            .collect()
            .await;
        for (id, files) in built {
            let mut stored = Vec::new();
            for file in files {
                let size = u64::try_from(file.body().len()).unwrap();
                let name = file.name().clone();
                self.handle
                    .write_segment_file(&id, &name, file.into_body().into())
                    .await
                    .unwrap();
                stored.push(SegmentFile::new(name, size));
            }
            self.segments.push(SegmentEntry::new(
                id,
                Level::L2,
                BuiltBy::new(Some(ivf.kind().clone()), BTreeSet::new()),
                ivf.model_id().cloned(),
                stored,
            ));
        }
    }

    async fn measure(&self, probes: usize, candidates: usize) -> Measured {
        let ivf = self.ivf(probes, candidates);
        let recorder = Recorder::new(self.handle.clone());
        let source = recorder.source();
        search(&ivf, &source, &self.segments, "q0", TOP).await;
        recorder.take();
        let mut measured = Measured::default();
        for (index, truth) in self.truths.iter().enumerate() {
            let hits = search(&ivf, &source, &self.segments, &format!("q{index}"), TOP).await;
            let found = hits
                .iter()
                .filter(|hit| {
                    let segment: usize = hit
                        .segment()
                        .as_str()
                        .trim_start_matches('s')
                        .parse()
                        .unwrap();
                    let row = usize::try_from(hit.row()).unwrap();
                    truth.contains(&(segment * PER_SEGMENT + row))
                })
                .count();
            measured.found += found;
            let reads = recorder.take();
            measured.most_rounds = measured.most_rounds.max(rounds(&reads));
            for read in reads {
                let round = read.round.min(2) - 1;
                *measured.reads.get_mut(round).unwrap() += 1;
                *measured.bytes.get_mut(round).unwrap() += read.range.len();
            }
        }
        measured
    }
}

#[derive(Debug, Default)]
struct Measured {
    found: usize,
    most_rounds: usize,
    reads: [usize; 2],
    bytes: [u64; 2],
}

impl Measured {
    fn recall(&self) -> f64 {
        let found = f64::from(u32::try_from(self.found).unwrap());
        found / f64::from(u32::try_from(QUERIES * TOP).unwrap())
    }

    fn report(&self, label: &str, probes: usize, candidates: usize) -> String {
        let queries = f64::from(u32::try_from(QUERIES).unwrap());
        let per_query = |total: usize| f64::from(u32::try_from(total).unwrap()) / queries;
        let kib = |total: u64| f64::from(u32::try_from(total).unwrap()) / queries / 1024.0;
        format!(
            "{label} probes {probes:>2} candidates {:>3}: recall@10 {:.3}, reads {:.1} + {:.1}, KiB {:.1} + {:.1}",
            candidates * TOP,
            self.recall(),
            per_query(self.reads.first().copied().unwrap_or(0)),
            per_query(self.reads.get(1).copied().unwrap_or(0)),
            kib(self.bytes.first().copied().unwrap_or(0)),
            kib(self.bytes.get(1).copied().unwrap_or(0)),
        )
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn recall_against_exact_search_across_segments() {
    let fixture = Fixture::new(11, Metric::Cosine).await;
    let mut table = BTreeMap::new();
    for probes in [4, 10, 20] {
        for candidates in [5, 10, 20] {
            let measured = fixture.measure(probes, candidates).await;
            println!("{}", measured.report("cosine", probes, candidates));
            assert_eq!(measured.most_rounds, 2, "a warm search takes two rounds");
            table.insert((probes, candidates), measured.recall());
        }
    }
    let defaults = table[&(10, 10)];
    assert!(defaults >= 0.85, "recall@10 at the defaults {defaults}");
    let more_candidates = table[&(10, 20)];
    assert!(
        more_candidates >= 0.95,
        "recall@10 with 200 candidates {more_candidates}"
    );
    assert!(table[&(4, 5)] <= defaults, "{table:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn recall_of_vectors_compared_by_dot_product() {
    let fixture = Fixture::new(23, Metric::Dot).await;
    let mut found = Vec::new();
    for (probes, candidates) in [(10, 10), (10, 20)] {
        let measured = fixture.measure(probes, candidates).await;
        println!("{}", measured.report("dot", probes, candidates));
        found.push(measured.recall());
    }
    assert!(found[0] >= 0.65, "recall@10 at the defaults {}", found[0]);
    assert!(
        found[1] >= 0.9,
        "recall@10 with 200 candidates {}",
        found[1]
    );
}
