use std::collections::BTreeSet;
use std::num::NonZeroUsize;

use crate::format::{BinaryVector, QuantizedVector};

use crate::centroids::dot;
use crate::testdata::Corpus;
use crate::{Centroids, KMeans};

const TOPICS: usize = 16;
const CLUSTERS: usize = 32;
const PER_TOPIC: usize = 100;
const DIMENSIONS: usize = 128;
const SPREAD: f32 = 1.0;
const QUERIES: usize = 40;
const TOP: usize = 10;

struct Fixture {
    corpus: Corpus,
    binary: Vec<BinaryVector>,
    int8: Vec<QuantizedVector>,
    queries: Vec<Vec<f32>>,
}

fn fixture(seed: u64) -> Fixture {
    let mut corpus = Corpus::new(seed, TOPICS, PER_TOPIC, DIMENSIONS, SPREAD);
    let binary = (0..corpus.len())
        .map(|index| BinaryVector::quantize(corpus.vector(index)).unwrap())
        .collect();
    let int8 = (0..corpus.len())
        .map(|index| QuantizedVector::quantize(corpus.vector(index)).unwrap())
        .collect();
    let queries = (0..QUERIES).map(|_| corpus.query(SPREAD)).collect();
    Fixture {
        corpus,
        binary,
        int8,
        queries,
    }
}

fn top_by_score(scored: Vec<(usize, f32)>, count: usize) -> Vec<usize> {
    let mut scored = scored;
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    scored
        .into_iter()
        .take(count)
        .map(|(index, _)| index)
        .collect()
}

fn exact(fixture: &Fixture, query: &[f32], candidates: &[usize]) -> Vec<usize> {
    let scored = candidates
        .iter()
        .map(|index| (*index, dot(fixture.corpus.vector(*index), query)))
        .collect();
    top_by_score(scored, TOP)
}

fn everything(fixture: &Fixture) -> Vec<usize> {
    (0..fixture.corpus.len()).collect()
}

fn binary_then_int8(
    fixture: &Fixture,
    query: &[f32],
    candidates: &[usize],
    keep: usize,
) -> Vec<usize> {
    let query_bits = BinaryVector::quantize(query).unwrap();
    let mut by_hamming: Vec<(u32, usize)> = candidates
        .iter()
        .map(|index| {
            let distance = fixture.binary[*index].hamming(&query_bits).unwrap();
            (distance, *index)
        })
        .collect();
    by_hamming.sort_unstable();
    let rescored = by_hamming
        .into_iter()
        .take(keep)
        .map(|(_, index)| (index, fixture.int8[index].dot(query).unwrap()))
        .collect();
    top_by_score(rescored, TOP)
}

fn recall(fixture: &Fixture, mut answer: impl FnMut(&[f32]) -> Vec<usize>) -> f64 {
    let all = everything(fixture);
    let mut hits = 0_usize;
    for query in &fixture.queries {
        let truth: BTreeSet<usize> = exact(fixture, query, &all).into_iter().collect();
        hits = hits.saturating_add(
            answer(query)
                .iter()
                .filter(|index| truth.contains(index))
                .count(),
        );
    }
    f64::from(u32::try_from(hits).unwrap())
        / f64::from(u32::try_from(fixture.queries.len().saturating_mul(TOP)).unwrap())
}

fn probed(
    fixture: &Fixture,
    centroids: &Centroids,
    assignments: &[usize],
    query: &[f32],
    probes: usize,
) -> Vec<usize> {
    let near = centroids.nearest(query, probes).unwrap();
    (0..fixture.corpus.len())
        .filter(|index| near.contains(&assignments[*index]))
        .collect()
}

fn trained(fixture: &Fixture, clusters: usize) -> (Centroids, Vec<usize>) {
    let centroids = KMeans::new(NonZeroUsize::new(clusters).unwrap())
        .train(fixture.corpus.vectors(), DIMENSIONS)
        .unwrap();
    let assignments = (0..fixture.corpus.len())
        .map(|index| centroids.assign(fixture.corpus.vector(index)).unwrap())
        .collect();
    (centroids, assignments)
}

#[test]
fn binary_scan_then_int8_rescoring_finds_the_exact_neighbours() {
    let fixture = fixture(11);
    let all = everything(&fixture);
    let measured = recall(&fixture, |query| {
        binary_then_int8(&fixture, query, &all, TOP * 10)
    });
    assert!(measured >= 0.9, "recall@10 {measured}");
}

#[test]
fn probing_the_nearest_clusters_finds_the_exact_neighbours() {
    let fixture = fixture(11);
    let (centroids, assignments) = trained(&fixture, CLUSTERS);
    let mut previous = 0.0;
    for probes in [1, 2, 4] {
        let measured = recall(&fixture, |query| {
            let candidates = probed(&fixture, &centroids, &assignments, query, probes);
            exact(&fixture, query, &candidates)
        });
        assert!(
            measured >= previous,
            "{probes} probes {measured} after {previous}"
        );
        previous = measured;
    }
    assert!(
        previous >= 0.95,
        "recall@10 with 4 of {CLUSTERS} clusters {previous}"
    );
    let combined = recall(&fixture, |query| {
        let candidates = probed(&fixture, &centroids, &assignments, query, 4);
        binary_then_int8(&fixture, query, &candidates, TOP * 10)
    });
    assert!(
        combined >= 0.9,
        "recall@10 with 4 clusters and both stages {combined}"
    );
}
