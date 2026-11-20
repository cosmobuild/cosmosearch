use std::num::NonZeroUsize;

use crate::centroids::dot;
use crate::random::SplitMix64;
use crate::{Centroids, Error, Result};

/// Training rows kept per cluster. Training on a sample of this size costs the same however
/// large the segment is, and 256 rows a cluster is the amount commonly used for the
/// partitioning step of an IVF index, since the centres of a cluster settle long before
/// every row is seen.
const SAMPLE_PER_CLUSTER: usize = 256;

/// Most Lloyd iterations. Assignments of clustered embeddings stop changing well before
/// this; the cap bounds the cost where they do not.
const MAX_ITERATIONS: usize = 25;

/// Candidates drawn for each k-means++ seed beyond the first, plus `log2(clusters)`. Keeping
/// the candidate that lowers the total distance most is greedy k-means++, which rarely puts
/// two seeds in one cluster where plain k-means++ often does.
const BASE_SEED_CANDIDATES: usize = 2;

/// The seed used unless one is set, so runs are repeatable without anyone choosing a number.
const DEFAULT_SEED: u64 = 0x636f_736d_6f73_6561;

/// Spherical k-means: it clusters by angle, which fits cosine and dot-product search on the
/// normalized vectors an embedding gives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KMeans {
    clusters: NonZeroUsize,
    seed: u64,
}

impl KMeans {
    /// k-means that finds up to `clusters` clusters.
    pub fn new(clusters: NonZeroUsize) -> Self {
        KMeans {
            clusters,
            seed: DEFAULT_SEED,
        }
    }

    /// Sets the seed. The same seed and vectors always give the same centroids.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Finds centroids for `vectors`, rows of `dimensions` values laid out one after the
    /// other. It trains on a sample of at most 256 rows per cluster, chosen by the seed,
    /// seeds centroids with greedy k-means++, and runs Lloyd iterations until assignments stop
    /// changing or 25 are done. A cluster that ends up empty is re-seeded from the sample
    /// row farthest from its centroid. Asking for more clusters than there are vectors gives
    /// one cluster per vector.
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    ///
    /// use cosmosearch_vector::KMeans;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let vectors = [1.0, 0.1, 0.9, 0.0, -1.0, 0.1, -0.9, 0.0];
    /// let centroids = KMeans::new(NonZeroUsize::MIN.saturating_add(1)).train(&vectors, 2)?;
    /// assert_ne!(centroids.assign(&[1.0, 0.0])?, centroids.assign(&[-1.0, 0.0])?);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Clustering`] if `dimensions` is zero, `vectors` is empty or does not
    /// hold a whole number of rows, or a value is not finite.
    pub fn train(&self, vectors: &[f32], dimensions: usize) -> Result<Centroids> {
        let count = validate(vectors, dimensions)?;
        let clusters = self.clusters.get().min(count);
        let sample_size = count.min(clusters.saturating_mul(SAMPLE_PER_CLUSTER));
        let mut rng = SplitMix64::new(self.seed);
        let sample = draw_sample(vectors, dimensions, count, sample_size, &mut rng);
        let rows: Vec<&[f32]> = sample.chunks_exact(dimensions).collect();
        let mut centroids = seed_plus_plus(&rows, dimensions, clusters, &mut rng);
        refine(&rows, &mut centroids);
        Ok(centroids)
    }
}

fn validate(vectors: &[f32], dimensions: usize) -> Result<usize> {
    if dimensions == 0 || vectors.is_empty() {
        return Err(invalid(
            "there must be at least one vector of one dimension",
        ));
    }
    if vectors.len().checked_rem(dimensions) != Some(0) {
        return Err(invalid("the values are not a whole number of rows"));
    }
    if !vectors.iter().all(|value| value.is_finite()) {
        return Err(invalid("a value is not finite"));
    }
    Ok(vectors.chunks_exact(dimensions).len())
}

fn draw_sample(
    vectors: &[f32],
    dimensions: usize,
    count: usize,
    size: usize,
    rng: &mut SplitMix64,
) -> Vec<f32> {
    let mut indices: Vec<usize> = (0..count).collect();
    if size < count {
        for position in 0..size {
            let pick = position.saturating_add(rng.below(count.saturating_sub(position)));
            indices.swap(position, pick);
        }
        indices.truncate(size);
    }
    let mut sample = Vec::with_capacity(size.saturating_mul(dimensions));
    for index in indices {
        if let Some(row) = vectors.chunks_exact(dimensions).nth(index) {
            sample.extend_from_slice(row);
        }
    }
    for row in sample.chunks_exact_mut(dimensions) {
        normalize(row);
    }
    sample
}

pub(crate) fn normalize(row: &mut [f32]) -> bool {
    let norm = dot(row, row).sqrt();
    if !norm.is_normal() {
        return false;
    }
    for value in row {
        *value /= norm;
    }
    true
}

fn seed_plus_plus(
    rows: &[&[f32]],
    dimensions: usize,
    clusters: usize,
    rng: &mut SplitMix64,
) -> Centroids {
    let candidates = usize::try_from(clusters.ilog2())
        .unwrap_or(0)
        .saturating_add(BASE_SEED_CANDIDATES);
    let mut values = Vec::with_capacity(clusters.saturating_mul(dimensions));
    let mut similarity = vec![f32::NEG_INFINITY; rows.len()];
    let mut chosen = rng.below(rows.len());
    for seeded in 1..=clusters {
        let Some(row) = rows.get(chosen) else {
            break;
        };
        values.extend_from_slice(row);
        for (best, other) in similarity.iter_mut().zip(rows) {
            *best = best.max(dot(row, other));
        }
        if seeded < clusters {
            chosen = best_candidate(rows, &similarity, candidates, rng);
        }
    }
    Centroids::from_trusted(dimensions, values)
}

fn best_candidate(
    rows: &[&[f32]],
    similarity: &[f32],
    candidates: usize,
    rng: &mut SplitMix64,
) -> usize {
    (0..candidates)
        .map(|_| pick_by_distance(similarity, rng))
        .map(|index| (index, potential_with(rows, similarity, index)))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map_or(0, |(index, _)| index)
}

fn potential_with(rows: &[&[f32]], similarity: &[f32], candidate: usize) -> f64 {
    let Some(candidate) = rows.get(candidate) else {
        return f64::INFINITY;
    };
    similarity
        .iter()
        .zip(rows)
        .map(|(best, row)| squared_distance(best.max(dot(candidate, row))))
        .sum()
}

fn squared_distance(similarity: f32) -> f64 {
    let distance = f64::from((1.0 - similarity).max(0.0));
    distance * distance
}

fn pick_by_distance(similarity: &[f32], rng: &mut SplitMix64) -> usize {
    let weights: Vec<f64> = similarity
        .iter()
        .map(|best| squared_distance(*best))
        .collect();
    let total: f64 = weights.iter().sum();
    let target = rng.unit() * total;
    let mut cumulative = 0.0;
    for (index, weight) in weights.iter().enumerate() {
        cumulative += weight;
        if cumulative > target {
            return index;
        }
    }
    rng.below(similarity.len())
}

fn refine(rows: &[&[f32]], centroids: &mut Centroids) {
    let dimensions = centroids.dimensions();
    let clusters = centroids.len();
    let mut assignments: Vec<usize> = Vec::new();
    for _ in 0..MAX_ITERATIONS {
        let (next, mut scores): (Vec<usize>, Vec<f32>) =
            rows.iter().map(|row| centroids.best(row)).unzip();
        if next == assignments {
            break;
        }
        assignments = next;
        let mut sums = vec![0.0_f32; clusters.saturating_mul(dimensions)];
        let mut counts = vec![0_usize; clusters];
        for (row, cluster) in rows.iter().zip(&assignments) {
            if let Some(sum) = sums.chunks_exact_mut(dimensions).nth(*cluster) {
                for (total, value) in sum.iter_mut().zip(*row) {
                    *total += value;
                }
            }
            if let Some(count) = counts.get_mut(*cluster) {
                *count = count.saturating_add(1);
            }
        }
        for (cluster, (sum, count)) in sums.chunks_exact_mut(dimensions).zip(&counts).enumerate() {
            let usable = *count > 0 && normalize(sum);
            let source = if usable {
                Some(&*sum)
            } else {
                farthest(&mut scores).and_then(|index| rows.get(index).copied())
            };
            if let (Some(source), Some(target)) = (source, centroids.get_mut(cluster)) {
                target.copy_from_slice(source);
            }
        }
    }
}

fn farthest(scores: &mut [f32]) -> Option<usize> {
    let (index, _) = scores
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))?;
    if let Some(score) = scores.get_mut(index) {
        *score = f32::INFINITY;
    }
    Some(index)
}

fn invalid(reason: &'static str) -> Error {
    Error::Clustering { reason }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use proptest::prelude::*;

    use super::*;
    use crate::testdata::Corpus;

    fn clusters(count: usize) -> NonZeroUsize {
        NonZeroUsize::new(count).unwrap()
    }

    fn nearest_by_distance(centroids: &Centroids, vector: &[f32]) -> usize {
        let mut unit_vector = vector.to_vec();
        normalize(&mut unit_vector);
        let distance = |row: &[f32]| -> f32 {
            row.iter()
                .zip(&unit_vector)
                .map(|(a, b)| (a - b) * (a - b))
                .sum()
        };
        (0..centroids.len())
            .min_by(|a, b| {
                distance(centroids.get(*a).unwrap())
                    .total_cmp(&distance(centroids.get(*b).unwrap()))
            })
            .unwrap()
    }

    #[test]
    fn refuses_input_that_cannot_be_clustered() {
        let kmeans = KMeans::new(clusters(2));
        for (vectors, dimensions) in [
            (vec![], 2),
            (vec![1.0, 2.0], 0),
            (vec![1.0, 2.0, 3.0], 2),
            (vec![1.0, f32::INFINITY], 2),
        ] {
            let err = kmeans.train(&vectors, dimensions).unwrap_err();
            assert!(matches!(err, Error::Clustering { .. }), "{err}");
        }
    }

    #[test]
    fn asking_for_more_clusters_than_vectors_gives_one_per_vector() {
        let centroids = KMeans::new(clusters(10))
            .train(&[1.0, 0.0, 0.0, 1.0, -1.0, 0.0], 2)
            .unwrap();
        assert_eq!(centroids.len(), 3);
        let assigned: BTreeSet<usize> = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
            .iter()
            .map(|vector| centroids.assign(vector).unwrap())
            .collect();
        assert_eq!(assigned.len(), 3);
    }

    #[test]
    fn identical_vectors_still_give_the_clusters_asked_for() {
        let vectors = [0.6, 0.8].repeat(10);
        let centroids = KMeans::new(clusters(4)).train(&vectors, 2).unwrap();
        assert_eq!(centroids.len(), 4);
        assert!(centroids.values().iter().all(|value| value.is_finite()));
    }

    #[test]
    fn a_large_input_trains_on_a_bounded_sample() {
        let corpus = Corpus::new(5, 2, 1500, 16, 0.3);
        let centroids = KMeans::new(clusters(2))
            .train(corpus.vectors(), 16)
            .unwrap();
        let first = centroids.assign(corpus.vector(0)).unwrap();
        let other = centroids.assign(corpus.vector(1)).unwrap();
        assert_ne!(first, other);
    }

    #[test]
    fn centroids_are_unit_vectors() {
        let corpus = Corpus::new(9, 4, 30, 24, 1.0);
        let centroids = KMeans::new(clusters(6))
            .train(corpus.vectors(), 24)
            .unwrap();
        for row in centroids.values().as_chunks::<24>().0 {
            assert!((dot(row, row) - 1.0).abs() < 1e-4);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(24))]

        #[test]
        fn the_same_seed_gives_the_same_centroids(
            seed in any::<u64>(),
            data_seed in any::<u64>(),
            count in 1_usize..12,
        ) {
            let corpus = Corpus::new(data_seed, 3, 25, 16, 1.0);
            let kmeans = KMeans::new(clusters(count)).with_seed(seed);
            let first = kmeans.train(corpus.vectors(), 16).unwrap();
            let second = kmeans.train(corpus.vectors(), 16).unwrap();
            prop_assert_eq!(first, second);
        }

        #[test]
        fn every_vector_is_assigned_to_its_nearest_centroid(
            data_seed in any::<u64>(),
            count in 1_usize..10,
        ) {
            let corpus = Corpus::new(data_seed, 4, 20, 16, 1.0);
            let centroids = KMeans::new(clusters(count)).train(corpus.vectors(), 16).unwrap();
            for index in 0..corpus.len() {
                let vector = corpus.vector(index);
                prop_assert_eq!(
                    centroids.assign(vector).unwrap(),
                    nearest_by_distance(&centroids, vector)
                );
            }
        }

        #[test]
        fn well_separated_topics_are_recovered(
            data_seed in any::<u64>(),
            topics in 2_usize..7,
            dimensions in 32_usize..64,
        ) {
            let corpus = Corpus::new(data_seed, topics, 30, dimensions, 0.3);
            let centroids = KMeans::new(clusters(topics))
                .with_seed(data_seed.rotate_left(7))
                .train(corpus.vectors(), dimensions)
                .unwrap();
            let mut cluster_of_topic = BTreeSet::new();
            for topic in 0..topics {
                let assigned: BTreeSet<usize> = (0..corpus.len())
                    .filter(|index| corpus.topic(*index) == topic)
                    .map(|index| centroids.assign(corpus.vector(index)).unwrap())
                    .collect();
                prop_assert_eq!(assigned.len(), 1, "topic {} is split", topic);
                cluster_of_topic.extend(assigned);
            }
            prop_assert_eq!(cluster_of_topic.len(), topics);
        }
    }
}
