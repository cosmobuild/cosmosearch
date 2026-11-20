use crate::centroids::dot;
use crate::random::SplitMix64;

const UNIFORM_RMS: f32 = 0.577_350_3;

#[expect(
    clippy::cast_possible_truncation,
    reason = "a value in -1..1 is exactly representable to the precision a test needs"
)]
pub(crate) fn unit(rng: &mut SplitMix64) -> f32 {
    (rng.unit() * 2.0 - 1.0) as f32
}

fn normalized(mut values: Vec<f32>) -> Vec<f32> {
    let norm = dot(&values, &values).sqrt();
    for value in &mut values {
        *value /= norm;
    }
    values
}

fn noisy(rng: &mut SplitMix64, center: &[f32], spread: f32) -> Vec<f32> {
    let per_value = spread * UNIFORM_RMS.recip()
        / f32::from(u16::try_from(center.len()).unwrap_or(u16::MAX)).sqrt();
    normalized(
        center
            .iter()
            .map(|value| value + per_value * unit(rng))
            .collect(),
    )
}

pub(crate) struct Corpus {
    dimensions: usize,
    centers: Vec<Vec<f32>>,
    vectors: Vec<f32>,
    topics: Vec<usize>,
    rng: SplitMix64,
}

impl Corpus {
    pub(crate) fn new(
        seed: u64,
        topics: usize,
        per_topic: usize,
        dimensions: usize,
        spread: f32,
    ) -> Self {
        let mut rng = SplitMix64::new(seed);
        let centers: Vec<Vec<f32>> = (0..topics)
            .map(|_| normalized((0..dimensions).map(|_| unit(&mut rng)).collect()))
            .collect();
        let mut vectors = Vec::new();
        let mut assigned = Vec::new();
        for _ in 0..per_topic {
            for (topic, center) in centers.iter().enumerate() {
                vectors.extend(noisy(&mut rng, center, spread));
                assigned.push(topic);
            }
        }
        Corpus {
            dimensions,
            centers,
            vectors,
            topics: assigned,
            rng,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.topics.len()
    }

    pub(crate) fn vectors(&self) -> &[f32] {
        &self.vectors
    }

    pub(crate) fn vector(&self, index: usize) -> &[f32] {
        self.vectors
            .chunks_exact(self.dimensions)
            .nth(index)
            .unwrap()
    }

    pub(crate) fn topic(&self, index: usize) -> usize {
        self.topics[index]
    }

    pub(crate) fn query(&mut self, spread: f32) -> Vec<f32> {
        let topic = self.rng.below(self.centers.len());
        noisy(&mut self.rng, &self.centers[topic], spread)
    }
}
