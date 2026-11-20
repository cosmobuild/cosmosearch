use std::collections::{BTreeMap, BinaryHeap};
use std::ops::Range;
use std::sync::Arc;

use crate::format::{IvfHead, QuantizedVector, hamming_bytes};
use cosmosearch_core::{Hit, IndexError, RowSet, SegmentFileName, SegmentId};

use crate::centroids::dot;

/// A segment a search touches: where its file is, its head and the rows it may return.
#[derive(Debug, Clone)]
pub(crate) struct Target {
    pub(crate) id: SegmentId,
    pub(crate) name: SegmentFileName,
    pub(crate) head: Arc<IvfHead>,
    pub(crate) rows: Arc<RowSet>,
}

impl Target {
    fn allows(&self, position: usize) -> bool {
        self.head
            .rows()
            .get(position)
            .is_some_and(|row| self.rows.contains(u64::from(*row)))
    }
}

/// One cluster chosen for scanning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Probe {
    pub(crate) target: usize,
    pub(crate) positions: Range<usize>,
}

/// Bytes read for consecutive positions of one target.
#[derive(Debug, Clone)]
pub(crate) struct Block {
    pub(crate) target: usize,
    pub(crate) positions: Range<usize>,
    pub(crate) bytes: Vec<u8>,
}

/// A vector kept by the binary scan. Ordered by distance first, so a heap of candidates
/// drops the farthest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Candidate {
    pub(crate) distance: u32,
    pub(crate) target: usize,
    pub(crate) position: usize,
}

/// The `budget` clusters nearest `query` over every target together, skipping clusters
/// with no vector a search may return.
pub(crate) fn choose(query: &[f32], targets: &[Target], budget: usize) -> Vec<Probe> {
    let mut scored: Vec<(f32, Probe)> = Vec::new();
    for (index, target) in targets.iter().enumerate() {
        for cluster in 0..target.head.clusters() {
            let (Some(centroid), Some(positions)) =
                (target.head.centroid(cluster), target.head.cluster(cluster))
            else {
                continue;
            };
            if !positions.is_empty() {
                let probe = Probe {
                    target: index,
                    positions,
                };
                scored.push((dot(centroid, query), probe));
            }
        }
    }
    scored.sort_by(|a, b| {
        b.0.total_cmp(&a.0)
            .then_with(|| a.1.target.cmp(&b.1.target))
            .then_with(|| a.1.positions.start.cmp(&b.1.positions.start))
    });
    scored
        .into_iter()
        .map(|(_, probe)| probe)
        .filter(|probe| {
            targets.get(probe.target).is_some_and(|target| {
                probe
                    .positions
                    .clone()
                    .any(|position| target.allows(position))
            })
        })
        .take(budget)
        .collect()
}

/// Joins ranges that touch or lie at most `max_gap` apart, so each becomes one read.
pub(crate) fn merge(mut ranges: Vec<Range<usize>>, max_gap: usize) -> Vec<Range<usize>> {
    ranges.sort_by_key(|range| (range.start, range.end));
    let mut merged: Vec<Range<usize>> = Vec::with_capacity(ranges.len());
    for range in ranges {
        if let Some(last) = merged.last_mut()
            && range.start <= last.end.saturating_add(max_gap)
        {
            last.end = last.end.max(range.end);
        } else {
            merged.push(range);
        }
    }
    merged
}

/// Scans the probed clusters' binary vectors in `blocks` and keeps the `keep` nearest to
/// `query_bits` by Hamming distance. Rows a search may not return are skipped, so they take
/// no place.
///
/// # Errors
///
/// Returns [`IndexError::Malformed`] if a block does not hold whole vectors or a probed
/// cluster was not read.
pub(crate) fn scan(
    query_bits: &[u8],
    targets: &[Target],
    probes: &[Probe],
    blocks: &[Block],
    keep: usize,
) -> Result<Vec<Candidate>, IndexError> {
    let mut heap: BinaryHeap<Candidate> = BinaryHeap::with_capacity(keep.saturating_add(1));
    for probe in probes {
        let target = targets.get(probe.target).ok_or_else(not_read)?;
        let block = blocks
            .iter()
            .find(|block| {
                block.target == probe.target
                    && block.positions.start <= probe.positions.start
                    && probe.positions.end <= block.positions.end
            })
            .ok_or_else(not_read)?;
        let vectors = target
            .head
            .binary_vectors(&block.bytes)
            .map_err(crate::ivf::format_error)?;
        if vectors.len() != block.positions.len() {
            return Err(not_read());
        }
        let skip = probe.positions.start.saturating_sub(block.positions.start);
        let found = vectors.skip(skip).zip(probe.positions.clone());
        for (bits, position) in found {
            if !target.allows(position) {
                continue;
            }
            heap.push(Candidate {
                distance: hamming_bytes(bits, query_bits),
                target: probe.target,
                position,
            });
            if heap.len() > keep {
                heap.pop();
            }
        }
    }
    Ok(heap.into_sorted_vec())
}

/// The candidates of each target, by position.
pub(crate) fn by_target(candidates: &[Candidate]) -> BTreeMap<usize, Vec<usize>> {
    let mut grouped: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for candidate in candidates {
        grouped
            .entry(candidate.target)
            .or_default()
            .push(candidate.position);
    }
    for positions in grouped.values_mut() {
        positions.sort_unstable();
    }
    grouped
}

/// Scores the candidates among `positions` of `target` with their int8 `vectors`.
///
/// # Errors
///
/// Returns [`IndexError::Malformed`] if a vector is missing or has other dimensions.
pub(crate) fn rescore(
    query: &[f32],
    target: &Target,
    positions: &Range<usize>,
    vectors: &[QuantizedVector],
    candidates: &[usize],
) -> Result<Vec<Hit>, IndexError> {
    let mut hits = Vec::new();
    for position in candidates
        .iter()
        .filter(|position| positions.contains(position))
    {
        let vector = vectors
            .get(position.saturating_sub(positions.start))
            .ok_or_else(not_read)?;
        let row = target.head.rows().get(*position).ok_or_else(not_read)?;
        let score = vector.dot(query).map_err(crate::ivf::format_error)?;
        hits.push(Hit::new(target.id.clone(), u64::from(*row), score));
    }
    Ok(hits)
}

/// Orders hits best first and keeps `limit`.
pub(crate) fn best(mut hits: Vec<Hit>, limit: usize) -> Vec<Hit> {
    hits.sort_by(|a, b| {
        b.score()
            .total_cmp(&a.score())
            .then_with(|| a.segment().cmp(b.segment()))
            .then_with(|| a.row().cmp(&b.row()))
    });
    hits.truncate(limit);
    hits
}

fn not_read() -> IndexError {
    IndexError::Malformed {
        reason: "an IVF read does not hold the vectors asked for",
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use crate::format::{IvfFile, IvfVector};

    use super::*;

    type Cluster = (Vec<f32>, Vec<(u32, Vec<f32>)>);

    fn target(id: &str, clusters: Vec<Cluster>) -> (Target, Vec<u8>) {
        let clusters = clusters
            .into_iter()
            .map(|(centroid, members)| {
                let members = members
                    .iter()
                    .map(|(row, vector)| IvfVector::quantize(*row, vector).unwrap())
                    .collect();
                (centroid, members)
            })
            .collect();
        let file = IvfFile::new(NonZeroU32::new(2).unwrap(), clusters).unwrap();
        let mut rows = RowSet::new();
        rows.insert(0..100);
        let target = Target {
            id: SegmentId::new(id).unwrap(),
            name: SegmentFileName::new(format!("ivf-{id}.bin")).unwrap(),
            head: Arc::new(file.head().clone()),
            rows: Arc::new(rows),
        };
        (target, file.encode().unwrap())
    }

    fn axis_segments() -> Vec<Target> {
        let (a, _) = target(
            "a",
            vec![
                (vec![1.0, 0.0], vec![(0, vec![1.0, 0.1])]),
                (vec![0.0, 1.0], vec![(1, vec![0.1, 1.0])]),
            ],
        );
        let (b, _) = target(
            "b",
            vec![
                (vec![0.9, 0.1], vec![(0, vec![1.0, 0.2])]),
                (vec![-1.0, 0.0], vec![(1, vec![-1.0, 0.0])]),
                (vec![0.95, 0.05], vec![]),
            ],
        );
        vec![a, b]
    }

    #[test]
    fn the_budget_goes_to_the_nearest_clusters_of_all_segments() {
        let targets = axis_segments();
        let chosen: Vec<(usize, usize)> = choose(&[1.0, 0.0], &targets, 2)
            .iter()
            .map(|probe| (probe.target, probe.positions.start))
            .collect();
        assert_eq!(chosen, [(0, 0), (1, 0)]);
        assert_eq!(choose(&[1.0, 0.0], &targets, 9).len(), 4);
        assert_eq!(choose(&[1.0, 0.0], &targets, 0).len(), 0);
    }

    #[test]
    fn a_cluster_with_no_allowed_row_takes_no_probe() {
        let mut targets = axis_segments();
        let mut only_second = RowSet::new();
        only_second.insert(1..2);
        targets[0].rows = Arc::new(only_second);
        let chosen: Vec<(usize, usize)> = choose(&[1.0, 0.0], &targets, 2)
            .iter()
            .map(|probe| (probe.target, probe.positions.start))
            .collect();
        assert_eq!(chosen, [(1, 0), (0, 1)]);
    }

    #[test]
    fn ranges_within_the_gap_are_read_together() {
        assert_eq!(merge(vec![], 4), Vec::<Range<usize>>::new());
        assert_eq!(merge(vec![5..6, 0..2, 2..3], 0), [0..3, 5..6]);
        assert_eq!(merge(vec![5..6, 0..2, 2..3], 2), vec![0..6]);
        assert_eq!(merge(vec![0..10, 3..4], 0), vec![0..10]);
    }

    #[test]
    fn the_scan_keeps_the_nearest_allowed_vectors() {
        let (mut segment, bytes) = target(
            "a",
            vec![(
                vec![1.0, 0.0],
                vec![
                    (0, vec![1.0, 1.0]),
                    (1, vec![1.0, -1.0]),
                    (2, vec![-1.0, -1.0]),
                ],
            )],
        );
        let range = segment.head.binary_range(0..3).unwrap();
        let read = bytes
            [usize::try_from(range.start()).unwrap()..usize::try_from(range.end()).unwrap()]
            .to_vec();
        let block = Block {
            target: 0,
            positions: 0..3,
            bytes: read,
        };
        let probe = Probe {
            target: 0,
            positions: 0..3,
        };
        let query = [0b11];
        let positions = |found: Vec<Candidate>| -> Vec<usize> {
            found.iter().map(|candidate| candidate.position).collect()
        };
        let targets = vec![segment.clone()];
        let all = scan(
            &query,
            &targets,
            std::slice::from_ref(&probe),
            std::slice::from_ref(&block),
            2,
        );
        assert_eq!(positions(all.unwrap()), [0, 1]);

        let mut without_first = RowSet::new();
        without_first.insert(1..3);
        segment.rows = Arc::new(without_first);
        let targets = vec![segment];
        let some = scan(
            &query,
            &targets,
            std::slice::from_ref(&probe),
            std::slice::from_ref(&block),
            2,
        );
        assert_eq!(positions(some.unwrap()), [1, 2]);

        let short = Block {
            bytes: block.bytes[..2].to_vec(),
            ..block
        };
        scan(&query, &targets, &[probe], &[short], 2).unwrap_err();
    }

    #[test]
    fn hits_are_ordered_by_score_then_segment_and_row() {
        let a = SegmentId::new("a").unwrap();
        let b = SegmentId::new("b").unwrap();
        let hits = vec![
            Hit::new(b.clone(), 1, 0.5),
            Hit::new(a.clone(), 2, 0.5),
            Hit::new(a, 1, 0.9),
            Hit::new(b, 0, -0.2),
        ];
        let ordered: Vec<(String, u64)> = best(hits, 3)
            .iter()
            .map(|hit| (hit.segment().to_string(), hit.row()))
            .collect();
        assert_eq!(
            ordered,
            [
                ("a".to_owned(), 1),
                ("a".to_owned(), 2),
                ("b".to_owned(), 1)
            ]
        );
    }
}
