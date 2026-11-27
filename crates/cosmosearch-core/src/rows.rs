use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::Arc;

use crate::SegmentId;

const RANGE_BYTES: usize = 40;

/// A set of chunk rows of one segment, kept as ranges since a file's chunks are adjacent.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RowSet {
    ranges: BTreeMap<u64, u64>,
}

impl RowSet {
    /// An empty set.
    pub fn new() -> Self {
        RowSet::default()
    }

    /// About how many bytes the set holds in memory.
    pub fn heap_bytes(&self) -> usize {
        self.ranges
            .len()
            .saturating_mul(RANGE_BYTES)
            .saturating_add(size_of::<Self>())
    }

    /// Adds `rows`.
    pub fn insert(&mut self, rows: Range<u64>) {
        if rows.is_empty() {
            return;
        }
        let (mut start, mut end) = (rows.start, rows.end);
        if let Some((&first, last)) = self.ranges.last_key_value()
            && first <= start
        {
            if start <= *last {
                let last = (*last).max(end);
                self.ranges.insert(first, last);
            } else {
                self.ranges.insert(start, end);
            }
            return;
        }
        let touching: Vec<(u64, u64)> = self
            .ranges
            .range(..=end)
            .rev()
            .take_while(|(_, last)| **last >= start)
            .map(|(first, last)| (*first, *last))
            .collect();
        for (first, last) in touching {
            self.ranges.remove(&first);
            start = start.min(first);
            end = end.max(last);
        }
        self.ranges.insert(start, end);
    }

    /// Adds every row of `other`.
    pub fn extend(&mut self, other: &RowSet) {
        for (start, end) in &other.ranges {
            self.insert(*start..*end);
        }
    }

    /// Whether `row` is in the set.
    pub fn contains(&self, row: u64) -> bool {
        self.ranges
            .range(..=row)
            .next_back()
            .is_some_and(|(_, end)| row < *end)
    }

    /// Whether every row in `rows` is in the set.
    pub fn covers(&self, rows: Range<u64>) -> bool {
        rows.is_empty()
            || self
                .ranges
                .range(..=rows.start)
                .next_back()
                .is_some_and(|(_, end)| rows.end <= *end)
    }

    /// Whether the set holds no rows.
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }
}

/// The chunk rows a search may score and return, by segment. A segment not named has none.
/// Row sets are shared, not copied.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AllowedRows {
    segments: BTreeMap<SegmentId, Arc<RowSet>>,
}

impl AllowedRows {
    /// Allows nothing.
    pub fn new() -> Self {
        AllowedRows::default()
    }

    /// Allows `rows` of `segment` as well as what is allowed already.
    pub fn allow(&mut self, segment: &SegmentId, rows: Arc<RowSet>) {
        if rows.is_empty() {
            return;
        }
        if let Some(allowed) = self.segments.get_mut(segment) {
            if !Arc::ptr_eq(allowed, &rows) && allowed != &rows {
                Arc::make_mut(allowed).extend(&rows);
            }
        } else {
            self.segments.insert(segment.clone(), rows);
        }
    }

    /// The rows allowed in `segment`, if any.
    pub fn rows(&self, segment: &SegmentId) -> Option<&Arc<RowSet>> {
        self.segments.get(segment)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use proptest::prelude::*;

    use super::*;

    type Case<'a> = (&'a [(u64, u64)], &'a [u64]);

    fn set(ranges: &[(u64, u64)]) -> RowSet {
        let mut rows = RowSet::new();
        for (start, end) in ranges {
            rows.insert(*start..*end);
        }
        rows
    }

    fn members(rows: &RowSet) -> Vec<u64> {
        (0..20).filter(|row| rows.contains(*row)).collect()
    }

    #[test]
    fn ranges_merge_in_any_order() {
        let cases: [Case<'_>; 6] = [
            (&[], &[]),
            (&[(3, 3)], &[]),
            (&[(2, 4)], &[2, 3]),
            (&[(2, 4), (4, 6)], &[2, 3, 4, 5]),
            (&[(8, 10), (2, 4), (3, 9)], &[2, 3, 4, 5, 6, 7, 8, 9]),
            (
                &[(0, 1), (5, 6), (10, 12), (1, 11)],
                &(0..12).collect::<Vec<_>>(),
            ),
        ];
        for (ranges, expected) in cases {
            let rows = set(ranges);
            assert_eq!(members(&rows), expected, "{ranges:?}");
            assert_eq!(rows.is_empty(), expected.is_empty());
        }
        assert_eq!(set(&[(0, 1), (1, 2), (2, 3)]), set(&[(0, 3)]));
    }

    #[test]
    fn a_segment_not_named_allows_nothing_and_allowing_twice_adds_up() {
        let a = SegmentId::new("a").unwrap();
        let b = SegmentId::new("b").unwrap();
        let mut allowed = AllowedRows::new();
        let first = Arc::new(set(&[(0, 2)]));
        allowed.allow(&a, Arc::clone(&first));
        allowed.allow(&a, Arc::new(set(&[(5, 6)])));
        allowed.allow(&b, Arc::new(RowSet::new()));
        assert_eq!(
            allowed.rows(&a).map(AsRef::as_ref),
            Some(&set(&[(0, 2), (5, 6)]))
        );
        assert_eq!(*first, set(&[(0, 2)]));
        assert_eq!(allowed.rows(&b), None);
    }

    #[test]
    fn a_set_allowed_once_is_shared_and_not_copied() {
        let a = SegmentId::new("a").unwrap();
        let rows = Arc::new(set(&[(0, 4)]));
        let mut allowed = AllowedRows::new();
        allowed.allow(&a, Arc::clone(&rows));
        allowed.allow(&a, Arc::clone(&rows));
        assert!(Arc::ptr_eq(allowed.rows(&a).unwrap(), &rows));
    }

    #[test]
    fn covers_only_ranges_held_whole() {
        let rows = set(&[(2, 5), (7, 9)]);
        let cases = [
            (0..0, true),
            (2..5, true),
            (3..4, true),
            (7..9, true),
            (2..6, false),
            (1..3, false),
            (4..8, false),
            (9..10, false),
        ];
        for (range, expected) in cases {
            assert_eq!(rows.covers(range.clone()), expected, "{range:?}");
        }
        assert!(!RowSet::new().covers(0..1));
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        #[test]
        fn insert_and_covers_agree_with_a_plain_set(
            ranges in prop::collection::vec((0u64..40, 0u64..8), 0..12),
            probe in (0u64..48, 0u64..10),
        ) {
            let mut rows = RowSet::new();
            let mut plain = BTreeSet::new();
            for (start, len) in &ranges {
                let end = start.saturating_add(*len);
                rows.insert(*start..end);
                plain.extend(*start..end);
            }
            for row in 0..50 {
                prop_assert_eq!(rows.contains(row), plain.contains(&row));
            }
            let (start, len) = probe;
            let end = start.saturating_add(len);
            let expected = (start..end).all(|row| plain.contains(&row));
            prop_assert_eq!(rows.covers(start..end), expected);
        }
    }
}
