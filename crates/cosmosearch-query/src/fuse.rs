use std::collections::{BTreeMap, BTreeSet};

use cosmosearch_core::IndexKind;

use crate::{Error, Result};

/// The rank constant of reciprocal rank fusion unless set. Cormack, Clarke and Büttcher (2009)
/// found 60 works well across collections: it damps the weight of the very first ranks, so one
/// list's top hit cannot outweigh agreement between lists.
const DEFAULT_K: f64 = 60.0;
const DEFAULT_WEIGHT: f64 = 1.0;

/// The rank constant and the weight of each index kind's list.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Fusion {
    k: f64,
    weights: BTreeMap<IndexKind, f64>,
}

impl Default for Fusion {
    fn default() -> Self {
        Fusion {
            k: DEFAULT_K,
            weights: BTreeMap::new(),
        }
    }
}

impl Fusion {
    pub(crate) fn set_k(&mut self, k: f64) {
        self.k = k;
    }

    pub(crate) fn set_weight(&mut self, kind: IndexKind, weight: f64) {
        self.weights.insert(kind, weight);
    }

    pub(crate) fn k(&self) -> f64 {
        self.k
    }

    pub(crate) fn weight(&self, kind: &IndexKind) -> f64 {
        self.weights.get(kind).copied().unwrap_or(DEFAULT_WEIGHT)
    }

    /// Refuses a rank constant or weight that is not finite and above zero.
    pub(crate) fn check(&self) -> Result<()> {
        if !usable(self.k) {
            return Err(Error::InvalidFusion {
                kind: None,
                value: self.k,
            });
        }
        for (kind, weight) in &self.weights {
            if !usable(*weight) {
                return Err(Error::InvalidFusion {
                    kind: Some(kind.clone()),
                    value: *weight,
                });
            }
        }
        Ok(())
    }
}

fn usable(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

/// Fuses ranked lists, best first in each and each with its weight, into one list of at most
/// `limit` items with their reciprocal rank fusion scores, best first.
///
/// An item scores the sum over the lists holding it of `weight / (k + rank)`, ranks counting
/// from 1. A repeat within one list counts only where it first appears. Equal scores are
/// ordered by the items themselves, ascending, so the order never depends on the input order.
///
/// `covers(list, item)` says whether the list at that position could have held the item. A
/// list that could not, such as vectors for a passage of a shard that has none, says nothing
/// about the item, so the item's score is scaled up by the weight of all lists over the weight
/// of those that cover it: it scores as if the lists that could not see it agreed with those
/// that could. A list that covers an item and leaves it out adds nothing.
pub(crate) fn fuse<T: Ord + Clone>(
    lists: &[(f64, Vec<T>)],
    k: f64,
    limit: usize,
    covers: impl Fn(usize, &T) -> bool,
) -> Vec<(T, f32)> {
    let mut scores: BTreeMap<&T, f64> = BTreeMap::new();
    let mut held: Vec<BTreeSet<&T>> = Vec::with_capacity(lists.len());
    for (weight, list) in lists {
        let mut seen = BTreeSet::new();
        let distinct = list.iter().filter(|item| seen.insert(*item));
        for (index, item) in distinct.enumerate() {
            *scores.entry(item).or_default() += weight / (k + rank(index));
        }
        held.push(seen);
    }
    let all: f64 = lists.iter().map(|(weight, _)| weight).sum();
    for (item, score) in &mut scores {
        let covering: f64 = lists
            .iter()
            .zip(&held)
            .enumerate()
            .filter(|(position, (_, seen))| seen.contains(*item) || covers(*position, item))
            .map(|(_, ((weight, _), _))| weight)
            .sum();
        if covering > 0.0 && covering < all {
            *score *= all / covering;
        }
    }
    let mut fused: Vec<(&T, f64)> = scores.into_iter().collect();
    fused.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    fused.truncate(limit);
    fused
        .into_iter()
        .map(|(item, score)| (item.clone(), narrow(score)))
        .collect()
}

fn rank(index: usize) -> f64 {
    u32::try_from(index)
        .ok()
        .and_then(|index| index.checked_add(1))
        .map_or(f64::from(u32::MAX), f64::from)
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "a fused score lies between 0 and the number of lists, where f32 keeps ranks apart"
)]
fn narrow(score: f64) -> f32 {
    score as f32
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    fn rrf(ranks: &[u32]) -> f32 {
        narrow(
            ranks
                .iter()
                .map(|rank| 1.0 / (60.0 + f64::from(*rank)))
                .sum(),
        )
    }

    fn fuse_default<T: Ord + Clone>(lists: &[Vec<T>], limit: usize) -> Vec<(T, f32)> {
        let fusion = Fusion::default();
        let kind = IndexKind::new("any").unwrap();
        let weighted: Vec<(f64, Vec<T>)> = lists
            .iter()
            .map(|list| (fusion.weight(&kind), list.clone()))
            .collect();
        fuse(&weighted, fusion.k(), limit, |_, _| true)
    }

    fn check(lists: &[&[&str]], limit: usize, expected: &[(&str, f32)]) {
        let lists: Vec<Vec<&str>> = lists.iter().map(|list| list.to_vec()).collect();
        let fused = fuse_default(&lists, limit);
        let items: Vec<&str> = fused.iter().map(|(item, _)| *item).collect();
        let wanted: Vec<&str> = expected.iter().map(|(item, _)| *item).collect();
        assert_eq!(items, wanted, "order");
        for ((item, score), (_, want)) in fused.iter().zip(expected) {
            assert!((score - want).abs() < 1e-7, "{item}: {score} != {want}");
        }
    }

    #[test]
    fn a_single_list_keeps_its_order() {
        check(
            &[&["c", "a", "b"]],
            10,
            &[("c", rrf(&[1])), ("a", rrf(&[2])), ("b", rrf(&[3]))],
        );
    }

    #[test]
    fn agreement_outranks_one_list_at_similar_ranks() {
        check(
            &[&["a", "b", "c"], &["d", "b", "e"]],
            10,
            &[
                ("b", rrf(&[2, 2])),
                ("a", rrf(&[1])),
                ("d", rrf(&[1])),
                ("c", rrf(&[3])),
                ("e", rrf(&[3])),
            ],
        );
    }

    #[test]
    fn one_passage_from_two_lists_is_one_hit() {
        check(
            &[&["x", "y"], &["y", "x"]],
            10,
            &[("x", rrf(&[1, 2])), ("y", rrf(&[1, 2]))],
        );
    }

    #[test]
    fn a_repeat_within_a_list_counts_once() {
        check(
            &[&["a", "a", "b"]],
            10,
            &[("a", rrf(&[1])), ("b", rrf(&[2]))],
        );
    }

    #[test]
    fn ties_are_ordered_by_item() {
        check(
            &[&["b", "a"], &["a", "b"]],
            10,
            &[("a", rrf(&[1, 2])), ("b", rrf(&[1, 2]))],
        );
        check(
            &[&["z"], &["m"], &["a"]],
            10,
            &[("a", rrf(&[1])), ("m", rrf(&[1])), ("z", rrf(&[1]))],
        );
    }

    #[test]
    fn a_list_that_cannot_hold_an_item_does_not_count_against_it() {
        let lists = vec![(1.0, vec!["far", "near"]), (1.0, vec!["near", "other"])];
        let second_covers_all_but_far = |list: usize, item: &&str| list == 0 || *item != "far";
        let fused = fuse(&lists, 60.0, 10, second_covers_all_but_far);
        let expected = [
            ("far", narrow(2.0 / 61.0)),
            ("near", rrf(&[2, 1])),
            ("other", rrf(&[2])),
        ];
        assert_eq!(fused, expected);

        let covered = fuse(&lists, 60.0, 10, |_, _| true);
        assert_eq!(covered[0], ("near", rrf(&[2, 1])));
        assert_eq!(covered[1], ("far", rrf(&[1])));
    }

    #[test]
    fn coverage_scales_by_the_weights_of_the_lists() {
        let lists = vec![(1.0, vec!["a"]), (3.0, vec!["b"])];
        let fused = fuse(&lists, 60.0, 10, |list, item: &&str| {
            list == 0 || *item == "b"
        });
        assert_eq!(fused[0], ("a", narrow(4.0 / 61.0)));
        assert_eq!(fused[1], ("b", narrow(3.0 / 61.0)));
    }

    #[test]
    fn an_empty_list_changes_nothing() {
        let expected = [("a", rrf(&[1])), ("b", rrf(&[2]))];
        check(&[&["a", "b"]], 10, &expected);
        check(&[&["a", "b"], &[]], 10, &expected);
        check(&[&[], &["a", "b"]], 10, &expected);
        check(&[], 10, &[]);
    }

    #[test]
    fn the_result_is_cut_to_the_limit() {
        check(
            &[&["a", "b", "c"], &["c"]],
            2,
            &[("c", rrf(&[3, 1])), ("a", rrf(&[1]))],
        );
        check(&[&["a", "b"]], 0, &[]);
    }

    #[test]
    fn the_defaults_are_k_60_and_weight_1_for_every_kind() {
        let fusion = Fusion::default();
        fusion.check().unwrap();
        assert!((fusion.k() - 60.0).abs() < f64::EPSILON);
        for name in ["bm25", "ivf-model"] {
            let weight = fusion.weight(&IndexKind::new(name).unwrap());
            assert!((weight - 1.0).abs() < f64::EPSILON, "{name}");
        }
        check(
            &[&["a", "b"], &["b"]],
            10,
            &[
                ("b", narrow(1.0 / 62.0 + 1.0 / 61.0)),
                ("a", narrow(1.0 / 61.0)),
            ],
        );
    }

    #[test]
    fn a_weight_scales_its_list_and_can_reorder() {
        let bm25 = IndexKind::new("bm25").unwrap();
        let ivf = IndexKind::new("ivf-model").unwrap();
        let lists = |fusion: &Fusion| {
            vec![
                (fusion.weight(&bm25), vec!["a", "b"]),
                (fusion.weight(&ivf), vec!["b", "a"]),
            ]
        };
        let mut fusion = Fusion::default();
        let even = fuse(&lists(&fusion), fusion.k(), 10, |_, _| true);
        assert_eq!(even[0].0, "a", "a tie is ordered by item");

        fusion.set_weight(ivf.clone(), 2.0);
        fusion.set_k(20.0);
        fusion.check().unwrap();
        let fused = fuse(&lists(&fusion), fusion.k(), 10, |_, _| true);
        let items: Vec<&str> = fused.iter().map(|(item, _)| *item).collect();
        assert_eq!(items, ["b", "a"]);
        let b = narrow(1.0 / 22.0 + 2.0 / 21.0);
        let a = narrow(1.0 / 21.0 + 2.0 / 22.0);
        assert!((fused[0].1 - b).abs() < 1e-7);
        assert!((fused[1].1 - a).abs() < 1e-7);
    }

    #[test]
    fn a_k_or_weight_that_is_not_finite_and_positive_is_refused() {
        let kind = IndexKind::new("bm25").unwrap();
        for bad in [0.0, -0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut fusion = Fusion::default();
            fusion.set_k(bad);
            let err = fusion.check().unwrap_err();
            assert!(
                matches!(err, Error::InvalidFusion { kind: None, .. }),
                "k {bad}: {err:?}"
            );

            let mut fusion = Fusion::default();
            fusion.set_weight(kind.clone(), bad);
            let err = fusion.check().unwrap_err();
            assert!(
                matches!(&err, Error::InvalidFusion { kind: Some(found), .. } if *found == kind),
                "weight {bad}: {err:?}"
            );
        }
        let mut fusion = Fusion::default();
        fusion.set_k(f64::MIN_POSITIVE);
        fusion.set_weight(kind, 0.5);
        fusion.check().unwrap();
    }

    fn lists() -> impl Strategy<Value = Vec<Vec<u8>>> {
        prop::collection::vec(prop::collection::vec(0u8..20, 0..15), 0..4)
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn the_fused_order_is_a_permutation_of_the_union(lists in lists()) {
            let fused = fuse_default(&lists, usize::MAX);
            let items: BTreeSet<u8> = fused.iter().map(|(item, _)| *item).collect();
            let union: BTreeSet<u8> = lists.iter().flatten().copied().collect();
            prop_assert_eq!(items.len(), fused.len());
            prop_assert_eq!(items, union);
            prop_assert!(fused.windows(2).all(|pair| pair[0].1 >= pair[1].1));
        }

        #[test]
        fn adding_a_list_never_lowers_a_score(
            lists in lists(),
            extra in prop::collection::vec(0u8..20, 0..15),
        ) {
            let before: BTreeMap<u8, f32> = fuse_default(&lists, usize::MAX).into_iter().collect();
            let mut more = lists;
            more.push(extra);
            let after: BTreeMap<u8, f32> = fuse_default(&more, usize::MAX).into_iter().collect();
            for (item, score) in before {
                prop_assert!(after[&item] >= score, "{item}");
            }
        }

        #[test]
        fn a_cut_is_a_prefix_of_the_whole(lists in lists(), limit in 0usize..25) {
            let whole = fuse_default(&lists, usize::MAX);
            let cut = fuse_default(&lists, limit);
            prop_assert_eq!(cut.as_slice(), &whole[..cut.len()]);
            prop_assert_eq!(cut.len(), limit.min(whole.len()));
        }
    }
}
