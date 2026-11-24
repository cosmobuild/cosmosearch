use std::time::Duration;

use cosmosearch_core::{Filter, SearchIdentity, Timestamp};
use cosmosearch_format::FileEntry;

use crate::{Error, Result, SearchHit};

/// A nudge up the results for the candidates whose file qualifies, set per search.
///
/// A candidate's multiplier is `1 + weight × how much its file qualifies`, from 0 to 1, and
/// the multipliers of a search's boosts multiply together. A candidate at rank `r` stands at
/// `r / multiplier`, so with a weight of 1 a qualifying candidate at rank 6 stands at 3,
/// above one at rank 4. Boosts place the candidates fusion found: a search with boosts fuses
/// twice as many candidates as it would without, and the boosts choose among them which reach
/// the reranker's shortlist and the results. After reranking, they place the reranked hits
/// again by their rerank ranks, so a boost means the same with a reranker or without one. A
/// boost never adds a candidate fusion did not find.
#[derive(Debug, Clone, PartialEq)]
pub struct Boost {
    rule: Rule,
    weight: f64,
}

#[derive(Debug, Clone, PartialEq)]
enum Rule {
    Matching(Filter),
    Recency { now: Timestamp, half_life: Duration },
}

impl Boost {
    /// Boosts the candidates whose file passes `filter` by `weight`: with a weight of 1, they
    /// count as ranked twice as high.
    pub fn matching(filter: Filter, weight: f64) -> Self {
        Boost {
            rule: Rule::Matching(filter),
            weight,
        }
    }

    /// Boosts the candidates by how recently their file was modified before `now`: a file
    /// modified at `now` or later gets the whole `weight`, and one modified `half_life`
    /// earlier half of it. A file whose modified time is not known gets nothing.
    ///
    /// The result cache keys on `now`, so a search given the current time to the millisecond
    /// is never answered from it. Rounding `now`, for example to the hour, lets repeated
    /// searches share a cached result.
    pub fn recency(now: Timestamp, half_life: Duration, weight: f64) -> Self {
        Boost {
            rule: Rule::Recency { now, half_life },
            weight,
        }
    }

    /// Refuses a weight that is not finite and at least zero, and a half-life of zero.
    pub(crate) fn check(&self) -> Result<()> {
        let half_life_ok = match &self.rule {
            Rule::Matching(_) => true,
            Rule::Recency { half_life, .. } => !half_life.is_zero(),
        };
        if self.weight.is_finite() && self.weight >= 0.0 && half_life_ok {
            Ok(())
        } else {
            Err(Error::InvalidBoost {
                weight: self.weight,
            })
        }
    }

    /// What a candidate's fused score is multiplied by.
    pub(crate) fn multiplier(&self, file: &FileEntry) -> f64 {
        self.weight.mul_add(self.share(file), 1.0)
    }

    fn share(&self, file: &FileEntry) -> f64 {
        match &self.rule {
            Rule::Matching(filter) => {
                if filter.matches(file.attributes(), file.modified()) {
                    1.0
                } else {
                    0.0
                }
            }
            Rule::Recency { now, half_life } => file.modified().map_or(0.0, |modified| {
                let age = now.millis().saturating_sub(modified.millis()).max(0);
                let half_lives = millis_f64(age) / half_life.as_secs_f64() / 1_000.0;
                0.5_f64.powf(half_lives)
            }),
        }
    }

    pub(crate) fn add_to(&self, identity: SearchIdentity) -> SearchIdentity {
        let identity = identity.number(self.weight.to_bits());
        match &self.rule {
            Rule::Matching(filter) => identity.number(0).filter(filter),
            Rule::Recency { now, half_life } => identity
                .number(1)
                .number(now.millis().cast_unsigned())
                .number(u64::try_from(half_life.as_nanos()).unwrap_or(u64::MAX)),
        }
    }
}

/// How many candidates a search fuses when it wants `wanted` and has `boosts`: with boosts,
/// twice as many, but no more than [`SearchRequest::MAX_LIMIT`] unless `wanted` is more.
///
/// [`SearchRequest::MAX_LIMIT`]: crate::SearchRequest::MAX_LIMIT
pub(crate) fn pool(wanted: usize, boosts: &[Boost]) -> usize {
    if boosts.is_empty() {
        return wanted;
    }
    wanted
        .saturating_mul(2)
        .min(wanted.max(crate::SearchRequest::MAX_LIMIT))
}

/// The best `wanted` of the fused `hits`, placed by their boosts when the search has any.
pub(crate) fn choose(hits: Vec<SearchHit>, boosts: &[Boost], wanted: usize) -> Vec<SearchHit> {
    if boosts.is_empty() {
        return hits;
    }
    let mut chosen = place(hits, SearchHit::boost);
    chosen.truncate(wanted);
    chosen
}

/// `items`, ranked best first, placed again by their multipliers: the item at rank `r`,
/// counting from 1, stands at `r / multiplier`, and items standing level keep their order.
pub(crate) fn place<T>(items: Vec<T>, multiplier: impl Fn(&T) -> f64) -> Vec<T> {
    let mut standing: Vec<(f64, T)> = items
        .into_iter()
        .enumerate()
        .map(|(index, item)| (rank(index) / multiplier(&item), item))
        .collect();
    standing.sort_by(|a, b| a.0.total_cmp(&b.0));
    standing.into_iter().map(|(_, item)| item).collect()
}

fn rank(index: usize) -> f64 {
    u32::try_from(index)
        .ok()
        .and_then(|index| index.checked_add(1))
        .map_or(f64::from(u32::MAX), f64::from)
}

#[expect(
    clippy::cast_precision_loss,
    reason = "an age in milliseconds loses precision only beyond 285,000 years"
)]
fn millis_f64(millis: i64) -> f64 {
    millis as f64
}

#[cfg(test)]
mod tests {
    use cosmosearch_core::{AttributeValue, Attributes, ETag, ManifestVersion};

    use super::*;

    fn file(modified: Option<i64>, owner: &str) -> FileEntry {
        FileEntry::pending(
            "docs/a.txt",
            5,
            ETag::new("e").unwrap(),
            ManifestVersion::new(3).unwrap(),
        )
        .with_modified(modified.map(Timestamp::from_millis))
        .with_attributes(Attributes::from([(
            "owner".to_owned(),
            AttributeValue::One(owner.to_owned()),
        )]))
    }

    #[test]
    fn a_matching_boost_multiplies_only_files_that_pass() {
        let boost = Boost::matching(Filter::new().equals("owner", "legal"), 1.0);
        assert!((boost.multiplier(&file(None, "legal")) - 2.0).abs() < 1e-12);
        assert!((boost.multiplier(&file(None, "sales")) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_recency_boost_halves_with_each_half_life() {
        let day = Duration::from_hours(24);
        let now = Timestamp::from_millis(10 * 86_400_000);
        let boost = Boost::recency(now, day, 2.0);
        let cases = [
            (Some(10 * 86_400_000), 3.0),
            (Some(11 * 86_400_000), 3.0),
            (Some(9 * 86_400_000), 2.0),
            (Some(8 * 86_400_000), 1.5),
            (None, 1.0),
        ];
        for (modified, expected) in cases {
            let got = boost.multiplier(&file(modified, "x"));
            assert!((got - expected).abs() < 1e-9, "{modified:?}: {got}");
        }
    }

    #[test]
    fn a_multiplier_divides_the_rank_and_level_items_keep_their_order() {
        let items = vec![
            ("a", 1.0),
            ("b", 1.0),
            ("c", 1.0),
            ("d", 1.0),
            ("e", 1.0),
            ("f", 2.0),
        ];
        let placed: Vec<&str> = place(items, |item| item.1)
            .into_iter()
            .map(|item| item.0)
            .collect();
        assert_eq!(placed, ["a", "b", "c", "f", "d", "e"]);
        let level = vec![("a", 1.0), ("b", 2.0)];
        let placed: Vec<&str> = place(level, |item| item.1)
            .into_iter()
            .map(|item| item.0)
            .collect();
        assert_eq!(placed, ["a", "b"]);
        let huge = vec![("a", 1.0), ("b", f64::INFINITY), ("c", f64::INFINITY)];
        let placed: Vec<&str> = place(huge, |item| item.1)
            .into_iter()
            .map(|item| item.0)
            .collect();
        assert_eq!(placed, ["b", "c", "a"]);
    }

    #[test]
    fn a_pool_is_twice_as_deep_with_boosts_up_to_the_largest_limit() {
        let boosts = [Boost::matching(Filter::new(), 1.0)];
        assert_eq!(pool(20, &[]), 20);
        assert_eq!(pool(0, &boosts), 0);
        assert_eq!(pool(20, &boosts), 40);
        assert_eq!(pool(600, &boosts), 1_000);
        assert_eq!(pool(1_500, &boosts), 1_500);
    }

    #[test]
    fn refuses_weights_below_zero_or_not_finite_and_a_zero_half_life() {
        let now = Timestamp::from_millis(0);
        Boost::matching(Filter::new(), 0.0).check().unwrap();
        for boost in [
            Boost::matching(Filter::new(), -0.5),
            Boost::matching(Filter::new(), f64::NAN),
            Boost::matching(Filter::new(), f64::INFINITY),
            Boost::recency(now, Duration::ZERO, 1.0),
        ] {
            assert!(
                matches!(boost.check(), Err(Error::InvalidBoost { .. })),
                "{boost:?}"
            );
        }
    }
}
