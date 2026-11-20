use cosmosearch_core::{Action, HeatPolicy, Level, ShardCost, ShardId};

/// A promotion a policy asked for.
#[derive(Debug, Clone, PartialEq)]
pub struct PlannedPromotion {
    shard: ShardId,
    level: Level,
    value: f64,
    cost: u64,
}

impl PlannedPromotion {
    /// The shard to promote.
    pub fn shard(&self) -> &ShardId {
        &self.shard
    }

    /// The level to build.
    pub fn level(&self) -> Level {
        self.level
    }

    /// What the policy valued the promotion at.
    pub fn value(&self) -> f64 {
        self.value
    }

    /// What building it costs.
    pub fn cost(&self) -> u64 {
        self.cost
    }

    fn value_per_cost(&self) -> f64 {
        self.value / unit(self.cost)
    }
}

/// A demotion a policy asked for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedDemotion {
    shard: ShardId,
    level: Level,
}

impl PlannedDemotion {
    /// The shard to demote.
    pub fn shard(&self) -> &ShardId {
        &self.shard
    }

    /// The level it falls back to.
    pub fn level(&self) -> Level {
        self.level
    }
}

/// What one round of indexing does about heat: the promotions that fit a spend limit, taken
/// by value for each unit of cost, those left for a later plan, and the demotions.
///
/// The limit holds within this plan only. The library has no coordinator, so processes that
/// plan at once each spend up to their own limit unless the host gives one of them the job.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SpendPlan {
    promotions: Vec<PlannedPromotion>,
    deferred: Vec<PlannedPromotion>,
    demotions: Vec<PlannedDemotion>,
    spent: u64,
}

impl SpendPlan {
    /// Asks `policy` about each of `shards` and fills `limit` with the promotions worth most
    /// for each unit of cost, best first. A promotion that does not fit what is left is
    /// deferred and cheaper ones after it are still tried, so the costs taken never add up to
    /// more than `limit`. Demotions cost nothing and are all planned.
    pub fn fill(policy: &dyn HeatPolicy, shards: &[ShardCost], limit: u64) -> Self {
        let mut wanted = Vec::new();
        let mut plan = SpendPlan::default();
        for candidate in shards {
            match policy.decide(candidate.shard(), candidate.level(), candidate.cost()) {
                Action::Promote { level, value } if value.is_finite() && value > 0.0 => {
                    wanted.push(PlannedPromotion {
                        shard: candidate.shard().clone(),
                        level,
                        value,
                        cost: candidate.cost(),
                    });
                }
                Action::Demote { level } => plan.demotions.push(PlannedDemotion {
                    shard: candidate.shard().clone(),
                    level,
                }),
                Action::Promote { .. } | Action::Keep | _ => {}
            }
        }
        wanted.sort_by(|a, b| {
            b.value_per_cost()
                .total_cmp(&a.value_per_cost())
                .then_with(|| a.cost.cmp(&b.cost))
                .then_with(|| a.shard.cmp(&b.shard))
        });
        for promotion in wanted {
            match plan.spent.checked_add(promotion.cost) {
                Some(total) if total <= limit => {
                    plan.spent = total;
                    plan.promotions.push(promotion);
                }
                Some(_) | None => plan.deferred.push(promotion),
            }
        }
        plan
    }

    /// The promotions to run, best value for each unit of cost first.
    pub fn promotions(&self) -> &[PlannedPromotion] {
        &self.promotions
    }

    /// The promotions the limit left out, in the same order, for a later plan.
    pub fn deferred(&self) -> &[PlannedPromotion] {
        &self.deferred
    }

    /// The demotions to run.
    pub fn demotions(&self) -> &[PlannedDemotion] {
        &self.demotions
    }

    /// The cost of the promotions to run together, never above the limit.
    pub fn spent(&self) -> u64 {
        self.spent
    }
}

/// `cost` as a divisor, at least one unit so that a free promotion still has a ratio.
#[expect(
    clippy::cast_precision_loss,
    reason = "a cost above 2^53 loses units that cannot change the order"
)]
fn unit(cost: u64) -> f64 {
    cost.max(1) as f64
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use proptest::prelude::*;

    use super::*;

    #[derive(Debug, Default)]
    struct Fixed(BTreeMap<ShardId, f64>);

    impl HeatPolicy for Fixed {
        fn decide(&self, shard: &ShardId, level: Level, _: u64) -> Action {
            match self.0.get(shard) {
                Some(value) if level < Level::L2 => Action::Promote {
                    level: Level::L2,
                    value: *value,
                },
                None if level >= Level::L2 => Action::Demote { level: Level::L1 },
                Some(_) | None => Action::Keep,
            }
        }
    }

    fn shard(name: &str) -> ShardId {
        ShardId::new(name).unwrap()
    }

    fn names(promotions: &[PlannedPromotion]) -> Vec<&str> {
        promotions.iter().map(|p| p.shard().as_str()).collect()
    }

    #[test]
    fn takes_the_best_value_per_cost_first_and_keeps_trying_cheaper_ones() {
        let policy = Fixed(BTreeMap::from([
            (shard("big"), 90.0),
            (shard("dense"), 30.0),
            (shard("small"), 4.0),
            (shard("worthless"), 0.0),
            (shard("warm"), 1.0),
        ]));
        let shards = [
            ShardCost::new(shard("big"), Level::L1, 60),
            ShardCost::new(shard("dense"), Level::L1, 10),
            ShardCost::new(shard("small"), Level::L1, 5),
            ShardCost::new(shard("worthless"), Level::L1, 1),
            ShardCost::new(shard("cold"), Level::L2, 40),
            ShardCost::new(shard("warm"), Level::L2, 40),
            ShardCost::new(shard("unknown"), Level::L1, 1),
        ];
        let plan = SpendPlan::fill(&policy, &shards, 20);
        assert_eq!(names(plan.promotions()), ["dense", "small"]);
        assert_eq!(names(plan.deferred()), ["big"]);
        assert_eq!(plan.spent(), 15);
        let demoted: Vec<&str> = plan
            .demotions()
            .iter()
            .map(|d| d.shard().as_str())
            .collect();
        assert_eq!(demoted, ["cold"]);

        let all = SpendPlan::fill(&policy, &shards, u64::MAX);
        assert_eq!(names(all.promotions()), ["dense", "big", "small"]);
        assert_eq!(all.deferred(), []);
        let none = SpendPlan::fill(&policy, &shards, 0);
        assert_eq!(none.promotions(), []);
        assert_eq!(none.spent(), 0);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        #[test]
        fn a_plan_never_spends_more_than_its_limit(
            candidates in prop::collection::vec((0.0f64..100.0, any::<u64>()), 0..24),
            limit in any::<u64>(),
        ) {
            let mut policy = Fixed::default();
            let mut shards = Vec::new();
            for (index, (value, cost)) in candidates.iter().enumerate() {
                let id = shard(&format!("s{index}"));
                policy.0.insert(id.clone(), *value);
                shards.push(ShardCost::new(id, Level::L1, *cost));
            }
            let plan = SpendPlan::fill(&policy, &shards, limit);
            let spent = plan
                .promotions()
                .iter()
                .try_fold(0u64, |total, promotion| total.checked_add(promotion.cost()));
            prop_assert_eq!(spent, Some(plan.spent()));
            prop_assert!(plan.spent() <= limit);
            let wanted = candidates.iter().filter(|(value, _)| *value > 0.0).count();
            prop_assert_eq!(plan.promotions().len().saturating_add(plan.deferred().len()), wanted);
            for left in plan.deferred() {
                prop_assert!(plan.spent().checked_add(left.cost()).is_none_or(|total| total > limit));
            }
        }
    }
}
