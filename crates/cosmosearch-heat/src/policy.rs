use std::collections::BTreeMap;
use std::time::SystemTime;

use cosmosearch_core::{Action, HeatPolicy, Level, Manifest, ShardId, Signal};

use crate::table::MAX_COUNT;
use crate::{Heat, PolicySettings, Result};

const BYTES_PER_MEBIBYTE: f64 = 1_048_576.0;

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ThresholdPolicy>();
};

/// Promotes and demotes shards by their heat for each mebibyte of cost, with hysteresis.
///
/// A shard goes to L2 when that ratio reaches an upper threshold and back to L1 when it falls
/// below a lower one. Between the two it stays where it is, so a shard whose heat hovers does
/// not move up and down. A shard's heat here is the weighted sum of its signals at one
/// moment, plus a share of the same sum of the shards whose key ranges lie directly before
/// and after it. The policy holds one number per shard of the manifest it was made from and
/// reads nothing.
#[derive(Debug, Clone, PartialEq)]
pub struct ThresholdPolicy {
    values: BTreeMap<ShardId, f64>,
    promote_at: f64,
    demote_below: f64,
}

impl ThresholdPolicy {
    /// The policy over `heat` as it is at `at`, for the shards of `manifest`. A time before
    /// the heat's own counts as that time.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSettings`](crate::Error::InvalidSettings) for a weight, share
    /// or threshold that is negative or not finite, or a promotion threshold not above the
    /// demotion threshold.
    pub fn new(
        heat: &Heat,
        at: SystemTime,
        manifest: &Manifest,
        settings: &PolicySettings,
    ) -> Result<Self> {
        let mut ordered: Vec<(&str, &ShardId)> = manifest
            .shards()
            .iter()
            .map(|(id, shard)| (shard.range().start(), id))
            .collect();
        ordered.sort_unstable();
        let ordered: Vec<&ShardId> = ordered.into_iter().map(|(_, id)| id).collect();
        ThresholdPolicy::over(heat, at, &ordered, settings)
    }

    /// The policy for `shards`, given in the order of their key ranges.
    fn over(
        heat: &Heat,
        at: SystemTime,
        shards: &[&ShardId],
        settings: &PolicySettings,
    ) -> Result<Self> {
        settings.check()?;
        let at = at.max(heat.as_of());
        let own: Vec<f64> = shards
            .iter()
            .map(|shard| {
                let counts = heat.shard(shard, at);
                Signal::ALL
                    .iter()
                    .zip(settings.weights)
                    .map(|(signal, weight)| counts.get(*signal) * weight)
                    .sum::<f64>()
                    .min(MAX_COUNT)
            })
            .collect();
        let beside = |index: Option<usize>| index.and_then(|i| own.get(i)).copied().unwrap_or(0.0);
        let values = shards
            .iter()
            .zip(&own)
            .enumerate()
            .map(|(index, (shard, own))| {
                let neighbours = beside(index.checked_sub(1)) + beside(index.checked_add(1));
                let value = (own + settings.neighbour_share * neighbours).min(MAX_COUNT);
                ((*shard).clone(), value)
            })
            .collect();
        Ok(ThresholdPolicy {
            values,
            promote_at: settings.promote_at,
            demote_below: settings.demote_below,
        })
    }

    /// The heat `shard` is valued at, zero for a shard the manifest does not hold.
    pub fn value(&self, shard: &ShardId) -> f64 {
        self.values.get(shard).copied().unwrap_or(0.0)
    }
}

impl HeatPolicy for ThresholdPolicy {
    fn decide(&self, shard: &ShardId, level: Level, cost: u64) -> Action {
        let value = self.value(shard);
        let per_mebibyte = value / mebibytes(cost);
        if level < Level::L2 && per_mebibyte >= self.promote_at {
            Action::Promote {
                level: Level::L2,
                value,
            }
        } else if level >= Level::L2 && per_mebibyte < self.demote_below {
            Action::Demote { level: Level::L1 }
        } else {
            Action::Keep
        }
    }
}

/// `cost` bytes in mebibytes, at least one byte so that a shard with nothing to build still
/// has a ratio.
#[expect(
    clippy::cast_precision_loss,
    reason = "a cost above 2^53 bytes loses bytes that cannot change a decision"
)]
fn mebibytes(cost: u64) -> f64 {
    cost.max(1) as f64 / BYTES_PER_MEBIBYTE
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use proptest::prelude::*;

    use super::*;
    use crate::HalfLife;
    use crate::clock::system_time;
    use crate::table::{Table, slot};

    const START: u64 = 1_800_000_000_000;
    const MEBIBYTE: u64 = 1 << 20;
    const HOUR_MS: u64 = 3_600_000;
    const DAY: Duration = Duration::from_hours(24);

    fn shard(name: &str) -> ShardId {
        ShardId::new(name).unwrap()
    }

    fn heat(counts: &[(&str, Signal, f64)]) -> Heat {
        let half_life = HalfLife::new(DAY).unwrap();
        let mut table = Table::default();
        for (name, signal, amount) in counts {
            table.add_at(
                &shard(name),
                slot(*signal).unwrap(),
                *amount,
                START,
                half_life,
            );
        }
        Heat::from_table(table, half_life)
    }

    fn policy(
        heat: &Heat,
        at: SystemTime,
        shards: &[&str],
        settings: &PolicySettings,
    ) -> ThresholdPolicy {
        let ids: Vec<ShardId> = shards.iter().map(|name| shard(name)).collect();
        let ids: Vec<&ShardId> = ids.iter().collect();
        ThresholdPolicy::over(heat, at, &ids, settings).unwrap()
    }

    fn settings() -> PolicySettings {
        PolicySettings::new()
            .neighbour_share(0.0)
            .thresholds(1.0, 0.25)
    }

    fn apply(level: Level, action: Action) -> Level {
        match action {
            Action::Promote { level, .. } | Action::Demote { level } => level,
            Action::Keep | _ => level,
        }
    }

    #[test]
    fn signals_are_weighted_and_neighbours_share_their_heat() {
        let heat = heat(&[
            ("a", Signal::Search, 2.0),
            ("a", Signal::Read, 1.0),
            ("b", Signal::Unmet, 2.0),
            ("d", Signal::Hit, 4.0),
        ]);
        let policy = policy(
            &heat,
            system_time(START),
            &["a", "b", "c", "d"],
            &PolicySettings::new().neighbour_share(0.5),
        );
        let close = |a: f64, b: f64| (a - b).abs() < 1e-9;
        assert!(close(policy.value(&shard("a")), 5.0 + 0.5 * 10.0));
        assert!(close(policy.value(&shard("b")), 10.0 + 0.5 * 5.0));
        assert!(close(policy.value(&shard("c")), 0.5 * 14.0));
        assert!(close(policy.value(&shard("d")), 4.0));
        assert!(close(policy.value(&shard("unknown")), 0.0));
    }

    #[test]
    fn promotes_above_the_upper_threshold_and_demotes_below_the_lower() {
        let heat = heat(&[("a", Signal::Search, 8.0)]);
        let policy = policy(&heat, system_time(START), &["a"], &settings());
        let a = shard("a");
        let promote = Action::Promote {
            level: Level::L2,
            value: 8.0,
        };
        assert_eq!(policy.decide(&a, Level::L1, 8 * MEBIBYTE), promote);
        assert_eq!(policy.decide(&a, Level::L1, 9 * MEBIBYTE), Action::Keep);
        assert_eq!(policy.decide(&a, Level::L2, 9 * MEBIBYTE), Action::Keep);
        assert_eq!(policy.decide(&a, Level::L2, 32 * MEBIBYTE), Action::Keep);
        assert_eq!(
            policy.decide(&a, Level::L2, 33 * MEBIBYTE),
            Action::Demote { level: Level::L1 }
        );
        assert_eq!(policy.decide(&a, Level::L1, 33 * MEBIBYTE), Action::Keep);
        assert_eq!(policy.decide(&a, Level::L1, 0), promote);
        assert_eq!(policy.decide(&shard("cold"), Level::L1, 0), Action::Keep);
    }

    #[test]
    fn a_time_before_the_heat_does_not_inflate_it() {
        let heat = heat(&[("a", Signal::Search, 8.0)]);
        let early = system_time(START - 30 * 24 * HOUR_MS);
        let policy = policy(&heat, early, &["a"], &settings());
        assert!((policy.value(&shard("a")) - 8.0).abs() < 1e-9);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        #[test]
        fn heat_that_stays_the_same_moves_a_shard_at_most_once(
            amount in 0.0f64..64.0,
            cost in 0u64..64 * MEBIBYTE,
            promoted in any::<bool>(),
        ) {
            let heat = heat(&[("a", Signal::Search, amount)]);
            let policy = policy(&heat, system_time(START), &["a"], &settings());
            let a = shard("a");
            let start = if promoted { Level::L2 } else { Level::L1 };
            let settled = apply(start, policy.decide(&a, start, cost));
            prop_assert_eq!(policy.decide(&a, settled, cost), Action::Keep);
        }

        #[test]
        fn decaying_heat_promotes_once_and_demotes_once(
            amount in 0.0f64..64.0,
            cost in 1u64..64 * MEBIBYTE,
        ) {
            let heat = heat(&[("a", Signal::Search, amount)]);
            let a = shard("a");
            let mut level = Level::L1;
            let mut moves = Vec::new();
            for hours in 0..240u32 {
                let at = system_time(START.saturating_add(u64::from(hours).saturating_mul(HOUR_MS)));
                let policy = policy(&heat, at, &["a"], &settings());
                let next = apply(level, policy.decide(&a, level, cost));
                if next != level {
                    moves.push(next);
                    level = next;
                }
            }
            prop_assert!(
                moves.is_empty() || moves == [Level::L2] || moves == [Level::L2, Level::L1],
                "{:?}", moves
            );
        }
    }
}
