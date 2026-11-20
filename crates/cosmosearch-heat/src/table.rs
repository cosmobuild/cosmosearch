use std::collections::BTreeMap;

use cosmosearch_core::{ShardId, Signal};

use crate::HalfLife;

pub(crate) const SIGNALS: usize = Signal::ALL.len();

/// One decaying count per signal, all referring to the same time.
pub(crate) type Counts = [f64; SIGNALS];

/// The largest count kept: far beyond any heat real use records, with headroom below
/// `f64::MAX` so that sums stay finite and every stored count decodes.
pub(crate) const MAX_COUNT: f64 = 1e300;

pub(crate) fn slot(signal: Signal) -> Option<usize> {
    Signal::ALL.iter().position(|known| *known == signal)
}

/// Decaying counts per shard, all as of one time. A count `c` as of `t` is worth
/// `c * 2^-((u - t) / half-life)` at time `u`, so counts from any time add exactly once
/// brought to the same time.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Table {
    as_of: u64,
    shards: BTreeMap<ShardId, Counts>,
}

impl Table {
    pub(crate) fn from_parts(as_of: u64, shards: BTreeMap<ShardId, Counts>) -> Self {
        Table { as_of, shards }
    }

    pub(crate) fn as_of(&self) -> u64 {
        self.as_of
    }

    pub(crate) fn shards(&self) -> &BTreeMap<ShardId, Counts> {
        &self.shards
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.shards.is_empty()
    }

    pub(crate) fn len(&self) -> usize {
        self.shards.len()
    }

    /// Brings every count to `to`.
    pub(crate) fn decay_to(&mut self, to: u64, half_life: HalfLife) {
        if to == self.as_of {
            return;
        }
        let factor = half_life.factor(self.as_of, to);
        for counts in self.shards.values_mut() {
            for count in counts.iter_mut() {
                *count *= factor;
            }
        }
        self.as_of = to;
    }

    /// Adds `amount` of `slot` to `shard` at time `at`. The table stays at its own time unless
    /// `at` is more than a half-life past it, so recording costs the same however many shards
    /// the table holds.
    pub(crate) fn add_at(
        &mut self,
        shard: &ShardId,
        slot: usize,
        amount: f64,
        at: u64,
        half_life: HalfLife,
    ) {
        if self.shards.is_empty() {
            self.as_of = at;
        } else if at > self.as_of.saturating_add(half_life.millis()) {
            self.decay_to(at, half_life);
        }
        let worth = amount * half_life.factor(at, self.as_of);
        let counts = match self.shards.get_mut(shard) {
            Some(counts) => counts,
            None => self.shards.entry(shard.clone()).or_default(),
        };
        if let Some(count) = counts.get_mut(slot) {
            *count = saturating_add(*count, worth);
        }
    }

    /// Adds every count of `other`, bringing both to the later of their times.
    pub(crate) fn add(&mut self, other: &Table, half_life: HalfLife) {
        if other.is_empty() {
            return;
        }
        let to = if self.is_empty() {
            other.as_of
        } else {
            self.as_of.max(other.as_of)
        };
        self.decay_to(to, half_life);
        let factor = half_life.factor(other.as_of, to);
        for (shard, counts) in &other.shards {
            let mine = match self.shards.get_mut(shard) {
                Some(mine) => mine,
                None => self.shards.entry(shard.clone()).or_default(),
            };
            for (count, theirs) in mine.iter_mut().zip(counts) {
                *count = saturating_add(*count, theirs * factor);
            }
        }
    }

    /// Keeps only the shards `keep` accepts, returning how many were dropped.
    pub(crate) fn retain(&mut self, mut keep: impl FnMut(&ShardId) -> bool) -> usize {
        let before = self.shards.len();
        self.shards.retain(|shard, _| keep(shard));
        before.saturating_sub(self.shards.len())
    }

    /// The counts of `shard` brought to `at`.
    pub(crate) fn counts_at(&self, shard: &ShardId, at: u64, half_life: HalfLife) -> Counts {
        self.shards.get(shard).map_or([0.0; SIGNALS], |counts| {
            let factor = half_life.factor(self.as_of, at);
            counts.map(|count| count * factor)
        })
    }
}

fn saturating_add(count: f64, worth: f64) -> f64 {
    (count + worth).min(MAX_COUNT)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::codec;

    const HOUR: u64 = 3_600_000;
    const START: u64 = 1_800_000_000_000;

    fn shard(name: &str) -> ShardId {
        ShardId::new(name).unwrap()
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1.0)
    }

    fn hours(count: u64) -> HalfLife {
        HalfLife::new(Duration::from_hours(count)).unwrap()
    }

    #[test]
    fn a_count_halves_each_half_life() {
        let half_life = hours(1);
        let mut table = Table::default();
        table.add_at(&shard("a"), 0, 8.0, START, half_life);
        let a = shard("a");
        assert!(close(table.counts_at(&a, START + HOUR, half_life)[0], 4.0));
        assert!(close(
            table.counts_at(&a, START + 3 * HOUR, half_life)[0],
            1.0
        ));
        table.decay_to(START + 2 * HOUR, half_life);
        assert!(close(
            table.counts_at(&a, START + 2 * HOUR, half_life)[0],
            2.0
        ));
        assert!(close(
            table.counts_at(&shard("b"), START, half_life)[1],
            0.0
        ));
    }

    #[test]
    fn counts_from_different_times_add_exactly() {
        let half_life = hours(24);
        let a = shard("a");
        let mut early = Table::default();
        early.add_at(&a, 1, 3.0, START, half_life);
        let mut late = Table::default();
        late.add_at(&a, 1, 5.0, START + 30 * HOUR, half_life);
        late.add_at(&shard("b"), 2, 1.0, START + 30 * HOUR, half_life);

        let mut forward = early.clone();
        forward.add(&late, half_life);
        let mut backward = late.clone();
        backward.add(&early, half_life);

        let at = START + 50 * HOUR;
        let want = 3.0 * 2f64.powf(-50.0 / 24.0) + 5.0 * 2f64.powf(-20.0 / 24.0);
        for table in [&forward, &backward] {
            assert_eq!(table.as_of(), START + 30 * HOUR);
            assert!(close(table.counts_at(&a, at, half_life)[1], want));
            assert!(close(
                table.counts_at(&shard("b"), at, half_life)[2],
                2f64.powf(-20.0 / 24.0)
            ));
        }
    }

    #[test]
    fn events_recorded_out_of_order_or_far_apart_add_up_the_same() {
        let half_life = hours(10);
        let a = shard("a");
        let times = [START + 5 * HOUR, START, START + 40 * HOUR, START + HOUR];
        let mut table = Table::default();
        for time in times {
            table.add_at(&a, 3, 1.0, time, half_life);
        }
        let at = START + 60 * HOUR;
        let want: f64 = times.iter().map(|time| half_life.factor(*time, at)).sum();
        assert!(close(table.counts_at(&a, at, half_life)[3], want));
        assert_eq!(table.as_of(), START + 40 * HOUR);
    }

    #[test]
    fn a_longer_half_life_decays_more_slowly() {
        let a = shard("a");
        let at = START + 24 * HOUR;
        let worth = |half_life: HalfLife| {
            let mut table = Table::default();
            table.add_at(&a, 0, 1.0, START, half_life);
            table.counts_at(&a, at, half_life)[0]
        };
        assert!(close(worth(hours(24)), 0.5));
        assert!(close(worth(hours(168)), 2f64.powf(-1.0 / 7.0)));
        assert!(close(worth(hours(6)), 1.0 / 16.0));
    }

    #[test]
    fn huge_amounts_saturate_and_still_encode() {
        let half_life = hours(1);
        let a = shard("a");
        let mut snapshot = Table::default();
        snapshot.add_at(&a, 0, f64::MAX, START, half_life);
        snapshot.add_at(&a, 0, f64::MAX, START + HOUR / 2, half_life);
        let mut merged = snapshot.clone();
        merged.add(&snapshot, half_life);
        for table in [&snapshot, &merged] {
            assert_eq!(table.counts_at(&a, table.as_of(), half_life)[0], MAX_COUNT);
            let decoded = codec::decode_snapshot(&codec::encode_snapshot(table)).unwrap();
            assert_eq!(&decoded, table);
        }
    }

    #[test]
    fn retain_reports_what_it_dropped() {
        let half_life = hours(1);
        let mut table = Table::default();
        for name in ["a", "b", "c"] {
            table.add_at(&shard(name), 0, 1.0, START, half_life);
        }
        assert_eq!(table.retain(|id| id.as_str() != "b"), 1);
        assert_eq!(table.len(), 2);
    }
}
