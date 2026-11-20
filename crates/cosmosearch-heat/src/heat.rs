use std::collections::BTreeMap;
use std::time::SystemTime;

use cosmosearch_core::{ShardId, Signal};

use crate::HalfLife;
use crate::clock::{system_time, unix_ms};
use crate::table::{Counts, Table, slot};

/// The heat of each shard: for each signal, a count that halves every half-life.
#[derive(Debug, Clone, PartialEq)]
pub struct Heat {
    table: Table,
    half_life: HalfLife,
}

impl Heat {
    /// No heat at all.
    pub fn new(half_life: HalfLife) -> Self {
        Heat::from_table(Table::default(), half_life)
    }

    pub(crate) fn from_table(table: Table, half_life: HalfLife) -> Self {
        Heat { table, half_life }
    }

    pub(crate) fn into_table(self) -> Table {
        self.table
    }

    /// The half-life its counts decay with.
    pub fn half_life(&self) -> HalfLife {
        self.half_life
    }

    /// The time its stored counts refer to.
    pub fn as_of(&self) -> SystemTime {
        system_time(self.table.as_of())
    }

    /// How many shards have heat.
    pub fn len(&self) -> usize {
        self.table.len()
    }

    /// Whether no shard has heat.
    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    /// The heat of `shard` at time `at`, zero for a shard without heat.
    pub fn shard(&self, shard: &ShardId, at: SystemTime) -> ShardHeat {
        ShardHeat(self.table.counts_at(shard, unix_ms(at), self.half_life))
    }

    /// The heat of every shard that has any, at time `at`.
    pub fn shards(&self, at: SystemTime) -> BTreeMap<ShardId, ShardHeat> {
        let factor = self.half_life.factor(self.table.as_of(), unix_ms(at));
        self.table
            .shards()
            .iter()
            .map(|(shard, counts)| (shard.clone(), ShardHeat(counts.map(|c| c * factor))))
            .collect()
    }

    /// Adds `other`, bringing both to the later of their times, with this heat's half-life.
    pub fn add(&mut self, other: &Heat) {
        self.table.add(&other.table, self.half_life);
    }
}

/// The heat of one shard at one time: a decayed count per signal.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ShardHeat(Counts);

impl ShardHeat {
    /// The decayed count of `signal`.
    pub fn get(&self, signal: Signal) -> f64 {
        slot(signal)
            .and_then(|slot| self.0.get(slot))
            .copied()
            .unwrap_or(0.0)
    }
}
