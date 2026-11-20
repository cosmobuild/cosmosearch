use std::fmt;
use std::sync::Arc;

use crate::{Level, ShardId};

/// What drew attention to a shard. Heat keeps one decaying counter per signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum Signal {
    /// A search whose scope covered the shard.
    Search,
    /// A read or expand call on a ref inside the shard.
    Read,
    /// Demand the shard did not meet: a search ended inconclusive with the shard unexplored.
    Unmet,
    /// A host's hint that the shard is about to be explored.
    Warm,
    /// A search that returned a passage inside the shard.
    Hit,
}

impl Signal {
    /// Every signal, in a fixed order.
    pub const ALL: [Signal; 5] = [
        Signal::Search,
        Signal::Read,
        Signal::Unmet,
        Signal::Warm,
        Signal::Hit,
    ];
}

/// Takes the heat signals of a process as they happen. It is called on the search path, so
/// it records in memory and never waits on storage.
pub trait HeatSink: Send + Sync + fmt::Debug {
    /// Counts `signal` once for each of `shards`.
    fn record(&self, signal: Signal, shards: &[&ShardId]);
}

/// A shared handle on a heat sink.
#[derive(Debug, Clone)]
pub struct SharedHeat(Arc<dyn HeatSink>);

impl SharedHeat {
    /// Shares `sink`.
    pub fn new(sink: impl HeatSink + 'static) -> Self {
        SharedHeat(Arc::new(sink))
    }

    /// The sink.
    pub fn get(&self) -> &dyn HeatSink {
        self.0.as_ref()
    }
}

/// One shard a heat policy is asked about: its level now and what building it at the next
/// level costs, in the unit of the plan's spend limit, such as source bytes to embed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardCost {
    shard: ShardId,
    level: Level,
    cost: u64,
}

impl ShardCost {
    /// `shard`, now at `level`, whose next build costs `cost`.
    pub fn new(shard: ShardId, level: Level, cost: u64) -> Self {
        ShardCost { shard, level, cost }
    }

    /// The shard.
    pub fn shard(&self) -> &ShardId {
        &self.shard
    }

    /// Its level now.
    pub fn level(&self) -> Level {
        self.level
    }

    /// What its next build costs.
    pub fn cost(&self) -> u64 {
        self.cost
    }
}

/// What a heat policy decides for one shard.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum Action {
    /// Build the shard at `level`.
    Promote {
        /// The level to build.
        level: Level,
        /// What the promotion is worth, in the policy's own measure. A plan with a spend
        /// limit takes promotions by their value for each unit of cost.
        value: f64,
    },
    /// Drop the shard's search structures above `level`.
    Demote {
        /// The level the shard falls back to.
        level: Level,
    },
    /// Leave the shard as it is.
    Keep,
}

/// Decides from heat which shards are worth building at a higher level, and which no longer are.
pub trait HeatPolicy: Send + Sync + fmt::Debug {
    /// What to do with `shard`, now at `level`, whose build at the next level costs `cost`
    /// in the unit of the plan's spend limit, such as source bytes to embed.
    fn decide(&self, shard: &ShardId, level: Level, cost: u64) -> Action;
}
