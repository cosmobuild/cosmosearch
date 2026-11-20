//! Heat for cosmosearch: how much each shard of an index is used, as decaying counts.
//!
//! Each count halves every half-life, a week unless set, and is kept as a value and the time
//! it refers to, so counts recorded at any time by any process add exactly. A query process
//! takes a [`HeatWriter`] from the index's [`HeatStore`] and hands its [`HeatRecorder`] to
//! its readers, which record in memory as they search and read. The host calls
//! [`HeatWriter::flush`] now and then to write what was recorded as a snapshot, and runs
//! [`HeatStore::merge`] and [`HeatStore::sweep`] the way it runs indexing jobs. Merges fold
//! every writer's snapshots into a chain of merged versions, each counted exactly once
//! however many merges run, and [`HeatStore::merged`] reads the newest.
//!
//! A [`ThresholdPolicy`] turns heat into decisions: it promotes a shard whose heat for each
//! mebibyte of cost reaches an upper threshold and demotes one that falls below a lower one.
//! [`SpendPlan::fill`] asks a policy about each shard and takes the promotions worth most for
//! their cost within a spend limit.
//!
//! ```
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use std::time::SystemTime;
//!
//! use cosmosearch_core::{FileSource, HeatSink, Signal};
//! use cosmosearch_heat::{HeatSettings, HeatStore, MergeOutcome};
//! use cosmosearch_store::{CreateOutcome, MemoryStore, MetadataStore};
//!
//! let metadata = MetadataStore::new(MemoryStore::new(), "s3://example-bucket/index".parse()?)?;
//! let CreateOutcome::Created(index) =
//!     metadata.create(&"s3://example-bucket/docs/".parse()?, FileSource::Listing).await?
//! else {
//!     return Ok(());
//! };
//! let manifest = index.head().await?.ok_or("the index has no manifest")?;
//! let shard = manifest.shards().keys().next().ok_or("the index has no shard")?;
//!
//! let heat = HeatStore::new(&index, HeatSettings::new())?;
//! let mut writer = heat.writer()?;
//! writer.recorder().record(Signal::Read, &[shard]);
//! writer.flush().await?;
//! assert_eq!(heat.merge().await?, MergeOutcome::Committed(1));
//!
//! let merged = heat.merged().await?;
//! let reads = merged.heat().shard(shard, SystemTime::now()).get(Signal::Read);
//! assert!(reads > 0.99 && reads <= 1.0);
//! # Ok(())
//! # }
//! ```

#![warn(clippy::arithmetic_side_effects)]

mod clock;
mod codec;
mod error;
#[cfg(any(test, feature = "test-util"))]
mod fake;
mod half_life;
mod heat;
#[cfg(test)]
mod interleave;
mod merge;
mod merged;
mod policy;
mod policy_settings;
mod recorder;
mod settings;
mod snapshot;
mod spend;
mod store;
mod sweep;
mod table;
mod writer;
mod writer_id;

pub use clock::{Clock, SystemClock};
pub use error::{Error, Result};
#[cfg(any(test, feature = "test-util"))]
pub use fake::FakeClock;
pub use half_life::HalfLife;
pub use heat::{Heat, ShardHeat};
pub use merge::MergeOutcome;
pub use merged::MergedHeat;
pub use policy::ThresholdPolicy;
pub use policy_settings::PolicySettings;
pub use recorder::HeatRecorder;
pub use settings::HeatSettings;
pub use snapshot::Snapshot;
pub use spend::{PlannedDemotion, PlannedPromotion, SpendPlan};
pub use store::HeatStore;
pub use sweep::HeatSweep;
pub use writer::HeatWriter;
pub use writer_id::WriterId;
