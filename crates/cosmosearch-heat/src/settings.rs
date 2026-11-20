use std::time::Duration;

use crate::clock::SharedClock;
use crate::{Clock, Error, HalfLife, Result, SystemClock};

const WEEK: Duration = Duration::from_hours(7 * 24);

/// An hour of snapshots at one flush a minute, longer than merges are expected to lag.
const HOUR_OF_FLUSHES: usize = 60;

/// How heat decays, how long a writer's snapshots wait to be merged, how many a writer keeps
/// in memory, and the clock.
///
/// Every process of an index should use the same half-life and retention.
#[derive(Debug, Clone)]
pub struct HeatSettings {
    pub(crate) half_life: HalfLife,
    pub(crate) writer_retention: Duration,
    pub(crate) kept_snapshots: usize,
    pub(crate) clock: SharedClock,
}

impl Default for HeatSettings {
    fn default() -> Self {
        HeatSettings {
            half_life: HalfLife::default(),
            writer_retention: WEEK,
            kept_snapshots: HOUR_OF_FLUSHES,
            clock: SharedClock::new(SystemClock::new()),
        }
    }
}

impl HeatSettings {
    /// A half-life of a week, a writer retention of a week, 60 kept snapshots and the system
    /// clock.
    pub fn new() -> Self {
        HeatSettings::default()
    }

    /// Sets the half-life, a week unless set.
    #[must_use]
    pub fn half_life(mut self, half_life: HalfLife) -> Self {
        self.half_life = half_life;
        self
    }

    /// Sets the writer retention, a week unless set. A writer takes a new id when its id is
    /// half this old, and a merge drops writers older than this, so merges must run more
    /// often than every half retention or the snapshots of a writer that stopped are lost.
    #[must_use]
    pub fn writer_retention(mut self, retention: Duration) -> Self {
        self.writer_retention = retention;
        self
    }

    /// Sets how many written snapshots a writer keeps in memory for
    /// [`HeatWriter::unmerged`] until a merge holds them, 60 unless set. Beyond that it drops
    /// the oldest, so its unmerged heat falls short of what it wrote until a merge takes them;
    /// it never counts anything twice.
    ///
    /// [`HeatWriter::unmerged`]: crate::HeatWriter::unmerged
    #[must_use]
    pub fn kept_snapshots(mut self, count: usize) -> Self {
        self.kept_snapshots = count;
        self
    }

    /// Sets the clock heat takes the time from, the system clock unless set.
    #[must_use]
    pub fn clock(mut self, clock: impl Clock + 'static) -> Self {
        self.clock = SharedClock::new(clock);
        self
    }

    pub(crate) fn check(&self) -> Result<()> {
        if self.writer_retention < Duration::from_millis(2) {
            return Err(Error::InvalidSettings {
                reason: "the writer retention must be at least two milliseconds",
            });
        }
        Ok(())
    }

    pub(crate) fn retention_ms(&self) -> u64 {
        u64::try_from(self.writer_retention.as_millis()).unwrap_or(u64::MAX)
    }

    pub(crate) fn now_ms(&self) -> u64 {
        self.clock.now_ms()
    }
}
