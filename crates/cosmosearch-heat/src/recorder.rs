use std::mem;
use std::sync::{Arc, Mutex, MutexGuard};

use cosmosearch_core::{HeatSink, ShardId, Signal};

use crate::clock::SharedClock;
use crate::table::{Table, slot};
use crate::{Error, HalfLife, Result};

/// Records the heat of one process in memory, to be written by its [`HeatWriter`]. Clones
/// share the same heat, so readers and the host can each hold one; recording never touches
/// the store.
///
/// [`HeatWriter`]: crate::HeatWriter
#[derive(Debug, Clone)]
pub struct HeatRecorder(Arc<Recording>);

#[derive(Debug)]
struct Recording {
    clock: SharedClock,
    half_life: HalfLife,
    pending: Mutex<Table>,
}

impl HeatRecorder {
    pub(crate) fn new(clock: SharedClock, half_life: HalfLife) -> Self {
        HeatRecorder(Arc::new(Recording {
            clock,
            half_life,
            pending: Mutex::new(Table::default()),
        }))
    }

    /// Adds `amount` of `signal` to `shard` now, for heat that is not one count per event,
    /// such as priors written at bootstrap.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidAmount`] for an amount that is negative or not finite.
    pub fn add(&self, shard: &ShardId, signal: Signal, amount: f64) -> Result<()> {
        if !amount.is_finite() || amount < 0.0 {
            return Err(Error::InvalidAmount);
        }
        self.add_each(signal, &[shard], amount);
        Ok(())
    }

    fn add_each(&self, signal: Signal, shards: &[&ShardId], amount: f64) {
        let Some(slot) = slot(signal) else {
            return;
        };
        let now = self.0.clock.now_ms();
        let half_life = self.0.half_life;
        let mut pending = self.pending();
        for shard in shards {
            pending.add_at(shard, slot, amount, now, half_life);
        }
    }

    /// Takes the heat recorded since the last call.
    pub(crate) fn take(&self) -> Table {
        mem::take(&mut *self.pending())
    }

    /// The heat recorded since the last [`HeatRecorder::take`].
    pub(crate) fn pending_table(&self) -> Table {
        self.pending().clone()
    }

    #[expect(
        clippy::expect_used,
        reason = "a poisoned lock means an earlier panic, and its state is not trusted"
    )]
    fn pending(&self) -> MutexGuard<'_, Table> {
        self.0
            .pending
            .lock()
            .expect("no thread panicked while holding the lock")
    }
}

impl HeatSink for HeatRecorder {
    fn record(&self, signal: Signal, shards: &[&ShardId]) {
        self.add_each(signal, shards, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::clock::{system_time, unix_ms};
    use crate::{Clock, FakeClock};

    const HOUR: Duration = Duration::from_secs(3_600);

    #[test]
    fn records_with_the_clock_it_was_given() {
        let clock = FakeClock::new(system_time(1_800_000_000_000));
        let half_life = HalfLife::new(HOUR).unwrap();
        let recorder = HeatRecorder::new(SharedClock::new(clock.clone()), half_life);
        let (a, b) = (ShardId::new("a").unwrap(), ShardId::new("b").unwrap());
        recorder.record(Signal::Search, &[&a, &b]);
        clock.advance(HOUR);
        recorder.record(Signal::Search, &[&a]);
        recorder.add(&b, Signal::Warm, 4.0).unwrap();
        assert!(matches!(
            recorder.add(&b, Signal::Warm, -1.0),
            Err(Error::InvalidAmount)
        ));
        assert!(matches!(
            recorder.add(&b, Signal::Warm, f64::NAN),
            Err(Error::InvalidAmount)
        ));

        let at = unix_ms(clock.now() + HOUR);
        let pending = recorder.take();
        let search = pending.counts_at(&a, at, half_life)[0];
        assert!((search - 0.75).abs() < 1e-12, "{search}");
        let warm = pending.counts_at(&b, at, half_life)[3];
        assert!((warm - 2.0).abs() < 1e-12, "{warm}");
        assert!(recorder.take().is_empty());
        assert!(recorder.pending_table().is_empty());
    }
}
