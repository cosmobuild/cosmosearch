use std::fmt;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Where heat takes the time from. The host can supply its own, for tests or for replaying
/// a log.
pub trait Clock: Send + Sync + fmt::Debug {
    /// The time now.
    fn now(&self) -> SystemTime;
}

/// The system's clock.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct SystemClock;

impl SystemClock {
    /// The system's clock.
    pub fn new() -> Self {
        SystemClock
    }
}

impl Clock for SystemClock {
    fn now(&self) -> SystemTime {
        SystemTime::now()
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SharedClock(Arc<dyn Clock>);

impl SharedClock {
    pub(crate) fn new(clock: impl Clock + 'static) -> Self {
        SharedClock(Arc::new(clock))
    }

    pub(crate) fn now_ms(&self) -> u64 {
        unix_ms(self.0.now())
    }
}

/// Milliseconds since the Unix epoch; a time before it counts as the epoch.
pub(crate) fn unix_ms(time: SystemTime) -> u64 {
    time.duration_since(UNIX_EPOCH).map_or(0, |since| {
        u64::try_from(since.as_millis()).unwrap_or(u64::MAX)
    })
}

pub(crate) fn system_time(ms: u64) -> SystemTime {
    UNIX_EPOCH
        .checked_add(Duration::from_millis(ms))
        .unwrap_or(UNIX_EPOCH)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn times_round_trip_through_milliseconds() {
        let time = UNIX_EPOCH + Duration::from_millis(1_800_000_000_123);
        assert_eq!(unix_ms(time), 1_800_000_000_123);
        assert_eq!(system_time(unix_ms(time)), time);
        assert_eq!(unix_ms(UNIX_EPOCH - Duration::from_secs(1)), 0);
    }
}
