use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// A point in time to the millisecond, counted from the Unix epoch; negative before it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Timestamp(i64);

impl Timestamp {
    /// The time `millis` milliseconds after the Unix epoch.
    pub fn from_millis(millis: i64) -> Self {
        Timestamp(millis)
    }

    /// Milliseconds since the Unix epoch.
    pub fn millis(self) -> i64 {
        self.0
    }

    /// The same time as a [`SystemTime`], or `None` if this platform cannot hold it.
    pub fn to_system_time(self) -> Option<SystemTime> {
        let distance = Duration::from_millis(self.0.unsigned_abs());
        if self.0 >= 0 {
            UNIX_EPOCH.checked_add(distance)
        } else {
            UNIX_EPOCH.checked_sub(distance)
        }
    }
}

impl From<SystemTime> for Timestamp {
    /// Drops what is below a millisecond, and saturates for times beyond the range.
    fn from(time: SystemTime) -> Self {
        match time.duration_since(UNIX_EPOCH) {
            Ok(after) => Timestamp(i64::try_from(after.as_millis()).unwrap_or(i64::MAX)),
            Err(before) => {
                let millis = i64::try_from(before.duration().as_millis()).unwrap_or(i64::MAX);
                Timestamp(millis.saturating_neg())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_to_and_from_system_time_at_millisecond_precision() {
        let cases = [
            (
                UNIX_EPOCH + Duration::from_micros(1_700_000_000_123_456),
                1_700_000_000_123,
            ),
            (UNIX_EPOCH - Duration::from_secs(1_001), -1_001_000),
            (UNIX_EPOCH, 0),
        ];
        for (time, millis) in cases {
            let stamp = Timestamp::from(time);
            assert_eq!(stamp.millis(), millis);
            assert_eq!(Timestamp::from(stamp.to_system_time().unwrap()), stamp);
        }
    }
}
