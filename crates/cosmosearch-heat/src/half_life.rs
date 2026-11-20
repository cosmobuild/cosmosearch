use std::time::Duration;

use crate::{Error, Result};

const WEEK: Duration = Duration::from_hours(7 * 24);
const MILLIS_PER_SECOND: f64 = 1_000.0;

/// The time over which heat halves when nothing adds to it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct HalfLife(Duration);

impl HalfLife {
    /// A half-life of `duration`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSettings`] for a half-life under one millisecond.
    pub fn new(duration: Duration) -> Result<Self> {
        if duration < Duration::from_millis(1) {
            return Err(Error::InvalidSettings {
                reason: "the half-life must be at least one millisecond",
            });
        }
        Ok(HalfLife(duration))
    }

    /// The half-life as a duration.
    pub fn get(self) -> Duration {
        self.0
    }

    pub(crate) fn millis(self) -> u64 {
        u64::try_from(self.0.as_millis()).unwrap_or(u64::MAX)
    }

    /// What a count at `from` is worth at `to`: below one later, above one earlier.
    pub(crate) fn factor(self, from_ms: u64, to_ms: u64) -> f64 {
        let elapsed = as_f64(to_ms) - as_f64(from_ms);
        (-elapsed / (self.0.as_secs_f64() * MILLIS_PER_SECOND)).exp2()
    }
}

impl Default for HalfLife {
    /// One week.
    fn default() -> Self {
        HalfLife(WEEK)
    }
}

#[expect(
    clippy::cast_precision_loss,
    reason = "Unix milliseconds stay below 2^53, where an f64 is exact, for another 285,000 years"
)]
fn as_f64(ms: u64) -> f64 {
    ms as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_count_halves_every_half_life() {
        let half_life = HalfLife::new(Duration::from_secs(10)).unwrap();
        assert!((half_life.factor(1_000, 11_000) - 0.5).abs() < 1e-15);
        assert!((half_life.factor(1_000, 21_000) - 0.25).abs() < 1e-15);
        assert!((half_life.factor(11_000, 1_000) - 2.0).abs() < 1e-15);
        assert!((half_life.factor(5, 5) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn the_default_is_a_week_and_zero_is_refused() {
        assert_eq!(HalfLife::default().get(), WEEK);
        assert!(matches!(
            HalfLife::new(Duration::from_micros(999)),
            Err(Error::InvalidSettings { .. })
        ));
    }
}
