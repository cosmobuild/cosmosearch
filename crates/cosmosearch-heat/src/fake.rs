use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, SystemTime};

use crate::Clock;

/// A clock for tests that stands still until it is moved. Clones share one time.
#[derive(Debug, Clone)]
pub struct FakeClock(Arc<Mutex<SystemTime>>);

impl FakeClock {
    /// A clock showing `start`.
    pub fn new(start: SystemTime) -> Self {
        FakeClock(Arc::new(Mutex::new(start)))
    }

    /// Moves the clock forward by `by`.
    pub fn advance(&self, by: Duration) {
        let mut now = self.locked();
        *now = now.checked_add(by).unwrap_or(*now);
    }

    /// Sets the clock to `to`, which may be earlier.
    pub fn set(&self, to: SystemTime) {
        *self.locked() = to;
    }

    #[expect(
        clippy::expect_used,
        reason = "a poisoned lock means an earlier panic, and its state is not trusted"
    )]
    fn locked(&self) -> MutexGuard<'_, SystemTime> {
        self.0
            .lock()
            .expect("no thread panicked while holding the lock")
    }
}

impl Clock for FakeClock {
    fn now(&self) -> SystemTime {
        *self.locked()
    }
}
