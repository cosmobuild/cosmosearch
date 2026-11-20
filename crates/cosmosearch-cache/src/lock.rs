use std::sync::{Mutex, MutexGuard};

#[expect(
    clippy::expect_used,
    reason = "a poisoned lock means an earlier panic, and its state is not trusted"
)]
pub(crate) fn locked<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .expect("no thread panicked while holding the lock")
}
