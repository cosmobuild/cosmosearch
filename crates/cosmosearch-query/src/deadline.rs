use tokio::time::{self, Instant};

/// Runs `work` until `deadline`, giving `None` if the deadline passes first. Work that is
/// done when first polled finishes even after the deadline.
pub(crate) async fn within<F: Future>(deadline: Option<Instant>, work: F) -> Option<F::Output> {
    match deadline {
        None => Some(work.await),
        Some(deadline) => time::timeout_at(deadline, work).await.ok(),
    }
}
