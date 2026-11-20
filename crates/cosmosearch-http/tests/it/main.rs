//! Contract tests for `cosmosearch-http`, against a fake server on localhost.

#![expect(
    clippy::unwrap_used,
    reason = "helpers in a test binary fail the test by panicking"
)]

mod embedder;
mod guard;
mod reranker;
mod retry;
mod server;

/// Everything an error shows: its message and `Debug` output, and those of every cause.
fn shown(error: &dyn std::error::Error) -> String {
    let mut parts = Vec::new();
    let mut next = Some(error);
    while let Some(error) = next {
        parts.push(format!("{error} {error:?}"));
        next = error.source();
    }
    parts.join(" ")
}
