//! Contract tests for `cosmosearch-store`.

#![expect(
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used,
    reason = "helpers in a test binary fail the test by panicking"
)]

mod backend;
mod collection;
mod conformance;
mod lifecycle;
#[cfg(feature = "s3")]
mod s3_responses;
mod support;
mod swap;
