//! Contract tests for `cosmosearch-vector`.

#![expect(
    clippy::panic,
    clippy::unwrap_used,
    reason = "helpers in a test binary fail the test by panicking"
)]

mod ivf;
mod lookup;
mod recall;
mod reuse;
mod s3;
mod segments;
mod support;
mod sweep;
