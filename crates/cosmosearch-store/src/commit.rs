use bytes::Bytes;
use tracing::warn;

use crate::{Error, ObjectKey, Result, WriteStore};

const WRITE_ATTEMPTS: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Confirmed {
    Mine,
    Other,
    Unknown,
}

/// A key reported as taken is not trusted until read back: a store can report a conflict
/// with nothing visible, and a retried write can find its own first attempt.
pub(crate) async fn create_confirmed(
    backend: &dyn WriteStore,
    key: &ObjectKey,
    body: &Bytes,
    is_mine: impl Fn(&[u8]) -> Result<bool>,
) -> Result<Confirmed> {
    let mut may_have_landed = false;
    for _ in 0..WRITE_ATTEMPTS {
        match backend.put_create(key, body.clone()).await {
            Ok(()) => return Ok(Confirmed::Mine),
            Err(Error::AlreadyExists { .. } | Error::OutcomeUnknown { .. }) => {
                may_have_landed = true;
            }
            Err(err) if may_have_landed => {
                warn!(error = %err, "giving up on a write that may have landed");
                return Ok(Confirmed::Unknown);
            }
            Err(err) => return Err(err),
        }
        match backend.get(key).await {
            Ok(found) if is_mine(&found)? => return Ok(Confirmed::Mine),
            Ok(_) => return Ok(Confirmed::Other),
            Err(Error::NotFound { .. } | Error::Backend { .. }) => {}
            Err(err) => {
                warn!(error = %err, "could not read back a write that may have landed");
                return Ok(Confirmed::Unknown);
            }
        }
    }
    Ok(Confirmed::Unknown)
}
