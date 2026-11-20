use cosmosearch_core::{RerankCause, RerankError};
use thiserror::Error;

use crate::connection::{JsonFault, Setup, Transport};

#[derive(Debug, Error)]
pub(crate) enum Failure {
    #[error(transparent)]
    Setup(#[from] Setup),
    #[error(transparent)]
    Transport(#[from] Transport),
    #[error("the rerank request cannot be encoded: {0}")]
    Encode(JsonFault),
    #[error("the endpoint's answer is not rerank JSON: {0}")]
    Malformed(JsonFault),
    #[error("the endpoint answered with an error in place of scores")]
    Answered,
    #[error("the endpoint returned {returned} scores for {sent} passages")]
    Count { sent: usize, returned: usize },
    #[error("the endpoint returned an index out of range or twice")]
    Index,
    #[error("the endpoint returned a score that is not finite")]
    Score,
}

impl From<Failure> for RerankError {
    fn from(failure: Failure) -> Self {
        RerankError::Model(RerankCause::new(failure))
    }
}
