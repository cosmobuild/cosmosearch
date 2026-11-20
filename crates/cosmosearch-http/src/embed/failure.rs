use cosmosearch_core::{EmbedCause, EmbedError};
use thiserror::Error;

use crate::connection::{JsonFault, Setup, Transport};

#[derive(Debug, Error)]
pub(crate) enum Failure {
    #[error(transparent)]
    Setup(#[from] Setup),
    #[error(transparent)]
    Transport(#[from] Transport),
    #[error("the embeddings request cannot be encoded: {0}")]
    Encode(JsonFault),
    #[error("the endpoint's answer is not embeddings JSON: {0}")]
    Malformed(JsonFault),
    #[error("the endpoint answered with an error in place of embeddings")]
    Answered,
    #[error("the endpoint returned {returned} vectors for {sent} texts")]
    Count { sent: usize, returned: usize },
    #[error("the endpoint returned an index out of range or twice")]
    Index,
    #[error("the endpoint returned a vector of {returned} dimensions, not {expected}")]
    Dimensions { expected: usize, returned: usize },
    #[error("the endpoint returned a vector with a non-finite value or, to normalize, zero length")]
    Length,
}

impl From<Failure> for EmbedError {
    fn from(failure: Failure) -> Self {
        EmbedError::Model(EmbedCause::new(failure))
    }
}
