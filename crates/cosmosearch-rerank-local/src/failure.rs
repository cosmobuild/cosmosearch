use std::error::Error as StdError;

use cosmosearch_core::{RerankCause, RerankError};

pub(crate) fn failed(cause: impl Into<Box<dyn StdError + Send + Sync>>) -> RerankError {
    RerankError::Model(RerankCause::new(cause))
}
