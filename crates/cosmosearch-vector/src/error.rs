use cosmosearch_core::Metric;
use thiserror::Error;

/// Errors from the vector store and the IVF index.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// The store failed.
    #[error(transparent)]
    Store(#[from] cosmosearch_store::Error),

    /// Vectors could not be encoded, or a vector is invalid.
    #[error(transparent)]
    Format(#[from] crate::format::Error),

    /// An id or range is invalid.
    #[error(transparent)]
    Core(#[from] cosmosearch_core::Error),

    /// The vectors given cannot be clustered or compared.
    #[error("the vectors cannot be clustered: {reason}")]
    Clustering {
        /// What is wrong with them.
        reason: &'static str,
    },

    /// The embedding identity's metric is one the index cannot score.
    #[error("the metric {metric:?} is not supported")]
    UnsupportedMetric {
        /// The metric.
        metric: Metric,
    },

    /// The time given lies before the Unix epoch.
    #[error("the time given lies before the Unix epoch")]
    TimeBeforeEpoch,
}

/// Result type of this crate.
pub type Result<T, E = Error> = std::result::Result<T, E>;
