use std::error::Error as StdError;
use std::fmt;
use std::num::NonZeroU32;
use std::sync::Arc;

use thiserror::Error;

use crate::{Error, Result, Usage};

/// What scores passages: the model, the exact revision of its weights and the most tokens
/// of a query and passage it reads together.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RerankerIdentity {
    model: String,
    revision: String,
    max_tokens: NonZeroU32,
}

impl RerankerIdentity {
    /// The identity of revision `revision` of `model`, reading at most `max_tokens` of each
    /// query and passage together.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidIdentity`] if the model or revision is empty or the token
    /// limit is zero.
    pub fn new(
        model: impl Into<String>,
        revision: impl Into<String>,
        max_tokens: u32,
    ) -> Result<Self> {
        let invalid = |reason| Error::InvalidIdentity { reason };
        let model = model.into();
        if model.is_empty() {
            return Err(invalid("the model must not be empty"));
        }
        let revision = revision.into();
        if revision.is_empty() {
            return Err(invalid("the revision must not be empty"));
        }
        let max_tokens =
            NonZeroU32::new(max_tokens).ok_or(invalid("the token limit must not be zero"))?;
        Ok(RerankerIdentity {
            model,
            revision,
            max_tokens,
        })
    }

    /// The model's name.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// The exact revision of the model's weights.
    pub fn revision(&self) -> &str {
        &self.revision
    }

    /// The most tokens of a query and passage the model reads together; the rest is cut.
    pub fn max_tokens(&self) -> NonZeroU32 {
        self.max_tokens
    }
}

/// One score per passage, in the order the passages were given, and what scoring cost.
#[derive(Debug, Clone, PartialEq)]
pub struct RerankScores {
    scores: Vec<f32>,
    usage: Usage,
}

impl RerankScores {
    /// Scores made with `usage`.
    pub fn new(scores: Vec<f32>, usage: Usage) -> Self {
        RerankScores { scores, usage }
    }

    /// The scores, higher meaning more relevant. They order passages of one query and mean
    /// nothing on their own.
    pub fn scores(&self) -> &[f32] {
        &self.scores
    }

    /// Takes the scores.
    pub fn into_scores(self) -> Vec<f32> {
        self.scores
    }

    /// What scoring cost, in tokens read.
    pub fn usage(&self) -> Usage {
        self.usage
    }
}

/// The cause of a failure inside a reranker.
#[derive(Debug)]
pub struct RerankCause(Box<dyn StdError + Send + Sync>);

impl RerankCause {
    /// Wraps a library's own error.
    pub fn new(source: impl Into<Box<dyn StdError + Send + Sync>>) -> Self {
        RerankCause(source.into())
    }
}

impl fmt::Display for RerankCause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl StdError for RerankCause {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.0.source()
    }
}

/// Errors from scoring passages.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum RerankError {
    /// The model could not score the passages.
    #[error("the reranking model failed")]
    Model(#[source] RerankCause),
}

/// A plug-in that scores how well passages answer a query, reading each passage together
/// with the query.
pub trait Reranker: Send + Sync + fmt::Debug {
    /// What gives the scores.
    fn identity(&self) -> &RerankerIdentity;

    /// Scores each of `passages` against `query`, giving one score per passage in order.
    /// Blocks for as long as scoring takes.
    ///
    /// # Errors
    ///
    /// Returns a [`RerankError`] if the passages cannot be scored.
    fn rerank(&self, query: &str, passages: &[&str]) -> Result<RerankScores, RerankError>;
}

/// A shared handle on a reranker, for running it on another thread.
#[derive(Debug, Clone)]
pub struct SharedReranker(Arc<dyn Reranker>);

impl SharedReranker {
    /// Shares `reranker`.
    pub fn new(reranker: impl Reranker + 'static) -> Self {
        SharedReranker(Arc::new(reranker))
    }

    /// The reranker.
    pub fn get(&self) -> &dyn Reranker {
        self.0.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_identity_keeps_its_fields() {
        let identity = RerankerIdentity::new("example-reranker", "rev-1", 512).unwrap();
        assert_eq!(identity.model(), "example-reranker");
        assert_eq!(identity.revision(), "rev-1");
        assert_eq!(identity.max_tokens().get(), 512);
    }

    #[test]
    fn rejects_empty_names_and_a_zero_token_limit() {
        for (model, revision, max_tokens) in [("", "r", 8), ("m", "", 8), ("m", "r", 0)] {
            assert!(matches!(
                RerankerIdentity::new(model, revision, max_tokens),
                Err(Error::InvalidIdentity { .. })
            ));
        }
    }

    #[test]
    fn an_error_keeps_its_cause() {
        let error = RerankError::Model(RerankCause::new("the endpoint answered HTTP 500"));
        assert_eq!(error.to_string(), "the reranking model failed");
        let cause = StdError::source(&error).unwrap();
        assert_eq!(cause.to_string(), "the endpoint answered HTTP 500");
    }
}
