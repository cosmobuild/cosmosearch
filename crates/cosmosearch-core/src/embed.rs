use std::error::Error as StdError;
use std::fmt;
use std::num::NonZeroU32;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{Error, ModelId, Result, Usage};

const MODEL_ID_PREFIX: &str = "emb-";
const MODEL_ID_HEX: usize = 32;

/// How two vectors of one embedding identity are compared.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum Metric {
    /// The cosine of the angle between them.
    Cosine,
    /// Their dot product.
    Dot,
}

impl Metric {
    fn as_str(self) -> &'static str {
        match self {
            Metric::Cosine => "cosine",
            Metric::Dot => "dot",
        }
    }
}

/// Everything that decides the vector a text gets. Vectors are comparable, and reusable,
/// only between equal identities.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EmbeddingIdentity {
    model: String,
    revision: String,
    dimensions: NonZeroU32,
    normalized: bool,
    metric: Metric,
    query_prefix: String,
    document_prefix: String,
    max_tokens: NonZeroU32,
}

impl EmbeddingIdentity {
    /// Starts an identity for revision `revision` of `model`, giving vectors of `dimensions`
    /// from inputs cut to `max_tokens`. It is normalized and compared by cosine unless set
    /// otherwise.
    pub fn builder(
        model: impl Into<String>,
        revision: impl Into<String>,
        dimensions: u32,
        max_tokens: u32,
    ) -> EmbeddingIdentityBuilder {
        EmbeddingIdentityBuilder {
            model: model.into(),
            revision: revision.into(),
            dimensions,
            max_tokens,
            normalized: true,
            metric: Metric::Cosine,
            query_prefix: String::new(),
            document_prefix: String::new(),
        }
    }

    /// The model's name.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// The exact revision of the model's weights.
    pub fn revision(&self) -> &str {
        &self.revision
    }

    /// The length of every vector.
    pub fn dimensions(&self) -> NonZeroU32 {
        self.dimensions
    }

    /// Whether every vector has length one.
    pub fn normalized(&self) -> bool {
        self.normalized
    }

    /// How vectors are compared.
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// Text put before a query.
    pub fn query_prefix(&self) -> &str {
        &self.query_prefix
    }

    /// Text put before a document.
    pub fn document_prefix(&self) -> &str {
        &self.document_prefix
    }

    /// The most tokens of an input the model reads; the rest is cut.
    pub fn max_tokens(&self) -> NonZeroU32 {
        self.max_tokens
    }

    /// The same identity with vectors cut to their first `dimensions`, for a model trained
    /// so its vectors can be shortened.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidIdentity`] if `dimensions` is zero or not below the current
    /// length.
    pub fn truncated(&self, dimensions: u32) -> Result<Self> {
        let dimensions = NonZeroU32::new(dimensions)
            .filter(|shorter| *shorter < self.dimensions)
            .ok_or(Error::InvalidIdentity {
                reason: "truncated dimensions must be above zero and below the current length",
            })?;
        Ok(EmbeddingIdentity {
            dimensions,
            ..self.clone()
        })
    }

    /// The short id of this whole identity, used to key stored vectors. Equal identities
    /// always give the same id, and any change to a field gives another.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidId`] only if the id format itself is broken, which is a bug.
    pub fn model_id(&self) -> Result<ModelId> {
        let mut hasher = blake3::Hasher::new();
        let dimensions = self.dimensions.get().to_string();
        let max_tokens = self.max_tokens.get().to_string();
        let normalized = if self.normalized { "1" } else { "0" };
        for field in [
            self.model.as_str(),
            self.revision.as_str(),
            &dimensions,
            normalized,
            self.metric.as_str(),
            &self.query_prefix,
            &self.document_prefix,
            &max_tokens,
        ] {
            hasher.update(&u64::try_from(field.len()).unwrap_or(u64::MAX).to_le_bytes());
            hasher.update(field.as_bytes());
        }
        let hex = hasher.finalize().to_hex();
        let short = hex.get(..MODEL_ID_HEX).unwrap_or_default();
        ModelId::new(format!("{MODEL_ID_PREFIX}{short}"))
    }
}

/// Builds an [`EmbeddingIdentity`].
#[derive(Debug, Clone)]
pub struct EmbeddingIdentityBuilder {
    model: String,
    revision: String,
    dimensions: u32,
    max_tokens: u32,
    normalized: bool,
    metric: Metric,
    query_prefix: String,
    document_prefix: String,
}

impl EmbeddingIdentityBuilder {
    /// Sets whether vectors have length one.
    #[must_use]
    pub fn normalized(mut self, normalized: bool) -> Self {
        self.normalized = normalized;
        self
    }

    /// Sets how vectors are compared.
    #[must_use]
    pub fn metric(mut self, metric: Metric) -> Self {
        self.metric = metric;
        self
    }

    /// Sets the text put before a query.
    #[must_use]
    pub fn query_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.query_prefix = prefix.into();
        self
    }

    /// Sets the text put before a document.
    #[must_use]
    pub fn document_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.document_prefix = prefix.into();
        self
    }

    /// Finishes the identity.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidIdentity`] if the model or revision is empty, or the
    /// dimensions or token limit is zero.
    pub fn build(self) -> Result<EmbeddingIdentity> {
        let invalid = |reason| Error::InvalidIdentity { reason };
        if self.model.is_empty() {
            return Err(invalid("the model must not be empty"));
        }
        if self.revision.is_empty() {
            return Err(invalid("the revision must not be empty"));
        }
        let dimensions =
            NonZeroU32::new(self.dimensions).ok_or(invalid("dimensions must not be zero"))?;
        let max_tokens =
            NonZeroU32::new(self.max_tokens).ok_or(invalid("the token limit must not be zero"))?;
        Ok(EmbeddingIdentity {
            model: self.model,
            revision: self.revision,
            dimensions,
            normalized: self.normalized,
            metric: self.metric,
            query_prefix: self.query_prefix,
            document_prefix: self.document_prefix,
            max_tokens,
        })
    }
}

/// Vectors for a list of texts, in the same order, and what making them cost.
#[derive(Debug, Clone, PartialEq)]
pub struct Embeddings {
    vectors: Vec<Vec<f32>>,
    usage: Usage,
}

impl Embeddings {
    /// Vectors made with `usage`.
    pub fn new(vectors: Vec<Vec<f32>>, usage: Usage) -> Self {
        Embeddings { vectors, usage }
    }

    /// The vectors, one per text.
    pub fn vectors(&self) -> &[Vec<f32>] {
        &self.vectors
    }

    /// Takes the vectors.
    pub fn into_vectors(self) -> Vec<Vec<f32>> {
        self.vectors
    }

    /// What making them cost, in tokens read.
    pub fn usage(&self) -> Usage {
        self.usage
    }
}

/// The cause of a failure inside an embedder.
#[derive(Debug)]
pub struct EmbedCause(Box<dyn StdError + Send + Sync>);

impl EmbedCause {
    /// Wraps a library's own error.
    pub fn new(source: impl Into<Box<dyn StdError + Send + Sync>>) -> Self {
        EmbedCause(source.into())
    }
}

impl fmt::Display for EmbedCause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl StdError for EmbedCause {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.0.source()
    }
}

/// Errors from embedding text.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum EmbedError {
    /// The model could not embed the texts.
    #[error("the embedding model failed")]
    Model(#[source] EmbedCause),
}

/// A plug-in that turns text into vectors of one embedding identity.
pub trait Embedder: Send + Sync + fmt::Debug {
    /// What decides the vectors; recorded with everything built from them.
    fn identity(&self) -> &EmbeddingIdentity;

    /// Embeds chunk texts for storage, giving one vector per text in order. Blocks for as
    /// long as embedding takes.
    ///
    /// # Errors
    ///
    /// Returns an [`EmbedError`] if the texts cannot be embedded.
    fn embed_documents(&self, texts: &[&str]) -> Result<Embeddings, EmbedError>;

    /// Embeds one search query. Blocks for as long as embedding takes.
    ///
    /// # Errors
    ///
    /// Returns an [`EmbedError`] if the query cannot be embedded.
    fn embed_query(&self, text: &str) -> Result<Embeddings, EmbedError>;
}

/// A shared handle on an embedder, for running it on another thread.
#[derive(Debug, Clone)]
pub struct SharedEmbedder(Arc<dyn Embedder>);

impl SharedEmbedder {
    /// Shares `embedder`.
    pub fn new(embedder: impl Embedder + 'static) -> Self {
        SharedEmbedder(Arc::new(embedder))
    }

    /// The embedder.
    pub fn get(&self) -> &dyn Embedder {
        self.0.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> EmbeddingIdentity {
        EmbeddingIdentity::builder("example-model", "rev-1", 768, 512)
            .query_prefix("query: ")
            .build()
            .unwrap()
    }

    #[test]
    fn the_model_id_covers_every_field() {
        let base = identity();
        let id = base.model_id().unwrap();
        assert_eq!(id, identity().model_id().unwrap());
        assert!(id.as_str().starts_with("emb-"));
        let changed = [
            EmbeddingIdentity::builder("other", "rev-1", 768, 512)
                .query_prefix("query: ")
                .build()
                .unwrap(),
            EmbeddingIdentity::builder("example-model", "rev-2", 768, 512)
                .query_prefix("query: ")
                .build()
                .unwrap(),
            base.truncated(256).unwrap(),
            EmbeddingIdentity::builder("example-model", "rev-1", 768, 256)
                .query_prefix("query: ")
                .build()
                .unwrap(),
            EmbeddingIdentity::builder("example-model", "rev-1", 768, 512)
                .build()
                .unwrap(),
            EmbeddingIdentity::builder("example-model", "rev-1", 768, 512)
                .query_prefix("query: ")
                .document_prefix("passage: ")
                .build()
                .unwrap(),
            EmbeddingIdentity::builder("example-model", "rev-1", 768, 512)
                .query_prefix("query: ")
                .normalized(false)
                .build()
                .unwrap(),
            EmbeddingIdentity::builder("example-model", "rev-1", 768, 512)
                .query_prefix("query: ")
                .metric(Metric::Dot)
                .build()
                .unwrap(),
        ];
        for other in changed {
            assert_ne!(other.model_id().unwrap(), id, "{other:?}");
        }
    }

    #[test]
    fn field_boundaries_are_part_of_the_id() {
        let a = EmbeddingIdentity::builder("ab", "c", 4, 8).build().unwrap();
        let b = EmbeddingIdentity::builder("a", "bc", 4, 8).build().unwrap();
        assert_ne!(a.model_id().unwrap(), b.model_id().unwrap());
    }

    #[test]
    fn truncation_only_shortens() {
        let base = identity();
        assert_eq!(base.truncated(256).unwrap().dimensions().get(), 256);
        for bad in [0, 768, 1024] {
            assert!(base.truncated(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn rejects_empty_names_and_zero_sizes() {
        for builder in [
            EmbeddingIdentity::builder("", "r", 4, 8),
            EmbeddingIdentity::builder("m", "", 4, 8),
            EmbeddingIdentity::builder("m", "r", 0, 8),
            EmbeddingIdentity::builder("m", "r", 4, 0),
        ] {
            assert!(matches!(
                builder.build(),
                Err(Error::InvalidIdentity { .. })
            ));
        }
    }

    #[test]
    fn the_identity_round_trips_as_json() {
        let base = identity();
        let json = serde_json::to_string(&base).unwrap();
        assert_eq!(
            serde_json::from_str::<EmbeddingIdentity>(&json).unwrap(),
            base
        );
    }
}
