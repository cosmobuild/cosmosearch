mod failure;
mod wire;

use std::fmt;
use std::num::NonZeroUsize;
use std::time::Duration;

use cosmosearch_core::{EmbedError, Embedder, EmbeddingIdentity, Embeddings, Usage};

use crate::ApiKey;
use crate::connection::{Connection, JsonFault, Settings};
use failure::Failure;
use wire::{EmbeddingsRequest, Expected};

const EMBEDDINGS_PATH: &str = "embeddings";
const DEFAULT_BATCH: NonZeroUsize = match NonZeroUsize::new(64) {
    Some(size) => size,
    None => NonZeroUsize::MIN,
};

/// Embeds texts by calling an OpenAI-compatible `POST {base}/embeddings` endpoint.
///
/// It blocks the calling thread for each request, so it must not run on an async runtime's
/// worker thread. Texts go in batches, one request at a time. A request that times out,
/// cannot connect or is answered 408, 429, 500, 502, 503, 504, 524 or 529 is tried again after a wait
/// that doubles each time, or as long as the endpoint's `Retry-After` asks, up to a minute.
pub struct HttpEmbedder {
    identity: EmbeddingIdentity,
    model: String,
    dimensions: Option<u32>,
    batch: NonZeroUsize,
    connection: Connection,
}

impl HttpEmbedder {
    /// Starts an embedder calling the endpoint under `base_url`, such as
    /// `https://api.example.com/v1`, for `model`, whose vectors are of `identity`. It sends
    /// no key, sends 64 texts a request, waits 30 seconds for an answer and tries a failed
    /// request 4 more times, first after half a second, unless set otherwise.
    pub fn builder(
        base_url: impl Into<String>,
        model: impl Into<String>,
        identity: EmbeddingIdentity,
    ) -> HttpEmbedderBuilder {
        HttpEmbedderBuilder {
            base_url: base_url.into(),
            model: model.into(),
            identity,
            request_dimensions: false,
            batch: DEFAULT_BATCH,
            settings: Settings::default(),
        }
    }

    fn embed(&self, prefix: &str, texts: &[&str]) -> Result<Embeddings, EmbedError> {
        let inputs: Vec<String> = texts.iter().map(|text| format!("{prefix}{text}")).collect();
        let mut vectors = Vec::with_capacity(inputs.len());
        let mut tokens = 0u64;
        for batch in inputs.chunks(self.batch.get()) {
            let (batch_vectors, batch_tokens) = self.embed_batch(batch)?;
            vectors.extend(batch_vectors);
            tokens = tokens.saturating_add(batch_tokens);
        }
        Ok(Embeddings::new(vectors, Usage::of_tokens(tokens)))
    }

    fn embed_batch(&self, inputs: &[String]) -> Result<(Vec<Vec<f32>>, u64), Failure> {
        let body = serde_json::to_vec(&EmbeddingsRequest {
            model: &self.model,
            input: inputs,
            dimensions: self.dimensions,
        })
        .map_err(|cause| Failure::Encode(JsonFault::of(&cause)))?;
        let answer = self.connection.post(&body)?;
        wire::parse(
            &answer,
            Expected {
                texts: inputs.len(),
                dimensions: usize::try_from(self.identity.dimensions().get()).unwrap_or(usize::MAX),
                normalized: self.identity.normalized(),
            },
        )
    }
}

impl fmt::Debug for HttpEmbedder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HttpEmbedder")
            .field("identity", &self.identity)
            .field("model", &self.model)
            .field("dimensions", &self.dimensions)
            .field("batch", &self.batch)
            .field("connection", &self.connection)
            .finish()
    }
}

impl Embedder for HttpEmbedder {
    fn identity(&self) -> &EmbeddingIdentity {
        &self.identity
    }

    fn embed_documents(&self, texts: &[&str]) -> Result<Embeddings, EmbedError> {
        self.embed(self.identity.document_prefix(), texts)
    }

    fn embed_query(&self, text: &str) -> Result<Embeddings, EmbedError> {
        self.embed(self.identity.query_prefix(), &[text])
    }
}

/// Builds an [`HttpEmbedder`].
#[derive(Debug, Clone)]
pub struct HttpEmbedderBuilder {
    base_url: String,
    model: String,
    identity: EmbeddingIdentity,
    request_dimensions: bool,
    batch: NonZeroUsize,
    settings: Settings,
}

impl HttpEmbedderBuilder {
    /// Sends `key` as a bearer token with every request.
    #[must_use]
    pub fn api_key(mut self, key: ApiKey) -> Self {
        self.settings.key = Some(key);
        self
    }

    /// Asks the endpoint for vectors of the identity's dimensions with the `dimensions`
    /// field, for a model that can shorten its vectors.
    #[must_use]
    pub fn request_dimensions(mut self) -> Self {
        self.request_dimensions = true;
        self
    }

    /// Sets how many texts go in one request.
    #[must_use]
    pub fn batch_size(mut self, size: NonZeroUsize) -> Self {
        self.batch = size;
        self
    }

    /// Sets how long one request may take, from connecting to reading the whole answer.
    #[must_use]
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.settings.timeout = timeout;
        self
    }

    /// Allows a base URL over plain `http`, which sends the key and the texts unencrypted:
    /// only for an endpoint on this machine or on a network the host trusts.
    #[must_use]
    pub fn allow_http(mut self) -> Self {
        self.settings.allow_http = true;
        self
    }

    /// Sets how many more times a request that failed for a passing reason is tried.
    #[must_use]
    pub fn retries(mut self, retries: u32) -> Self {
        self.settings.retries = retries;
        self
    }

    /// Sets the wait before the first retry, which doubles for each one after it, when the
    /// endpoint sends no `Retry-After`.
    #[must_use]
    pub fn retry_wait(mut self, wait: Duration) -> Self {
        self.settings.retry_wait = wait;
        self
    }

    /// Finishes the embedder.
    ///
    /// # Errors
    ///
    /// Returns [`EmbedError::Model`] if the model name is empty, the base URL is not an https
    /// URL, or an http URL without [`allow_http`](Self::allow_http), or holds a user name,
    /// password, query or fragment, the key is empty or holds characters a header cannot
    /// carry, or the HTTP client cannot be built.
    pub fn build(self) -> Result<HttpEmbedder, EmbedError> {
        let connection =
            Connection::new(&self.base_url, EMBEDDINGS_PATH, &self.model, self.settings)
                .map_err(Failure::from)?;
        let dimensions = self
            .request_dimensions
            .then(|| self.identity.dimensions().get());
        Ok(HttpEmbedder {
            identity: self.identity,
            model: self.model,
            dimensions,
            batch: self.batch,
            connection,
        })
    }
}
