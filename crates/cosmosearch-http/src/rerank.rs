mod failure;
mod wire;

use std::fmt;
use std::time::Duration;

use cosmosearch_core::{RerankError, RerankScores, Reranker, RerankerIdentity, Usage};

use crate::ApiKey;
use crate::connection::{Connection, JsonFault, Settings};
use failure::Failure;
use wire::RerankRequest;

const RERANK_PATH: &str = "rerank";

/// Scores passages by calling a `POST {base}/rerank` endpoint, which takes a model, a query
/// and documents and answers with a relevance score for each document's index.
///
/// It blocks the calling thread for each request, so it must not run on an async runtime's
/// worker thread. All passages of one call go in one request. A request that times out,
/// cannot connect or is answered 408, 429, 500, 502, 503, 504, 524 or 529 is tried again after a wait
/// that doubles each time, or as long as the endpoint's `Retry-After` asks, up to a minute.
pub struct HttpReranker {
    identity: RerankerIdentity,
    model: String,
    connection: Connection,
}

impl HttpReranker {
    /// Starts a reranker calling the endpoint under `base_url`, such as
    /// `https://api.example.com/v1`, for `model`, whose scores are of `identity`. It sends no
    /// key, waits 30 seconds for an answer and tries a failed request 4 more times, first
    /// after half a second, unless set otherwise.
    pub fn builder(
        base_url: impl Into<String>,
        model: impl Into<String>,
        identity: RerankerIdentity,
    ) -> HttpRerankerBuilder {
        HttpRerankerBuilder {
            base_url: base_url.into(),
            model: model.into(),
            identity,
            settings: Settings::default(),
        }
    }

    fn request(&self, query: &str, passages: &[&str]) -> Result<(Vec<f32>, u64), Failure> {
        let body = serde_json::to_vec(&RerankRequest {
            model: &self.model,
            query,
            documents: passages,
            top_n: passages.len(),
        })
        .map_err(|cause| Failure::Encode(JsonFault::of(&cause)))?;
        let answer = self.connection.post(&body)?;
        wire::parse(&answer, passages.len())
    }
}

impl fmt::Debug for HttpReranker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HttpReranker")
            .field("identity", &self.identity)
            .field("model", &self.model)
            .field("connection", &self.connection)
            .finish()
    }
}

impl Reranker for HttpReranker {
    fn identity(&self) -> &RerankerIdentity {
        &self.identity
    }

    fn rerank(&self, query: &str, passages: &[&str]) -> Result<RerankScores, RerankError> {
        if passages.is_empty() {
            return Ok(RerankScores::new(Vec::new(), Usage::new()));
        }
        let (scores, tokens) = self.request(query, passages)?;
        Ok(RerankScores::new(scores, Usage::of_tokens(tokens)))
    }
}

/// Builds an [`HttpReranker`].
#[derive(Debug, Clone)]
pub struct HttpRerankerBuilder {
    base_url: String,
    model: String,
    identity: RerankerIdentity,
    settings: Settings,
}

impl HttpRerankerBuilder {
    /// Sends `key` as a bearer token with every request.
    #[must_use]
    pub fn api_key(mut self, key: ApiKey) -> Self {
        self.settings.key = Some(key);
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

    /// Finishes the reranker.
    ///
    /// # Errors
    ///
    /// Returns [`RerankError::Model`] if the model name is empty, the base URL is not an https
    /// URL, or an http URL without [`allow_http`](Self::allow_http), or holds a user name,
    /// password, query or fragment, the key is empty or holds characters a header cannot
    /// carry, or the HTTP client cannot be built.
    pub fn build(self) -> Result<HttpReranker, RerankError> {
        let connection = Connection::new(&self.base_url, RERANK_PATH, &self.model, self.settings)
            .map_err(Failure::from)?;
        Ok(HttpReranker {
            identity: self.identity,
            model: self.model,
            connection,
        })
    }
}
