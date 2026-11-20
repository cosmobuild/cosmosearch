use std::collections::hash_map::RandomState;
use std::error::Error as StdError;
use std::fmt;
use std::hash::{BuildHasher, Hasher};
use std::io::{self, Read};
use std::time::Duration;

use reqwest::Url;
use reqwest::blocking::{Client, Response};
use reqwest::header::{CONTENT_TYPE, RETRY_AFTER};
use reqwest::redirect::Policy;
use serde_json::error::Category;
use thiserror::Error;

use crate::ApiKey;

const JSON: &str = "application/json";
pub(crate) const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);
pub(crate) const DEFAULT_RETRIES: u32 = 4;
pub(crate) const DEFAULT_RETRY_WAIT: Duration = Duration::from_millis(500);
const MAX_WAIT: Duration = Duration::from_secs(60);
/// The largest answer read, far above what a batch of vectors or scores needs.
pub(crate) const MAX_ANSWER_BYTES: u64 = 64 << 20;
/// Statuses that report a passing failure. 524 is a timeout at a gateway's edge network and
/// 529 an overloaded provider; gateways send both though neither is a standard status.
const RETRIED_STATUSES: [u16; 8] = [408, 429, 500, 502, 503, 504, 524, 529];

/// Where JSON failed to encode or parse, without the text: `serde_json`'s own message can
/// quote the value it stopped at, which may be document text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct JsonFault {
    kind: Category,
    line: usize,
    column: usize,
}

impl JsonFault {
    pub(crate) fn of(error: &serde_json::Error) -> Self {
        JsonFault {
            kind: error.classify(),
            line: error.line(),
            column: error.column(),
        }
    }
}

impl fmt::Display for JsonFault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = match self.kind {
            Category::Io => "an I/O error",
            Category::Syntax => "a syntax error",
            Category::Data => "a value of the wrong shape",
            Category::Eof => "an early end",
        };
        if self.line == 0 {
            f.write_str(kind)
        } else {
            write!(f, "{kind} at line {}, column {}", self.line, self.column)
        }
    }
}

/// Whether a body that is not the expected answer is an error object, `{"error": ...}`, which
/// some gateways send with status 200 when the provider fails after accepting the request.
/// The error itself is not kept: its message can quote the input.
pub(crate) fn holds_error(body: &[u8]) -> bool {
    #[derive(serde::Deserialize)]
    struct ErrorBody {
        #[expect(dead_code, reason = "only its presence is checked")]
        error: serde::de::IgnoredAny,
    }
    serde_json::from_slice::<ErrorBody>(body).is_ok()
}

#[derive(Debug, Error)]
pub(crate) enum Setup {
    #[error("the model name must not be empty")]
    Model,
    #[error("the base URL cannot be parsed")]
    BaseUrl(#[source] Box<dyn StdError + Send + Sync>),
    #[error("the base URL must use http or https")]
    Scheme,
    #[error("the base URL uses plain http, which needs `allow_http`")]
    PlainHttp,
    #[error("the base URL must not hold a user name, password, query or fragment")]
    Credentials,
    #[error("the API key must be visible ASCII and not empty")]
    ApiKey,
    #[error("the HTTP client cannot be built")]
    Client(#[source] reqwest::Error),
    #[error("building the HTTP client panicked")]
    ClientPanicked,
}

#[derive(Debug, Error)]
pub(crate) enum Transport {
    #[error("the request timed out")]
    Timeout(#[source] reqwest::Error),
    #[error("the request failed")]
    Request(#[source] reqwest::Error),
    #[error("the endpoint answered HTTP {0}")]
    Status(u16),
    #[error("the answer is larger than {MAX_ANSWER_BYTES} bytes")]
    TooLarge,
    #[error("the answer could not be read")]
    Read(#[source] io::Error),
}

#[derive(Debug, Clone)]
pub(crate) struct Settings {
    pub(crate) key: Option<ApiKey>,
    pub(crate) timeout: Duration,
    pub(crate) retries: u32,
    pub(crate) retry_wait: Duration,
    pub(crate) allow_http: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            key: None,
            timeout: DEFAULT_TIMEOUT,
            retries: DEFAULT_RETRIES,
            retry_wait: DEFAULT_RETRY_WAIT,
            allow_http: false,
        }
    }
}

pub(crate) struct Connection {
    endpoint: Url,
    key: Option<ApiKey>,
    retries: u32,
    retry_wait: Duration,
    client: Client,
}

struct Failed {
    failure: Transport,
    retry: bool,
    retry_after: Option<Duration>,
}

impl Connection {
    pub(crate) fn new(
        base_url: &str,
        path: &str,
        model: &str,
        settings: Settings,
    ) -> Result<Self, Setup> {
        if model.is_empty() {
            return Err(Setup::Model);
        }
        let base = base_url.trim_end_matches('/');
        let endpoint = Url::parse(&format!("{base}/{path}"))
            .map_err(|source| Setup::BaseUrl(source.into()))?;
        match endpoint.scheme() {
            "https" => {}
            "http" if settings.allow_http => {}
            "http" => return Err(Setup::PlainHttp),
            _ => return Err(Setup::Scheme),
        }
        if !endpoint.username().is_empty()
            || endpoint.password().is_some()
            || endpoint.query().is_some()
            || endpoint.fragment().is_some()
        {
            return Err(Setup::Credentials);
        }
        if settings.key.as_ref().is_some_and(|key| !key.is_valid()) {
            return Err(Setup::ApiKey);
        }
        let client = build_client(settings.timeout)?;
        Ok(Connection {
            endpoint,
            key: settings.key,
            retries: settings.retries,
            retry_wait: settings.retry_wait,
            client,
        })
    }

    pub(crate) fn post(&self, body: &[u8]) -> Result<Vec<u8>, Transport> {
        let mut attempt = 0u32;
        loop {
            match self.try_post(body.to_vec()) {
                Ok(answer) => return Ok(answer),
                Err(failed) if failed.retry && attempt < self.retries => {
                    pause(self.wait(attempt, failed.retry_after));
                    attempt = attempt.saturating_add(1);
                }
                Err(failed) => return Err(failed.failure),
            }
        }
    }

    fn try_post(&self, body: Vec<u8>) -> Result<Vec<u8>, Failed> {
        let mut request = self
            .client
            .post(self.endpoint.clone())
            .header(CONTENT_TYPE, JSON)
            .body(body);
        if let Some(key) = &self.key {
            request = request.bearer_auth(key.expose());
        }
        let response = request.send().map_err(of_transport)?;
        let status = response.status();
        if !status.is_success() {
            return Err(Failed {
                failure: Transport::Status(status.as_u16()),
                retry: RETRIED_STATUSES.contains(&status.as_u16()),
                retry_after: retry_after(&response),
            });
        }
        read_answer(response)
    }

    fn wait(&self, attempt: u32, retry_after: Option<Duration>) -> Duration {
        if let Some(asked) = retry_after {
            return asked.min(MAX_WAIT);
        }
        let backoff = self
            .retry_wait
            .saturating_mul(2u32.saturating_pow(attempt))
            .min(MAX_WAIT);
        jittered(backoff)
    }
}

/// Builds the client on a thread of its own, since building a blocking client on a thread
/// that runs an async runtime panics.
fn build_client(timeout: Duration) -> Result<Client, Setup> {
    let build = || {
        Client::builder()
            .timeout(timeout)
            .redirect(Policy::none())
            .build()
    };
    std::thread::scope(|scope| scope.spawn(build).join())
        .map_err(|_panic| Setup::ClientPanicked)?
        .map_err(Setup::Client)
}

impl fmt::Debug for Connection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Connection")
            .field("endpoint", &self.endpoint.as_str())
            .field("key", &self.key)
            .field("retries", &self.retries)
            .field("retry_wait", &self.retry_wait)
            .finish_non_exhaustive()
    }
}

fn of_transport(error: reqwest::Error) -> Failed {
    let retry = !(error.is_builder() || error.is_redirect());
    let failure = if error.is_timeout() {
        Transport::Timeout(error)
    } else {
        Transport::Request(error)
    };
    Failed {
        failure,
        retry,
        retry_after: None,
    }
}

fn read_answer(response: Response) -> Result<Vec<u8>, Failed> {
    let too_large = || Failed {
        failure: Transport::TooLarge,
        retry: false,
        retry_after: None,
    };
    if response
        .content_length()
        .is_some_and(|len| len > MAX_ANSWER_BYTES)
    {
        return Err(too_large());
    }
    let mut answer = Vec::new();
    response
        .take(MAX_ANSWER_BYTES.saturating_add(1))
        .read_to_end(&mut answer)
        .map_err(|error| Failed {
            failure: Transport::Read(error),
            retry: true,
            retry_after: None,
        })?;
    if u64::try_from(answer.len()).map_or(true, |len| len > MAX_ANSWER_BYTES) {
        return Err(too_large());
    }
    Ok(answer)
}

fn retry_after(response: &Response) -> Option<Duration> {
    let value = response.headers().get(RETRY_AFTER)?.to_str().ok()?;
    value.trim().parse::<u64>().ok().map(Duration::from_secs)
}

fn jittered(wait: Duration) -> Duration {
    let random = RandomState::new().build_hasher().finish();
    let thousandths = u32::try_from(random % 1_000).unwrap_or(0);
    let fraction = 0.5 + f64::from(thousandths) / 2_000.0;
    wait.mul_f64(fraction)
}

#[expect(
    clippy::disallowed_methods,
    reason = "the plug-ins block by contract and run off the async runtime, so a retry waits on its own thread"
)]
fn pause(wait: Duration) {
    std::thread::sleep(wait);
}
