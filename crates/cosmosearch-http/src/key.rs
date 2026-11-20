use std::fmt;

/// A secret key for an endpoint, sent as a bearer token. Its `Debug` output hides it, and it
/// never appears in an error.
#[derive(Clone, PartialEq, Eq)]
pub struct ApiKey(String);

impl ApiKey {
    /// Wraps `key`.
    pub fn new(key: impl Into<String>) -> Self {
        ApiKey(key.into())
    }

    pub(crate) fn expose(&self) -> &str {
        &self.0
    }

    pub(crate) fn is_valid(&self) -> bool {
        !self.0.is_empty() && self.0.bytes().all(|byte| byte.is_ascii_graphic())
    }
}

impl fmt::Debug for ApiKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ApiKey(<redacted>)")
    }
}
