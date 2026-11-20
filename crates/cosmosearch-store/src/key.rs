use std::fmt;

use crate::{Error, Result};

/// The key of an object within a backend. Object keys are not OS paths.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ObjectKey(String);

impl ObjectKey {
    /// Creates a key.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidKey`] for an empty key.
    pub fn new(key: impl Into<String>) -> Result<Self> {
        let key = key.into();
        if key.is_empty() {
            return Err(Error::InvalidKey {
                key,
                reason: "a key must not be empty",
            });
        }
        Ok(ObjectKey(key))
    }

    /// The key as a string.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The key of `name` under this key, treated as a directory.
    #[must_use]
    pub fn child(&self, name: &str) -> ObjectKey {
        ObjectKey(format!("{}/{name}", self.0))
    }

    /// Whether this key sits under `prefix`, treated as a directory.
    pub fn is_under(&self, prefix: &ObjectKey) -> bool {
        self.0
            .strip_prefix(prefix.as_str())
            .is_some_and(|rest| rest.starts_with('/'))
    }

    /// The last `/`-separated part of the key.
    pub fn file_name(&self) -> &str {
        self.0.rsplit('/').next().unwrap_or(&self.0)
    }
}

impl fmt::Display for ObjectKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
