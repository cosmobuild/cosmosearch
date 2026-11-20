use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

/// The entity tag of one version of an object, kept without its quotes.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct ETag(String);

impl ETag {
    /// Creates an ETag from its bare or quoted form.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidETag`] for an empty value, or one with quotes inside,
    /// whitespace, control characters or `#`.
    pub fn new(value: impl AsRef<str>) -> Result<Self> {
        let raw = value.as_ref();
        let bare = raw
            .strip_prefix('"')
            .and_then(|v| v.strip_suffix('"'))
            .unwrap_or(raw);
        let valid = !bare.is_empty()
            && bare
                .chars()
                .all(|c| c.is_ascii_graphic() && c != '"' && c != '#');
        if valid {
            Ok(ETag(bare.to_owned()))
        } else {
            Err(Error::InvalidETag(raw.to_owned()))
        }
    }

    /// The bare value, without quotes.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The quoted value, as used in `ETag`, `If-Match` and `If-None-Match` headers.
    pub fn quoted(&self) -> String {
        format!("\"{}\"", self.0)
    }
}

impl fmt::Display for ETag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl FromStr for ETag {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        ETag::new(s)
    }
}

impl TryFrom<String> for ETag {
    type Error = Error;

    fn try_from(value: String) -> Result<Self> {
        ETag::new(value)
    }
}

impl From<ETag> for String {
    fn from(etag: ETag) -> Self {
        etag.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_one_pair_of_quotes() {
        let etag = ETag::new("\"9f3a\"").unwrap();
        assert_eq!(etag.as_str(), "9f3a");
        assert_eq!(etag.quoted(), "\"9f3a\"");
        assert_eq!(ETag::new("9f3a").unwrap(), etag);
    }

    #[test]
    fn accepts_store_formats() {
        for value in [
            "\"d41d8cd98f00b204e9800998ecf8427e-12\"",
            "7c132f6-65fb5446c8fee-2",
        ] {
            ETag::new(value).unwrap();
        }
    }

    #[test]
    fn rejects_invalid_values() {
        for value in [
            "",
            "\"\"",
            "a b",
            "a\"b",
            "W/\"weak\"",
            "a#b",
            "\"unclosed",
            "tab\t",
        ] {
            ETag::new(value).unwrap_err();
        }
    }

    #[test]
    fn serializes_as_a_bare_string() {
        let etag = ETag::new("\"abc\"").unwrap();
        assert_eq!(serde_json::to_string(&etag).unwrap(), "\"abc\"");
        let back: ETag = serde_json::from_str("\"abc\"").unwrap();
        assert_eq!(back, etag);
        serde_json::from_str::<ETag>("\"a b\"").unwrap_err();
    }
}
