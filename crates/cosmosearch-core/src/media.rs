use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

const MAX_PART_LEN: usize = 127;

/// A file's format as a media type, such as `application/pdf`, in lower case and without
/// parameters.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct MediaType(String);

impl MediaType {
    /// Creates a media type from a value such as `Application/PDF` or
    /// `text/plain; charset=utf-8`, dropping parameters and lowering the case.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidMediaType`] unless the value is a type and a subtype joined by
    /// `/`, each of 1 to 127 ASCII letters, digits or `!#$&-^_.+`.
    pub fn new(value: impl AsRef<str>) -> Result<Self> {
        let raw = value.as_ref();
        let essence = raw.split(';').next().unwrap_or_default().trim();
        let valid = essence
            .split_once('/')
            .is_some_and(|(kind, subtype)| valid_part(kind) && valid_part(subtype));
        if valid {
            Ok(MediaType(essence.to_ascii_lowercase()))
        } else {
            Err(Error::InvalidMediaType(raw.to_owned()))
        }
    }

    /// The media type as a string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

fn valid_part(part: &str) -> bool {
    (1..=MAX_PART_LEN).contains(&part.len())
        && part.bytes().all(|b| {
            b.is_ascii_alphanumeric()
                || matches!(
                    b,
                    b'!' | b'#' | b'$' | b'&' | b'-' | b'^' | b'_' | b'.' | b'+'
                )
        })
}

impl fmt::Display for MediaType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl FromStr for MediaType {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        MediaType::new(s)
    }
}

impl TryFrom<String> for MediaType {
    type Error = Error;

    fn try_from(value: String) -> Result<Self> {
        MediaType::new(value)
    }
}

impl From<MediaType> for String {
    fn from(value: MediaType) -> Self {
        value.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_the_essence_in_lower_case() {
        let cases = [
            ("application/pdf", "application/pdf"),
            ("Application/PDF", "application/pdf"),
            ("text/plain; charset=utf-8", "text/plain"),
            (" text/markdown ", "text/markdown"),
            (
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ),
        ];
        for (input, expected) in cases {
            assert_eq!(MediaType::new(input).unwrap().as_str(), expected, "{input}");
        }
    }

    #[test]
    fn rejects_values_that_are_not_a_type_and_subtype() {
        for input in [
            "",
            "pdf",
            "/pdf",
            "application/",
            "a/b/c",
            "text/pl ain",
            "tëxt/plain",
        ] {
            assert!(
                matches!(MediaType::new(input), Err(Error::InvalidMediaType(_))),
                "{input}"
            );
        }
    }
}
