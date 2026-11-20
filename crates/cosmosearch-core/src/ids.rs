use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

const MAX_ID_LEN: usize = 128;

fn validate_id(kind: &'static str, value: &str) -> Result<()> {
    let reason = if value.is_empty() {
        Some("must not be empty")
    } else if value.len() > MAX_ID_LEN {
        Some("must be at most 128 characters")
    } else if value.starts_with('.') {
        Some("must not start with `.`")
    } else if !value
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
    {
        Some("may contain only ASCII letters, digits, `.`, `_` and `-`")
    } else {
        None
    };
    match reason {
        Some(reason) => Err(Error::InvalidId {
            kind,
            value: value.to_owned(),
            reason,
        }),
        None => Ok(()),
    }
}

macro_rules! id_type {
    ($(#[$doc:meta])* $name:ident, $kind:literal) => {
        $(#[$doc])*
        #[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
        #[serde(try_from = "String", into = "String")]
        pub struct $name(String);

        impl $name {
            /// Creates the id, checking that it is a valid path segment: 1 to 128 ASCII
            /// letters, digits, `.`, `_` or `-`, not starting with `.`.
            ///
            /// # Errors
            ///
            /// Returns [`Error::InvalidId`] for a rejected value.
            pub fn new(value: impl Into<String>) -> Result<Self> {
                let value = value.into();
                validate_id($kind, &value)?;
                Ok($name(value))
            }

            /// The id as a string.
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }

        impl FromStr for $name {
            type Err = Error;

            fn from_str(s: &str) -> Result<Self> {
                $name::new(s)
            }
        }

        impl TryFrom<String> for $name {
            type Error = Error;

            fn try_from(value: String) -> Result<Self> {
                $name::new(value)
            }
        }

        impl From<$name> for String {
            fn from(id: $name) -> Self {
                id.0
            }
        }
    };
}

id_type!(
    /// Identifies a shard within one index.
    ShardId,
    "shard id"
);

id_type!(
    /// Identifies a segment within one index.
    SegmentId,
    "segment id"
);

id_type!(
    /// Identifies one index across its life.
    IndexId,
    "index id"
);

id_type!(
    /// Identifies one commit attempt.
    CommitId,
    "commit id"
);

id_type!(
    /// The short id of a full embedding identity.
    ModelId,
    "model id"
);

id_type!(
    /// The name of one file of a segment, relative to its segment directory.
    SegmentFileName,
    "segment file name"
);

id_type!(
    /// Names the kind of index a segment was built for, such as `bm25`.
    IndexKind,
    "index kind"
);

id_type!(
    /// Names an extractor and the output it produces, such as `pdf-1`.
    ExtractorProfile,
    "extractor profile"
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_path_safe_ids() {
        for value in ["contracts-2019", "a", "shard_01.v2", &"x".repeat(128)] {
            assert_eq!(ShardId::new(value).unwrap().as_str(), value);
        }
    }

    #[test]
    fn rejects_unsafe_ids() {
        for value in [
            "",
            ".hidden",
            "..",
            "a/b",
            "a b",
            "ü",
            "a#b",
            &"x".repeat(129),
        ] {
            SegmentId::new(value).unwrap_err();
        }
    }

    #[test]
    fn error_names_the_kind() {
        let err = ShardId::new("a/b").unwrap_err().to_string();
        assert!(err.starts_with("invalid shard id `a/b`"), "{err}");
    }

    #[test]
    fn serde_validates() {
        let id: ShardId = serde_json::from_str("\"s1\"").unwrap();
        assert_eq!(serde_json::to_string(&id).unwrap(), "\"s1\"");
        serde_json::from_str::<ShardId>("\"a/b\"").unwrap_err();
    }
}
