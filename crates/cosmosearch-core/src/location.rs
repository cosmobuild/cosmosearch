use std::fmt::{self, Write as _};
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

/// The backend a [`Location`] lives on, printed as the location's URL scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum Scheme {
    /// S3 or an S3-compatible store; the authority is the bucket.
    S3,
    /// The local filesystem; the key is an absolute path without its leading `/`.
    File,
}

impl Scheme {
    const ALL: [Scheme; 2] = [Scheme::S3, Scheme::File];

    /// The scheme as printed, without `://`.
    pub fn as_str(self) -> &'static str {
        match self {
            Scheme::S3 => "s3",
            Scheme::File => "file",
        }
    }

    fn check_authority(self, authority: &str) -> Option<&'static str> {
        match self {
            Scheme::S3 if authority.is_empty() => Some("the bucket is empty"),
            Scheme::S3 if !authority.bytes().all(is_authority_byte) => {
                Some("the bucket has a character that is not allowed")
            }
            Scheme::File if !authority.is_empty() => Some("a file location has no authority"),
            Scheme::S3 | Scheme::File => None,
        }
    }
}

fn is_authority_byte(b: u8) -> bool {
    b.is_ascii_graphic() && !matches!(b, b'/' | b'#' | b'%' | b'@')
}

/// An object or prefix on some backend: `scheme://authority/key`. An empty key is the root.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct Location {
    scheme: Scheme,
    authority: String,
    key: String,
}

impl Location {
    /// Creates a location.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidLocation`] if the authority does not suit the scheme.
    pub fn new(
        scheme: Scheme,
        authority: impl Into<String>,
        key: impl Into<String>,
    ) -> Result<Self> {
        let (authority, key) = (authority.into(), key.into());
        if let Some(reason) = scheme.check_authority(&authority) {
            return Err(Error::InvalidLocation {
                input: format!("{}://{authority}/{key}", scheme.as_str()),
                reason,
            });
        }
        Ok(Location {
            scheme,
            authority,
            key,
        })
    }

    /// The backend this location lives on.
    pub fn scheme(&self) -> Scheme {
        self.scheme
    }

    /// The bucket for S3, empty for the local filesystem.
    pub fn authority(&self) -> &str {
        &self.authority
    }

    /// The object key or prefix, unescaped.
    pub fn key(&self) -> &str {
        &self.key
    }

    /// The key without trailing slashes, as used for a prefix.
    pub fn prefix_key(&self) -> &str {
        self.key.strip_suffix('/').unwrap_or(&self.key)
    }

    /// Whether `other` is this prefix or sits inside it.
    pub fn contains(&self, other: &Location) -> bool {
        let prefix = self.prefix_key();
        self.scheme == other.scheme
            && self.authority == other.authority
            && (prefix.is_empty()
                || other
                    .prefix_key()
                    .strip_prefix(prefix)
                    .is_some_and(|rest| rest.is_empty() || rest.starts_with('/')))
    }

    /// Whether both name the same prefix, ignoring a trailing slash.
    pub fn same_prefix(&self, other: &Location) -> bool {
        self.scheme == other.scheme
            && self.authority == other.authority
            && self.prefix_key() == other.prefix_key()
    }
}

impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}://{}/", self.scheme.as_str(), self.authority)?;
        for c in self.key.chars() {
            if c == '%' || c == '#' || c.is_control() {
                let mut buf = [0u8; 4];
                for byte in c.encode_utf8(&mut buf).bytes() {
                    write!(f, "%{byte:02X}")?;
                }
            } else {
                f.write_char(c)?;
            }
        }
        Ok(())
    }
}

impl FromStr for Location {
    type Err = Error;

    fn from_str(input: &str) -> Result<Self> {
        let invalid = |reason| Error::InvalidLocation {
            input: input.to_owned(),
            reason,
        };
        let (name, rest) = input
            .split_once("://")
            .ok_or_else(|| invalid("missing `://`"))?;
        let scheme = Scheme::ALL
            .into_iter()
            .find(|scheme| scheme.as_str() == name)
            .ok_or_else(|| invalid("unknown scheme"))?;
        let (authority, key) = rest
            .split_once('/')
            .ok_or_else(|| invalid("missing `/` after the authority"))?;
        let key = percent_decode(key).ok_or_else(|| invalid("bad `%` escape in the key"))?;
        if let Some(reason) = scheme.check_authority(authority) {
            return Err(invalid(reason));
        }
        Ok(Location {
            scheme,
            authority: authority.to_owned(),
            key,
        })
    }
}

impl TryFrom<String> for Location {
    type Error = Error;

    fn try_from(value: String) -> Result<Self> {
        value.parse()
    }
}

impl From<Location> for String {
    fn from(location: Location) -> Self {
        location.to_string()
    }
}

fn percent_decode(s: &str) -> Option<String> {
    let mut out = Vec::with_capacity(s.len());
    let mut bytes = s.bytes();
    while let Some(b) = bytes.next() {
        if b == b'%' {
            let hi = char::from(bytes.next()?).to_digit(16)?;
            let lo = char::from(bytes.next()?).to_digit(16)?;
            out.push(u8::try_from(hi.checked_mul(16)?.checked_add(lo)?).ok()?);
        } else {
            out.push(b);
        }
    }
    String::from_utf8(out).ok()
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn parses_each_scheme() {
        let s3: Location = "s3://example-bucket/docs/a.txt".parse().unwrap();
        assert_eq!(
            (s3.scheme(), s3.authority(), s3.key()),
            (Scheme::S3, "example-bucket", "docs/a.txt")
        );
        let file: Location = "file:///srv/docs/a.txt".parse().unwrap();
        assert_eq!(
            (file.scheme(), file.authority(), file.key()),
            (Scheme::File, "", "srv/docs/a.txt")
        );
    }

    #[test]
    fn allows_an_empty_key_for_a_root_prefix() {
        let root: Location = "s3://example-bucket/".parse().unwrap();
        assert_eq!(root.key(), "");
        assert_eq!(root.to_string(), "s3://example-bucket/");
    }

    #[test]
    fn a_trailing_slash_names_the_same_prefix() {
        let check = |a: &str, b: &str, same: bool| {
            let (a, b): (Location, Location) = (a.parse().unwrap(), b.parse().unwrap());
            assert_eq!(a.same_prefix(&b), same);
        };
        check("s3://b/docs/", "s3://b/docs", true);
        check("s3://b/docs", "s3://b/docs2", false);
        check("s3://b/docs", "s3://c/docs", false);
        check("s3://b/docs", "file:///docs", false);
    }

    #[test]
    fn containment_respects_path_segments() {
        let check = |outer: &str, inner: &str, expected: bool| {
            let (outer, inner): (Location, Location) =
                (outer.parse().unwrap(), inner.parse().unwrap());
            assert_eq!(outer.contains(&inner), expected);
        };
        check("s3://b/", "s3://b/docs/", true);
        check("s3://b/docs", "s3://b/docs/", true);
        check("s3://b/docs/", "s3://b/docs/a", true);
        check("s3://b/docs", "s3://b/docs2", false);
        check("s3://b/docs", "s3://b/", false);
        check("s3://b/", "s3://c/docs", false);
    }

    #[test]
    fn rejects_malformed_locations() {
        for input in [
            "",
            "example-bucket/k",
            "s3:/example-bucket/k",
            "ftp://example-bucket/k",
            "s3://example-bucket",
            "s3:///k",
            "s3://a#b/k",
            "s3://a b/k",
            "file://host/k",
            "s3://example-bucket/k%2",
            "s3://example-bucket/k%zz",
            "s3://example-bucket/k%FF",
        ] {
            input.parse::<Location>().unwrap_err();
        }
    }

    #[test]
    fn new_checks_the_authority() {
        Location::new(Scheme::S3, "", "k").unwrap_err();
        Location::new(Scheme::File, "host", "k").unwrap_err();
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        #[test]
        fn any_key_round_trips(key in any::<String>()) {
            let location = Location::new(Scheme::S3, "example-bucket", key).unwrap();
            let printed = location.to_string();
            let parsed: Location = printed.parse().unwrap();
            prop_assert_eq!(&parsed, &location);
            prop_assert_eq!(parsed.to_string(), printed);
        }
    }
}
