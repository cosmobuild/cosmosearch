use std::fmt;

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

const MAX_VERSION: u64 = 999_999_999_999;

/// The version of a manifest, starting at 1, printed as `v000000000123.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "u64", into = "u64")]
pub struct ManifestVersion(u64);

impl ManifestVersion {
    /// The version of an index's first manifest.
    pub const FIRST: ManifestVersion = ManifestVersion(1);

    /// Creates a version.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidVersion`] for zero or a value over 12 digits.
    pub fn new(value: u64) -> Result<Self> {
        if (1..=MAX_VERSION).contains(&value) {
            Ok(ManifestVersion(value))
        } else {
            Err(Error::InvalidVersion(value.to_string()))
        }
    }

    /// The version number.
    pub fn get(self) -> u64 {
        self.0
    }

    /// The version after this one.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidVersion`] if there is no next version.
    pub fn next(self) -> Result<Self> {
        ManifestVersion::new(self.0.saturating_add(1))
    }

    /// The version before this one, or `None` for the first.
    pub fn previous(self) -> Option<Self> {
        ManifestVersion::new(self.0.saturating_sub(1)).ok()
    }

    /// The manifest's file name, such as `v000000000123.json`.
    pub fn file_name(self) -> String {
        format!("v{:012}.json", self.0)
    }

    /// Parses a file name made by [`ManifestVersion::file_name`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidVersion`] for any other name.
    pub fn from_file_name(name: &str) -> Result<Self> {
        let invalid = || Error::InvalidVersion(name.to_owned());
        let digits = name
            .strip_prefix('v')
            .and_then(|rest| rest.strip_suffix(".json"))
            .filter(|digits| digits.len() == 12 && digits.bytes().all(|b| b.is_ascii_digit()))
            .ok_or_else(invalid)?;
        let value = digits
            .parse()
            .map_err(|_err: std::num::ParseIntError| invalid())?;
        ManifestVersion::new(value)
    }
}

impl fmt::Display for ManifestVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl TryFrom<u64> for ManifestVersion {
    type Error = Error;

    fn try_from(value: u64) -> Result<Self> {
        ManifestVersion::new(value)
    }
}

impl From<ManifestVersion> for u64 {
    fn from(version: ManifestVersion) -> Self {
        version.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_names_round_trip_and_sort() {
        let low = ManifestVersion::new(9).unwrap();
        let high = low.next().unwrap();
        assert_eq!(low.file_name(), "v000000000009.json");
        assert_eq!(
            ManifestVersion::from_file_name(&high.file_name()).unwrap(),
            high
        );
        assert!(low.file_name() < high.file_name());
    }

    #[test]
    fn rejects_bad_values_and_names() {
        ManifestVersion::new(0).unwrap_err();
        ManifestVersion::new(MAX_VERSION)
            .unwrap()
            .next()
            .unwrap_err();
        for name in [
            "",
            "v1.json",
            "v000000000000.json",
            "v00000000001a.json",
            "x000000000001.json",
            "v000000000001.jsn",
            "v0000000000001.json",
        ] {
            ManifestVersion::from_file_name(name).unwrap_err();
        }
    }

    #[test]
    fn json_validates() {
        serde_json::from_str::<ManifestVersion>("0").unwrap_err();
        assert_eq!(serde_json::to_string(&ManifestVersion::FIRST).unwrap(), "1");
    }
}
