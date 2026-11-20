use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

/// An index level, from the file table every file gets to the vector index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Level {
    /// The file table: key, size, ETag, status and host attributes.
    L0,
    /// The keyword index, such as BM25.
    L1,
    /// The vector index, such as IVF.
    L2,
}

impl Level {
    /// All levels, lowest first.
    pub const ALL: [Level; 3] = [Level::L0, Level::L1, Level::L2];

    /// The level's name, `L0` to `L2`.
    pub fn as_str(self) -> &'static str {
        match self {
            Level::L0 => "L0",
            Level::L1 => "L1",
            Level::L2 => "L2",
        }
    }
}

impl fmt::Display for Level {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Level {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        Level::ALL
            .into_iter()
            .find(|level| level.as_str() == s)
            .ok_or_else(|| Error::InvalidLevel(s.to_owned()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_text_and_json() {
        for level in Level::ALL {
            assert_eq!(level.to_string().parse::<Level>().unwrap(), level);
            let json = serde_json::to_string(&level).unwrap();
            assert_eq!(json, format!("\"{level}\""));
            assert_eq!(serde_json::from_str::<Level>(&json).unwrap(), level);
        }
    }

    #[test]
    fn orders_lowest_first() {
        assert!(Level::L0 < Level::L1 && Level::L1 < Level::L2);
    }

    #[test]
    fn rejects_unknown_names() {
        for name in ["", "l0", "L3", "L1 "] {
            name.parse::<Level>().unwrap_err();
        }
    }
}
