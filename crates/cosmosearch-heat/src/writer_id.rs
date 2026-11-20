use std::fmt;
use std::time::SystemTime;

use cosmosearch_store::new_commit_id;

use crate::Result;
use crate::clock::system_time;

const MAX_RANDOM_LEN: usize = 64;

/// The id a process writes its heat snapshots under: the Unix millisecond it was taken, a
/// hyphen and a random part. A process takes a new one on every start, so a restart never
/// reuses a sequence number.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WriterId {
    born_ms: u64,
    random: String,
}

impl WriterId {
    pub(crate) fn new(born_ms: u64) -> Result<Self> {
        Ok(WriterId {
            born_ms,
            random: new_commit_id()?.as_str().to_owned(),
        })
    }

    /// Reads an id in its printed form, refusing any other spelling of it.
    pub(crate) fn parse(text: &str) -> Option<Self> {
        let (born, random) = text.split_once('-')?;
        let born_ms: u64 = born.parse().ok()?;
        let canonical = born_ms.to_string() == born;
        let random_ok = !random.is_empty()
            && random.len() <= MAX_RANDOM_LEN
            && random
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
        (canonical && random_ok).then(|| WriterId {
            born_ms,
            random: random.to_owned(),
        })
    }

    /// When the id was taken.
    pub fn born(&self) -> SystemTime {
        system_time(self.born_ms)
    }

    pub(crate) fn born_ms(&self) -> u64 {
        self.born_ms
    }
}

impl fmt::Display for WriterId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}-{}", self.born_ms, self.random)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_id_parses_back_and_others_are_refused() {
        let id = WriterId::new(1_800_000_000_123).unwrap();
        assert_eq!(WriterId::parse(&id.to_string()), Some(id.clone()));
        assert!(id.to_string().starts_with("1800000000123-"));
        assert_ne!(WriterId::new(1_800_000_000_123).unwrap(), id);
        for text in [
            "",
            "-",
            "12-",
            "-ab",
            "012-ab",
            "12-AB",
            "12-xy",
            "12-ab-cd",
            "99999999999999999999999-ab",
        ] {
            assert_eq!(WriterId::parse(text), None, "{text}");
        }
    }
}
