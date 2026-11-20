use std::fmt;

use uuid::Uuid;

const SUFFIX: &str = ".pack";

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct PackName {
    created_ms: u64,
    random: String,
}

impl PackName {
    pub(crate) fn new(created_ms: u64) -> Self {
        PackName {
            created_ms,
            random: Uuid::new_v4().simple().to_string(),
        }
    }

    pub(crate) fn parse(name: &str) -> Option<Self> {
        let (created, random) = name.strip_suffix(SUFFIX)?.split_once('-')?;
        let valid = |text: &str, allowed: fn(&u8) -> bool| {
            !text.is_empty() && text.as_bytes().iter().all(allowed)
        };
        if !valid(created, u8::is_ascii_digit) || !valid(random, u8::is_ascii_hexdigit) {
            return None;
        }
        Some(PackName {
            created_ms: created.parse().ok()?,
            random: random.to_owned(),
        })
    }

    pub(crate) fn created_ms(&self) -> u64 {
        self.created_ms
    }
}

impl fmt::Display for PackName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}-{}{SUFFIX}", self.created_ms, self.random)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_name_parses_back() {
        let name = PackName::new(1_700_000_000_123);
        assert_eq!(PackName::parse(&name.to_string()), Some(name.clone()));
        assert!(name.to_string().starts_with("1700000000123-"));
    }

    #[test]
    fn refuses_other_names() {
        for name in [
            "",
            ".pack",
            "123.pack",
            "-abc.pack",
            "123-.pack",
            "123-abc",
            "12a-abc.pack",
            "123-xyz.pack",
            "99999999999999999999999-abc.pack",
        ] {
            assert_eq!(PackName::parse(name), None, "{name}");
        }
    }
}
