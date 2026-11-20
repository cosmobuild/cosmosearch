use std::collections::BTreeSet;
use std::ops::Bound;

use crate::KeyRange;

/// The part of an index's data a search covers: the files whose object keys start with any
/// of its prefixes.
///
/// A search opens only the shards whose key ranges hold such keys, and counts only those
/// shards as searched.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Scope {
    prefixes: BTreeSet<String>,
}

impl Default for Scope {
    fn default() -> Self {
        Scope::all()
    }
}

impl Scope {
    /// Every file.
    pub fn all() -> Self {
        Scope::prefixes(&[""])
    }

    /// The files whose keys start with `prefix`, such as `docs/contracts/`.
    pub fn prefix(prefix: &str) -> Self {
        Scope::prefixes(&[prefix])
    }

    /// The files whose keys start with any of `prefixes`. With no prefixes, no file is in it.
    pub fn prefixes(prefixes: &[&str]) -> Self {
        let sorted: BTreeSet<&str> = prefixes.iter().copied().collect();
        let mut kept: BTreeSet<String> = BTreeSet::new();
        for prefix in sorted {
            if kept
                .last()
                .is_none_or(|last| !prefix.starts_with(last.as_str()))
            {
                kept.insert(prefix.to_owned());
            }
        }
        Scope { prefixes: kept }
    }

    /// Whether every file is in it.
    pub fn is_all(&self) -> bool {
        self.prefixes.contains("")
    }

    /// Whether the file with `key` is in it.
    pub fn contains(&self, key: &str) -> bool {
        self.prefixes
            .range::<str, _>((Bound::Unbounded, Bound::Included(key)))
            .next_back()
            .is_some_and(|prefix| key.starts_with(prefix.as_str()))
    }

    /// Whether some key in `range` is in it.
    pub fn overlaps(&self, range: &KeyRange) -> bool {
        self.prefixes
            .iter()
            .any(|prefix| prefix_overlaps(prefix, range))
    }

    /// Whether every key in `range` is in it.
    pub fn covers(&self, range: &KeyRange) -> bool {
        self.prefixes
            .iter()
            .any(|prefix| prefix_covers(prefix, range))
    }

    /// The prefixes, none starting with another, in order.
    pub(crate) fn sorted_prefixes(&self) -> impl Iterator<Item = &str> {
        self.prefixes.iter().map(String::as_str)
    }

    pub(crate) fn len(&self) -> usize {
        self.prefixes.len()
    }
}

/// Whether a key starting with `prefix` falls in `range`: the first such key at or after the
/// range's start must come before its end.
fn prefix_overlaps(prefix: &str, range: &KeyRange) -> bool {
    let start = range.start();
    let first = if start.starts_with(prefix) {
        start
    } else if start < prefix {
        prefix
    } else {
        return false;
    };
    range.end().is_none_or(|end| first < end)
}

/// Whether every key in `range` starts with `prefix`: the range starts with it and ends no
/// later than the first key after all those that do.
fn prefix_covers(prefix: &str, range: &KeyRange) -> bool {
    if !range.start().starts_with(prefix) {
        return false;
    }
    match successor(prefix) {
        None => true,
        Some(after) => range
            .end()
            .is_some_and(|end| end.as_bytes() <= after.as_slice()),
    }
}

/// The smallest byte string above every string starting with `prefix`, or `None` when no
/// string is, as for the empty prefix.
fn successor(prefix: &str) -> Option<Vec<u8>> {
    let mut bytes = prefix.as_bytes().to_vec();
    while let Some(last) = bytes.pop() {
        if let Some(next) = last.checked_add(1) {
            bytes.push(next);
            return Some(bytes);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    fn range(start: &str, end: Option<&str>) -> KeyRange {
        KeyRange::new(start, end.map(str::to_owned)).unwrap()
    }

    #[test]
    fn keeps_only_prefixes_no_other_contains() {
        let scope = Scope::prefixes(&["docs/b/", "docs/", "logs/", "docs/a/x", "logs/"]);
        let kept: Vec<&str> = scope.sorted_prefixes().collect();
        assert_eq!(kept, ["docs/", "logs/"]);
        assert_eq!(scope, Scope::prefixes(&["logs/", "docs/"]));
        assert!(Scope::prefixes(&["docs/", ""]).is_all());
        assert!(!Scope::prefix("docs/").is_all());
    }

    #[test]
    fn contains_the_keys_under_its_prefixes() {
        let scope = Scope::prefixes(&["docs/a/", "docs/c"]);
        let cases = [
            ("docs/a/1.txt", true),
            ("docs/a/", true),
            ("docs/a", false),
            ("docs/b/1.txt", false),
            ("docs/c", true),
            ("docs/cat.txt", true),
            ("docs/d.txt", false),
            ("", false),
        ];
        for (key, expected) in cases {
            assert_eq!(scope.contains(key), expected, "{key}");
        }
        assert!(Scope::all().contains(""));
        assert!(!Scope::prefixes(&[]).contains("docs/a"));
    }

    #[test]
    fn overlaps_and_covers_key_ranges() {
        let scope = Scope::prefix("docs/b/");
        let cases = [
            (range("", None), true, false),
            (range("docs/a", Some("docs/b/")), false, false),
            (range("docs/a", Some("docs/b/0")), true, false),
            (range("docs/b/", Some("docs/b0")), true, true),
            (range("docs/b/m", Some("docs/b/z")), true, true),
            (range("docs/b/m", None), true, false),
            (range("docs/b0", None), false, false),
            (range("docs/c", None), false, false),
        ];
        for (range, overlaps, covers) in cases {
            assert_eq!(scope.overlaps(&range), overlaps, "{range:?}");
            assert_eq!(scope.covers(&range), covers, "{range:?}");
        }
        assert!(Scope::all().covers(&range("", None)));
        assert!(!Scope::prefixes(&[]).overlaps(&range("", None)));
    }

    fn key() -> impl Strategy<Value = String> {
        "[ab/\u{e9}\u{10ffff}]{0,4}"
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(512))]

        #[test]
        fn range_checks_agree_with_the_keys(
            prefixes in prop::collection::vec(key(), 0..3),
            start in key(),
            end in prop::option::of(key()),
            keys in prop::collection::vec(key(), 0..24),
        ) {
            prop_assume!(end.as_ref().is_none_or(|end| start < *end));
            let range = range(&start, end.as_deref());
            let prefixes_owned = prefixes.clone();
            let prefixes: Vec<&str> = prefixes.iter().map(String::as_str).collect();
            let scope = Scope::prefixes(&prefixes);
            for key in keys.iter().chain(&prefixes_owned).chain([&start]) {
                let any = prefixes.iter().any(|prefix| key.starts_with(prefix));
                prop_assert_eq!(scope.contains(key), any);
                if range.contains(key) {
                    if any {
                        prop_assert!(scope.overlaps(&range));
                    } else {
                        prop_assert!(!scope.covers(&range));
                    }
                }
            }
            if scope.covers(&range) {
                prop_assert!(scope.overlaps(&range));
            }
        }
    }
}
