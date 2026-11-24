use std::collections::BTreeSet;
use std::ops::{Bound, RangeBounds};

use crate::{AttributeValue, Attributes, Number, Timestamp};

/// Which files a search may look at, judged by their host attributes and modified time.
///
/// Every condition must hold. A filter with no conditions lets every file through.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Filter {
    conditions: Vec<(Target, Condition)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Target {
    Attribute(String),
    Modified,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Condition {
    AnyOf(BTreeSet<String>),
    Numbers(Bound<Number>, Bound<Number>),
    Times(Bound<Timestamp>, Bound<Timestamp>),
    Never,
}

impl Filter {
    /// A filter every file passes.
    pub fn new() -> Self {
        Filter::default()
    }

    /// Also requires attribute `name` to equal `value`, or as a list to contain it.
    #[must_use]
    pub fn equals(self, name: impl Into<String>, value: &str) -> Self {
        self.any_of(name, &[value])
    }

    /// Also requires attribute `name` to equal any of `values`, or as a list to contain any of
    /// them. With no values, no file passes.
    #[must_use]
    pub fn any_of(self, name: impl Into<String>, values: &[&str]) -> Self {
        let values = values.iter().map(|value| (*value).to_owned()).collect();
        self.with(Target::Attribute(name.into()), Condition::AnyOf(values))
    }

    /// Also requires attribute `name` to be a number within `range`, such as `10.0..=20.0`.
    /// With a bound that is NaN or infinite, no file passes.
    #[must_use]
    pub fn numbers(self, name: impl Into<String>, range: impl RangeBounds<f64>) -> Self {
        let bound = |bound: Bound<&f64>| match bound {
            Bound::Included(value) => Number::new(*value).map(Bound::Included),
            Bound::Excluded(value) => Number::new(*value).map(Bound::Excluded),
            Bound::Unbounded => Some(Bound::Unbounded),
        };
        let condition = match (bound(range.start_bound()), bound(range.end_bound())) {
            (Some(start), Some(end)) => Condition::Numbers(start, end),
            _ => Condition::Never,
        };
        self.with(Target::Attribute(name.into()), condition)
    }

    /// Also requires attribute `name` to be a time within `range`.
    #[must_use]
    pub fn times(self, name: impl Into<String>, range: impl RangeBounds<Timestamp>) -> Self {
        let condition = Condition::Times(range.start_bound().cloned(), range.end_bound().cloned());
        self.with(Target::Attribute(name.into()), condition)
    }

    /// Also requires the file's modified time to be within `range`, such as
    /// `since..` for files changed since a time. A file whose modified time is not known
    /// does not pass.
    #[must_use]
    pub fn modified(self, range: impl RangeBounds<Timestamp>) -> Self {
        let condition = Condition::Times(range.start_bound().cloned(), range.end_bound().cloned());
        self.with(Target::Modified, condition)
    }

    fn with(mut self, target: Target, condition: Condition) -> Self {
        self.conditions.push((target, condition));
        self
    }

    /// Whether this filter lets every file through.
    pub fn is_empty(&self) -> bool {
        self.conditions.is_empty()
    }

    /// The conditions, each encoded and then sorted, so that filters that mean the same give
    /// the same parts.
    pub(crate) fn identity_parts(&self) -> Vec<Vec<u8>> {
        let mut parts: Vec<Vec<u8>> = self
            .conditions
            .iter()
            .map(|(target, condition)| {
                let mut part = Vec::new();
                match target {
                    Target::Attribute(name) => {
                        part.push(0);
                        push_text(&mut part, name);
                    }
                    Target::Modified => part.push(1),
                }
                match condition {
                    Condition::AnyOf(values) => {
                        part.push(0);
                        push_len(&mut part, values.len());
                        for value in values {
                            push_text(&mut part, value);
                        }
                    }
                    Condition::Numbers(start, end) => {
                        part.push(1);
                        push_bound(&mut part, *start, |n| n.get().to_bits().to_le_bytes());
                        push_bound(&mut part, *end, |n| n.get().to_bits().to_le_bytes());
                    }
                    Condition::Times(start, end) => {
                        part.push(2);
                        push_bound(&mut part, *start, |t| t.millis().to_le_bytes());
                        push_bound(&mut part, *end, |t| t.millis().to_le_bytes());
                    }
                    Condition::Never => part.push(3),
                }
                part
            })
            .collect();
        parts.sort_unstable();
        parts
    }

    /// Whether a file with these attributes and this modified time passes.
    pub fn matches(&self, attributes: &Attributes, modified: Option<Timestamp>) -> bool {
        self.conditions
            .iter()
            .all(|(target, condition)| match target {
                Target::Attribute(name) => attributes
                    .get(name)
                    .is_some_and(|value| holds(condition, value)),
                Target::Modified => modified.is_some_and(|time| holds_time(condition, time)),
            })
    }
}

fn holds(condition: &Condition, value: &AttributeValue) -> bool {
    match (condition, value) {
        (Condition::AnyOf(values), AttributeValue::One(one)) => values.contains(one),
        (Condition::AnyOf(values), AttributeValue::Many(many)) => {
            many.iter().any(|item| values.contains(item))
        }
        (Condition::Numbers(start, end), AttributeValue::Number(number)) => {
            (*start, *end).contains(number)
        }
        (Condition::Times(..), AttributeValue::Time(time)) => holds_time(condition, *time),
        _ => false,
    }
}

fn holds_time(condition: &Condition, time: Timestamp) -> bool {
    match condition {
        Condition::Times(start, end) => (*start, *end).contains(&time),
        Condition::AnyOf(_) | Condition::Numbers(..) | Condition::Never => false,
    }
}

fn push_len(part: &mut Vec<u8>, len: usize) {
    part.extend_from_slice(&u64::try_from(len).unwrap_or(u64::MAX).to_le_bytes());
}

fn push_text(part: &mut Vec<u8>, text: &str) {
    push_len(part, text.len());
    part.extend_from_slice(text.as_bytes());
}

fn push_bound<T: Copy>(part: &mut Vec<u8>, bound: Bound<T>, bytes: impl Fn(T) -> [u8; 8]) {
    match bound {
        Bound::Included(value) => {
            part.push(0);
            part.extend_from_slice(&bytes(value));
        }
        Bound::Excluded(value) => {
            part.push(1);
            part.extend_from_slice(&bytes(value));
        }
        Bound::Unbounded => part.push(2),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn number(value: f64) -> AttributeValue {
        AttributeValue::Number(Number::new(value).unwrap())
    }

    fn time(millis: i64) -> Timestamp {
        Timestamp::from_millis(millis)
    }

    fn attributes() -> Attributes {
        Attributes::from([
            ("owner".to_owned(), AttributeValue::One("legal".to_owned())),
            (
                "labels".to_owned(),
                AttributeValue::Many(vec!["draft".to_owned(), "finance".to_owned()]),
            ),
            ("amount".to_owned(), number(250.0)),
            ("signed".to_owned(), AttributeValue::Time(time(1_000))),
        ])
    }

    #[test]
    fn checks_single_values_and_lists_against_any_of_several_values() {
        let cases = [
            (Filter::new(), true),
            (Filter::new().equals("owner", "legal"), true),
            (Filter::new().equals("owner", "sales"), false),
            (Filter::new().any_of("owner", &["sales", "legal"]), true),
            (Filter::new().any_of("labels", &["finance", "hr"]), true),
            (Filter::new().any_of("labels", &["hr", "ops"]), false),
            (Filter::new().any_of("owner", &[]), false),
            (Filter::new().equals("missing", "legal"), false),
            (Filter::new().equals("amount", "250"), false),
            (
                Filter::new()
                    .equals("owner", "legal")
                    .equals("labels", "draft"),
                true,
            ),
            (
                Filter::new()
                    .equals("owner", "legal")
                    .equals("labels", "hr"),
                false,
            ),
        ];
        for (filter, expected) in cases {
            assert_eq!(filter.matches(&attributes(), None), expected, "{filter:?}");
        }
    }

    #[test]
    fn checks_numbers_and_times_against_ranges() {
        let cases = [
            (Filter::new().numbers("amount", 100.0..=250.0), true),
            (Filter::new().numbers("amount", 100.0..250.0), false),
            (Filter::new().numbers("amount", 250.0..), true),
            (Filter::new().numbers("amount", ..f64::NAN), false),
            (Filter::new().numbers("amount", f64::NEG_INFINITY..), false),
            (Filter::new().numbers("owner", ..), false),
            (Filter::new().numbers("signed", ..), false),
            (Filter::new().times("signed", time(1_000)..), true),
            (Filter::new().times("signed", ..time(1_000)), false),
            (Filter::new().times("amount", ..), false),
        ];
        for (filter, expected) in cases {
            assert_eq!(filter.matches(&attributes(), None), expected, "{filter:?}");
        }
    }

    #[test]
    fn checks_the_modified_time_and_fails_a_file_without_one() {
        let filter = Filter::new().modified(time(500)..time(2_000));
        assert!(filter.matches(&Attributes::new(), Some(time(500))));
        assert!(!filter.matches(&Attributes::new(), Some(time(2_000))));
        assert!(!filter.matches(&Attributes::new(), None));
        assert!(!Filter::new().modified(..).matches(&Attributes::new(), None));
    }

    #[test]
    fn a_file_without_attributes_passes_only_an_empty_filter() {
        assert!(Filter::new().matches(&Attributes::new(), None));
        assert!(
            !Filter::new()
                .equals("owner", "legal")
                .matches(&Attributes::new(), None)
        );
    }

    #[test]
    fn filters_that_mean_the_same_give_the_same_parts() {
        let one = Filter::new()
            .any_of("owner", &["b", "a"])
            .numbers("amount", 1.0..2.0)
            .modified(time(5)..);
        let other = Filter::new()
            .modified(time(5)..)
            .numbers("amount", 1.0..2.0)
            .any_of("owner", &["a", "b"]);
        assert_eq!(one.identity_parts(), other.identity_parts());
        let cases = [
            Filter::new().numbers("amount", 1.0..=2.0),
            Filter::new().times("amount", time(5)..),
            Filter::new().times("modified", time(5)..),
            Filter::new().modified(time(5)..),
            Filter::new().modified(time(6)..),
        ];
        for (i, a) in cases.iter().enumerate() {
            for b in &cases[i + 1..] {
                assert_ne!(a.identity_parts(), b.identity_parts(), "{a:?} {b:?}");
            }
        }
    }
}
