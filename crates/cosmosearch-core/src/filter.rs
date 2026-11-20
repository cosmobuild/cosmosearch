use std::collections::BTreeSet;

use crate::{AttributeValue, Attributes};

/// Which files a search may look at, judged by their host attributes.
///
/// Every condition must hold. A filter with no conditions lets every file through.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Filter {
    conditions: Vec<(String, BTreeSet<String>)>,
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
    pub fn any_of(mut self, name: impl Into<String>, values: &[&str]) -> Self {
        let values = values.iter().map(|value| (*value).to_owned()).collect();
        self.conditions.push((name.into(), values));
        self
    }

    /// Whether this filter lets every file through.
    pub fn is_empty(&self) -> bool {
        self.conditions.is_empty()
    }

    /// The conditions by attribute name and then by values, so that filters that mean the
    /// same compare the same.
    pub(crate) fn sorted_conditions(&self) -> Vec<(&str, &BTreeSet<String>)> {
        let mut conditions: Vec<(&str, &BTreeSet<String>)> = self
            .conditions
            .iter()
            .map(|(name, values)| (name.as_str(), values))
            .collect();
        conditions.sort_unstable();
        conditions
    }

    /// Whether a file with these attributes passes.
    pub fn matches(&self, attributes: &Attributes) -> bool {
        self.conditions.iter().all(|(name, values)| {
            attributes.get(name).is_some_and(|value| match value {
                AttributeValue::One(one) => values.contains(one),
                AttributeValue::Many(many) => many.iter().any(|item| values.contains(item)),
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn attributes() -> Attributes {
        Attributes::from([
            ("owner".to_owned(), AttributeValue::One("legal".to_owned())),
            (
                "labels".to_owned(),
                AttributeValue::Many(vec!["draft".to_owned(), "finance".to_owned()]),
            ),
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
            assert_eq!(filter.matches(&attributes()), expected, "{filter:?}");
        }
    }

    #[test]
    fn a_file_without_attributes_passes_only_an_empty_filter() {
        assert!(Filter::new().matches(&Attributes::new()));
        assert!(
            !Filter::new()
                .equals("owner", "legal")
                .matches(&Attributes::new())
        );
    }
}
