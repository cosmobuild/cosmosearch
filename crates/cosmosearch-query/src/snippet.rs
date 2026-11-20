use std::collections::BTreeSet;
use std::fmt;
use std::ops::Range;
use std::sync::Arc;

/// The passage of a hit, read back from its file, with where the query's words are in it.
#[derive(Clone, PartialEq, Eq)]
pub struct Snippet {
    text: Arc<str>,
    matches: Vec<Range<usize>>,
}

impl Snippet {
    /// The snippet of `text` for `query`: every word of the text that is a word of the
    /// query, whatever its case, is a match. A word is a run of letters and digits.
    pub(crate) fn new(text: Arc<str>, query: &str) -> Self {
        let wanted: BTreeSet<String> = words(query)
            .filter_map(|range| query.get(range))
            .map(str::to_lowercase)
            .collect();
        let matches = words(&text)
            .filter(|range| {
                text.get(range.clone())
                    .is_some_and(|word| wanted.contains(&word.to_lowercase()))
            })
            .collect();
        Snippet { text, matches }
    }

    /// The passage text.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// The byte ranges of [`Snippet::text`] that hold a word of the query, in order.
    pub fn matches(&self) -> &[Range<usize>] {
        &self.matches
    }

    /// About how many bytes the snippet holds, for a cache's budget.
    pub(crate) fn weight(&self) -> usize {
        let matches = self.matches.len().saturating_mul(size_of::<Range<usize>>());
        self.text.len().saturating_add(matches)
    }
}

impl fmt::Debug for Snippet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Snippet")
            .field("bytes", &self.text.len())
            .field("matches", &self.matches.len())
            .finish()
    }
}

/// The byte range of each run of letters and digits in `text`.
fn words(text: &str) -> impl Iterator<Item = Range<usize>> {
    let mut chars = text.char_indices().peekable();
    std::iter::from_fn(move || {
        let (start, _) = chars.find(|(_, c)| c.is_alphanumeric())?;
        let mut end = text.len();
        while let Some((at, c)) = chars.peek().copied() {
            if !c.is_alphanumeric() {
                end = at;
                break;
            }
            chars.next();
        }
        Some(start..end)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matched(text: &str, query: &str) -> Vec<String> {
        let snippet = Snippet::new(Arc::from(text), query);
        snippet
            .matches()
            .iter()
            .map(|range| snippet.text().get(range.clone()).unwrap().to_owned())
            .collect()
    }

    #[test]
    fn marks_whole_words_of_the_query_whatever_their_case() {
        let cases: [(&str, &str, &[&str]); 6] = [
            (
                "Refund policy: refunds within 30 days.",
                "refund 30",
                &["Refund", "30"],
            ),
            ("the Policy", "POLICY, the", &["the", "Policy"]),
            ("prix d'été en café", "été café", &["été", "café"]),
            ("nothing here", "refund", &[]),
            ("", "refund", &[]),
            ("refund", "", &[]),
        ];
        for (text, query, expected) in cases {
            assert_eq!(matched(text, query), expected, "{text:?} for {query:?}");
        }
    }

    #[test]
    fn debug_output_shows_no_text() {
        let shown = format!("{:?}", Snippet::new(Arc::from("secret words"), "secret"));
        assert!(!shown.contains("secret"), "{shown}");
    }
}
