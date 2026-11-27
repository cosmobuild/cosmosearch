use std::collections::BTreeSet;
use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use cosmosearch_core::{Query, SharedIndex};

/// The passage of a hit, read back from its file, with where the query's words are in it.
#[derive(Clone, PartialEq, Eq)]
pub struct Snippet {
    text: Arc<str>,
    matches: Vec<Range<usize>>,
}

impl Snippet {
    /// The snippet of `text` with `matches` marked.
    pub(crate) fn new(text: Arc<str>, matches: Vec<Range<usize>>) -> Self {
        Snippet { text, matches }
    }

    /// The passage text.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// The byte ranges of [`Snippet::text`] that hold a word of the query, in order. Words
    /// are read as the engine's keyword index reads them, so with stemming `refunds` is a
    /// match for `refund`; with no keyword index, a word matches when it equals a word of the
    /// query whatever its case.
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

/// The byte ranges of `text` that hold a word of `query` as `indexes` read words, merged and
/// in order, leaving out any range that is empty or not whole characters of `text`. With no
/// index that reads words, every word equal to a word of the query, whatever its case, a
/// word being a run of letters and digits.
pub(crate) fn matches<'a>(
    indexes: impl Iterator<Item = &'a SharedIndex>,
    query: &Query,
    text: &str,
) -> Vec<Range<usize>> {
    let mut found: Option<Vec<Range<usize>>> = None;
    for index in indexes {
        if let Some(ranges) = index.get().word_matches(query, text) {
            found.get_or_insert_with(Vec::new).extend(
                ranges
                    .into_iter()
                    .filter(|range| !range.is_empty() && text.get(range.clone()).is_some()),
            );
        }
    }
    let Some(mut found) = found else {
        return whole_words(text, query.text());
    };
    found.sort_unstable_by_key(|range| (range.start, range.end));
    let mut merged: Vec<Range<usize>> = Vec::with_capacity(found.len());
    for range in found {
        match merged.last_mut() {
            Some(last) if range.start <= last.end => last.end = last.end.max(range.end),
            _ => merged.push(range),
        }
    }
    merged
}

/// Every word of `text` that is a word of `query`, whatever its case.
fn whole_words(text: &str, query: &str) -> Vec<Range<usize>> {
    let wanted: BTreeSet<String> = words(query)
        .filter_map(|range| query.get(range))
        .map(str::to_lowercase)
        .collect();
    words(text)
        .filter(|range| {
            text.get(range.clone())
                .is_some_and(|word| wanted.contains(&word.to_lowercase()))
        })
        .collect()
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
    use async_trait::async_trait;
    use cosmosearch_core::{
        AllowedRows, Built, Chunk, Hits, Index, IndexError, IndexKind, Level, SegmentEntry,
        SegmentId, SegmentSource, Usage,
    };

    use super::*;

    #[derive(Debug)]
    struct Marks {
        kind: IndexKind,
        ranges: Option<Vec<Range<usize>>>,
    }

    #[async_trait]
    impl Index for Marks {
        fn kind(&self) -> &IndexKind {
            &self.kind
        }

        fn level(&self) -> Level {
            Level::L1
        }

        fn word_matches(&self, _query: &Query, _text: &str) -> Option<Vec<Range<usize>>> {
            self.ranges.clone()
        }

        async fn build(
            &self,
            _segment: &SegmentId,
            _chunks: &[Chunk],
        ) -> Result<Built, IndexError> {
            Err(IndexError::Malformed {
                reason: "not built in these tests",
            })
        }

        async fn search(
            &self,
            _source: &SegmentSource,
            _segments: &[SegmentEntry],
            _allowed: &AllowedRows,
            _query: &Query,
            _limit: usize,
        ) -> Result<Hits, IndexError> {
            Ok(Hits::new(Vec::new(), Usage::new()))
        }
    }

    fn marked(text: &str, ranges: Vec<Option<Vec<Range<usize>>>>) -> Vec<&str> {
        let indexes: Vec<SharedIndex> = ranges
            .into_iter()
            .enumerate()
            .map(|(at, ranges)| {
                SharedIndex::new(Marks {
                    kind: IndexKind::new(format!("marks-{at}")).unwrap(),
                    ranges,
                })
            })
            .collect();
        matches(indexes.iter(), &Query::new("refund"), text)
            .into_iter()
            .map(|range| text.get(range).unwrap())
            .collect()
    }

    #[test]
    fn marks_of_several_indexes_are_merged_in_order() {
        let text = "refund policy";
        assert_eq!(
            marked(
                text,
                vec![Some(vec![7..13, 0..6]), None, Some(vec![2..6, 8..10])]
            ),
            ["refund", "policy"]
        );
        assert_eq!(
            marked(text, vec![Some(vec![0..3, 1..2]), Some(vec![3..6, 4..5])]),
            ["refund"]
        );
    }

    #[test]
    fn whole_words_are_marked_only_when_no_index_reads_words() {
        assert_eq!(
            marked("refund policy", vec![None, Some(Vec::new())]),
            Vec::<&str>::new()
        );
        assert_eq!(marked("refund policy", vec![None]), ["refund"]);
    }

    #[test]
    fn ranges_that_are_not_whole_characters_of_the_text_are_left_out() {
        let text = "café x";
        #[expect(clippy::reversed_empty_ranges, reason = "a plug-in may return one")]
        let reversed = 6..2;
        assert_eq!(
            marked(text, vec![Some(vec![0..4, 5..20, 3..3, reversed, 6..7])]),
            ["x"]
        );
    }

    fn matched(text: &str, query: &str) -> Vec<String> {
        matches(std::iter::empty(), &Query::new(query), text)
            .into_iter()
            .map(|range| text.get(range).unwrap().to_owned())
            .collect()
    }

    #[test]
    fn with_no_keyword_index_marks_whole_words_of_the_query_whatever_their_case() {
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
        let shown = format!(
            "{:?}",
            Snippet::new(Arc::from("secret words"), std::iter::once(0..6).collect())
        );
        assert!(!shown.contains("secret"), "{shown}");
    }
}
