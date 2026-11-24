use cosmosearch_core::{ExtractFailure, ExtractLimits, Extraction, Extractor, ExtractorProfile};

use crate::Result;
use crate::split::{Span, decode, passages};

const TEXT_EXTENSIONS: [&str; 1] = ["txt"];
const TEXT_MEDIA_TYPES: [&str; 1] = ["text/plain"];

/// Extracts plain text, grouping paragraphs into chunks.
#[derive(Debug, Clone)]
pub struct PlainText(ExtractorProfile);

impl PlainText {
    /// Creates the extractor.
    ///
    /// # Errors
    ///
    /// Returns an error if the fixed profile id is not valid, which is a bug.
    pub fn new() -> Result<Self> {
        Ok(PlainText(ExtractorProfile::new("text-2")?))
    }
}

impl Extractor for PlainText {
    fn profile(&self) -> &ExtractorProfile {
        &self.0
    }

    fn extensions(&self) -> &[&str] {
        &TEXT_EXTENSIONS
    }

    fn media_types(&self) -> &[&str] {
        &TEXT_MEDIA_TYPES
    }

    fn extract(&self, file: &[u8], _: &ExtractLimits) -> Result<Extraction, ExtractFailure> {
        let (text, start) = decode(file)?;
        let whole = Span {
            start,
            end: text.len(),
        };
        Ok(Extraction::new(passages(text, whole, &[])))
    }
}

#[cfg(test)]
mod tests {
    use cosmosearch_core::{Locator, MAX_CHUNK_BYTES, Passage};
    use proptest::prelude::*;

    use super::*;
    use crate::split::{assert_faithful, tokens};

    fn extract(file: &str) -> Vec<Passage> {
        PlainText::new()
            .unwrap()
            .extract(file.as_bytes(), &ExtractLimits::default())
            .unwrap()
            .into_passages()
    }

    fn texts(found: &[Passage]) -> Vec<&str> {
        found.iter().map(Passage::text).collect()
    }

    #[test]
    fn damaged_files_give_passages_or_an_error_and_never_a_panic() {
        let good = "First paragraph with é and words.\n\nSecond one\r\nwith a break.\n\n\n\tThird."
            .as_bytes();
        cosmosearch_core::extract_damaged(&PlainText::new().unwrap(), good);
    }

    #[test]
    fn groups_short_paragraphs_and_keeps_the_original_bytes() {
        let file = "first line\nsecond line\n\n  third\n";
        let found = extract(file);
        assert_eq!(texts(&found), ["first line\nsecond line\n\n  third"]);
        assert_faithful(file, &found);
    }

    #[test]
    fn starts_a_new_chunk_when_a_paragraph_would_not_fit() {
        let paragraph = "a".repeat(MAX_CHUNK_BYTES - 10);
        let file = format!("{paragraph}\n\n{paragraph}");
        let found = extract(&file);
        assert_eq!(found.len(), 2);
        assert_faithful(&file, &found);
    }

    #[test]
    fn prefers_to_cut_a_long_paragraph_at_a_sentence() {
        let sentence = format!("{}. ", "a ".repeat(400).trim_end());
        let file = sentence.repeat(4);
        let found = extract(&file);
        assert!(found.len() > 1);
        assert!(found[0].text().ends_with('.'), "{}", found[0].text());
        assert_faithful(&file, &found);
    }

    #[test]
    fn never_cuts_inside_a_character() {
        let file = "é".repeat(MAX_CHUNK_BYTES * 2);
        assert_faithful(&file, &extract(&file));
    }

    #[test]
    fn leaves_a_byte_order_mark_out_of_every_range() {
        let file = "\u{feff}hello";
        let found = extract(file);
        assert_eq!(texts(&found), ["hello"]);
        let Locator::Bytes(range) = found[0].locator() else {
            panic!("expected a byte locator");
        };
        assert_eq!(range.start(), 3);
    }

    #[test]
    fn only_the_first_byte_order_mark_is_special() {
        let file = "\u{feff}\u{feff}x";
        let found = extract(file);
        assert_eq!(texts(&found), ["\u{feff}x"]);
        assert_faithful(file, &found);
    }

    #[test]
    fn handles_windows_line_endings() {
        let file = "one\r\n\r\ntwo\r\n";
        assert_faithful(file, &extract(file));
    }

    #[test]
    fn finds_nothing_in_a_blank_file() {
        for file in ["", " \n\t\n", "\u{feff}"] {
            assert_eq!(extract(file).len(), 0, "{file:?}");
        }
    }

    #[test]
    fn refuses_invalid_utf8() {
        let failure = PlainText::new()
            .unwrap()
            .extract(b"ok \xff\xfe", &ExtractLimits::default())
            .unwrap_err();
        assert_eq!(failure, ExtractFailure::Malformed);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn every_passage_is_an_exact_slice_and_no_text_is_lost(file in tokens()) {
            assert_faithful(&file, &extract(&file));
        }
    }
}
