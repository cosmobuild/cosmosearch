use std::ops::Range;

use cosmosearch_core::{ByteRange, ExtractFailure, Locator, Passage, text_ranges};

const BYTE_ORDER_MARK: char = '\u{feff}';

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Span {
    pub(crate) start: usize,
    pub(crate) end: usize,
}

/// Decodes a file, returning its text and where the content starts after any byte order mark.
pub(crate) fn decode(file: &[u8]) -> Result<(&str, usize), ExtractFailure> {
    let Ok(text) = std::str::from_utf8(file) else {
        return Err(ExtractFailure::Malformed);
    };
    let start = if text.starts_with(BYTE_ORDER_MARK) {
        BYTE_ORDER_MARK.len_utf8()
    } else {
        0
    };
    Ok((text, start))
}

/// The lines of `text` from `start`, each with its offset and line ending.
pub(crate) fn lines_from(text: &str, start: usize) -> impl Iterator<Item = (usize, &str)> {
    let body = text.get(start..).unwrap_or_default();
    body.split_inclusive('\n').scan(start, |offset, line| {
        let at = *offset;
        *offset = at.saturating_add(line.len());
        Some((at, line))
    })
}

/// Turns one section of `text` into passages no larger than the chunk size.
pub(crate) fn passages(text: &str, section: Span, heading_path: &[String]) -> Vec<Passage> {
    text_ranges(text, section.start..section.end)
        .into_iter()
        .filter_map(|range| passage(text, range, heading_path))
        .collect()
}

fn passage(text: &str, range: Range<usize>, heading_path: &[String]) -> Option<Passage> {
    let start = u64::try_from(range.start).ok()?;
    let end = u64::try_from(range.end).ok()?;
    let body = text.get(range)?;
    let range = ByteRange::new(start, end).ok()?;
    Some(Passage::new(
        Locator::Bytes(range),
        heading_path.to_vec(),
        body,
    ))
}

#[cfg(test)]
pub(crate) fn assert_faithful(file: &str, found: &[Passage]) {
    use cosmosearch_core::MAX_CHUNK_BYTES;

    let mut previous_end = 0;
    let mut kept = String::new();
    for passage in found {
        let Locator::Bytes(range) = passage.locator() else {
            panic!("expected a byte locator");
        };
        let start = usize::try_from(range.start()).unwrap();
        let end = usize::try_from(range.end()).unwrap();
        assert_eq!(file.get(start..end), Some(passage.text()));
        assert!(start >= previous_end, "passages overlap at {start}");
        assert!(passage.text().len() <= MAX_CHUNK_BYTES, "passage too large");
        assert!(!passage.is_blank(), "blank passage at {start}");
        previous_end = end;
        kept.extend(passage.text().chars().filter(|c| !c.is_whitespace()));
    }
    let expected: String = file
        .strip_prefix(BYTE_ORDER_MARK)
        .unwrap_or(file)
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect();
    assert_eq!(kept, expected, "text was lost or added");
}

#[cfg(test)]
pub(crate) fn tokens() -> impl proptest::strategy::Strategy<Value = String> {
    use proptest::prelude::*;
    let long = "x".repeat(2_500);
    let long_accents = "é".repeat(1_300);
    let tokens = vec![
        "word ",
        "other. ",
        "\n",
        "\n\n",
        "\r\n",
        "# Title\n",
        "## Sub ##\n",
        "#nospace\n",
        "```\n",
        "~~~~\n",
        "---\n",
        "===\n",
        "    # indented\n",
        "\t",
        "\u{feff}",
        "ü ",
        Box::leak(long.into_boxed_str()),
        Box::leak(long_accents.into_boxed_str()),
    ];
    prop::collection::vec(prop::sample::select(tokens), 0..300).prop_map(|parts| parts.concat())
}
