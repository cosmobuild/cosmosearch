use std::ops::Range;

use crate::MAX_CHUNK_BYTES;

const SENTENCE_ENDS: [&str; 3] = [". ", "! ", "? "];

/// Paragraphs this short that cannot share a chunk with the paragraph after them lead its first
/// piece instead of standing alone, so a title above a long table stays with the table.
const LEAD_BYTES: usize = MAX_CHUNK_BYTES / 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Span {
    start: usize,
    end: usize,
}

impl Span {
    fn len(self) -> usize {
        self.end.saturating_sub(self.start)
    }

    fn trimmed(self, text: &str) -> Option<Span> {
        let slice = text.get(self.start..self.end)?;
        let body = slice.trim();
        if body.is_empty() {
            return None;
        }
        let lead = slice.len().saturating_sub(slice.trim_start().len());
        let start = self.start.saturating_add(lead);
        Some(Span {
            start,
            end: start.saturating_add(body.len()),
        })
    }
}

/// Splits part of `text` into ranges of at most [`MAX_CHUNK_BYTES`], in order.
///
/// Paragraphs are grouped up to that size, and a longer paragraph is cut at a line, then a
/// sentence, then a word. A short group that the next paragraph does not fit beside is cut
/// together with it, so it leads the first piece. Every range is trimmed, and only whitespace
/// is left out.
pub fn text_ranges(text: &str, within: Range<usize>) -> Vec<Range<usize>> {
    let section = Span {
        start: within.start,
        end: within.end,
    };
    pack(text, section)
        .into_iter()
        .map(|span| span.start..span.end)
        .collect()
}

fn paragraphs(text: &str, section: Span) -> Vec<Span> {
    let mut found = Vec::new();
    let mut open: Option<Span> = None;
    let mut at = section.start;
    let body = text.get(section.start..section.end).unwrap_or_default();
    for line in body.split_inclusive('\n') {
        if line.trim().is_empty() {
            found.extend(open.take());
        } else {
            let lead = line.len().saturating_sub(line.trim_start().len());
            let start = open.map_or(at.saturating_add(lead), |span| span.start);
            let end = at.saturating_add(line.trim_end().len());
            open = Some(Span { start, end });
        }
        at = at.saturating_add(line.len());
    }
    found.extend(open);
    found
}

fn pack(text: &str, section: Span) -> Vec<Span> {
    let mut packed = Vec::new();
    let mut open: Option<Span> = None;
    for paragraph in paragraphs(text, section) {
        let joined = open.map(|current| Span {
            start: current.start,
            end: paragraph.end,
        });
        match (open.take(), joined) {
            (Some(_), Some(joined)) if joined.len() <= MAX_CHUNK_BYTES => open = Some(joined),
            (Some(current), Some(joined)) if current.len() <= LEAD_BYTES => {
                packed.extend(split_long(text, joined));
            }
            (current, _) => {
                packed.extend(current);
                if paragraph.len() > MAX_CHUNK_BYTES {
                    packed.extend(split_long(text, paragraph));
                } else {
                    open = Some(paragraph);
                }
            }
        }
    }
    packed.extend(open);
    packed
}

fn split_long(text: &str, paragraph: Span) -> Vec<Span> {
    let mut pieces = Vec::new();
    let mut start = paragraph.start;
    while paragraph.end.saturating_sub(start) > MAX_CHUNK_BYTES {
        let limit = text.floor_char_boundary(start.saturating_add(MAX_CHUNK_BYTES));
        let Some(window) = text.get(start..limit).filter(|w| !w.is_empty()) else {
            break;
        };
        let cut = start.saturating_add(best_cut(window));
        pieces.extend(Span { start, end: cut }.trimmed(text));
        start = cut;
    }
    pieces.extend(
        Span {
            start,
            end: paragraph.end,
        }
        .trimmed(text),
    );
    pieces
}

fn best_cut(window: &str) -> usize {
    let half = window.len() / 2;
    let line = window.rfind('\n').map(|at| at.saturating_add(1));
    let sentence = SENTENCE_ENDS
        .iter()
        .filter_map(|end| window.rfind(end))
        .max()
        .map(|at| at.saturating_add(1));
    let word = window.rfind(char::is_whitespace);
    line.filter(|at| *at >= half)
        .or(sentence.filter(|at| *at >= half))
        .or(word.filter(|at| *at > 0))
        .unwrap_or(window.len())
}

#[cfg(test)]
mod tests {
    use std::fmt::Write as _;

    use super::*;

    fn slices(text: &str) -> Vec<&str> {
        text_ranges(text, 0..text.len())
            .into_iter()
            .map(|range| text.get(range).unwrap())
            .collect()
    }

    #[test]
    fn groups_paragraphs_and_trims() {
        assert_eq!(slices("  one\ntwo\n\nthree \n"), ["one\ntwo\n\nthree"]);
    }

    #[test]
    fn stays_within_the_range_given() {
        let text = "aaa bbb ccc";
        let found = text_ranges(text, 4..7);
        assert_eq!(found.first(), Some(&(4..7)));
        assert_eq!(found.len(), 1);
    }

    #[test]
    fn finds_nothing_in_blank_text() {
        assert_eq!(slices(" \n\t\n").len(), 0);
    }

    #[test]
    fn a_short_title_leads_a_paragraph_too_long_to_share_a_chunk_with() {
        let title = "Consolidated Balance Sheets";
        let rows = |count: usize| -> String {
            let mut table = String::new();
            for row in 0..count {
                writeln!(table, "Line item {row:03} 1,234 5,678").unwrap();
            }
            table
        };
        for table in [rows(76), rows(200)] {
            let text = format!("{title}\n\n{table}");
            let found = slices(&text);
            assert!(found.len() > 1, "{}", table.len());
            assert!(found[0].starts_with(title));
            assert!(found[0].contains("Line item 000"));
            assert!(found.iter().all(|s| s.len() <= MAX_CHUNK_BYTES));
            let kept: usize = found.iter().map(|s| s.split_whitespace().count()).sum();
            assert_eq!(kept, text.split_whitespace().count());
        }

        let long_intro = "Intro words here. ".repeat(40);
        let text = format!("{long_intro}\n\n{}", rows(70));
        let found = slices(&text);
        assert_eq!(found[0], long_intro.trim());
    }

    #[test]
    fn cuts_long_text_within_the_limit_and_loses_only_whitespace() {
        let text = "word ".repeat(MAX_CHUNK_BYTES);
        let found = slices(&text);
        assert!(found.len() > 1);
        assert!(found.iter().all(|s| s.len() <= MAX_CHUNK_BYTES));
        let kept: usize = found.iter().map(|s| s.split_whitespace().count()).sum();
        assert_eq!(kept, MAX_CHUNK_BYTES);
    }
}
