use std::collections::{BTreeMap, BTreeSet};

use crate::markup::{Line, Page};

const EDGE_LINES: usize = 2;
const MIN_PAGES: usize = 3;
const MIN_REPEATS: usize = 3;
const PAGES_PER_REPEAT: usize = 4;
const ROMAN_DIGITS: [&str; 2] = ["ivxlcdm", "IVXLCDM"];

/// Removes running headers and footers: lines among the first or last few of a page whose
/// text, with numbers ignored, sits there on at least a quarter of the pages.
pub(crate) fn remove_running_lines(pages: &mut [Page]) {
    if pages.len() < MIN_PAGES {
        return;
    }
    let mut repeats: BTreeMap<String, usize> = BTreeMap::new();
    for page in pages.iter() {
        let keys: BTreeSet<String> = edge_lines(&page.lines)
            .map(|line| key(&line.text))
            .collect();
        for key in keys {
            let count = repeats.entry(key).or_default();
            *count = count.saturating_add(1);
        }
    }
    let needed = MIN_REPEATS.max(pages.len().div_ceil(PAGES_PER_REPEAT));
    let running: BTreeSet<String> = repeats
        .into_iter()
        .filter(|(_, count)| *count >= needed)
        .map(|(key, _)| key)
        .collect();
    if running.is_empty() {
        return;
    }
    for page in pages.iter_mut() {
        let edges = edge_positions(&page.lines);
        let mut at = 0usize;
        page.lines.retain(|line| {
            let running_edge = edges.contains(&at) && running.contains(&key(&line.text));
            at = at.saturating_add(1);
            !running_edge
        });
    }
}

fn edge_positions(lines: &[Line]) -> BTreeSet<usize> {
    let filled: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, line)| !line.text.is_empty())
        .map(|(at, _)| at)
        .collect();
    let tail = filled.len().saturating_sub(EDGE_LINES).max(EDGE_LINES);
    let first = filled.iter().take(EDGE_LINES);
    let last = filled.iter().skip(tail);
    first.chain(last).copied().collect()
}

fn edge_lines(lines: &[Line]) -> impl Iterator<Item = &Line> {
    let edges = edge_positions(lines);
    lines
        .iter()
        .enumerate()
        .filter(move |(at, _)| edges.contains(at))
        .map(|(_, line)| line)
}

fn key(text: &str) -> String {
    let roman = ROMAN_DIGITS
        .iter()
        .any(|digits| text.chars().all(|c| digits.contains(c)));
    if roman && !text.is_empty() {
        return "#".to_owned();
    }
    let mut key = String::with_capacity(text.len());
    for c in text.chars() {
        if !c.is_ascii_digit() {
            key.push(c);
        } else if !key.ends_with('#') {
            key.push('#');
        }
    }
    key
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use super::*;

    fn pages(texts: &[Vec<String>]) -> Vec<Page> {
        texts
            .iter()
            .zip(1u32..)
            .map(|(lines, number)| Page {
                number: NonZeroU32::new(number).unwrap(),
                lines: lines
                    .iter()
                    .map(|text| Line {
                        level: None,
                        text: text.clone(),
                    })
                    .collect(),
            })
            .collect()
    }

    fn texts(pages: &[Page]) -> Vec<Vec<&str>> {
        pages
            .iter()
            .map(|page| page.lines.iter().map(|line| line.text.as_str()).collect())
            .collect()
    }

    fn lines(texts: &[&str]) -> Vec<String> {
        texts.iter().map(|text| (*text).to_owned()).collect()
    }

    #[test]
    fn ignores_numbers_and_roman_page_numbers() {
        assert_eq!(key("Page 12 of 120"), "Page # of #");
        assert_eq!(key("SP 800-53r5"), "SP #-#r#");
        assert_eq!(key("xiv"), "#");
        assert_eq!(key("XIV"), "#");
        assert_eq!(key("42"), "#");
        assert_eq!(key("Mix"), "Mix");
    }

    #[test]
    fn removes_lines_repeated_at_the_edges_of_pages() {
        let texts_in: Vec<Vec<String>> =
            [("1", "one"), ("2", "two"), ("iii", "three"), ("4", "four")]
                .iter()
                .map(|(number, body)| lines(&["Report 2024", "", body, number]))
                .collect();
        let mut pages = pages(&texts_in);
        remove_running_lines(&mut pages);
        assert_eq!(
            texts(&pages),
            [["", "one"], ["", "two"], ["", "three"], ["", "four"]]
        );
    }

    #[test]
    fn keeps_a_repeated_line_in_the_middle_of_a_page() {
        let texts_in: Vec<Vec<String>> = ["a", "b", "c", "d"]
            .iter()
            .map(|letter| {
                let edge = |name: &str| format!("{name} {letter}");
                vec![
                    edge("top"),
                    edge("first"),
                    "DISCUSSION".to_owned(),
                    edge("last"),
                    edge("end"),
                ]
            })
            .collect();
        let mut pages = pages(&texts_in);
        remove_running_lines(&mut pages);
        assert!(texts(&pages).iter().all(|lines| lines.len() == 5));
    }

    #[test]
    fn needs_a_quarter_of_the_pages_and_at_least_three() {
        for (count, with_header, removed) in [(20, 4, false), (20, 5, true), (2, 2, false)] {
            let texts_in: Vec<Vec<String>> = (0..count)
                .map(|n| {
                    let top = if n < with_header { "Header" } else { "other" };
                    vec![top.to_owned(), format!("body {}", "x".repeat(n))]
                })
                .collect();
            let mut pages = pages(&texts_in);
            remove_running_lines(&mut pages);
            let first = texts(&pages)[0][0].to_owned();
            assert_eq!(first != "Header", removed, "{count} pages, {with_header}");
        }
    }
}
