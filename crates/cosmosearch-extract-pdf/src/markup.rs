use std::num::NonZeroU32;

const INLINE_TAGS: [&str; 4] = ["<u>", "</u>", "<sup>", "</sup>"];
const MAX_HEADING_LEVEL: usize = 6;
/// A run of this many dots or more is a leader joining a label to its figure, not text.
const LEADER_DOTS: usize = 3;

/// The lines of one page, counting pages from 1.
pub(crate) struct Page {
    pub(crate) number: NonZeroU32,
    pub(crate) lines: Vec<Line>,
}

/// One line of a page with its Markdown removed. Blank text separates paragraphs.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Line {
    pub(crate) level: Option<usize>,
    pub(crate) text: String,
}

pub(crate) fn clean_page(markdown: &str) -> Vec<Line> {
    markdown.lines().filter_map(clean_line).collect()
}

fn clean_line(line: &str) -> Option<Line> {
    let trimmed = line.trim();
    if is_table_rule(trimmed) {
        return None;
    }
    let (level, body) = split_heading(trimmed);
    let mut plain = without_links(body);
    for tag in INLINE_TAGS {
        plain = plain.replace(tag, "");
    }
    let plain = without_leaders(&without_emphasis(&plain));
    let plain = if trimmed.starts_with('|') {
        plain.replace('|', " ")
    } else {
        plain
    };
    Some(Line {
        level,
        text: plain.split_whitespace().collect::<Vec<_>>().join(" "),
    })
}

/// `text` without dot leaders and with runs of whitespace made one space.
pub(crate) fn plain_text(text: &str) -> String {
    without_leaders(text)
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn is_table_rule(line: &str) -> bool {
    line.starts_with('|')
        && line.contains('-')
        && line.chars().all(|c| matches!(c, '|' | '-' | ':' | ' '))
}

fn split_heading(line: &str) -> (Option<usize>, &str) {
    let after = line.trim_start_matches('#');
    let level = line.len().saturating_sub(after.len());
    if (1..=MAX_HEADING_LEVEL).contains(&level) && after.starts_with(' ') {
        (Some(level), after)
    } else {
        (None, line)
    }
}

fn without_links(line: &str) -> String {
    let mut plain = String::with_capacity(line.len());
    let mut rest = line;
    while let Some(link) = next_link(rest) {
        let before = rest.get(..link.open).unwrap_or_default();
        plain.push_str(before.strip_suffix('!').unwrap_or(before));
        plain.push_str(link.label);
        rest = rest.get(link.end..).unwrap_or_default();
    }
    plain.push_str(rest);
    plain
}

struct Link<'a> {
    open: usize,
    label: &'a str,
    end: usize,
}

fn next_link(text: &str) -> Option<Link<'_>> {
    let middle = text.find("](")?;
    let open = text.get(..middle)?.rfind('[')?;
    let length = text.get(middle..)?.find(')')?;
    Some(Link {
        open,
        label: text.get(open.saturating_add(1)..middle)?,
        end: middle.saturating_add(length).saturating_add(1),
    })
}

/// `line` with each dot leader, as in `Cash.........$ 4,258`, turned into a space.
fn without_leaders(line: &str) -> String {
    let mut plain = String::with_capacity(line.len());
    let mut dots = 0usize;
    for c in line.chars() {
        if c == '.' {
            dots = dots.saturating_add(1);
            continue;
        }
        push_dots(&mut plain, dots);
        dots = 0;
        plain.push(c);
    }
    push_dots(&mut plain, dots);
    plain
}

fn push_dots(plain: &mut String, dots: usize) {
    if dots >= LEADER_DOTS {
        plain.push(' ');
    } else {
        plain.push_str(&".".repeat(dots));
    }
}

fn without_emphasis(line: &str) -> String {
    let mut plain = String::with_capacity(line.len());
    let mut chars = line.chars().peekable();
    let mut previous: Option<char> = None;
    while let Some(c) = chars.next() {
        if c != '*' {
            plain.push(c);
            previous = Some(c);
            continue;
        }
        let mut run = 1usize;
        while chars.next_if_eq(&'*').is_some() {
            run = run.saturating_add(1);
        }
        let spaced_before = previous.is_none_or(char::is_whitespace);
        let spaced_after = chars.peek().is_none_or(|next| next.is_whitespace());
        if spaced_before && spaced_after {
            plain.push_str(&"*".repeat(run));
            previous = Some('*');
        }
    }
    plain
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(line: &str) -> String {
        clean_page(line).remove(0).text
    }

    #[test]
    fn removes_inline_markup() {
        let cases = [
            ("**bold** and *italic*", "bold and italic"),
            ("**one** **two**", "one two"),
            ("***R* **> a**", "R > a"),
            ("<u>AC-11</u>, <u>AC-12</u>", "AC-11, AC-12"),
            ("x<sup>2</sup>", "x2"),
            (
                "see [the guide](https://example.com/a) now",
                "see the guide now",
            ),
            ("![a chart](x.png) b", "a chart b"),
            ("[*Assignment: value*]", "[Assignment: value]"),
        ];
        for (markdown, expected) in cases {
            assert_eq!(text(markdown), expected, "{markdown:?}");
        }
    }

    #[test]
    fn keeps_a_real_asterisk() {
        assert_eq!(text("2 * 3 = 6"), "2 * 3 = 6");
    }

    #[test]
    fn reads_headings() {
        let line = clean_page("## **Bias**").remove(0);
        assert_eq!(
            line,
            Line {
                level: Some(2),
                text: "Bias".to_owned()
            }
        );
        for markdown in ["#nospace", "####### seven", "plain"] {
            assert_eq!(clean_page(markdown).remove(0).level, None, "{markdown:?}");
        }
    }

    #[test]
    fn turns_tables_into_words_and_drops_their_rules() {
        let lines = clean_page("|Finding|Yes|\n|---|:-:|\n|a b|c|");
        let texts: Vec<_> = lines.iter().map(|l| l.text.as_str()).collect();
        assert_eq!(texts, ["Finding Yes", "a b c"]);
    }

    #[test]
    fn turns_dot_leaders_into_spaces() {
        assert_eq!(
            text("Cash and cash equivalents........$ 4,258"),
            "Cash and cash equivalents $ 4,258"
        );
        assert_eq!(text("Total.... 12"), "Total 12");
        assert_eq!(
            text("See note 1.. and 3.5 or e.g."),
            "See note 1.. and 3.5 or e.g."
        );
        assert_eq!(text("Assets..."), "Assets");
    }

    #[test]
    fn keeps_blank_lines_between_paragraphs() {
        let texts: Vec<_> = clean_page("one\n\ntwo")
            .into_iter()
            .map(|l| l.text)
            .collect();
        assert_eq!(texts, ["one", "", "two"]);
    }

    #[test]
    fn leaves_unbalanced_link_syntax_alone() {
        assert_eq!(text("a ](b c"), "a ](b c");
    }
}
