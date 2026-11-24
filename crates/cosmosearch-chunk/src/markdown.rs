use cosmosearch_core::{
    ExtractFailure, ExtractLimits, Extraction, Extractor, ExtractorProfile, Passage,
};

use crate::Result;
use crate::split::{Span, decode, lines_from, passages};

const MARKDOWN_EXTENSIONS: [&str; 2] = ["md", "markdown"];
const MARKDOWN_MEDIA_TYPES: [&str; 2] = ["text/markdown", "text/x-markdown"];
const FRONT_MATTER_ENDS: [&str; 2] = ["---", "..."];
const MAX_HEADING_LEVEL: usize = 6;
const MAX_INDENT: usize = 3;
const MIN_FENCE: usize = 3;

/// Extracts Markdown, one section per heading, with each section's heading path.
#[derive(Debug, Clone)]
pub struct Markdown(ExtractorProfile);

impl Markdown {
    /// Creates the extractor.
    ///
    /// # Errors
    ///
    /// Returns an error if the fixed profile id is not valid, which is a bug.
    pub fn new() -> Result<Self> {
        Ok(Markdown(ExtractorProfile::new("markdown-2")?))
    }
}

impl Extractor for Markdown {
    fn profile(&self) -> &ExtractorProfile {
        &self.0
    }

    fn extensions(&self) -> &[&str] {
        &MARKDOWN_EXTENSIONS
    }

    fn media_types(&self) -> &[&str] {
        &MARKDOWN_MEDIA_TYPES
    }

    fn extract(&self, file: &[u8], _: &ExtractLimits) -> Result<Extraction, ExtractFailure> {
        let (text, start) = decode(file)?;
        let mut sections = Sections::new(text, start);
        sections.scan();
        Ok(Extraction::new(sections.found))
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Fence {
    marker: char,
    length: usize,
}

struct Sections<'a> {
    text: &'a str,
    section_start: usize,
    headings: Vec<(usize, String)>,
    found: Vec<Passage>,
}

impl<'a> Sections<'a> {
    fn new(text: &'a str, start: usize) -> Self {
        Sections {
            text,
            section_start: start,
            headings: Vec::new(),
            found: Vec::new(),
        }
    }

    fn scan(&mut self) {
        let text = self.text;
        let body = self.skip_front_matter();
        let mut fence: Option<Fence> = None;
        let mut paragraph: Option<usize> = None;
        for (at, line) in lines_from(text, body) {
            if let Some(open) = fence {
                if closes_fence(line, open) {
                    fence = None;
                }
            } else if let Some(open) = opens_fence(line) {
                fence = Some(open);
                paragraph = None;
            } else if let Some((level, title)) = atx_heading(line) {
                self.open_section(at, level, title.to_owned());
                paragraph = None;
            } else if let Some(level) = setext_level(line) {
                if let Some(start) = paragraph.take() {
                    let title = text.get(start..at).unwrap_or_default();
                    let title = title.split_whitespace().collect::<Vec<_>>().join(" ");
                    self.open_section(start, level, title);
                }
            } else if line.trim().is_empty() {
                paragraph = None;
            } else {
                paragraph.get_or_insert(at);
            }
        }
        self.close_section(text.len());
    }

    fn skip_front_matter(&mut self) -> usize {
        let start = self.section_start;
        let mut lines = lines_from(self.text, start);
        if lines
            .next()
            .is_none_or(|(_, line)| line.trim_end() != "---")
        {
            return start;
        }
        let end = lines
            .find(|(_, line)| FRONT_MATTER_ENDS.contains(&line.trim_end()))
            .map(|(at, line)| at.saturating_add(line.len()));
        let Some(end) = end else {
            return start;
        };
        self.close_section(end);
        self.section_start = end;
        end
    }

    fn open_section(&mut self, at: usize, level: usize, title: String) {
        self.close_section(at);
        self.headings.retain(|(open, _)| *open < level);
        if !title.is_empty() {
            self.headings.push((level, title));
        }
        self.section_start = at;
    }

    fn close_section(&mut self, end: usize) {
        let path: Vec<String> = self.headings.iter().map(|(_, t)| t.clone()).collect();
        let section = Span {
            start: self.section_start,
            end,
        };
        self.found.extend(passages(self.text, section, &path));
    }
}

fn unindented(line: &str) -> Option<&str> {
    let rest = line.trim_start_matches(' ');
    (line.len().saturating_sub(rest.len()) <= MAX_INDENT).then_some(rest)
}

fn opens_fence(line: &str) -> Option<Fence> {
    let rest = unindented(line)?;
    let marker = rest.chars().next().filter(|c| matches!(c, '`' | '~'))?;
    let after = rest.trim_start_matches(marker);
    let length = rest.len().saturating_sub(after.len());
    let valid = length >= MIN_FENCE && (marker == '~' || !after.contains('`'));
    valid.then_some(Fence { marker, length })
}

fn closes_fence(line: &str, open: Fence) -> bool {
    unindented(line).is_some_and(|rest| {
        let rest = rest.trim_end();
        rest.len() >= open.length && rest.chars().all(|c| c == open.marker)
    })
}

fn atx_heading(line: &str) -> Option<(usize, &str)> {
    let rest = unindented(line)?;
    let after = rest.trim_start_matches('#');
    let level = rest.len().saturating_sub(after.len());
    let separated = after.is_empty() || after.starts_with(char::is_whitespace);
    if !(1..=MAX_HEADING_LEVEL).contains(&level) || !separated {
        return None;
    }
    let body = after.trim();
    let stripped = body.trim_end_matches('#');
    let closed = stripped.is_empty() || stripped.ends_with(char::is_whitespace);
    Some((level, if closed { stripped.trim_end() } else { body }))
}

fn setext_level(line: &str) -> Option<usize> {
    let rest = unindented(line)?.trim_end();
    let marker = rest.chars().next()?;
    let level = match marker {
        '=' => 1,
        '-' => 2,
        _ => return None,
    };
    rest.chars().all(|c| c == marker).then_some(level)
}

#[cfg(test)]
mod tests {
    use cosmosearch_core::ETag;
    use proptest::prelude::*;

    use super::*;
    use crate::extract_chunks;
    use crate::split::{assert_faithful, tokens};

    fn extract(file: &str) -> Vec<Passage> {
        Markdown::new()
            .unwrap()
            .extract(file.as_bytes(), &ExtractLimits::default())
            .unwrap()
            .into_passages()
    }

    fn check(file: &str, expected: &[(&[&str], &str)]) {
        let found = extract(file);
        let actual: Vec<(Vec<&str>, &str)> = found
            .iter()
            .map(|p| {
                let path = p.heading_path().iter().map(String::as_str).collect();
                (path, p.text())
            })
            .collect();
        let expected: Vec<(Vec<&str>, &str)> =
            expected.iter().map(|(p, t)| (p.to_vec(), *t)).collect();
        assert_eq!(actual, expected, "{file:?}");
        assert_faithful(file, &found);
    }

    #[test]
    fn damaged_files_give_passages_or_an_error_and_never_a_panic() {
        let good = b"# Terms\n\nIntro text.\n\n## Refunds\n\n- within [30 days](https://example.com)\n- `code`\n\n```\nfenced # not a heading\n```\n\n> quoted\n\nSetext\n===\n";
        cosmosearch_core::extract_damaged(&Markdown::new().unwrap(), good);
    }

    #[test]
    fn follows_the_heading_hierarchy() {
        check(
            "intro\n\n# A\none\n\n## B\ntwo\n\n### C\nthree\n\n## D\nfour\n\n# E\nfive\n",
            &[
                (&[], "intro"),
                (&["A"], "# A\none"),
                (&["A", "B"], "## B\ntwo"),
                (&["A", "B", "C"], "### C\nthree"),
                (&["A", "D"], "## D\nfour"),
                (&["E"], "# E\nfive"),
            ],
        );
    }

    #[test]
    fn reads_setext_headings() {
        check(
            "Top\n===\none\n\nSub\nnext\n---\ntwo\n",
            &[
                (&["Top"], "Top\n===\none"),
                (&["Top", "Sub next"], "Sub\nnext\n---\ntwo"),
            ],
        );
    }

    #[test]
    fn a_rule_after_a_blank_line_is_not_a_heading() {
        check("one\n\n---\n\ntwo\n", &[(&[], "one\n\n---\n\ntwo")]);
    }

    #[test]
    fn ignores_headings_inside_code_fences() {
        check(
            "# A\n```\n# not a heading\n```\n~~~~\n# nor this\n~~~\n~~~~\nend\n",
            &[(
                &["A"],
                "# A\n```\n# not a heading\n```\n~~~~\n# nor this\n~~~\n~~~~\nend",
            )],
        );
    }

    #[test]
    fn an_unclosed_fence_runs_to_the_end() {
        check("# A\n```\n# hidden\n", &[(&["A"], "# A\n```\n# hidden")]);
    }

    #[test]
    fn recognises_only_real_atx_headings() {
        check(
            "#nospace\n\n    # code\n\n####### seven\n",
            &[(&[], "#nospace\n\n    # code\n\n####### seven")],
        );
        check(
            "   ## Indented ##\nx\n",
            &[(&["Indented"], "## Indented ##\nx")],
        );
        check("# C#\nx\n", &[(&["C#"], "# C#\nx")]);
        check("#\nx\n", &[(&[], "#\nx")]);
    }

    #[test]
    fn keeps_front_matter_out_of_the_headings() {
        check(
            "---\ntitle: x\n---\n# A\nbody\n",
            &[(&[], "---\ntitle: x\n---"), (&["A"], "# A\nbody")],
        );
        check("---\nno close\n", &[(&[], "---\nno close")]);
    }

    #[test]
    fn a_long_section_keeps_its_path_on_every_chunk() {
        let file = format!("# A\n\n{}", "word ".repeat(1_500));
        let found = extract(&file);
        assert!(found.len() > 1);
        assert!(found.iter().all(|p| p.heading_path() == ["A"]));
        assert_faithful(&file, &found);
    }

    #[test]
    fn a_range_read_gives_back_each_chunk() {
        let file = "# Refunds\n\nAsk within 30 days.\n";
        let chunks = extract_chunks(
            &Markdown::new().unwrap(),
            &"s3://example-bucket/a.md".parse().unwrap(),
            &ETag::new("9f3a").unwrap(),
            file.as_bytes(),
            &ExtractLimits::default(),
        )
        .unwrap()
        .into_chunks();
        assert_eq!(
            chunks[0].source().to_string(),
            format!(
                "s3://example-bucket/a.md#bytes=0-{}@etag=9f3a",
                file.trim_end().len()
            )
        );
        assert_eq!(chunks[0].text(), file.trim_end());
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn every_passage_is_an_exact_slice_and_no_text_is_lost(file in tokens()) {
            assert_faithful(&file, &extract(&file));
        }
    }
}
