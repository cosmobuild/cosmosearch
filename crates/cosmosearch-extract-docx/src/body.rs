use std::num::NonZeroU32;

use cosmosearch_core::{ElementPath, ExtractFailure};
use quick_xml::events::BytesStart;

use crate::styles::{HeadingStyles, outline_level};
use crate::xml::{Step, XmlReader, attribute};

/// Elements whose content is left out: drawings and text boxes, alternative copies, and
/// tracked deletions, moves and earlier formatting.
const SKIPPED: [&[u8]; 8] = [
    b"drawing",
    b"pict",
    b"object",
    b"Fallback",
    b"del",
    b"moveFrom",
    b"pPrChange",
    b"rPrChange",
];

/// One paragraph of the body with its text and, outside tables, its heading level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Paragraph {
    pub(crate) path: ElementPath,
    pub(crate) heading: Option<u8>,
    pub(crate) text: String,
}

#[derive(Default)]
struct Counters {
    child: u32,
    paragraph: u32,
    row: u32,
    cell: u32,
    cell_paragraph: u32,
}

struct Open {
    path: ElementPath,
    style: Option<String>,
    outline: Option<u8>,
    text: String,
}

/// Reads the paragraphs of `word/document.xml` in document order.
pub(crate) fn paragraphs(
    xml: &[u8],
    styles: &HeadingStyles,
    max_depth: usize,
) -> Result<Vec<Paragraph>, ExtractFailure> {
    let mut reader = XmlReader::new(xml, max_depth);
    let mut body = Body::new(styles);
    let mut body_depth: Option<usize> = None;
    let mut skip_from: Option<usize> = None;
    loop {
        let step = reader.next()?;
        let depth = reader.depth();
        if let Some(from) = skip_from {
            if matches!(step, Step::Close(_)) && depth < from {
                skip_from = None;
            }
            continue;
        }
        match step {
            Step::Open { element, empty } => {
                let name = element.local_name();
                let name = name.as_ref();
                let Some(at) = body_depth else {
                    if name == b"body" && !empty {
                        body_depth = Some(depth);
                    }
                    continue;
                };
                let parent = if empty {
                    depth
                } else {
                    depth.saturating_sub(1)
                };
                if parent == at {
                    body.next_child()?;
                }
                if SKIPPED.contains(&name) {
                    if !empty {
                        skip_from = Some(depth);
                    }
                    continue;
                }
                body.open(name, &element, empty, depth)?;
            }
            Step::Close(name) if name == b"body" => body_depth = None,
            Step::Close(name) => body.close(&name, depth),
            Step::Text(text) => body.text(&text),
            Step::End => return Ok(body.found),
        }
    }
}

struct Body<'styles> {
    styles: &'styles HeadingStyles,
    count: Counters,
    tables: usize,
    run_depth: Option<usize>,
    in_text: bool,
    open: Option<Open>,
    found: Vec<Paragraph>,
}

impl<'styles> Body<'styles> {
    fn new(styles: &'styles HeadingStyles) -> Self {
        Body {
            styles,
            count: Counters::default(),
            tables: 0,
            run_depth: None,
            in_text: false,
            open: None,
            found: Vec::new(),
        }
    }

    fn next_child(&mut self) -> Result<(), ExtractFailure> {
        self.count = Counters {
            child: next(self.count.child)?,
            ..Counters::default()
        };
        Ok(())
    }

    fn open(
        &mut self,
        name: &[u8],
        element: &BytesStart<'_>,
        empty: bool,
        depth: usize,
    ) -> Result<(), ExtractFailure> {
        let in_run = self.run_depth.is_some();
        match name {
            b"tbl" if !empty => self.tables = self.tables.saturating_add(1),
            b"tr" if self.tables == 1 => {
                self.count.row = next(self.count.row)?;
                self.count.cell = 0;
            }
            b"tc" if self.tables == 1 => {
                self.count.cell = next(self.count.cell)?;
                self.count.cell_paragraph = 0;
            }
            b"p" => {
                let paragraph = Open {
                    path: self.next_paragraph()?,
                    style: None,
                    outline: None,
                    text: String::new(),
                };
                if empty {
                    self.found.push(self.finish(paragraph));
                } else {
                    self.open = Some(paragraph);
                }
            }
            b"pStyle" => {
                if let Some(paragraph) = self.open.as_mut() {
                    paragraph.style = attribute(element, b"val");
                }
            }
            b"outlineLvl" => {
                if let Some(paragraph) = self.open.as_mut() {
                    paragraph.outline = attribute(element, b"val")
                        .as_deref()
                        .and_then(outline_level);
                }
            }
            b"r" if !empty => self.run_depth = Some(depth),
            b"t" if !empty && in_run => self.in_text = true,
            b"tab" if in_run => self.push('\t'),
            b"br" | b"cr" if in_run => self.push('\n'),
            b"noBreakHyphen" if in_run => self.push('-'),
            _ => {}
        }
        Ok(())
    }

    fn next_paragraph(&mut self) -> Result<ElementPath, ExtractFailure> {
        let count = &mut self.count;
        if self.tables == 0 {
            count.paragraph = next(count.paragraph)?;
            return Ok(ElementPath::in_body(
                non_zero(count.child)?,
                non_zero(count.paragraph)?,
            ));
        }
        count.cell_paragraph = next(count.cell_paragraph)?;
        Ok(ElementPath::in_cell(
            non_zero(count.child)?,
            non_zero(count.row)?,
            non_zero(count.cell)?,
            non_zero(count.cell_paragraph)?,
        ))
    }

    fn close(&mut self, name: &[u8], depth: usize) {
        match name {
            b"tbl" => self.tables = self.tables.saturating_sub(1),
            b"t" => self.in_text = false,
            b"r" if self.run_depth.is_some_and(|run| depth < run) => self.run_depth = None,
            b"p" => {
                if let Some(paragraph) = self.open.take() {
                    self.found.push(self.finish(paragraph));
                }
            }
            _ => {}
        }
    }

    fn text(&mut self, text: &str) {
        if self.in_text
            && let Some(paragraph) = self.open.as_mut()
        {
            paragraph.text.push_str(text);
        }
    }

    fn push(&mut self, c: char) {
        if let Some(paragraph) = self.open.as_mut() {
            paragraph.text.push(c);
        }
    }

    fn finish(&self, paragraph: Open) -> Paragraph {
        let style_level = paragraph
            .style
            .as_deref()
            .and_then(|id| self.styles.level(id));
        Paragraph {
            path: paragraph.path,
            heading: (self.tables == 0)
                .then_some(paragraph.outline.or(style_level))
                .flatten(),
            text: paragraph.text,
        }
    }
}

fn next(count: u32) -> Result<u32, ExtractFailure> {
    count.checked_add(1).ok_or(ExtractFailure::Malformed)
}

fn non_zero(count: u32) -> Result<NonZeroU32, ExtractFailure> {
    NonZeroU32::new(count).ok_or(ExtractFailure::Malformed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn located(body: &str) -> Vec<(String, Option<u8>, String)> {
        let xml = format!(r#"<w:document xmlns:w="w"><w:body>{body}</w:body></w:document>"#);
        paragraphs(xml.as_bytes(), &HeadingStyles::default(), 64)
            .unwrap()
            .into_iter()
            .map(|p| (p.path.to_string(), p.heading, p.text))
            .collect()
    }

    fn p(text: &str) -> String {
        format!("<w:p><w:r><w:t>{text}</w:t></w:r></w:p>")
    }

    fn cell(content: &str) -> String {
        format!("<w:tc><w:tcPr/>{content}</w:tc>")
    }

    #[test]
    fn counting_rules_are_frozen() {
        let nested = format!(
            "<w:tbl><w:tr>{}{}</w:tr></w:tbl>",
            cell(&p("n1")),
            cell(&p("n2"))
        );
        let body = [
            p("first"),
            "<w:bookmarkStart w:id=\"0\"/>".to_owned(),
            format!(
                "<w:sdt><w:sdtContent>{}<w:p/>{}</w:sdtContent></w:sdt>",
                p("a"),
                p("b")
            ),
            format!(
                "<w:tbl><w:tblPr/><w:tr>{}{}</w:tr><w:tr>{}</w:tr></w:tbl>",
                cell(&p("r1c1")),
                cell(&format!("{}{}", p("r1c2 one"), p("r1c2 two"))),
                cell(&format!("{}{nested}{}", p("before"), p("after"))),
            ),
            "<w:sectPr/>".to_owned(),
        ]
        .concat();
        let found: Vec<(String, String)> = located(&body)
            .into_iter()
            .map(|(path, _, text)| (path, text))
            .collect();
        let expected = [
            ("b1/p1", "first"),
            ("b3/p1", "a"),
            ("b3/p2", ""),
            ("b3/p3", "b"),
            ("b4/r1/c1/p1", "r1c1"),
            ("b4/r1/c2/p1", "r1c2 one"),
            ("b4/r1/c2/p2", "r1c2 two"),
            ("b4/r2/c1/p1", "before"),
            ("b4/r2/c1/p2", "n1"),
            ("b4/r2/c1/p3", "n2"),
            ("b4/r2/c1/p4", "after"),
        ];
        let expected: Vec<(String, String)> = expected
            .iter()
            .map(|(path, text)| ((*path).to_owned(), (*text).to_owned()))
            .collect();
        assert_eq!(found, expected);
    }

    #[test]
    fn reads_runs_tabs_and_breaks_and_leaves_out_what_is_not_body_text() {
        let body = concat!(
            r#"<w:p><w:pPr><w:tabs><w:tab w:val="left"/></w:tabs></w:pPr>"#,
            "<w:r><w:t>one</w:t><w:tab/><w:t>two</w:t><w:br/><w:t>three</w:t></w:r>",
            "<w:r><w:instrText>TOC</w:instrText><w:noBreakHyphen/></w:r>",
            "<w:del><w:r><w:delText>gone</w:delText></w:r></w:del>",
            "<w:r><w:drawing><w:txbxContent><w:p><w:r><w:t>box</w:t></w:r></w:p></w:txbxContent></w:drawing></w:r>",
            "</w:p>",
        );
        let found = located(body);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].2, "one\ttwo\nthree-");
    }

    #[test]
    fn headings_come_from_outline_levels_outside_tables_only() {
        let heading =
            r#"<w:p><w:pPr><w:outlineLvl w:val="1"/></w:pPr><w:r><w:t>H</w:t></w:r></w:p>"#;
        let body = format!("{heading}<w:tbl><w:tr><w:tc>{heading}</w:tc></w:tr></w:tbl>");
        let levels: Vec<Option<u8>> = located(&body).into_iter().map(|(_, h, _)| h).collect();
        assert_eq!(levels, [Some(2), None]);
    }

    #[test]
    fn a_paragraph_in_a_table_outside_a_cell_is_malformed() {
        let xml = r#"<w:document xmlns:w="w"><w:body><w:tbl><w:p/></w:tbl></w:body></w:document>"#;
        assert_eq!(
            paragraphs(xml.as_bytes(), &HeadingStyles::default(), 64).unwrap_err(),
            ExtractFailure::Malformed
        );
    }
}
