use cosmosearch_core::{
    ElementPath, ExtractFailure, ExtractLimits, Extraction, Extractor, ExtractorProfile, Locator,
    MAX_CHUNK_BYTES, Passage, text_ranges,
};

use crate::archive::Archive;
use crate::body::{Paragraph, paragraphs};
use crate::styles::HeadingStyles;

const DOCX_EXTENSIONS: [&str; 1] = ["docx"];
const DOCUMENT_PART: &str = "word/document.xml";
const STYLES_PART: &str = "word/styles.xml";

/// Extracts the body of docx files: paragraphs and table cells in document order, with the
/// heading path from paragraph styles.
///
/// Each passage is located by the [`ElementPath`] of its first paragraph. Headers, footers,
/// footnotes, comments and text boxes are not read.
#[derive(Debug, Clone)]
pub struct Docx(ExtractorProfile);

impl Docx {
    /// Creates the extractor.
    ///
    /// # Errors
    ///
    /// Returns an error if the fixed profile id is not valid, which is a bug.
    pub fn new() -> Result<Self, cosmosearch_core::Error> {
        Ok(Docx(ExtractorProfile::new("docx-2")?))
    }
}

impl Extractor for Docx {
    fn profile(&self) -> &ExtractorProfile {
        &self.0
    }

    fn extensions(&self) -> &[&str] {
        &DOCX_EXTENSIONS
    }

    fn extract(&self, file: &[u8], limits: &ExtractLimits) -> Result<Extraction, ExtractFailure> {
        let mut archive = Archive::open(file, limits)?;
        let document = archive
            .part(DOCUMENT_PART)?
            .ok_or(ExtractFailure::Malformed)?;
        let styles = match archive.part(STYLES_PART)? {
            Some(xml) => HeadingStyles::parse(&xml, limits.depth())?,
            None => HeadingStyles::default(),
        };
        let found = paragraphs(&document, &styles, limits.depth())?;
        Ok(Extraction::new(passages(found)))
    }
}

#[derive(Default)]
struct Builder {
    headings: Vec<(u8, String)>,
    section: Vec<(ElementPath, String)>,
    found: Vec<Passage>,
}

fn passages(paragraphs: Vec<Paragraph>) -> Vec<Passage> {
    let mut builder = Builder::default();
    for paragraph in paragraphs {
        let text = paragraph.text.trim();
        if text.is_empty() {
            continue;
        }
        if let Some(level) = paragraph.heading {
            builder.finish_section();
            builder.headings.retain(|(open, _)| *open < level);
            builder.headings.push((level, text.to_owned()));
        }
        builder.section.push((paragraph.path, text.to_owned()));
    }
    builder.finish_section();
    builder.found
}

impl Builder {
    fn finish_section(&mut self) {
        let path: Vec<String> = self.headings.iter().map(|(_, t)| t.clone()).collect();
        let mut open: Option<(ElementPath, String)> = None;
        for (element, text) in std::mem::take(&mut self.section) {
            if text.len() > MAX_CHUNK_BYTES {
                self.push(open.take(), &path);
                for range in text_ranges(&text, 0..text.len()) {
                    if let Some(piece) = text.get(range) {
                        self.push(Some((element, piece.to_owned())), &path);
                    }
                }
                continue;
            }
            open = match open.take() {
                Some((first, mut joined))
                    if joined.len().saturating_add(1).saturating_add(text.len())
                        <= MAX_CHUNK_BYTES =>
                {
                    joined.push('\n');
                    joined.push_str(&text);
                    Some((first, joined))
                }
                Some(full) => {
                    self.push(Some(full), &path);
                    Some((element, text))
                }
                None => Some((element, text)),
            };
        }
        self.push(open, &path);
    }

    fn push(&mut self, passage: Option<(ElementPath, String)>, path: &[String]) {
        if let Some((element, text)) = passage {
            self.found
                .push(Passage::new(Locator::Element(element), path.to_vec(), text));
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use cosmosearch_core::LimitKind;

    use super::*;
    use crate::fixture::{Part, docx, zip};

    const STYLES: &str = r#"<w:styles xmlns:w="w">
        <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/></w:style>
        <w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/></w:style>
    </w:styles>"#;
    const HOSTILE_TIME: Duration = Duration::from_secs(2);

    fn p(text: &str) -> String {
        format!("<w:p><w:r><w:t>{text}</w:t></w:r></w:p>")
    }

    fn heading(level: u8, text: &str) -> String {
        format!(
            r#"<w:p><w:pPr><w:pStyle w:val="Heading{level}"/></w:pPr><w:r><w:t>{text}</w:t></w:r></w:p>"#
        )
    }

    fn extractor() -> Docx {
        Docx::new().unwrap()
    }

    fn extract(file: &[u8], limits: &ExtractLimits) -> Result<Vec<Passage>, ExtractFailure> {
        extractor()
            .extract(file, limits)
            .map(Extraction::into_passages)
    }

    fn shape(found: &[Passage]) -> Vec<(String, String, String)> {
        found
            .iter()
            .map(|p| {
                (
                    p.locator().to_string(),
                    p.heading_path().join(" > "),
                    p.text().to_owned(),
                )
            })
            .collect()
    }

    fn check_reads_back(file: &[u8]) -> usize {
        let limits = ExtractLimits::default();
        let found = extract(file, &limits).unwrap();
        let again = extract(file, &limits).unwrap();
        for passage in &found {
            let same: Vec<&str> = found
                .iter()
                .filter(|p| p.locator() == passage.locator())
                .map(Passage::text)
                .collect();
            let resolved: Vec<&str> = again
                .iter()
                .filter(|p| p.locator() == passage.locator())
                .map(Passage::text)
                .collect();
            assert_eq!(resolved, same, "{}", passage.locator());
        }
        found.len()
    }

    #[test]
    fn damaged_files_give_passages_or_an_error_and_never_a_panic() {
        let body = format!(
            "{}{}<w:tbl><w:tr><w:tc>{}</w:tc></w:tr></w:tbl>{}",
            heading(1, "Terms"),
            p("Refunds within 30 days &amp; more"),
            p("cell"),
            heading(2, "Fees"),
        );
        cosmosearch_core::extract_damaged(&extractor(), &docx(&body, Some(STYLES)));
        let limits = ExtractLimits::default();
        for (part, text) in [("body", body.as_str()), ("styles", STYLES)] {
            for at in 0..text.len() {
                for byte in [b'<', b'>', b'"', b'/', b'&', 0xff] {
                    let mut bad = text.as_bytes().to_vec();
                    bad[at] = byte;
                    let bad = String::from_utf8_lossy(&bad);
                    let file = if part == "body" {
                        docx(&bad, Some(STYLES))
                    } else {
                        docx(&body, Some(&bad))
                    };
                    let _outcome = extract(&file, &limits);
                }
            }
        }
    }

    #[test]
    fn groups_paragraphs_by_section_with_heading_paths() {
        let body = [
            p("Intro"),
            heading(1, "Scope"),
            p("All systems."),
            heading(2, "Cloud"),
            p("Hosted."),
            "<w:tbl><w:tr><w:tc><w:p/></w:tc><w:tc>".to_owned(),
            p("Cell text"),
            "</w:tc></w:tr></w:tbl>".to_owned(),
            heading(1, "Roles"),
            p("Owners."),
        ]
        .concat();
        let found = extract(&docx(&body, Some(STYLES)), &ExtractLimits::default()).unwrap();
        let expected = [
            ("element=b1/p1", "", "Intro"),
            ("element=b2/p1", "Scope", "Scope\nAll systems."),
            (
                "element=b4/p1",
                "Scope > Cloud",
                "Cloud\nHosted.\nCell text",
            ),
            ("element=b7/p1", "Roles", "Roles\nOwners."),
        ];
        let expected: Vec<(String, String, String)> = expected
            .iter()
            .map(|(l, h, t)| ((*l).to_owned(), (*h).to_owned(), (*t).to_owned()))
            .collect();
        assert_eq!(shape(&found), expected);
    }

    #[test]
    fn chunks_stay_within_the_limit_and_each_resolves_by_its_locator() {
        let sentence = "Access is reviewed every quarter by the system owner. ";
        let long = sentence.repeat(80);
        let mut body = p(&long);
        for _ in 0..120 {
            body.push_str(&p(sentence));
        }
        let file = docx(&body, None);
        let found = extract(&file, &ExtractLimits::default()).unwrap();
        assert!(found.len() > 4);
        assert!(found.iter().all(|p| p.text().len() <= MAX_CHUNK_BYTES));
        assert!(
            found
                .iter()
                .take(2)
                .all(|p| p.locator().to_string() == "element=b1/p1")
        );
        check_reads_back(&file);
    }

    #[test]
    fn without_a_styles_part_only_heading_ids_are_headings() {
        let body = format!(
            "{}{}",
            heading(2, "Known"),
            r#"<w:p><w:pPr><w:pStyle w:val="Custom"/></w:pPr><w:r><w:t>Body</w:t></w:r></w:p>"#
        );
        let found = extract(&docx(&body, None), &ExtractLimits::default()).unwrap();
        assert_eq!(found[0].heading_path(), ["Known"]);
        assert_eq!(found[0].text(), "Known\nBody");
    }

    #[test]
    fn a_missing_document_or_a_bad_zip_is_malformed() {
        let no_document = zip(&[Part::new("word/styles.xml", STYLES.as_bytes())]);
        for file in [no_document.as_slice(), b"PK\x03\x04", b""] {
            assert_eq!(
                extract(file, &ExtractLimits::default()).unwrap_err(),
                ExtractFailure::Malformed
            );
        }
    }

    #[test]
    fn enforces_archive_and_depth_limits() {
        let file = docx(&p("x").repeat(50), None);
        let two_parts = zip(&[Part::new("a", b""), Part::new("b", b"")]);
        let lying = zip(&[Part {
            declared: Some(u32::MAX),
            ..Part::new("word/document.xml", b"<w:document/>")
        }]);
        let b = ExtractLimits::builder;
        let cases = [
            (
                &two_parts,
                b().archive_entries(1),
                LimitKind::ArchiveEntries,
            ),
            (&file, b().archive_bytes(100), LimitKind::ArchiveBytes),
            (&file, b().depth(4), LimitKind::Depth),
            (&lying, b(), LimitKind::ArchiveBytes),
        ];
        for (file, builder, kind) in cases {
            assert_eq!(
                extract(file, &builder.build().unwrap()).unwrap_err(),
                ExtractFailure::LimitExceeded(kind),
                "{kind}"
            );
        }
    }

    #[test]
    fn hostile_files_fail_cleanly_and_quickly() {
        const HEAD: &str = r#"<?xml version="1.0" encoding="UTF-8"?>"#;
        const OPEN: &str = r#"<w:document xmlns:w="w"><w:body>"#;
        const CLOSE: &str = "</w:body></w:document>";
        let laughs = format!(
            r#"{HEAD}<!DOCTYPE d [<!ENTITY a "aaaaaaaaaa"><!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;"><!ENTITY c "&b;&b;&b;&b;&b;&b;&b;&b;&b;&b;"><!ENTITY d "&c;&c;&c;&c;&c;&c;&c;&c;&c;&c;">]>{OPEN}<w:p><w:r><w:t>&d;&d;&d;</w:t></w:r></w:p>{CLOSE}"#
        );
        let external = format!(
            r#"{HEAD}<!DOCTYPE d [<!ENTITY x SYSTEM "file:///etc/passwd">]>{OPEN}<w:p><w:r><w:t>&x;</w:t></w:r></w:p>{CLOSE}"#
        );
        let mut bad_utf8 = format!("{HEAD}{OPEN}<w:p><w:r><w:t>").into_bytes();
        bad_utf8.extend_from_slice(&[0xff, 0xfe, 0xc3]);
        bad_utf8.extend_from_slice(format!("</w:t></w:r></w:p>{CLOSE}").as_bytes());
        let unbalanced = format!("{HEAD}{OPEN}<w:p><w:r><w:t>text</w:r></w:t></w:p>{CLOSE}");
        let document = |body: &[u8]| zip(&[Part::new(DOCUMENT_PART, body)]);
        let control = docx(&p("text"), None);
        let truncated = control.get(..control.len() / 2).unwrap().to_vec();
        let cases = [
            ("billion-laughs", document(laughs.as_bytes())),
            ("external-entity", document(external.as_bytes())),
            ("bad-utf8", document(&bad_utf8)),
            ("unbalanced-tags", document(unbalanced.as_bytes())),
            ("garbage", b"\x00\x01 not an archive".repeat(64)),
            ("truncated", truncated),
        ];
        for (name, file) in cases {
            let started = Instant::now();
            let failure = extract(&file, &ExtractLimits::default()).unwrap_err();
            let took = started.elapsed();
            assert_eq!(failure, ExtractFailure::Malformed, "{name}");
            assert!(took < HOSTILE_TIME, "{name} took {took:?}");
        }
    }
}
