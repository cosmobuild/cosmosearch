use std::collections::HashSet;
use std::num::NonZeroU32;

use cosmosearch_core::{
    ExtractFailure, ExtractLimits, Extraction, Extractor, ExtractorProfile, LimitKind, Locator,
    Passage, text_ranges,
};
use pdf_inspector::extractor::extract_text_with_positions_mem_pages;
use pdf_inspector::{PageMarkdown, classify_pdf_mem, extract_pages_markdown_mem};

use crate::decoded;
use crate::ligature::{close_gaps, has_broken_ligatures};
use crate::markup::{Page, clean_page};
use crate::rows::{Placed, is_split, rows};
use crate::running::remove_running_lines;

const PDF_EXTENSIONS: [&str; 1] = ["pdf"];
const PDF_MEDIA_TYPES: [&str; 1] = ["application/pdf"];
const MIN_LINES_TO_JUDGE_HEADINGS: usize = 20;
const MAX_HEADINGS_PER_THREE_LINES: usize = 1;
const THREE_LINES: usize = 3;

/// Extracts PDFs with a text layer. Pages without usable text, such as scans, are left out
/// and counted as unread parts.
#[derive(Debug, Clone)]
pub struct Pdf(ExtractorProfile);

impl Pdf {
    /// Creates the extractor.
    ///
    /// # Errors
    ///
    /// Returns an error if the fixed profile id is not valid, which is a bug.
    pub fn new() -> Result<Self, cosmosearch_core::Error> {
        Ok(Pdf(ExtractorProfile::new("pdf-4")?))
    }
}

impl Extractor for Pdf {
    fn profile(&self) -> &ExtractorProfile {
        &self.0
    }

    fn extensions(&self) -> &[&str] {
        &PDF_EXTENSIONS
    }

    fn media_types(&self) -> &[&str] {
        &PDF_MEDIA_TYPES
    }

    fn extract(&self, file: &[u8], limits: &ExtractLimits) -> Result<Extraction, ExtractFailure> {
        decoded::check(file, limits.decoded_bytes())?;
        let Ok(classified) = classify_pdf_mem(file) else {
            return Err(ExtractFailure::Malformed);
        };
        if usize::try_from(classified.page_count).map_or(true, |pages| pages > limits.pages()) {
            return Err(ExtractFailure::LimitExceeded(LimitKind::Pages));
        }
        let Ok(extraction) = extract_pages_markdown_mem(file, None) else {
            return Err(ExtractFailure::Malformed);
        };
        let unread = extraction.pages.iter().filter(|page| unread(page)).count();
        let mut pages: Vec<Page> = extraction
            .pages
            .iter()
            .filter(|page| !page.needs_ocr)
            .filter_map(|page| {
                let number = page.page.checked_add(1).and_then(NonZeroU32::new)?;
                Some(Page {
                    number,
                    lines: clean_page(&page.markdown),
                })
            })
            .collect();
        rebuild_split_pages(file, &mut pages);
        let unread = u32::try_from(unread).unwrap_or(u32::MAX);
        Ok(Extraction::new(passages(pages)).with_unread_parts(unread))
    }
}

/// Whether a page held something the parser could not read as text: a scan, garbled or
/// vector text, or text it judged unusable. A blank page is not unread.
fn unread(page: &PageMarkdown) -> bool {
    page.needs_ocr && (page.ocr_reason.is_some() || !page.markdown.trim().is_empty())
}

/// Rebuilds, from where their text sits, the lines of the pages whose tables the parser read
/// column by column. If the positions cannot be read the parser's lines stay.
fn rebuild_split_pages(file: &[u8], pages: &mut [Page]) {
    let split: HashSet<u32> = pages
        .iter()
        .filter(|page| is_split(&page.lines))
        .map(|page| page.number.get())
        .collect();
    if split.is_empty() {
        return;
    }
    let Ok(items) = extract_text_with_positions_mem_pages(file, Some(&split)) else {
        return;
    };
    for page in pages.iter_mut() {
        let number = page.number.get();
        if !split.contains(&number) {
            continue;
        }
        let placed: Vec<Placed> = items
            .iter()
            .filter_map(|item| Placed::of(item, number))
            .collect();
        let rebuilt = rows(placed, &page.lines);
        if !rebuilt.is_empty() {
            page.lines = rebuilt;
        }
    }
}

fn passages(mut pages: Vec<Page>) -> Vec<Passage> {
    remove_running_lines(&mut pages);
    let texts = pages
        .iter()
        .flat_map(|page| page.lines.iter().map(|line| line.text.as_str()));
    if has_broken_ligatures(texts) {
        for line in pages.iter_mut().flat_map(|page| page.lines.iter_mut()) {
            line.text = close_gaps(&line.text);
        }
    }
    if headings_are_unreliable(&pages) {
        for line in pages.iter_mut().flat_map(|page| page.lines.iter_mut()) {
            line.level = None;
        }
    }
    let mut builder = Builder::default();
    let mut last = None;
    for page in pages {
        last = Some(page.number);
        builder.add_page(page);
    }
    if let Some(number) = last {
        builder.finish_section(number);
    }
    builder.found
}

fn headings_are_unreliable(pages: &[Page]) -> bool {
    let lines = pages
        .iter()
        .flat_map(|page| page.lines.iter())
        .filter(|line| !line.text.is_empty());
    let (total, headings) = lines.fold((0usize, 0usize), |(total, headings), line| {
        (
            total.saturating_add(1),
            headings.saturating_add(usize::from(line.level.is_some())),
        )
    });
    total >= MIN_LINES_TO_JUDGE_HEADINGS
        && headings.saturating_mul(THREE_LINES) > total.saturating_mul(MAX_HEADINGS_PER_THREE_LINES)
}

#[derive(Default)]
struct Builder {
    headings: Vec<(usize, String)>,
    section: String,
    has_body: bool,
    found: Vec<Passage>,
}

impl Builder {
    fn add_page(&mut self, page: Page) {
        let mut page_has_body = false;
        for line in page.lines {
            if let Some(level) = line.level
                && !line.text.is_empty()
            {
                if self.has_body {
                    self.finish_section(page.number);
                }
                self.headings.retain(|(open, _)| *open < level);
                self.headings.push((level, line.text.clone()));
            } else if !line.text.is_empty() {
                self.has_body = true;
                page_has_body = true;
            }
            self.section.push_str(&line.text);
            self.section.push('\n');
        }
        if self.has_body || !page_has_body {
            self.finish_section(page.number);
        }
    }

    fn finish_section(&mut self, number: NonZeroU32) {
        self.has_body = false;
        let section = std::mem::take(&mut self.section);
        let path: Vec<String> = self.headings.iter().map(|(_, t)| t.clone()).collect();
        for range in text_ranges(&section, 0..section.len()) {
            if let Some(text) = section.get(range) {
                self.found
                    .push(Passage::new(Locator::Page(number), path.clone(), text));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::markup::Line;

    fn page(number: u32, lines: &[(Option<usize>, &str)]) -> Page {
        Page {
            number: NonZeroU32::new(number).unwrap(),
            lines: lines
                .iter()
                .map(|(level, text)| Line {
                    level: *level,
                    text: (*text).to_owned(),
                })
                .collect(),
        }
    }

    #[expect(
        clippy::arithmetic_side_effects,
        clippy::format_push_string,
        reason = "a fixture builder working on a few small test pages"
    )]
    fn pdf_with_pages(pages: &[&str]) -> Vec<u8> {
        let mut objects = vec![
            "<< /Type /Catalog /Pages 2 0 R >>".to_owned(),
            String::new(),
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_owned(),
        ];
        let mut kids = Vec::new();
        for text in pages {
            if *text == "[image]" {
                let content = "q 612 0 0 792 0 0 cm /Im1 Do Q";
                objects.push(format!(
                    "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {} 0 R \
                     /Resources << /XObject << /Im1 {} 0 R >> >> >>",
                    objects.len() + 2,
                    objects.len() + 3
                ));
                kids.push(format!("{} 0 R", objects.len()));
                objects.push(format!(
                    "<< /Length {} >>\nstream\n{content}\nendstream",
                    content.len()
                ));
                objects.push(
                    "<< /Type /XObject /Subtype /Image /Width 1000 /Height 1000 /ColorSpace /DeviceGray \
                     /BitsPerComponent 8 /Length 1 >>\nstream\nx\nendstream"
                        .to_owned(),
                );
                continue;
            }
            let content = format!("BT /F1 12 Tf 72 720 Td ({text}) Tj ET");
            objects.push(format!(
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {} 0 R \
                 /Resources << /Font << /F1 3 0 R >> >> >>",
                objects.len() + 2
            ));
            kids.push(format!("{} 0 R", objects.len()));
            objects.push(format!(
                "<< /Length {} >>\nstream\n{content}\nendstream",
                content.len()
            ));
        }
        objects[1] = format!(
            "<< /Type /Pages /Kids [{}] /Count {} >>",
            kids.join(" "),
            pages.len()
        );
        let mut file = String::from("%PDF-1.7\n");
        let mut offsets = Vec::new();
        for (index, object) in objects.iter().enumerate() {
            offsets.push(file.len());
            file.push_str(&format!("{} 0 obj\n{object}\nendobj\n", index + 1));
        }
        let xref = file.len();
        file.push_str(&format!(
            "xref\n0 {}\n0000000000 65535 f \n",
            objects.len() + 1
        ));
        for offset in offsets {
            file.push_str(&format!("{offset:010} 00000 n \n"));
        }
        file.push_str(&format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n",
            objects.len() + 1
        ));
        file.into_bytes()
    }

    #[expect(
        clippy::arithmetic_side_effects,
        reason = "a fixture builder working on one small test page"
    )]
    fn pdf_with_compressed_page(text: &str, decoded_bytes: usize, extra: &str) -> Vec<u8> {
        use std::io::Write;

        let mut content = format!("BT /F1 12 Tf 72 720 Td ({text}) Tj ET").into_bytes();
        content.resize(decoded_bytes, b' ');
        let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::best());
        encoder.write_all(&content).unwrap();
        let stream = encoder.finish().unwrap();
        let objects: [Vec<u8>; 5] = [
            b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R \
              /Resources << /Font << /F1 5 0 R >> >> >>"
                .to_vec(),
            [
                format!(
                    "<< /Length {} /Filter /FlateDecode {extra} >>\nstream\n",
                    stream.len()
                )
                .as_bytes(),
                &stream,
                b"\nendstream",
            ]
            .concat(),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec(),
        ];
        let mut file = b"%PDF-1.7\n".to_vec();
        let mut offsets = Vec::new();
        for (index, object) in objects.iter().enumerate() {
            offsets.push(file.len());
            file.extend_from_slice(format!("{} 0 obj\n", index + 1).as_bytes());
            file.extend_from_slice(object);
            file.extend_from_slice(b"\nendobj\n");
        }
        let xref = file.len();
        file.extend_from_slice(b"xref\n0 6\n0000000000 65535 f \n");
        for offset in offsets {
            file.extend_from_slice(format!("{offset:010} 00000 n \n").as_bytes());
        }
        file.extend_from_slice(
            format!("trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n").as_bytes(),
        );
        file
    }

    #[test]
    fn refuses_streams_that_decode_past_the_limit_before_parsing_them() {
        const MIB: usize = 1024 * 1024;
        let small = ExtractLimits::builder()
            .decoded_bytes(1024 * 1024)
            .build()
            .unwrap();
        let over = ExtractFailure::LimitExceeded(LimitKind::DecodedBytes);
        for extra in ["", "/Type /XObject /Subtype /Image"] {
            let bomb = pdf_with_compressed_page("Refunds take thirty days.", 2 * MIB, extra);
            assert!(bomb.len() < 16 * 1024, "{}", bomb.len());
            assert_eq!(extract(&bomb, &small).unwrap_err(), over, "{extra}");
            let fits = pdf_with_compressed_page("Refunds take thirty days.", MIB / 2, extra);
            assert_eq!(
                extract(&fits, &small).unwrap()[0].text(),
                "Refunds take thirty days.",
                "{extra}"
            );
        }
    }

    #[test]
    fn a_page_that_is_only_an_image_is_counted_unread_and_a_blank_page_is_not() {
        let file = pdf_with_pages(&["Refunds take thirty days.", "[image]", ""]);
        let found = extractor()
            .extract(&file, &ExtractLimits::default())
            .unwrap();
        assert_eq!(found.unread_parts(), 1);
        assert_eq!(found.passages().len(), 1);
    }

    fn extractor() -> Pdf {
        Pdf::new().unwrap()
    }

    fn extract(file: &[u8], limits: &ExtractLimits) -> Result<Vec<Passage>, ExtractFailure> {
        extractor()
            .extract(file, limits)
            .map(Extraction::into_passages)
    }

    #[test]
    fn each_passage_is_located_by_its_page() {
        let file = pdf_with_pages(&["first page words", "second page words"]);
        let found = extract(&file, &ExtractLimits::default()).unwrap();
        let located: Vec<(String, &str)> = found
            .iter()
            .map(|p| (p.locator().to_string(), p.text()))
            .collect();
        assert_eq!(
            located,
            [
                ("page=1".to_owned(), "first page words"),
                ("page=2".to_owned(), "second page words"),
            ]
        );
    }

    #[test]
    fn extraction_repeats_exactly() {
        let file = pdf_with_pages(&["same words each time"]);
        let limits = ExtractLimits::default();
        assert_eq!(
            extract(&file, &limits).unwrap(),
            extract(&file, &limits).unwrap()
        );
    }

    #[test]
    fn refuses_too_many_pages() {
        let file = pdf_with_pages(&["one", "two", "three"]);
        let limits = ExtractLimits::builder().pages(2).build().unwrap();
        assert_eq!(
            extract(&file, &limits).unwrap_err(),
            ExtractFailure::LimitExceeded(LimitKind::Pages)
        );
    }

    #[test]
    fn a_file_that_is_not_a_pdf_is_malformed() {
        for file in [&b"not a pdf"[..], b"", b"%PDF-1.7\ngarbage"] {
            assert_eq!(
                extract(file, &ExtractLimits::default()).unwrap_err(),
                ExtractFailure::Malformed,
                "{file:?}"
            );
        }
    }

    #[test]
    fn a_page_with_no_text_gives_no_passages() {
        let file = pdf_with_pages(&[""]);
        assert_eq!(extract(&file, &ExtractLimits::default()).unwrap().len(), 0);
    }

    #[test]
    fn headings_carry_over_pages_and_nest() {
        let found = passages(vec![
            page(
                1,
                &[(Some(1), "A"), (None, "one"), (Some(2), "B"), (None, "two")],
            ),
            page(2, &[(None, "three"), (Some(1), "C"), (None, "four")]),
        ]);
        let shape: Vec<(String, Vec<&str>, &str)> = found
            .iter()
            .map(|p| {
                let path = p.heading_path().iter().map(String::as_str).collect();
                (p.locator().to_string(), path, p.text())
            })
            .collect();
        let expected = |page: &str, path: &[&'static str], text: &'static str| {
            (page.to_owned(), path.to_vec(), text)
        };
        assert_eq!(
            shape,
            [
                expected("page=1", &["A"], "A\none"),
                expected("page=1", &["A", "B"], "B\ntwo"),
                expected("page=2", &["A", "B"], "three"),
                expected("page=2", &["C"], "C\nfour"),
            ]
        );
    }

    #[test]
    fn a_heading_with_no_text_joins_the_section_after_it() {
        let found = passages(vec![
            page(1, &[(Some(1), "Part"), (Some(2), "Topic"), (None, "words")]),
            page(2, &[(None, "more"), (Some(2), "Last")]),
            page(3, &[(None, "end"), (Some(2), "Tail")]),
        ]);
        let shape: Vec<(String, String, &str)> = found
            .iter()
            .map(|p| {
                (
                    p.locator().to_string(),
                    p.heading_path().join(">"),
                    p.text(),
                )
            })
            .collect();
        let expected =
            |page: &str, path: &str, text: &'static str| (page.to_owned(), path.to_owned(), text);
        assert_eq!(
            shape,
            [
                expected("page=1", "Part>Topic", "Part\nTopic\nwords"),
                expected("page=2", "Part>Topic", "more"),
                expected("page=3", "Part>Last", "Last\nend"),
                expected("page=3", "Part>Tail", "Tail"),
            ]
        );
    }

    #[test]
    fn a_page_of_headings_alone_keeps_its_own_passage() {
        let found = passages(vec![
            page(1, &[(Some(1), "Title"), (Some(2), "Subtitle")]),
            page(2, &[(None, "body")]),
        ]);
        let located: Vec<(String, &str)> = found
            .iter()
            .map(|p| (p.locator().to_string(), p.text()))
            .collect();
        assert_eq!(
            located,
            [
                ("page=1".to_owned(), "Title\nSubtitle"),
                ("page=2".to_owned(), "body"),
            ]
        );
    }

    #[test]
    fn ignores_headings_when_too_many_lines_claim_to_be_one() {
        let many: Vec<(Option<usize>, &str)> = (0..30)
            .map(|i| ((i % 2 == 0).then_some(1), "ordinary body line"))
            .collect();
        let found = passages(vec![page(1, &many)]);
        assert!(found.iter().all(|p| p.heading_path().is_empty()));
        assert_eq!(found.len(), 1);

        let few: Vec<(Option<usize>, &str)> = (0..30)
            .map(|i| ((i % 5 == 0).then_some(1), "ordinary body line"))
            .collect();
        assert!(
            passages(vec![page(1, &few)])
                .iter()
                .all(|p| !p.heading_path().is_empty())
        );
    }

    #[test]
    fn repairs_ligatures_only_in_a_document_that_needs_it() {
        let broken = passages(vec![page(1, &[(None, "fi ndings fi nal")])]);
        assert_eq!(broken[0].text(), "findings final");
        let healthy = passages(vec![
            page(1, &[(None, "wifi network")]),
            page(2, &[(None, "final file filter")]),
        ]);
        assert_eq!(healthy[0].text(), "wifi network");
    }

    #[test]
    fn splits_a_long_page_and_keeps_the_page_on_each_part() {
        let long = "word ".repeat(1_500);
        let found = passages(vec![page(7, &[(None, long.trim_end())])]);
        assert!(found.len() > 1);
        assert!(found.iter().all(|p| p.locator().to_string() == "page=7"));
    }
}
