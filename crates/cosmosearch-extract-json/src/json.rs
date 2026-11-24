use std::num::NonZeroU64;

use cosmosearch_core::{
    ExtractFailure, ExtractLimits, Extraction, Extractor, ExtractorProfile, LimitKind, Locator,
    MAX_CHUNK_BYTES, Passage, text_ranges,
};
use serde_json::Deserializer;

use crate::node::Node;

const JSON_EXTENSIONS: [&str; 3] = ["json", "jsonl", "ndjson"];
const JSON_MEDIA_TYPES: [&str; 3] = [
    "application/json",
    "application/jsonl",
    "application/x-ndjson",
];
const BYTE_ORDER_MARK: &[u8] = b"\xEF\xBB\xBF";

/// Extracts JSON in UTF-8, and JSON Lines: a file holding several values is read as an array
/// of them.
///
/// A value that fits in one passage becomes one, its text a line per value inside it,
/// labelled with its path from there. A larger object or array is split between its members,
/// grouped while they fit. The heading path holds the reference tokens of the passage's
/// container, as in a JSON Pointer.
///
/// A passage is located by the number of its first value, counting every value from 1 in
/// document order, the outermost first.
#[derive(Debug, Clone)]
pub struct Json(ExtractorProfile);

impl Json {
    /// Creates the extractor.
    ///
    /// # Errors
    ///
    /// Returns an error if the fixed profile id is not valid, which is a bug.
    pub fn new() -> Result<Self, cosmosearch_core::Error> {
        Ok(Json(ExtractorProfile::new("json-2")?))
    }
}

impl Extractor for Json {
    fn profile(&self) -> &ExtractorProfile {
        &self.0
    }

    fn extensions(&self) -> &[&str] {
        &JSON_EXTENSIONS
    }

    fn media_types(&self) -> &[&str] {
        &JSON_MEDIA_TYPES
    }

    fn extract(&self, file: &[u8], limits: &ExtractLimits) -> Result<Extraction, ExtractFailure> {
        let file = file.strip_prefix(BYTE_ORDER_MARK).unwrap_or(file);
        let values: Vec<Node> = Deserializer::from_slice(file)
            .into_iter::<Node>()
            .collect::<Result<_, _>>()
            .map_err(|_cause| ExtractFailure::Malformed)?;
        let root = match <[Node; 1]>::try_from(values) {
            Ok([value]) => value,
            Err(values) if values.is_empty() => return Ok(Extraction::new(Vec::new())),
            Err(values) => Node::Array(values),
        };
        if depth(&root) > limits.depth() {
            return Err(ExtractFailure::LimitExceeded(LimitKind::Depth));
        }
        let mut walker = Walker {
            limits: *limits,
            next: NonZeroU64::MIN,
            chars: 0,
            path: Vec::new(),
            found: Vec::new(),
        };
        walker.split(&root)?;
        Ok(Extraction::new(walker.found))
    }
}

struct Walker {
    limits: ExtractLimits,
    next: NonZeroU64,
    chars: usize,
    path: Vec<String>,
    found: Vec<Passage>,
}

impl Walker {
    /// Adds `value` as one passage if it fits, or else its members.
    fn split(&mut self, value: &Node) -> Result<(), ExtractFailure> {
        let first = self.next;
        if let Some(text) = render(value) {
            self.skip(value)?;
            return self.push(first, text);
        }
        self.step()?;
        let members = value.members();
        if members.is_empty() {
            let text = value.scalar();
            for range in text_ranges(text, 0..text.len()) {
                if let Some(piece) = text.get(range) {
                    self.push(first, piece.to_owned())?;
                }
            }
            return Ok(());
        }
        let mut open: Option<(NonZeroU64, String)> = None;
        for (token, member) in members {
            let start = self.next;
            let mut text = String::new();
            if render_into(member, &token, &mut text) {
                self.skip(member)?;
                open = match open.take() {
                    Some((at, mut joined))
                        if joined.len().saturating_add(1).saturating_add(text.len())
                            <= MAX_CHUNK_BYTES =>
                    {
                        joined.push('\n');
                        joined.push_str(&text);
                        Some((at, joined))
                    }
                    Some((at, joined)) => {
                        self.push(at, joined)?;
                        Some((start, text))
                    }
                    None => Some((start, text)),
                };
            } else {
                if let Some((at, joined)) = open.take() {
                    self.push(at, joined)?;
                }
                self.path.push(token);
                self.split(member)?;
                self.path.pop();
            }
        }
        if let Some((at, joined)) = open {
            self.push(at, joined)?;
        }
        Ok(())
    }

    fn push(&mut self, first: NonZeroU64, text: String) -> Result<(), ExtractFailure> {
        self.chars = self.chars.saturating_add(text.chars().count());
        if self.chars > self.limits.chars() {
            return Err(ExtractFailure::LimitExceeded(LimitKind::Chars));
        }
        self.found
            .push(Passage::new(Locator::Value(first), self.path.clone(), text));
        Ok(())
    }

    fn step(&mut self) -> Result<(), ExtractFailure> {
        self.next = self.next.checked_add(1).ok_or(ExtractFailure::Malformed)?;
        Ok(())
    }

    /// Moves past `value` and everything inside it.
    fn skip(&mut self, value: &Node) -> Result<(), ExtractFailure> {
        self.step()?;
        for (_, member) in value.members() {
            self.skip(member)?;
        }
        Ok(())
    }
}

/// How many objects and arrays are nested at the deepest point of `value`.
fn depth(value: &Node) -> usize {
    let inner = value
        .members()
        .into_iter()
        .map(|(_, member)| depth(member))
        .max();
    inner.map_or(0, |deepest| deepest.saturating_add(1))
}

/// The text of `value` as one passage, or `None` if it is longer than one passage may be.
fn render(value: &Node) -> Option<String> {
    let mut text = String::new();
    render_into(value, "", &mut text).then_some(text)
}

/// Appends a line per value inside `value`, labelled with its path below `label`. Stops and
/// returns `false` as soon as `out` is longer than one passage may be.
fn render_into(value: &Node, label: &str, out: &mut String) -> bool {
    let members = value.members();
    if members.is_empty() {
        if !out.is_empty() {
            out.push('\n');
        }
        if !label.is_empty() {
            out.push_str(label);
            out.push_str(": ");
        }
        out.push_str(value.scalar());
        return out.len() <= MAX_CHUNK_BYTES;
    }
    for (token, member) in members {
        let inner = if label.is_empty() {
            token
        } else {
            format!("{label}/{token}")
        };
        if !render_into(member, &inner, out) {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extractor() -> Json {
        Json::new().unwrap()
    }

    fn extract_with(file: &str, limits: &ExtractLimits) -> Result<Vec<Passage>, ExtractFailure> {
        extractor()
            .extract(file.as_bytes(), limits)
            .map(Extraction::into_passages)
    }

    fn extract(file: &str) -> Vec<Passage> {
        extract_with(file, &ExtractLimits::default()).unwrap()
    }

    fn shape(found: &[Passage]) -> Vec<(String, String, String)> {
        found
            .iter()
            .map(|p| {
                (
                    p.locator().to_string(),
                    p.heading_path().join("/"),
                    p.text().to_owned(),
                )
            })
            .collect()
    }

    fn expect(rows: &[(&str, &str, &str)]) -> Vec<(String, String, String)> {
        rows.iter()
            .map(|(l, h, t)| ((*l).to_owned(), (*h).to_owned(), (*t).to_owned()))
            .collect()
    }

    #[test]
    fn damaged_files_give_passages_or_an_error_and_never_a_panic() {
        let good = br#"{"title": "Terms", "items": [{"name": "refund \u00e9", "days": 30}, [1, 2.5e3, null, true]], "notes": {"a": {"b": "deep"}}}"#;
        cosmosearch_core::extract_damaged(&extractor(), good);
    }

    #[test]
    fn a_small_document_is_one_passage_with_labelled_lines() {
        let found =
            extract(r#"{"name": "Ada", "tags": ["x", "y"], "meta": {}, "n": null, "ok": true}"#);
        assert_eq!(
            shape(&found),
            expect(&[(
                "value=1",
                "",
                "name: Ada\ntags/0: x\ntags/1: y\nmeta: {}\nn: null\nok: true"
            )])
        );
    }

    #[test]
    fn a_large_array_is_grouped_and_numbered_in_document_order() {
        let note = "n".repeat(900);
        let item = |id: &str| format!(r#"{{"id": {id}, "note": "{note}"}}"#);
        let file = format!(
            r#"{{"title": "Report", "items": [{}, {}, {}], "end": 0}}"#,
            item("1"),
            item("2"),
            item("3")
        );
        let line = |index: &str, id: &str| format!("{index}/id: {id}\n{index}/note: {note}");
        assert_eq!(
            shape(&extract(&file)),
            vec![
                (
                    "value=2".to_owned(),
                    String::new(),
                    "title: Report".to_owned()
                ),
                (
                    "value=4".to_owned(),
                    "items".to_owned(),
                    format!("{}\n{}", line("0", "1"), line("1", "2"))
                ),
                ("value=10".to_owned(), "items".to_owned(), line("2", "3")),
                ("value=13".to_owned(), String::new(), "end: 0".to_owned()),
            ]
        );
    }

    #[test]
    fn a_long_string_is_split_and_keeps_its_value() {
        let file = format!(r#"{{"body": "{}"}}"#, "word ".repeat(1_000));
        let found = extract(&file);
        assert!(found.len() > 1);
        assert!(found.iter().all(|p| p.locator().to_string() == "value=2"));
        assert!(found.iter().all(|p| p.heading_path() == ["body"]));
        assert!(found.iter().all(|p| p.text().len() <= MAX_CHUNK_BYTES));
    }

    #[test]
    fn json_lines_read_as_an_array() {
        let found = extract("{\"a\": 1}\n\n{\"a\": 2}\n");
        assert_eq!(shape(&found), expect(&[("value=1", "", "0/a: 1\n1/a: 2")]));
    }

    #[test]
    fn every_passage_resolves_by_its_value() {
        let mut items = Vec::new();
        for id in 0..200 {
            items.push(format!(
                r#"{{"id": {id}, "text": "entry number {id} of the list"}}"#
            ));
        }
        let file = format!(
            r#"{{"list": [{}], "long": "{}"}}"#,
            items.join(","),
            "x ".repeat(2_000)
        );
        let found = extract(&file);
        let again = extract(&file);
        assert!(found.len() > 4);
        for passage in &found {
            let pick = |list: &[Passage]| -> Vec<String> {
                list.iter()
                    .filter(|p| p.locator() == passage.locator())
                    .map(|p| p.text().to_owned())
                    .collect()
            };
            assert_eq!(pick(&again), pick(&found));
        }
    }

    #[test]
    fn empty_input_has_no_text_and_broken_input_is_malformed() {
        assert_eq!(extract("\u{feff}  \n").len(), 0);
        for file in ["{", "[1,]", "{\"a\": 1} x"] {
            assert_eq!(
                extract_with(file, &ExtractLimits::default()).unwrap_err(),
                ExtractFailure::Malformed,
                "{file}"
            );
        }
        let bad = Json::new()
            .unwrap()
            .extract(b"\"\xff\"", &ExtractLimits::default());
        assert_eq!(bad.unwrap_err(), ExtractFailure::Malformed);
    }

    #[test]
    fn enforces_depth_and_character_limits() {
        let b = ExtractLimits::builder;
        let wide = format!("[{}]", vec!["\"x\""; 3_000].join(","));
        let cases = [
            ("[[[[1]]]]", b().depth(3), LimitKind::Depth),
            (wide.as_str(), b().chars(100), LimitKind::Chars),
        ];
        for (file, builder, kind) in cases {
            assert_eq!(
                extract_with(file, &builder.build().unwrap()).unwrap_err(),
                ExtractFailure::LimitExceeded(kind),
                "{kind}"
            );
        }
        extract_with("[[[1]]]", &b().depth(3).build().unwrap()).unwrap();
    }
}
