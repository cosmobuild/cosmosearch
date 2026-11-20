use std::borrow::Cow;

use cosmosearch_core::{ExtractFailure, LimitKind};
use quick_xml::escape::resolve_predefined_entity;
use quick_xml::events::{BytesStart, Event};
use quick_xml::{Reader, XmlVersion};

/// One step through an XML part, with only what the docx reader needs.
pub(crate) enum Step<'xml> {
    /// An element opens; `empty` when it closes at once.
    Open {
        element: BytesStart<'xml>,
        empty: bool,
    },
    /// An element closes.
    Close(Vec<u8>),
    /// Character data, with references resolved.
    Text(String),
    /// The end of the part.
    End,
}

/// Reads an XML part with a depth limit, refusing a DOCTYPE and any entity other than the
/// five XML predefines, so nothing is ever expanded.
pub(crate) struct XmlReader<'xml> {
    reader: Reader<&'xml [u8]>,
    depth: usize,
    max_depth: usize,
}

impl<'xml> XmlReader<'xml> {
    pub(crate) fn new(xml: &'xml [u8], max_depth: usize) -> Self {
        XmlReader {
            reader: Reader::from_reader(xml),
            depth: 0,
            max_depth,
        }
    }

    /// How many elements are open.
    pub(crate) fn depth(&self) -> usize {
        self.depth
    }

    pub(crate) fn next(&mut self) -> Result<Step<'xml>, ExtractFailure> {
        loop {
            let Ok(event) = self.reader.read_event() else {
                return Err(ExtractFailure::Malformed);
            };
            match event {
                Event::Start(element) => {
                    self.depth = self.depth.saturating_add(1);
                    if self.depth > self.max_depth {
                        return Err(ExtractFailure::LimitExceeded(LimitKind::Depth));
                    }
                    return Ok(Step::Open {
                        element,
                        empty: false,
                    });
                }
                Event::Empty(element) => {
                    if self.depth >= self.max_depth {
                        return Err(ExtractFailure::LimitExceeded(LimitKind::Depth));
                    }
                    return Ok(Step::Open {
                        element,
                        empty: true,
                    });
                }
                Event::End(element) => {
                    self.depth = self.depth.saturating_sub(1);
                    return Ok(Step::Close(element.local_name().as_ref().to_vec()));
                }
                Event::Text(text) => {
                    let text = text
                        .xml10_content()
                        .map_err(|_cause| ExtractFailure::Malformed)?;
                    return Ok(Step::Text(text.into_owned()));
                }
                Event::CData(text) => {
                    let text = text.decode().map_err(|_cause| ExtractFailure::Malformed)?;
                    return Ok(Step::Text(text.into_owned()));
                }
                Event::GeneralRef(reference) => {
                    if let Ok(Some(c)) = reference.resolve_char_ref() {
                        return Ok(Step::Text(c.to_string()));
                    }
                    let name = reference
                        .decode()
                        .map_err(|_cause| ExtractFailure::Malformed)?;
                    let text = resolve_predefined_entity(&name).ok_or(ExtractFailure::Malformed)?;
                    return Ok(Step::Text(text.to_owned()));
                }
                Event::Eof if self.depth == 0 => return Ok(Step::End),
                Event::DocType(_) | Event::Eof => return Err(ExtractFailure::Malformed),
                Event::Comment(_) | Event::Decl(_) | Event::PI(_) => {}
            }
        }
    }
}

/// The unescaped value of the attribute with local name `name`.
pub(crate) fn attribute(element: &BytesStart<'_>, name: &[u8]) -> Option<String> {
    element
        .attributes()
        .flatten()
        .find(|attribute| attribute.key.local_name().as_ref() == name)
        .and_then(|attribute| attribute.normalized_value(XmlVersion::Implicit1_0).ok())
        .map(Cow::into_owned)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(xml: &str, max_depth: usize) -> Result<String, ExtractFailure> {
        let mut reader = XmlReader::new(xml.as_bytes(), max_depth);
        let mut out = String::new();
        loop {
            match reader.next()? {
                Step::Text(text) => out.push_str(&text),
                Step::End => return Ok(out),
                Step::Open { .. } | Step::Close(_) => {}
            }
        }
    }

    #[test]
    fn resolves_predefined_and_character_references() {
        assert_eq!(
            texts("<a>x &amp; &lt;y&gt; &#65;&#x42;</a>", 8).unwrap(),
            "x & <y> AB"
        );
    }

    #[test]
    fn refuses_a_doctype_unknown_entities_and_broken_markup() {
        for xml in [
            "<!DOCTYPE a [<!ENTITY e \"x\">]><a>&e;</a>",
            "<a>&e;</a>",
            "<a><b></a>",
            "<a>",
            "<a>\u{0}</b>",
        ] {
            assert_eq!(
                texts(xml, 8).unwrap_err(),
                ExtractFailure::Malformed,
                "{xml}"
            );
        }
    }

    #[test]
    fn enforces_the_depth_limit_on_open_and_empty_elements() {
        assert_eq!(texts("<a><b>x</b></a>", 2).unwrap(), "x");
        for xml in ["<a><b><c/></b></a>", "<a><b><c></c></b></a>"] {
            assert_eq!(
                texts(xml, 2).unwrap_err(),
                ExtractFailure::LimitExceeded(LimitKind::Depth),
                "{xml}"
            );
        }
    }

    #[test]
    fn reads_an_attribute_by_local_name() {
        let mut reader = XmlReader::new(br#"<w:pStyle w:val="Heading1"/>"#, 4);
        let Ok(Step::Open { element, empty }) = reader.next() else {
            panic!("expected an element");
        };
        assert!(empty);
        assert_eq!(attribute(&element, b"val").as_deref(), Some("Heading1"));
    }
}
