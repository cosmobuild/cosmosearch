use std::collections::BTreeMap;

use cosmosearch_core::ExtractFailure;

use crate::xml::{Step, XmlReader, attribute};

const OUTLINE_LEVELS: u8 = 9;
const MOST_STYLES_FOLLOWED: usize = 16;
const HEADING_ID: &str = "Heading";

#[derive(Default)]
struct Style {
    level: Option<u8>,
    based_on: Option<String>,
}

/// The heading level of each paragraph style, from 1 for the outermost.
///
/// A style is a heading when it is named `heading N` or sets an outline level, itself or
/// through the styles it is based on.
#[derive(Debug, Default)]
pub(crate) struct HeadingStyles(BTreeMap<String, u8>);

impl HeadingStyles {
    pub(crate) fn parse(xml: &[u8], max_depth: usize) -> Result<Self, ExtractFailure> {
        let mut styles: BTreeMap<String, Style> = BTreeMap::new();
        let mut reader = XmlReader::new(xml, max_depth);
        let mut current: Option<(String, Style)> = None;
        loop {
            match reader.next()? {
                Step::Open { element, empty } => {
                    let name = element.local_name();
                    match name.as_ref() {
                        b"style" if !empty => {
                            let paragraph =
                                attribute(&element, b"type").as_deref() == Some("paragraph");
                            current = attribute(&element, b"styleId")
                                .filter(|_| paragraph)
                                .map(|id| (id, Style::default()));
                        }
                        b"name" => {
                            if let Some((_, style)) = current.as_mut() {
                                let value = attribute(&element, b"val").unwrap_or_default();
                                style.level = style.level.or(heading_name_level(&value));
                            }
                        }
                        b"outlineLvl" => {
                            if let Some((_, style)) = current.as_mut() {
                                style.level = attribute(&element, b"val")
                                    .as_deref()
                                    .and_then(outline_level)
                                    .or(style.level);
                            }
                        }
                        b"basedOn" => {
                            if let Some((_, style)) = current.as_mut() {
                                style.based_on = attribute(&element, b"val");
                            }
                        }
                        _ => {}
                    }
                }
                Step::Close(name) if name == b"style" => {
                    if let Some((id, style)) = current.take() {
                        styles.insert(id, style);
                    }
                }
                Step::Close(_) | Step::Text(_) => {}
                Step::End => break,
            }
        }
        let mut levels = BTreeMap::new();
        for id in styles.keys() {
            if let Some(level) = resolve(&styles, id) {
                levels.insert(id.clone(), level);
            }
        }
        Ok(HeadingStyles(levels))
    }

    /// The heading level of the paragraph style `id`, if it is a heading. A style the styles
    /// part does not describe is a heading when its id is `HeadingN`.
    pub(crate) fn level(&self, id: &str) -> Option<u8> {
        self.0.get(id).copied().or_else(|| {
            let number = id
                .get(..HEADING_ID.len())
                .filter(|prefix| prefix.eq_ignore_ascii_case(HEADING_ID))
                .and(id.get(HEADING_ID.len()..))?;
            heading_number(number)
        })
    }
}

fn resolve(styles: &BTreeMap<String, Style>, id: &str) -> Option<u8> {
    let mut next = Some(id);
    for _ in 0..MOST_STYLES_FOLLOWED {
        let style = styles.get(next?)?;
        if style.level.is_some() {
            return style.level;
        }
        next = style.based_on.as_deref();
    }
    None
}

/// The heading level of a paragraph's own outline level, `0` being the outermost.
pub(crate) fn outline_level(value: &str) -> Option<u8> {
    let level: u8 = value.parse().ok()?;
    (level < OUTLINE_LEVELS).then(|| level.saturating_add(1))
}

fn heading_name_level(name: &str) -> Option<u8> {
    let (word, number) = name.split_once(' ')?;
    if !word.eq_ignore_ascii_case("heading") {
        return None;
    }
    heading_number(number)
}

fn heading_number(number: &str) -> Option<u8> {
    let level: u8 = number.parse().ok()?;
    (1..=OUTLINE_LEVELS).contains(&level).then_some(level)
}

#[cfg(test)]
mod tests {
    use super::*;

    const STYLES: &str = r#"<w:styles xmlns:w="w">
        <w:style w:type="paragraph" w:styleId="Normal"><w:name w:val="Normal"/></w:style>
        <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/></w:style>
        <w:style w:type="paragraph" w:styleId="LocalHeading2"><w:name w:val="heading 2"/></w:style>
        <w:style w:type="paragraph" w:styleId="Custom"><w:name w:val="Custom"/><w:pPr><w:outlineLvl w:val="2"/></w:pPr></w:style>
        <w:style w:type="paragraph" w:styleId="Derived"><w:basedOn w:val="Heading1"/></w:style>
        <w:style w:type="paragraph" w:styleId="Body"><w:pPr><w:outlineLvl w:val="9"/></w:pPr></w:style>
        <w:style w:type="paragraph" w:styleId="LoopA"><w:basedOn w:val="LoopB"/></w:style>
        <w:style w:type="paragraph" w:styleId="LoopB"><w:basedOn w:val="LoopA"/></w:style>
        <w:style w:type="character" w:styleId="Heading1Char"><w:name w:val="heading 1"/></w:style>
    </w:styles>"#;

    #[test]
    fn finds_headings_by_name_outline_level_and_base_style() {
        let styles = HeadingStyles::parse(STYLES.as_bytes(), 16).unwrap();
        let levels: Vec<(&str, Option<u8>)> = [
            "Normal",
            "Heading1",
            "LocalHeading2",
            "Custom",
            "Derived",
            "Body",
            "LoopA",
            "Heading1Char",
            "Missing",
            "Heading4",
            "Heading10",
        ]
        .into_iter()
        .map(|id| (id, styles.level(id)))
        .collect();
        assert_eq!(
            levels,
            [
                ("Normal", None),
                ("Heading1", Some(1)),
                ("LocalHeading2", Some(2)),
                ("Custom", Some(3)),
                ("Derived", Some(1)),
                ("Body", None),
                ("LoopA", None),
                ("Heading1Char", None),
                ("Missing", None),
                ("Heading4", Some(4)),
                ("Heading10", None),
            ]
        );
    }
}
