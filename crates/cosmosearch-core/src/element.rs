use std::fmt;
use std::num::NonZeroU32;

/// A paragraph of a word-processing document, by its position in the document body.
///
/// Printed as `b<child>/p<n>`, or `b<child>/r<row>/c<cell>/p<n>` inside a table, every number
/// counting from 1:
///
/// - `child` counts every element directly inside the body, whatever its kind.
/// - Outside tables, `n` counts the paragraphs inside that child in document order; a child
///   that is one paragraph is `p1`.
/// - Inside a table, `row` counts the rows of the outermost tables in that child, `cell` the
///   cells of that row, and `n` the paragraphs of that cell, including those of tables nested
///   in it.
/// - Text boxes are not counted, and their paragraphs are not read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ElementPath {
    child: NonZeroU32,
    cell: Option<(NonZeroU32, NonZeroU32)>,
    paragraph: NonZeroU32,
}

impl ElementPath {
    /// The `paragraph`-th paragraph outside tables in body element `child`.
    pub fn in_body(child: NonZeroU32, paragraph: NonZeroU32) -> Self {
        ElementPath {
            child,
            cell: None,
            paragraph,
        }
    }

    /// The `paragraph`-th paragraph of a table cell in body element `child`.
    pub fn in_cell(
        child: NonZeroU32,
        row: NonZeroU32,
        cell: NonZeroU32,
        paragraph: NonZeroU32,
    ) -> Self {
        ElementPath {
            child,
            cell: Some((row, cell)),
            paragraph,
        }
    }

    /// The element directly inside the body.
    pub fn child(self) -> NonZeroU32 {
        self.child
    }

    /// The table row, for a paragraph in a table.
    pub fn row(self) -> Option<NonZeroU32> {
        self.cell.map(|(row, _)| row)
    }

    /// The cell within its row, for a paragraph in a table.
    pub fn cell(self) -> Option<NonZeroU32> {
        self.cell.map(|(_, cell)| cell)
    }

    /// The paragraph within the body element or the cell.
    pub fn paragraph(self) -> NonZeroU32 {
        self.paragraph
    }

    pub(crate) fn parse(text: &str) -> Option<Self> {
        let parts: Vec<&str> = text.split('/').collect();
        let number = |part: &str, prefix: char| {
            part.strip_prefix(prefix)
                .and_then(crate::refs::parse_canonical_u64)
                .and_then(|n| u32::try_from(n).ok())
                .and_then(NonZeroU32::new)
        };
        match parts.as_slice() {
            [child, paragraph] => Some(ElementPath::in_body(
                number(child, 'b')?,
                number(paragraph, 'p')?,
            )),
            [child, row, cell, paragraph] => Some(ElementPath::in_cell(
                number(child, 'b')?,
                number(row, 'r')?,
                number(cell, 'c')?,
                number(paragraph, 'p')?,
            )),
            _ => None,
        }
    }
}

impl fmt::Display for ElementPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "b{}/", self.child)?;
        if let Some((row, cell)) = self.cell {
            write!(f, "r{row}/c{cell}/")?;
        }
        write!(f, "p{}", self.paragraph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n(value: u32) -> NonZeroU32 {
        NonZeroU32::new(value).unwrap()
    }

    #[test]
    fn prints_and_parses_both_forms() {
        let cases = [
            (ElementPath::in_body(n(3), n(1)), "b3/p1"),
            (
                ElementPath::in_cell(n(12), n(2), n(4), n(1)),
                "b12/r2/c4/p1",
            ),
        ];
        for (path, printed) in cases {
            assert_eq!(path.to_string(), printed);
            assert_eq!(ElementPath::parse(printed), Some(path));
        }
    }

    #[test]
    fn rejects_other_shapes() {
        for text in [
            "",
            "b1",
            "p1",
            "b0/p1",
            "b1/p0",
            "b01/p1",
            "b1/r1/p1",
            "b1/c1/r1/p1",
            "b1/p1/",
            "x1/p1",
        ] {
            assert_eq!(ElementPath::parse(text), None, "{text}");
        }
    }

    #[test]
    fn exposes_its_parts() {
        let path = ElementPath::in_cell(n(5), n(6), n(7), n(8));
        assert_eq!(
            (path.child(), path.row(), path.cell(), path.paragraph()),
            (n(5), Some(n(6)), Some(n(7)), n(8))
        );
        assert_eq!(ElementPath::in_body(n(1), n(2)).row(), None);
    }
}
