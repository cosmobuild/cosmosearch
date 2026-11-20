use std::collections::BTreeMap;

use crate::markup::{Line, plain_text};

/// How many figures on lines of figures alone mark a page whose tables the parser read
/// column by column, with all the figures apart from their labels.
const SPLIT_FIGURES: usize = 12;
/// Items whose baselines lie within this share of their height sit on one row.
const SAME_ROW: f32 = 0.5;
/// A gap between rows of more than this many line heights starts a new paragraph.
const PARAGRAPH_GAP: f32 = 1.8;
/// Text turned by less than this many degrees reads as horizontal.
const LEVEL_DEGREES: f32 = 1.0;
const FULL_TURN: f32 = 360.0;

/// A piece of text on a page and where it sits: its left edge and baseline in points, the
/// baseline growing upward, and its height.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Placed {
    pub(crate) text: String,
    pub(crate) x: f32,
    pub(crate) y: f32,
    pub(crate) height: f32,
}

impl Placed {
    /// The piece of `item`, if it is horizontal text on page `page`, counting from 1.
    pub(crate) fn of(item: &pdf_inspector::TextItem, page: u32) -> Option<Placed> {
        let level = item.rotation < LEVEL_DEGREES || item.rotation > FULL_TURN - LEVEL_DEGREES;
        (item.page == page && level && !item.text.trim().is_empty()).then(|| Placed {
            text: item.text.clone(),
            x: item.x,
            y: item.y,
            height: item.height,
        })
    }
}

/// Whether the parser's lines for a page hold many figures on lines of figures alone, as when
/// it reads a table column by column and its labels come after all its figures.
pub(crate) fn is_split(lines: &[Line]) -> bool {
    let mut figures = 0usize;
    for line in lines {
        let tokens: Vec<&str> = line.text.split_whitespace().collect();
        let numbers = tokens.iter().filter(|token| **token != "$").count();
        if numbers >= 2 && tokens.iter().all(|token| is_figure(token)) {
            figures = figures.saturating_add(numbers);
        }
    }
    figures >= SPLIT_FIGURES
}

fn is_figure(token: &str) -> bool {
    matches!(token, "$" | "\u{2014}" | "\u{2013}" | "-")
        || (token.chars().any(|c| c.is_ascii_digit())
            && token.chars().all(|c| {
                c.is_ascii_digit() || matches!(c, ',' | '.' | '(' | ')' | '%' | '-' | '$')
            }))
}

/// The lines of a page rebuilt from where its text sits: pieces on one baseline make a row,
/// read left to right, rows run top to bottom, and a wide gap starts a paragraph. A row whose
/// text matches one of `parsed`'s headings keeps that heading's level.
pub(crate) fn rows(mut placed: Vec<Placed>, parsed: &[Line]) -> Vec<Line> {
    placed.sort_by(|a, b| b.y.total_cmp(&a.y).then(a.x.total_cmp(&b.x)));
    let mut grouped: Vec<Vec<Placed>> = Vec::new();
    for piece in placed {
        match grouped.last_mut() {
            Some(row) if row.first().is_some_and(|first| same_row(first, &piece)) => {
                row.push(piece);
            }
            Some(_) | None => grouped.push(vec![piece]),
        }
    }
    let headings: BTreeMap<&str, usize> = parsed
        .iter()
        .filter_map(|line| Some((line.text.as_str(), line.level?)))
        .collect();
    let mut lines = Vec::with_capacity(grouped.len());
    let mut above: Option<(f32, f32)> = None;
    for mut row in grouped {
        row.sort_by(|a, b| a.x.total_cmp(&b.x));
        let Some(first) = row.first() else {
            continue;
        };
        let (y, height) = (first.y, row.iter().map(|p| p.height).fold(0.0, f32::max));
        if let Some((last_y, last_height)) = above
            && last_y - y > PARAGRAPH_GAP * last_height.max(height)
        {
            lines.push(Line {
                level: None,
                text: String::new(),
            });
        }
        above = Some((y, height));
        let joined: Vec<&str> = row.iter().map(|p| p.text.as_str()).collect();
        let text = plain_text(&joined.join(" "));
        let level = headings.get(text.as_str()).copied();
        lines.push(Line { level, text });
    }
    lines
}

fn same_row(first: &Placed, piece: &Placed) -> bool {
    (first.y - piece.y).abs() <= SAME_ROW * first.height.max(piece.height)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn placed(text: &str, x: f32, y: f32) -> Placed {
        Placed {
            text: text.to_owned(),
            x,
            y,
            height: 10.0,
        }
    }

    fn line(text: &str) -> Line {
        Line {
            level: None,
            text: text.to_owned(),
        }
    }

    #[test]
    fn a_page_of_figures_apart_from_their_labels_is_split() {
        let figures: Vec<Line> = ["$ 61,502 $ 75,956 $ 72,948", "23,818 17,624 13,885"]
            .iter()
            .chain(&["85,320 93,580 86,833", "(431) 346 61", "— 95,279 4.2%"])
            .map(|text| line(text))
            .collect();
        assert!(is_split(&figures));
        let labelled: Vec<Line> = ["Total revenue 85,320 93,580 86,833"; 10]
            .iter()
            .map(|text| line(text))
            .collect();
        assert!(!is_split(&labelled));
        let years = [line("2016 2015 2014"), line("52"), line("Revenue: Product")];
        assert!(!is_split(&years));
    }

    #[test]
    fn pieces_on_one_baseline_make_a_row_read_left_to_right() {
        let pieces = vec![
            placed("85,320", 300.0, 500.0),
            placed("Total revenue", 50.0, 501.0),
            placed("93,580", 400.0, 499.5),
            placed("INCOME STATEMENTS", 50.0, 560.0),
            placed("Cost of revenue..........", 50.0, 488.0),
            placed("32,780", 300.0, 488.0),
        ];
        let parsed = [Line {
            level: Some(2),
            text: "INCOME STATEMENTS".to_owned(),
        }];
        let found = rows(pieces, &parsed);
        let texts: Vec<&str> = found.iter().map(|l| l.text.as_str()).collect();
        assert_eq!(
            texts,
            [
                "INCOME STATEMENTS",
                "",
                "Total revenue 85,320 93,580",
                "Cost of revenue 32,780"
            ]
        );
        assert_eq!(found[0].level, Some(2));
        assert_eq!(found[2].level, None);
    }
}
