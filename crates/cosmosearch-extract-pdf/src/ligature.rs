const LIGATURES: [&str; 2] = ["fi", "fl"];

fn is_lowercase_at(text: &str, at: usize) -> bool {
    text.get(at..)
        .and_then(|rest| rest.chars().next())
        .is_some_and(char::is_lowercase)
}

fn ends_with_ligature(text: &str, at: usize) -> bool {
    at.checked_sub(2)
        .and_then(|start| text.get(start..at))
        .is_some_and(|before| LIGATURES.contains(&before))
}

fn gaps(text: &str) -> impl Iterator<Item = usize> {
    text.match_indices(' ')
        .map(|(at, _)| at)
        .filter(|at| ends_with_ligature(text, *at) && is_lowercase_at(text, at.saturating_add(1)))
}

fn intact(text: &str) -> usize {
    LIGATURES
        .iter()
        .flat_map(|ligature| text.match_indices(ligature))
        .filter(|(at, ligature)| is_lowercase_at(text, at.saturating_add(ligature.len())))
        .count()
}

/// Whether a document's text shows ligatures broken by a stray space, as "fi nding".
pub(crate) fn has_broken_ligatures<'a>(texts: impl Iterator<Item = &'a str> + Clone) -> bool {
    let broken: usize = texts.clone().map(|text| gaps(text).count()).sum();
    let whole: usize = texts.map(intact).sum();
    broken > whole
}

pub(crate) fn close_gaps(text: &str) -> String {
    let mut closed = String::with_capacity(text.len());
    let mut start = 0;
    for gap in gaps(text) {
        closed.push_str(text.get(start..gap).unwrap_or_default());
        start = gap.saturating_add(1);
    }
    closed.push_str(text.get(start..).unwrap_or_default());
    closed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closes_gaps_after_a_ligature() {
        let cases = [
            ("research fi ndings", "research findings"),
            ("defi ne artifi cial", "define artificial"),
            ("confl ict", "conflict"),
            ("fi", "fi"),
            ("fi Nally", "fi Nally"),
            ("fi 3", "fi 3"),
            ("no such thing", "no such thing"),
            ("añfi ñu", "añfiñu"),
        ];
        for (text, expected) in cases {
            assert_eq!(close_gaps(text), expected, "{text:?}");
        }
    }

    #[test]
    fn only_a_document_where_gaps_outnumber_whole_words_is_broken() {
        let broken = ["the fi ndings are fi nal"];
        assert!(has_broken_ligatures(broken.iter().copied()));
        let healthy = ["the findings of wifi networks are final"];
        assert!(!has_broken_ligatures(healthy.iter().copied()));
        let rare_real_gap = ["wifi network", "final file filter"];
        assert!(!has_broken_ligatures(rare_real_gap.iter().copied()));
        assert!(!has_broken_ligatures([""].iter().copied()));
    }
}
