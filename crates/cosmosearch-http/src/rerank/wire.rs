use serde::{Deserialize, Serialize};

use super::failure::Failure;
use crate::connection::{JsonFault, holds_error};

#[derive(Debug, Serialize)]
pub(crate) struct RerankRequest<'body> {
    pub(crate) model: &'body str,
    pub(crate) query: &'body str,
    pub(crate) documents: &'body [&'body str],
    pub(crate) top_n: usize,
}

#[derive(Debug, Deserialize)]
struct RerankResponse {
    results: Vec<Item>,
    #[serde(default)]
    usage: Option<TokenUsage>,
}

#[derive(Debug, Deserialize)]
struct Item {
    index: usize,
    relevance_score: f32,
}

#[derive(Debug, Deserialize)]
struct TokenUsage {
    #[serde(default)]
    total_tokens: Option<u64>,
}

/// The scores in `body` in the order of the `sent` passages, and the tokens it reports.
pub(crate) fn parse(body: &[u8], sent: usize) -> Result<(Vec<f32>, u64), Failure> {
    let response: RerankResponse = serde_json::from_slice(body).map_err(|cause| {
        if holds_error(body) {
            Failure::Answered
        } else {
            Failure::Malformed(JsonFault::of(&cause))
        }
    })?;
    if response.results.len() != sent {
        return Err(Failure::Count {
            sent,
            returned: response.results.len(),
        });
    }
    let mut slots: Vec<Option<f32>> = vec![None; sent];
    for item in response.results {
        let slot = slots.get_mut(item.index).ok_or(Failure::Index)?;
        if slot.is_some() {
            return Err(Failure::Index);
        }
        if !item.relevance_score.is_finite() {
            return Err(Failure::Score);
        }
        *slot = Some(item.relevance_score);
    }
    let scores: Vec<f32> = slots
        .into_iter()
        .collect::<Option<_>>()
        .ok_or(Failure::Index)?;
    let tokens = response
        .usage
        .and_then(|usage| usage.total_tokens)
        .unwrap_or(0);
    Ok((scores, tokens))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn puts_scores_in_passage_order() {
        let body = br#"{"results":[{"index":1,"relevance_score":0.9},
            {"index":0,"relevance_score":0.25}],"usage":{"total_tokens":31}}"#;
        let (scores, tokens) = parse(body, 2).unwrap();
        assert_eq!(scores, vec![0.25, 0.9]);
        assert_eq!(tokens, 31);
        let without_usage = br#"{"results":[{"index":0,"relevance_score":-3.5}]}"#;
        assert_eq!(parse(without_usage, 1).unwrap(), (vec![-3.5], 0));
    }

    #[test]
    fn rejects_bad_answers() {
        let cases: [&[u8]; 6] = [
            br#"{"results":[{"index":0,"relevance_score":0.5}]}"#,
            br#"{"results":[{"index":0,"relevance_score":0.5},{"index":0,"relevance_score":0.1}]}"#,
            br#"{"results":[{"index":0,"relevance_score":0.5},{"index":2,"relevance_score":0.1}]}"#,
            br#"{"results":[{"index":0,"relevance_score":0.5},{"index":1,"relevance_score":1e39}]}"#,
            br#"{"results":[{"index":0},{"index":1,"relevance_score":0.1}]}"#,
            br#"{"data":[]}"#,
        ];
        for body in cases {
            parse(body, 2).unwrap_err();
        }
    }
}
