use serde::{Deserialize, Serialize};

use super::failure::Failure;
use crate::connection::{JsonFault, holds_error};

#[derive(Debug, Serialize)]
pub(crate) struct EmbeddingsRequest<'body> {
    pub(crate) model: &'body str,
    pub(crate) input: &'body [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) dimensions: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingsResponse {
    data: Vec<Item>,
    #[serde(default)]
    usage: Option<TokenUsage>,
}

#[derive(Debug, Deserialize)]
struct Item {
    #[serde(default)]
    index: Option<usize>,
    embedding: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct TokenUsage {
    #[serde(default)]
    prompt_tokens: Option<u64>,
    #[serde(default)]
    total_tokens: Option<u64>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Expected {
    pub(crate) texts: usize,
    pub(crate) dimensions: usize,
    pub(crate) normalized: bool,
}

pub(crate) fn parse(body: &[u8], expected: Expected) -> Result<(Vec<Vec<f32>>, u64), Failure> {
    let response: EmbeddingsResponse = serde_json::from_slice(body).map_err(|cause| {
        if holds_error(body) {
            Failure::Answered
        } else {
            Failure::Malformed(JsonFault::of(&cause))
        }
    })?;
    if response.data.len() != expected.texts {
        return Err(Failure::Count {
            sent: expected.texts,
            returned: response.data.len(),
        });
    }
    let mut slots: Vec<Option<Vec<f32>>> = vec![None; expected.texts];
    for (position, item) in response.data.into_iter().enumerate() {
        let slot = slots
            .get_mut(item.index.unwrap_or(position))
            .ok_or(Failure::Index)?;
        if slot.is_some() {
            return Err(Failure::Index);
        }
        *slot = Some(checked(item.embedding, expected)?);
    }
    let vectors: Vec<Vec<f32>> = slots
        .into_iter()
        .collect::<Option<_>>()
        .ok_or(Failure::Index)?;
    let tokens = response
        .usage
        .and_then(|usage| usage.total_tokens.or(usage.prompt_tokens))
        .unwrap_or(0);
    Ok((vectors, tokens))
}

fn checked(mut vector: Vec<f32>, expected: Expected) -> Result<Vec<f32>, Failure> {
    if vector.len() != expected.dimensions {
        return Err(Failure::Dimensions {
            expected: expected.dimensions,
            returned: vector.len(),
        });
    }
    let length = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if !length.is_finite() || (expected.normalized && length <= 0.0) {
        return Err(Failure::Length);
    }
    if expected.normalized {
        for x in &mut vector {
            *x /= length;
        }
    }
    Ok(vector)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TWO_UNIT: Expected = Expected {
        texts: 2,
        dimensions: 2,
        normalized: true,
    };

    #[test]
    fn orders_by_index_and_normalizes() {
        let body = br#"{"data":[{"index":1,"embedding":[0,2]},{"index":0,"embedding":[3,4]}],
            "usage":{"prompt_tokens":5,"total_tokens":7}}"#;
        let (vectors, tokens) = parse(body, TWO_UNIT).unwrap();
        assert_eq!(vectors, vec![vec![0.6, 0.8], vec![0.0, 1.0]]);
        assert_eq!(tokens, 7);
    }

    #[test]
    fn keeps_raw_vectors_and_falls_back_on_prompt_tokens() {
        let body = br#"{"data":[{"index":0,"embedding":[3,4]},{"index":1,"embedding":[0,0]}],
            "usage":{"prompt_tokens":5}}"#;
        let raw = Expected {
            normalized: false,
            ..TWO_UNIT
        };
        let (vectors, tokens) = parse(body, raw).unwrap();
        assert_eq!(vectors, vec![vec![3.0, 4.0], vec![0.0, 0.0]]);
        assert_eq!(tokens, 5);
        let without_usage =
            br#"{"data":[{"index":0,"embedding":[1,0]},{"index":1,"embedding":[0,1]}]}"#;
        assert_eq!(parse(without_usage, TWO_UNIT).unwrap().1, 0);
    }

    #[test]
    fn takes_the_order_of_the_answer_when_it_gives_no_index() {
        let body = br#"{"object":"list","data":[{"object":"embedding","embedding":[3,4]},
            {"object":"embedding","embedding":[0,2]}],"usage":{"prompt_tokens":5,"total_tokens":5}}"#;
        let (vectors, _) = parse(body, TWO_UNIT).unwrap();
        assert_eq!(vectors, vec![vec![0.6, 0.8], vec![0.0, 1.0]]);
    }

    #[test]
    fn reports_an_error_object_apart_from_bad_json() {
        let error = br#"{"error":{"code":502,"message":"Provider returned error"}}"#;
        assert!(matches!(parse(error, TWO_UNIT), Err(Failure::Answered)));
        assert!(matches!(
            parse(b"{\"data\":", TWO_UNIT),
            Err(Failure::Malformed(_))
        ));
    }

    #[test]
    fn rejects_bad_answers() {
        let cases: [&[u8]; 6] = [
            br#"{"data":[{"index":0,"embedding":[1,0]}]}"#,
            br#"{"data":[{"index":0,"embedding":[1,0]},{"index":0,"embedding":[0,1]}]}"#,
            br#"{"data":[{"index":0,"embedding":[1,0]},{"index":2,"embedding":[0,1]}]}"#,
            br#"{"data":[{"index":0,"embedding":[1,0]},{"index":1,"embedding":[0,1,0]}]}"#,
            br#"{"data":[{"index":0,"embedding":[1,0]},{"index":1,"embedding":[0,0]}]}"#,
            br#"{"data":[{"index":0,"embedding":[1,0]},{"index":1,"embedding":[0,1e39]}]}"#,
        ];
        for body in cases {
            parse(body, TWO_UNIT).unwrap_err();
        }
    }
}
