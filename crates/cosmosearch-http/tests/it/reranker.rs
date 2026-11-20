use std::error::Error;
use std::time::Duration;

use cosmosearch_core::{RerankError, Reranker, RerankerIdentity};
use cosmosearch_http::{ApiKey, HttpReranker, HttpRerankerBuilder};
use serde_json::json;

use crate::server::{Reply, Server, serve};

const KEY: &str = "test-key-not-a-secret";

pub(super) fn identity() -> RerankerIdentity {
    RerankerIdentity::new("example-rerank", "2026-01", 4096).unwrap()
}

fn builder(server: &Server) -> HttpRerankerBuilder {
    HttpReranker::builder(&server.base, "example-rerank-v1", identity())
        .allow_http()
        .retries(0)
}

fn ok(results: &[(usize, f32)], tokens: u64) -> Reply {
    let results: Vec<_> = results
        .iter()
        .map(|(index, score)| json!({"index": index, "relevance_score": score}))
        .collect();
    let body = json!({"id": "r-1", "results": results, "usage": {"total_tokens": tokens}});
    Reply::Answer(200, body.to_string())
}

fn cause(error: &RerankError) -> String {
    error.source().unwrap().to_string()
}

#[test]
fn sends_the_model_query_passages_and_key() {
    let server = serve(vec![ok(&[(0, 0.5), (1, 0.25)], 12)]);
    let reranker = builder(&server).api_key(ApiKey::new(KEY)).build().unwrap();
    reranker
        .rerank(
            "refund policy",
            &["refunds take thirty days", "parcels ship"],
        )
        .unwrap();
    let request = server.next();
    assert!(request.head.starts_with("post /v1/rerank http/1.1\r\n"));
    assert!(
        request
            .head
            .contains(&format!("authorization: bearer {KEY}\r\n"))
    );
    assert!(request.head.contains("content-type: application/json\r\n"));
    assert_eq!(
        request.body,
        json!({
            "model": "example-rerank-v1",
            "query": "refund policy",
            "documents": ["refunds take thirty days", "parcels ship"],
            "top_n": 2,
        })
    );
}

#[test]
fn an_error_sent_with_status_200_is_reported_without_its_message() {
    let body = json!({"error": {"code": 502, "message": "provider failed on: refunds take"}});
    let server = serve(vec![Reply::Answer(200, body.to_string())]);
    let error = builder(&server)
        .build()
        .unwrap()
        .rerank("q", &["refunds take thirty days", "b"])
        .unwrap_err();
    assert!(
        cause(&error).contains("an error in place of scores"),
        "{error:?}"
    );
    assert!(!format!("{error:?}").contains("refunds"));
}

#[test]
fn sends_no_key_unless_set() {
    let server = serve(vec![ok(&[(0, 0.5)], 3)]);
    builder(&server)
        .build()
        .unwrap()
        .rerank("q", &["p"])
        .unwrap();
    assert!(!server.next().head.contains("authorization"));
}

#[test]
fn returns_scores_in_passage_order_with_tokens() {
    let server = serve(vec![ok(&[(2, 9.5), (0, 3.0), (1, -1.25)], 40)]);
    let scores = builder(&server)
        .build()
        .unwrap()
        .rerank("q", &["a", "b", "c"])
        .unwrap();
    assert_eq!(scores.scores(), [3.0, -1.25, 9.5]);
    assert_eq!(scores.usage().tokens(), 40);
}

#[test]
fn counts_no_tokens_when_the_endpoint_reports_none() {
    let body = json!({"results": [{"index": 0, "relevance_score": 0.5}]}).to_string();
    let server = serve(vec![Reply::Answer(200, body)]);
    let scores = builder(&server)
        .build()
        .unwrap()
        .rerank("q", &["a"])
        .unwrap();
    assert_eq!(scores.usage().tokens(), 0);
}

#[test]
fn sends_nothing_for_no_passages() {
    let server = serve(vec![]);
    let scores = builder(&server).build().unwrap().rerank("q", &[]).unwrap();
    assert_eq!(scores.scores(), &[] as &[f32]);
    assert_eq!(scores.usage().tokens(), 0);
}

#[test]
fn reports_failed_and_bad_answers_without_the_key() {
    let answer = |body: serde_json::Value| Reply::Answer(200, body.to_string());
    let cases = [
        (
            Reply::Answer(401, r#"{"error":"bad key"}"#.into()),
            "HTTP 401",
        ),
        (Reply::Answer(429, "{}".into()), "HTTP 429"),
        (Reply::Answer(500, "{}".into()), "HTTP 500"),
        (
            Reply::Answer(200, "{\"results\":".into()),
            "not rerank JSON",
        ),
        (
            answer(json!({"results": [{"index": 0}, {"index": 1}]})),
            "not rerank JSON",
        ),
        (ok(&[(0, 0.5)], 1), "1 scores for 2 passages"),
        (
            ok(&[(0, 0.5), (1, 0.5), (1, 0.5)], 1),
            "3 scores for 2 passages",
        ),
        (ok(&[(0, 0.5), (2, 0.5)], 1), "out of range or twice"),
        (ok(&[(1, 0.5), (1, 0.5)], 1), "out of range or twice"),
        (
            answer(json!({"results": [
                {"index": 0, "relevance_score": 0.5},
                {"index": 1, "relevance_score": 1e39},
            ]})),
            "not finite",
        ),
    ];
    for (reply, expected) in cases {
        let server = serve(vec![reply]);
        let error = builder(&server)
            .api_key(ApiKey::new(KEY))
            .build()
            .unwrap()
            .rerank("q", &["a", "b"])
            .unwrap_err();
        assert!(cause(&error).contains(expected), "{expected}: {error:?}");
        assert!(!format!("{error:?}").contains(KEY));
        assert!(!cause(&error).contains(KEY));
    }
}

#[test]
fn times_out() {
    let server = serve(vec![Reply::Hang]);
    let error = builder(&server)
        .timeout(Duration::from_millis(200))
        .build()
        .unwrap()
        .rerank("q", &["a"])
        .unwrap_err();
    assert!(cause(&error).contains("timed out"), "{error:?}");
}

#[test]
fn hides_the_key_and_rejects_bad_settings() {
    let builder = HttpReranker::builder("https://api.example.com/v1/", "m", identity())
        .api_key(ApiKey::new(KEY));
    assert!(!format!("{builder:?}").contains(KEY));
    let reranker = builder.build().unwrap();
    let debug = format!("{reranker:?}");
    assert!(
        debug.contains("https://api.example.com/v1/rerank"),
        "{debug}"
    );
    assert!(!debug.contains(KEY), "{debug}");
    assert_eq!(reranker.identity(), &identity());

    let bad = [
        HttpReranker::builder("not a url", "m", identity()),
        HttpReranker::builder("ftp://api.example.com", "m", identity()),
        HttpReranker::builder("https://api.example.com", "", identity()),
        HttpReranker::builder("https://api.example.com", "m", identity())
            .api_key(ApiKey::new("two words")),
        HttpReranker::builder("https://api.example.com", "m", identity()).api_key(ApiKey::new("")),
    ];
    for builder in bad {
        assert!(matches!(builder.build(), Err(RerankError::Model(_))));
    }
}

#[test]
fn a_bad_answer_never_shows_its_text() {
    let secret = "PRIVATE-PASSAGE-TEXT";
    let body = json!({"results": [{"index": 0, "relevance_score": secret}]}).to_string();
    let server = serve(vec![Reply::Answer(200, body)]);
    let error = builder(&server)
        .build()
        .unwrap()
        .rerank("q", &["a passage"])
        .unwrap_err();
    assert!(cause(&error).contains("not rerank JSON"), "{error:?}");
    let shown = crate::shown(&error);
    assert!(!shown.contains(secret), "{shown}");
}
