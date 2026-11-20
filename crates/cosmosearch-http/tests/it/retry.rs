use std::error::Error;
use std::time::{Duration, Instant};

use cosmosearch_core::{Embedder, EmbeddingIdentity, Reranker, RerankerIdentity};
use cosmosearch_http::{HttpEmbedder, HttpEmbedderBuilder, HttpReranker};
use serde_json::json;

use crate::server::{Reply, Server, serve};

const QUICK: Duration = Duration::from_millis(1);

fn embedder(server: &Server) -> HttpEmbedderBuilder {
    let identity = EmbeddingIdentity::builder("example-embed", "2026-01", 2, 512)
        .build()
        .unwrap();
    HttpEmbedder::builder(&server.base, "example-embed-v1", identity)
        .allow_http()
        .retry_wait(QUICK)
}

fn vector(tokens: u64) -> Reply {
    let body = json!({
        "data": [{"index": 0, "embedding": [0.6, 0.8]}],
        "usage": {"total_tokens": tokens},
    });
    Reply::Answer(200, body.to_string())
}

fn failed(status: u16) -> Reply {
    Reply::Answer(status, "{}".into())
}

#[test]
fn a_passing_failure_is_tried_again_until_it_succeeds() {
    let server = serve(vec![failed(429), failed(503), vector(7)]);
    let found = embedder(&server).build().unwrap().embed_query("q").unwrap();
    assert_eq!(found.vectors(), [vec![0.6, 0.8]]);
    assert_eq!(found.usage().tokens(), 7);
    for _ in 0..3 {
        assert_eq!(server.next().body["input"], json!(["q"]));
    }
}

#[test]
fn gateway_timeouts_and_overloads_are_tried_again() {
    let server = serve(vec![failed(524), failed(529), vector(3)]);
    let found = embedder(&server).build().unwrap().embed_query("q").unwrap();
    assert_eq!(found.vectors(), [vec![0.6, 0.8]]);
}

#[test]
fn retry_after_sets_the_wait() {
    let asking = Reply::Asking(429, "retry-after: 0".into(), "{}".into());
    let server = serve(vec![asking, vector(1)]);
    let started = Instant::now();
    embedder(&server)
        .retry_wait(Duration::from_secs(30))
        .build()
        .unwrap()
        .embed_query("q")
        .unwrap();
    assert!(started.elapsed() < Duration::from_secs(10));
}

#[test]
fn it_gives_up_after_its_retries_with_the_last_failure() {
    let server = serve(vec![failed(500), failed(502), failed(500)]);
    let error = embedder(&server)
        .retries(2)
        .build()
        .unwrap()
        .embed_query("q")
        .unwrap_err();
    assert!(
        error.source().unwrap().to_string().contains("HTTP 500"),
        "{error:?}"
    );
}

#[test]
fn a_refusal_is_not_tried_again() {
    for status in [400, 401, 403, 404, 422] {
        let server = serve(vec![failed(status)]);
        let error = embedder(&server)
            .build()
            .unwrap()
            .embed_query("q")
            .unwrap_err();
        let cause = error.source().unwrap().to_string();
        assert!(cause.contains(&format!("HTTP {status}")), "{cause}");
    }
}

#[test]
fn a_timed_out_request_is_tried_again() {
    let server = serve(vec![Reply::Hang, vector(3)]);
    let found = embedder(&server)
        .timeout(Duration::from_millis(200))
        .build()
        .unwrap()
        .embed_query("q")
        .unwrap();
    assert_eq!(found.usage().tokens(), 3);
}

#[test]
fn the_reranker_tries_again_too() {
    let answer = json!({"results": [{"index": 0, "relevance_score": 0.5}]});
    let server = serve(vec![failed(503), Reply::Answer(200, answer.to_string())]);
    let identity = RerankerIdentity::new("example-rerank", "2026-01", 512).unwrap();
    let scores = HttpReranker::builder(&server.base, "example-rerank-v1", identity)
        .allow_http()
        .retry_wait(QUICK)
        .build()
        .unwrap()
        .rerank("q", &["passage"])
        .unwrap();
    assert_eq!(scores.scores(), [0.5]);
}
