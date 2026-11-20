use std::error::Error;

use cosmosearch_core::{EmbedError, Embedder, EmbeddingIdentity};
use cosmosearch_http::{ApiKey, HttpEmbedder, HttpReranker};
use serde_json::json;

use crate::server::{Reply, Server, serve};

const LIMIT: usize = 64 << 20;

fn identity() -> EmbeddingIdentity {
    EmbeddingIdentity::builder("example-embed", "2026-01", 2, 512)
        .build()
        .unwrap()
}

fn embedder(server: &Server) -> HttpEmbedder {
    HttpEmbedder::builder(&server.base, "example-embed-v1", identity())
        .allow_http()
        .retries(0)
        .build()
        .unwrap()
}

fn cause(error: &EmbedError) -> String {
    error.source().unwrap().to_string()
}

#[test]
fn plain_http_needs_the_host_to_allow_it() {
    let refused = HttpEmbedder::builder("http://127.0.0.1:9/v1", "example-embed-v1", identity())
        .api_key(ApiKey::new("test-key-not-a-secret"))
        .build()
        .unwrap_err();
    assert!(cause(&refused).contains("plain http"), "{refused:?}");
    HttpEmbedder::builder("https://api.example.com/v1", "example-embed-v1", identity())
        .build()
        .unwrap();
}

#[test]
fn a_base_url_with_credentials_or_a_query_is_refused() {
    for base in [
        "https://user:pass@api.example.com/v1",
        "https://user@api.example.com/v1",
        "https://api.example.com/v1?key=secret",
        "https://api.example.com/v1#key=secret",
    ] {
        let refused = HttpEmbedder::builder(base, "example-embed-v1", identity())
            .build()
            .unwrap_err();
        assert!(cause(&refused).contains("user name"), "{refused:?}");
        assert!(!format!("{refused:?}").contains("secret"), "{refused:?}");
        let refused = HttpReranker::builder(base, "example-rerank-v1", super::reranker::identity())
            .build()
            .unwrap_err();
        assert!(!format!("{refused:?}").contains("secret"), "{refused:?}");
    }
}

#[tokio::test]
async fn the_plugins_can_be_built_and_dropped_inside_an_async_runtime() {
    let embedder =
        HttpEmbedder::builder("https://api.example.com/v1", "example-embed-v1", identity())
            .build()
            .unwrap();
    let reranker = HttpReranker::builder(
        "https://api.example.com/v1",
        "example-rerank-v1",
        super::reranker::identity(),
    )
    .build()
    .unwrap();
    drop((embedder, reranker));
}

#[test]
fn a_redirect_is_not_followed() {
    let moved = Reply::Asking(
        307,
        "location: http://127.0.0.1:9/v1/embeddings".into(),
        "{}".into(),
    );
    let server = serve(vec![moved]);
    let error = embedder(&server).embed_query("q").unwrap_err();
    assert!(cause(&error).contains("HTTP 307"), "{error:?}");
    assert_eq!(server.next().body["input"], json!(["q"]));
}

#[test]
fn an_answer_over_the_limit_is_refused() {
    let declared = u64::try_from(LIMIT).unwrap() + 1;
    for reply in [Reply::Declaring(declared), Reply::Unsized(LIMIT + 1)] {
        let server = serve(vec![reply]);
        let error = embedder(&server).embed_query("q").unwrap_err();
        assert!(cause(&error).contains("larger than"), "{error:?}");
    }
}
