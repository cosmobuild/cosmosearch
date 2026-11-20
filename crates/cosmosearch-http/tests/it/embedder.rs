use std::error::Error;
use std::num::NonZeroUsize;
use std::time::Duration;

use cosmosearch_core::{EmbedError, Embedder, EmbeddingIdentity};
use cosmosearch_http::{ApiKey, HttpEmbedder, HttpEmbedderBuilder};
use serde_json::json;

use crate::server::{Reply, Server, serve};

const KEY: &str = "test-key-not-a-secret";

fn identity(normalized: bool) -> EmbeddingIdentity {
    EmbeddingIdentity::builder("example-embed", "2026-01", 2, 512)
        .normalized(normalized)
        .query_prefix("query: ")
        .document_prefix("passage: ")
        .build()
        .unwrap()
}

fn builder(server: &Server) -> HttpEmbedderBuilder {
    HttpEmbedder::builder(&server.base, "example-embed-v1", identity(true))
        .allow_http()
        .retries(0)
}

fn ok(vectors: &[(usize, [f32; 2])], tokens: u64) -> Reply {
    let data: Vec<_> = vectors
        .iter()
        .map(|(index, embedding)| json!({"object": "embedding", "index": index, "embedding": embedding}))
        .collect();
    let body = json!({"data": data, "usage": {"prompt_tokens": tokens, "total_tokens": tokens}});
    Reply::Answer(200, body.to_string())
}

fn cause(error: &EmbedError) -> String {
    error.source().unwrap().to_string()
}

#[test]
fn sends_the_model_prefixed_inputs_dimensions_and_key() {
    let server = serve(vec![
        ok(&[(0, [1.0, 0.0]), (1, [0.0, 1.0])], 4),
        ok(&[(0, [1.0, 0.0])], 2),
    ]);
    let embedder = builder(&server)
        .api_key(ApiKey::new(KEY))
        .request_dimensions()
        .build()
        .unwrap();

    embedder.embed_documents(&["a", "b"]).unwrap();
    let documents = server.next();
    assert!(
        documents
            .head
            .starts_with("post /v1/embeddings http/1.1\r\n")
    );
    assert!(
        documents
            .head
            .contains(&format!("authorization: bearer {KEY}\r\n"))
    );
    assert!(
        documents
            .head
            .contains("content-type: application/json\r\n")
    );
    assert_eq!(
        documents.body,
        json!({"model": "example-embed-v1", "input": ["passage: a", "passage: b"], "dimensions": 2})
    );

    embedder.embed_query("q").unwrap();
    assert_eq!(server.next().body["input"], json!(["query: q"]));
}

#[test]
fn sends_no_key_or_dimensions_unless_set() {
    let server = serve(vec![ok(&[(0, [1.0, 0.0])], 1)]);
    let embedder = builder(&server).build().unwrap();
    embedder.embed_query("q").unwrap();
    let request = server.next();
    assert!(!request.head.contains("authorization"));
    assert_eq!(
        request.body,
        json!({"model": "example-embed-v1", "input": ["query: q"]})
    );
}

#[test]
fn returns_vectors_in_input_order_with_tokens() {
    let server = serve(vec![ok(&[(1, [0.0, 2.0]), (0, [3.0, 4.0])], 9)]);
    let embeddings = builder(&server)
        .build()
        .unwrap()
        .embed_documents(&["first", "second"])
        .unwrap();
    assert_eq!(embeddings.vectors(), [vec![0.6, 0.8], vec![0.0, 1.0]]);
    assert_eq!(embeddings.usage().tokens(), 9);
}

#[test]
fn splits_input_into_batches() {
    let server = serve(vec![
        ok(&[(0, [1.0, 0.0]), (1, [0.0, 1.0])], 3),
        ok(&[(0, [1.0, 0.0])], 2),
    ]);
    let embeddings = builder(&server)
        .batch_size(NonZeroUsize::new(2).unwrap())
        .build()
        .unwrap()
        .embed_documents(&["a", "b", "c"])
        .unwrap();
    assert_eq!(embeddings.vectors().len(), 3);
    assert_eq!(embeddings.usage().tokens(), 5);
    assert_eq!(
        server.next().body["input"],
        json!(["passage: a", "passage: b"])
    );
    assert_eq!(server.next().body["input"], json!(["passage: c"]));
}

#[test]
fn reports_failed_and_bad_answers_without_the_key() {
    let cases = [
        (
            Reply::Answer(401, r#"{"error":"bad key"}"#.into()),
            "HTTP 401",
        ),
        (Reply::Answer(429, "{}".into()), "HTTP 429"),
        (Reply::Answer(500, "{}".into()), "HTTP 500"),
        (
            Reply::Answer(200, "{\"data\":".into()),
            "not embeddings JSON",
        ),
        (
            Reply::Answer(
                200,
                json!({"data": [{"index": 0, "embedding": [1.0, 0.0, 0.0]}]}).to_string(),
            ),
            "3 dimensions, not 2",
        ),
        (
            ok(&[(0, [1.0, 0.0]), (1, [0.0, 1.0])], 2),
            "2 vectors for 1 texts",
        ),
    ];
    for (reply, expected) in cases {
        let server = serve(vec![reply]);
        let error = builder(&server)
            .api_key(ApiKey::new(KEY))
            .build()
            .unwrap()
            .embed_query("q")
            .unwrap_err();
        assert!(cause(&error).contains(expected), "{expected}: {error:?}");
        assert!(!format!("{error:?}").contains(KEY));
    }
}

#[test]
fn times_out() {
    let server = serve(vec![Reply::Hang]);
    let error = builder(&server)
        .timeout(Duration::from_millis(200))
        .build()
        .unwrap()
        .embed_query("q")
        .unwrap_err();
    assert!(cause(&error).contains("timed out"), "{error:?}");
}

#[test]
fn keeps_vectors_as_returned_when_not_normalized() {
    let server = serve(vec![ok(&[(0, [3.0, 4.0])], 1)]);
    let embeddings = HttpEmbedder::builder(&server.base, "example-embed-v1", identity(false))
        .allow_http()
        .build()
        .unwrap()
        .embed_query("q")
        .unwrap();
    assert_eq!(embeddings.vectors(), [vec![3.0, 4.0]]);
}

#[test]
fn hides_the_key_and_rejects_bad_settings() {
    let embedder = HttpEmbedder::builder("https://api.example.com/v1/", "m", identity(true))
        .api_key(ApiKey::new(KEY))
        .build()
        .unwrap();
    let debug = format!("{embedder:?}");
    assert!(
        debug.contains("https://api.example.com/v1/embeddings"),
        "{debug}"
    );
    assert!(!debug.contains(KEY), "{debug}");

    let bad = [
        HttpEmbedder::builder("not a url", "m", identity(true)),
        HttpEmbedder::builder("ftp://api.example.com", "m", identity(true)),
        HttpEmbedder::builder("https://api.example.com", "", identity(true)),
        HttpEmbedder::builder("https://api.example.com", "m", identity(true))
            .api_key(ApiKey::new("two words")),
        HttpEmbedder::builder("https://api.example.com", "m", identity(true))
            .api_key(ApiKey::new("")),
    ];
    for builder in bad {
        assert!(matches!(builder.build(), Err(EmbedError::Model(_))));
    }
}

#[test]
fn a_bad_answer_never_shows_its_text() {
    let secret = "PRIVATE-PASSAGE-TEXT";
    let replies = [
        json!({"data": [{"index": 0, "embedding": secret}]}).to_string(),
        format!("{{\"data\": [{{\"index\": 0, \"embedding\": [{secret}]}}]}}"),
    ];
    for body in replies {
        let server = serve(vec![Reply::Answer(200, body)]);
        let error = builder(&server)
            .build()
            .unwrap()
            .embed_query("q")
            .unwrap_err();
        assert!(cause(&error).contains("not embeddings JSON"), "{error:?}");
        let shown = crate::shown(&error);
        assert!(!shown.contains(secret), "{shown}");
    }
}
