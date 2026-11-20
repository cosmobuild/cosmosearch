use std::assert_matches;
use std::collections::VecDeque;
use std::future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bytes::Bytes;
use cosmosearch_core::{ByteRange, ETag};
use cosmosearch_store::{
    Error, ObjectKey, ObjectStoreBackend, ReadStore, Result, S3Options, WriteStore,
};
use futures_util::TryStreamExt;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinSet;

#[derive(Clone, Copy)]
enum Reply {
    Status(u16),
    Body(u16, &'static str),
    Hang,
}

struct Server {
    store: ObjectStoreBackend,
    requests: Arc<AtomicUsize>,
    _task: JoinSet<()>,
}

async fn read_request(stream: &mut TcpStream) {
    let mut seen = Vec::new();
    let mut chunk = [0; 4096];
    loop {
        let read = stream.read(&mut chunk).await.unwrap();
        if read == 0 {
            return;
        }
        seen.extend_from_slice(&chunk[..read]);
        if let Some(end) = seen.windows(4).position(|window| window == b"\r\n\r\n") {
            let head = String::from_utf8_lossy(&seen[..end]).to_lowercase();
            let body = head
                .lines()
                .find_map(|line| line.strip_prefix("content-length:"))
                .and_then(|value| value.trim().parse::<usize>().ok())
                .unwrap_or(0);
            if seen.len() >= end + 4 + body {
                return;
            }
        }
    }
}

async fn send(stream: &mut TcpStream, status: u16, body: &str) {
    let response = format!(
        "HTTP/1.1 {status} Scripted\r\nContent-Length: {}\r\nETag: \"e\"\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(response.as_bytes()).await.unwrap();
}

async fn serve(replies: &[Reply]) -> Server {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let requests = Arc::new(AtomicUsize::new(0));
    let queue = Mutex::new(replies.iter().copied().collect::<VecDeque<_>>());
    let counter = Arc::clone(&requests);
    let mut task = JoinSet::new();
    task.spawn(async move {
        loop {
            let (mut stream, _) = listener.accept().await.unwrap();
            read_request(&mut stream).await;
            counter.fetch_add(1, Ordering::SeqCst);
            let reply = queue.lock().unwrap().pop_front().unwrap();
            match reply {
                Reply::Status(status) => send(&mut stream, status, "").await,
                Reply::Body(status, body) => send(&mut stream, status, body).await,
                Reply::Hang => future::pending().await,
            }
        }
    });
    let options = S3Options::builder("example-bucket")
        .endpoint(endpoint)
        .credentials("local-id", "local-key")
        .allow_http(true)
        .max_retries(1)
        .timeout(Duration::from_millis(250))
        .build()
        .unwrap();
    Server {
        store: ObjectStoreBackend::s3(&options).unwrap(),
        requests,
        _task: task,
    }
}

fn key() -> ObjectKey {
    ObjectKey::new("a").unwrap()
}

type ExpectedError = fn(&Error) -> bool;

async fn check_create(replies: &[Reply], requests: usize, expected: ExpectedError) {
    let server = serve(replies).await;
    let err = server
        .store
        .put_create(&key(), Bytes::from_static(b"body"))
        .await
        .unwrap_err();
    assert!(expected(&err), "{err:?}");
    assert_eq!(
        server.requests.load(Ordering::SeqCst),
        requests,
        "the store sent an unexpected number of requests"
    );
}

#[tokio::test]
async fn a_refused_create_reports_the_key_as_taken() {
    let taken: ExpectedError = |err| matches!(err, Error::AlreadyExists { .. });
    check_create(&[Reply::Status(412)], 1, taken).await;
    check_create(&[Reply::Status(409)], 1, taken).await;
}

#[tokio::test]
async fn a_retried_create_can_find_its_own_first_attempt() {
    let taken: ExpectedError = |err| matches!(err, Error::AlreadyExists { .. });
    check_create(&[Reply::Status(503), Reply::Status(412)], 2, taken).await;
}

#[tokio::test]
async fn a_timed_out_create_is_unknown_and_not_retried() {
    let unknown: ExpectedError = |err| matches!(err, Error::OutcomeUnknown { .. });
    check_create(&[Reply::Hang], 1, unknown).await;
}

#[tokio::test]
async fn a_forbidden_create_is_a_definite_refusal() {
    let refused: ExpectedError = |err| matches!(err, Error::Refused { .. });
    check_create(&[Reply::Status(403)], 1, refused).await;
}

async fn read(reply: Reply) -> Result<Bytes> {
    let server = serve(&[reply]).await;
    let range = ByteRange::new(0, 1).unwrap();
    let etag = ETag::new("e").unwrap();
    server.store.read_range(&key(), range, &etag).await
}

#[tokio::test]
async fn pinned_reads_map_status_codes() {
    assert_matches!(read(Reply::Status(412)).await, Err(Error::Stale { .. }));
    assert_matches!(read(Reply::Status(404)).await, Err(Error::NotFound { .. }));
    assert_matches!(read(Reply::Status(403)).await, Err(Error::Refused { .. }));
}

#[tokio::test]
async fn one_unrepresentable_key_fails_the_whole_listing() {
    let listing = concat!(
        r#"<?xml version="1.0" encoding="UTF-8"?><ListBucketResult>"#,
        r#"<KeyCount>2</KeyCount><IsTruncated>false</IsTruncated>"#,
        r#"<Contents><Key>a//bad</Key><LastModified>2026-01-01T00:00:00.000Z</LastModified>"#,
        r#"<ETag>"e"</ETag><Size>1</Size></Contents>"#,
        r#"<Contents><Key>good</Key><LastModified>2026-01-01T00:00:00.000Z</LastModified>"#,
        r#"<ETag>"e"</ETag><Size>1</Size></Contents></ListBucketResult>"#,
    );
    let server = serve(&[Reply::Body(200, listing)]).await;
    let result: Result<Vec<_>> = server.store.list(None, None).try_collect().await;
    assert_matches!(result, Err(Error::Backend { .. }));
}

#[tokio::test]
async fn folder_markers_are_not_listed_as_files() {
    let listing = concat!(
        r#"<?xml version="1.0" encoding="UTF-8"?><ListBucketResult>"#,
        r#"<KeyCount>2</KeyCount><IsTruncated>false</IsTruncated>"#,
        r#"<Contents><Key>docs/sub/</Key><LastModified>2026-01-01T00:00:00.000Z</LastModified>"#,
        r#"<ETag>"e"</ETag><Size>0</Size></Contents>"#,
        r#"<Contents><Key>good</Key><LastModified>2026-01-01T00:00:00.000Z</LastModified>"#,
        r#"<ETag>"e"</ETag><Size>1</Size></Contents></ListBucketResult>"#,
    );
    let server = serve(&[Reply::Body(200, listing), Reply::Status(404)]).await;
    let found: Vec<String> = server
        .store
        .list(None, None)
        .map_ok(|info| info.key().to_string())
        .try_collect()
        .await
        .unwrap();
    assert_eq!(found, ["good"]);
}
