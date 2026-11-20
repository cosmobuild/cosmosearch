use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use cosmosearch_core::{
    Built, EmbedCause, EmbedError, Embedder, EmbeddingIdentity, Embeddings, FakeEmbedder, Index,
    Query, SegmentId, SegmentSource, SharedEmbedder,
};
use cosmosearch_store::{Fault, FaultStore, IndexHandle, MemoryStore, Op};
use cosmosearch_vector::{Ivf, VectorStore};

use crate::segments::{chunks, every_row, identity, search, store_segment};
use crate::support::create;

#[derive(Debug)]
struct Counting {
    inner: FakeEmbedder,
    texts: Arc<AtomicUsize>,
}

impl Embedder for Counting {
    fn identity(&self) -> &EmbeddingIdentity {
        self.inner.identity()
    }

    fn embed_documents(&self, texts: &[&str]) -> Result<Embeddings, EmbedError> {
        self.texts.fetch_add(texts.len(), Ordering::SeqCst);
        self.inner.embed_documents(texts)
    }

    fn embed_query(&self, text: &str) -> Result<Embeddings, EmbedError> {
        self.inner.embed_query(text)
    }
}

#[derive(Debug)]
struct Failing {
    inner: FakeEmbedder,
    given: AtomicUsize,
    limit: usize,
}

impl Embedder for Failing {
    fn identity(&self) -> &EmbeddingIdentity {
        self.inner.identity()
    }

    fn embed_documents(&self, texts: &[&str]) -> Result<Embeddings, EmbedError> {
        let given = self.given.fetch_add(texts.len(), Ordering::SeqCst) + texts.len();
        if given > self.limit {
            return Err(EmbedError::Model(EmbedCause::new(
                "the endpoint is overloaded",
            )));
        }
        self.inner.embed_documents(texts)
    }

    fn embed_query(&self, text: &str) -> Result<Embeddings, EmbedError> {
        self.inner.embed_query(text)
    }
}

struct Fixture {
    handle: IndexHandle,
    faults: FaultStore,
    texts: Arc<AtomicUsize>,
}

impl Fixture {
    async fn new() -> Self {
        let faults = FaultStore::new(MemoryStore::new());
        let handle = create(faults.clone(), "s3://example-bucket/meta".parse().unwrap()).await;
        Fixture {
            handle,
            faults,
            texts: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn plain(&self) -> Ivf {
        let embedder = Counting {
            inner: FakeEmbedder::new(identity("fake", 32)),
            texts: Arc::clone(&self.texts),
        };
        Ivf::new(SharedEmbedder::new(embedder)).unwrap()
    }

    fn reusing(&self) -> Ivf {
        self.plain()
            .vector_store(VectorStore::new(&self.handle).unwrap())
    }

    fn embedded(&self) -> usize {
        self.texts.swap(0, Ordering::SeqCst)
    }
}

fn passages(count: usize) -> Vec<String> {
    (0..count).map(|i| format!("passage number {i}")).collect()
}

async fn build(ivf: &Ivf, id: &str, texts: &[String]) -> Built {
    ivf.build(&SegmentId::new(id).unwrap(), &chunks(texts))
        .await
        .unwrap()
}

fn body(built: &Built) -> &[u8] {
    built.files().first().unwrap().body()
}

#[tokio::test]
async fn a_second_build_of_the_same_chunks_embeds_nothing_and_writes_the_same_vectors() {
    let f = Fixture::new().await;
    let ivf = f.reusing();
    let mut texts = passages(40);
    texts.push(texts[3].clone());

    let first = build(&ivf, "s1", &texts).await;
    assert_eq!(f.embedded(), 40, "a repeated chunk is embedded once");
    assert!(first.usage().tokens() > 0);

    let second = build(&ivf, "s2", &texts).await;
    assert_eq!(f.embedded(), 0);
    assert_eq!(second.usage().tokens(), 0);
    assert!(second.usage().object_reads() > 0);
    assert_eq!(body(&second), body(&first));

    let without_store = build(&f.plain(), "s3", &texts).await;
    assert_eq!(f.embedded(), 40);
    assert_eq!(body(&without_store), body(&first));

    let segment = store_segment(&f.handle, &ivf, "s4", &texts).await;
    assert_eq!(f.embedded(), 0);
    let source = SegmentSource::new(f.handle.clone());
    let one = std::slice::from_ref(&segment);
    let hits = search(&ivf, &source, one, &texts[7], 1).await;
    assert_eq!(hits[0].row(), 7);
    assert!((hits[0].score() - 1.0).abs() < 0.02, "{hits:?}");
    let found = ivf
        .search(&source, one, &every_row(one), &Query::new(&texts[7]), 1)
        .await
        .unwrap();
    assert_eq!(found.usage().tokens(), 3, "the query's words");
    assert_eq!(
        found.usage().object_reads(),
        0,
        "reads are the caller's to count"
    );
}

#[tokio::test]
async fn only_chunks_not_seen_before_are_embedded() {
    let f = Fixture::new().await;
    let ivf = f.reusing();
    let texts = passages(10);
    build(&ivf, "s1", &texts[..6]).await;
    assert_eq!(f.embedded(), 6);

    let both = build(&ivf, "s2", &texts).await;
    assert_eq!(f.embedded(), 4);
    let alone = build(&f.plain(), "s3", &texts).await;
    assert_eq!(body(&both), body(&alone));
}

#[tokio::test]
async fn a_vector_store_that_cannot_keep_vectors_costs_only_a_re_embed() {
    let f = Fixture::new().await;
    let ivf = f.reusing();
    let texts = passages(5);
    f.faults.inject(Op::PutCreate, "/vectors/", Fault::Crash);
    let first = build(&ivf, "s1", &texts).await;
    assert_eq!(f.embedded(), 5);

    let second = build(&ivf, "s2", &texts).await;
    assert_eq!(f.embedded(), 5);
    assert_eq!(body(&second), body(&first));
    build(&ivf, "s3", &texts).await;
    assert_eq!(f.embedded(), 0);
}

#[test]
fn an_ivf_reports_its_embedding_identity() {
    let ivf = Ivf::new(SharedEmbedder::new(FakeEmbedder::new(identity("fake", 8)))).unwrap();
    assert_eq!(
        ivf.model_id(),
        Some(&identity("fake", 8).model_id().unwrap())
    );
}

#[tokio::test]
async fn a_build_that_fails_late_keeps_the_vectors_it_finished() {
    let f = Fixture::new().await;
    let texts = passages(4_096 + 100);
    let failing = Failing {
        inner: FakeEmbedder::new(identity("fake", 32)),
        given: AtomicUsize::new(0),
        limit: 4_096,
    };
    let broken = Ivf::new(SharedEmbedder::new(failing))
        .unwrap()
        .vector_store(VectorStore::new(&f.handle).unwrap());
    broken
        .build(&SegmentId::new("s1").unwrap(), &chunks(&texts))
        .await
        .unwrap_err();

    build(&f.reusing(), "s2", &texts).await;
    assert_eq!(f.embedded(), 100, "only the texts after the kept group");
}
