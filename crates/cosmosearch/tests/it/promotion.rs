use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use cosmosearch::{
    Engine, SearchHit,
    core::{EmbedError, Embedder, EmbeddingIdentity, Embeddings, SharedEmbedder},
    index::{BuiltSegment, Extractors, Promotion, PublishOutcome},
    vector::{Ivf, VectorStore},
};
use cosmosearch_core::{
    ExtractLimits, FakeEmbedder, Level, SegmentId, ShardEntry, ShardId, SharedIndex,
};

use crate::reading::{Fixture, keys, search};

const FIRST_IN_TWO_LISTS: f32 = 2.0 / 61.0;

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

pub(super) struct Vectors {
    pub(super) fixture: Fixture,
    texts: Arc<AtomicUsize>,
}

impl Vectors {
    pub(super) async fn new() -> Self {
        Vectors {
            fixture: Fixture::new().await,
            texts: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub(super) fn ivf(&self) -> Ivf {
        let identity = EmbeddingIdentity::builder("fake", "r1", 32, 512)
            .build()
            .unwrap();
        let embedder = Counting {
            inner: FakeEmbedder::new(identity),
            texts: Arc::clone(&self.texts),
        };
        Ivf::new(SharedEmbedder::new(embedder))
            .unwrap()
            .vector_store(VectorStore::new(&self.fixture.index).unwrap())
    }

    pub(super) fn embedded(&self) -> usize {
        self.texts.swap(0, Ordering::SeqCst)
    }

    async fn engine(&self) -> Engine {
        self.fixture
            .builder()
            .index(self.ivf())
            .open()
            .await
            .unwrap()
    }

    pub(super) async fn build(&self, index: &SharedIndex, promotion: &Promotion) -> BuiltSegment {
        let indexer = &self.fixture.indexer;
        let mut files = Vec::new();
        for job in promotion.jobs() {
            files.push(
                indexer
                    .extract(
                        job.clone(),
                        &Extractors::builtin().unwrap(),
                        &ExtractLimits::default(),
                    )
                    .await
                    .unwrap(),
            );
        }
        indexer
            .build_promotion(index, promotion, files)
            .await
            .unwrap()
            .unwrap()
    }

    async fn promote(&self) -> BuiltSegment {
        let index = SharedIndex::new(self.ivf());
        let shard = ShardId::new("0").unwrap();
        let indexer = &self.fixture.indexer;
        let promotion = indexer.plan_promotion(&index, &shard).await.unwrap();
        let built = self.build(&index, &promotion).await;
        let outcome = indexer.publish(&built).await.unwrap();
        assert!(
            matches!(outcome, PublishOutcome::Published(_)),
            "the promotion was not published: {outcome:?}"
        );
        built
    }

    pub(super) async fn promoted(&self) -> Vec<SegmentId> {
        let head = self.fixture.index.head().await.unwrap().unwrap();
        head.shards()
            .values()
            .flat_map(ShardEntry::segments)
            .filter(|segment| segment.level() == Level::L2)
            .map(|segment| segment.id().clone())
            .collect()
    }
}

async fn found_by_both(engine: &Engine, text: &str, key: &str) {
    let results = search(engine, text).await;
    assert_eq!(
        keys(&results).first().copied(),
        Some(key),
        "the passage comes first"
    );
    let score = results.hits().first().map(SearchHit::score);
    assert!(
        score.is_some_and(|score| (score - FIRST_IN_TWO_LISTS).abs() < 1e-6),
        "both BM25 and IVF rank it first: {score:?}"
    );
    assert!(results.usage().tokens() > 0, "the query is embedded");
}

#[tokio::test]
async fn re_promotion_embeds_nothing() {
    let v = Vectors::new().await;
    v.fixture
        .index(&[
            ("docs/a.txt", b"alpha harbour lantern"),
            ("docs/b.txt", b"bravo meadow quartz"),
            ("docs/c.txt", b"charlie violet ember"),
        ])
        .await;

    let first = v.promote().await;
    assert_eq!(v.embedded(), 3);
    assert_eq!(first.usage().tokens(), 9);
    assert_eq!(v.promoted().await, [first.id().unwrap().clone()]);
    let engine = v.engine().await;
    found_by_both(&engine, "bravo meadow quartz", "docs/b.txt").await;

    let second = v.promote().await;
    assert_eq!(v.embedded(), 0);
    assert_eq!(second.usage().tokens(), 0);
    assert_eq!(v.promoted().await, [second.id().unwrap().clone()]);
    engine.refresh().await.unwrap();
    found_by_both(&engine, "alpha harbour lantern", "docs/a.txt").await;

    v.fixture
        .index(&[("docs/c.txt", b"charlie violet ember glows")])
        .await;
    let third = v.promote().await;
    assert_eq!(v.embedded(), 1, "only the changed passage");
    assert_eq!(third.usage().tokens(), 4);
    engine.refresh().await.unwrap();
    found_by_both(&engine, "charlie violet ember glows", "docs/c.txt").await;
}

#[tokio::test]
async fn named_files_embedded_eagerly_are_reused_by_the_promotion() {
    let v = Vectors::new().await;
    v.fixture
        .index(&[
            ("docs/a.txt", b"alpha harbour lantern"),
            ("docs/b.txt", b"bravo meadow quartz"),
        ])
        .await;
    let index = SharedIndex::new(v.ivf());
    let indexer = &v.fixture.indexer;
    let planned = indexer
        .plan_embedding(&index, &["docs/a.txt".to_owned()])
        .await
        .unwrap();
    let built = v.build(&index, &planned[0]).await;
    indexer.publish(&built).await.unwrap();
    assert_eq!(v.embedded(), 1);
    let engine = v
        .fixture
        .builder()
        .index(index.clone())
        .open()
        .await
        .unwrap();
    found_by_both(&engine, "alpha harbour lantern", "docs/a.txt").await;

    v.promote().await;
    assert_eq!(v.embedded(), 1, "only the file not embedded before");
    assert_eq!(v.promoted().await.len(), 1);
}

#[cfg(feature = "cache")]
#[tokio::test]
async fn engines_sharing_a_cache_search_warm_without_reads_until_a_purge() {
    let v = Vectors::new().await;
    v.fixture
        .index(&[
            ("docs/a.txt", b"alpha harbour lantern"),
            ("docs/b.txt", b"bravo meadow quartz"),
        ])
        .await;
    v.promote().await;
    let cache = cosmosearch::cache::Cache::builder(1 << 20)
        .build()
        .await
        .unwrap();
    let open = || async {
        v.fixture
            .builder()
            .index(v.ivf())
            .cache(cache.clone())
            .open()
            .await
            .unwrap()
    };

    let engine = open().await;
    let cold = search(&engine, "bravo meadow quartz").await;
    assert!(cold.usage().object_reads() > 0);
    let warm = search(&engine, "bravo meadow quartz").await;
    assert_eq!(
        warm.usage().object_reads(),
        0,
        "a warm repeat reads nothing"
    );
    assert_eq!(keys(&warm), keys(&cold));
    let other = search(&open().await, "alpha harbour lantern").await;
    assert_eq!(
        other.usage().object_reads(),
        0,
        "a second engine shares the blocks"
    );

    cache.purge(engine.index_id()).await.unwrap();
    let purged = search(&open().await, "bravo meadow quartz").await;
    assert!(
        purged.usage().object_reads() > 0,
        "a purged index is read again"
    );
    assert_eq!(keys(&purged), keys(&cold));
}
