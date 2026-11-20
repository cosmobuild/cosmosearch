use cosmosearch_core::{
    AllowedRows, Built, BuiltFile, Chunk, Hits, Index, IndexError, IndexKind, Level, ModelId,
    Query, SegmentEntry, SegmentFileName, SegmentId, SegmentSource, SharedIndex, Usage,
};
use cosmosearch_indexer::{FileJob, PublishOutcome, Rebuild};

use super::jobs::bm25;
use super::*;

#[derive(Debug)]
pub(super) struct Vectors {
    kind: IndexKind,
    model: ModelId,
}

#[async_trait]
impl Index for Vectors {
    fn kind(&self) -> &IndexKind {
        &self.kind
    }

    fn level(&self) -> Level {
        Level::L2
    }

    fn model_id(&self) -> Option<&ModelId> {
        Some(&self.model)
    }

    async fn build(&self, segment: &SegmentId, chunks: &[Chunk]) -> Result<Built, IndexError> {
        let name = SegmentFileName::new(format!("vec-{segment}.bin")).unwrap();
        let tokens = u64::try_from(chunks.len()).unwrap();
        Ok(Built::new(
            vec![BuiltFile::new(name, vec![1, 2, 3])],
            Usage::of_tokens(tokens),
        ))
    }

    async fn search(
        &self,
        _: &SegmentSource,
        _: &[SegmentEntry],
        _: &AllowedRows,
        _: &Query,
        _: usize,
    ) -> Result<Hits, IndexError> {
        Ok(Hits::default())
    }
}

pub(super) fn vectors(model: &str) -> SharedIndex {
    SharedIndex::new(Vectors {
        kind: IndexKind::new(format!("vec-{model}")).unwrap(),
        model: ModelId::new(model).unwrap(),
    })
}

fn plugin_for(rebuild: &Rebuild) -> SharedIndex {
    if rebuild.kind().as_str() == "bm25" {
        bm25()
    } else {
        vectors(rebuild.model_id().unwrap().as_str())
    }
}

impl Fixture {
    pub(super) async fn segments_at(&self, level: Level) -> Vec<SegmentEntry> {
        let head = self.index.head().await.unwrap().unwrap();
        let mut found: Vec<SegmentEntry> = head
            .shards()
            .values()
            .flat_map(|shard| shard.segments().iter().cloned())
            .filter(|segment| segment.level() == level)
            .collect();
        found.sort_by(|a, b| a.id().cmp(b.id()));
        found.dedup();
        found
    }
}

#[tokio::test]
async fn new_work_is_built_only_at_l1() {
    let f = fixture(FileSource::HostNamed).await;
    f.named_file("docs/a.txt", b"refund policy text").await;
    let plan = f.indexer.plan(10).await.unwrap();
    let files = f.extract_all(&plan).await;
    let err = f
        .indexer
        .build(&vectors("m1"), files, vec![])
        .await
        .unwrap_err();
    assert!(
        matches!(err, Error::WrongLevel { level: Level::L2 }),
        "{err}"
    );
}

#[tokio::test]
async fn a_segment_is_recorded_with_its_plugins_level_and_embedding_identity() {
    let f = fixture(FileSource::HostNamed).await;
    f.ready_files(&[
        ("docs/a.txt", b"refund policy text"),
        ("docs/b.txt", b"beta words"),
    ])
    .await;
    let index = vectors("m1");
    let promotion = f
        .indexer
        .plan_promotion(&index, &ShardId::new("0").unwrap())
        .await
        .unwrap();
    let built = f.build_promotion(&index, &promotion).await;
    let entry = built.entry().unwrap();
    assert_eq!(entry.level(), Level::L2);
    assert_eq!(entry.model_id(), Some(&ModelId::new("m1").unwrap()));
    assert_eq!(entry.built_by().index().unwrap().as_str(), "vec-m1");
    assert_eq!(built.usage().tokens(), 2);

    assert!(matches!(
        f.indexer.publish(&built).await.unwrap(),
        PublishOutcome::Published(_)
    ));
    let head = f.index.head().await.unwrap().unwrap();
    assert!(head.model_ids().contains(&ModelId::new("m1").unwrap()));
    assert_eq!(f.segments_at(Level::L2).await.len(), 1);
}

#[tokio::test]
async fn rebuilds_never_mix_segments_of_different_plugins() {
    let f = fixture(FileSource::HostNamed).await;
    f.named_file("docs/a.txt", b"zebracorn alpha").await;
    f.named_file("docs/b.txt", b"beta words").await;
    f.index_everything().await;
    assert!(matches!(
        f.promote(&vectors("m1")).await,
        PublishOutcome::Published(_)
    ));
    f.send(vec![remove("docs/a.txt")]).await;

    let rebuilds = f.indexer.plan_rebuilds().await.unwrap();
    assert_eq!(rebuilds.len(), 2, "{rebuilds:#?}");
    let lexical = f.segments_at(Level::L1).await;
    let vector_segments = f.segments_at(Level::L2).await;
    for rebuild in &rebuilds {
        let expected = if rebuild.kind().as_str() == "bm25" {
            &lexical
        } else {
            &vector_segments
        };
        assert_eq!(rebuild.replaces(), [expected[0].id().clone()]);
        let keys: Vec<&str> = rebuild.jobs().iter().map(FileJob::key).collect();
        assert_eq!(keys, ["docs/b.txt"]);
    }

    for rebuild in &rebuilds {
        let files = f.extract_jobs(rebuild).await;
        let built = f
            .indexer
            .build(&plugin_for(rebuild), files, rebuild.replaces().to_vec())
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(
            f.indexer.publish(&built).await.unwrap(),
            PublishOutcome::Published(_)
        ));
    }
    for level in [Level::L1, Level::L2] {
        let segments = f.segments_at(level).await;
        assert_eq!(segments.len(), 1);
        let chunks = f.chunks_of(&segments[0]).await;
        assert_eq!(chunks.file_versions().len(), 1);
    }
    assert_eq!(
        f.indexer.plan_rebuilds().await.unwrap(),
        Vec::<Rebuild>::new()
    );
}

#[tokio::test]
async fn a_rebuild_built_with_another_plugin_is_refused() {
    let f = fixture(FileSource::HostNamed).await;
    f.named_file("docs/a.txt", b"zebracorn").await;
    f.named_file("docs/b.txt", b"beta").await;
    f.index_everything().await;
    f.send(vec![remove("docs/a.txt")]).await;
    let rebuild = f.indexer.plan_rebuilds().await.unwrap().remove(0);
    let files = f.extract_jobs(&rebuild).await;
    let built = f
        .indexer
        .build(&vectors("m1"), files, rebuild.replaces().to_vec())
        .await
        .unwrap()
        .unwrap();
    let version = f.version().await;

    let err = f.indexer.publish(&built).await.unwrap_err();
    assert!(
        matches!(&err, Error::WrongPlugin { kind } if kind.as_str() == "bm25"),
        "{err}"
    );
    assert_eq!(f.version().await, version);
}
