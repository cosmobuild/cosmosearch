use cosmosearch_core::{Level, Scheme};
use cosmosearch_indexer::{ExtractedFile, FileJob, PublishOutcome};
use cosmosearch_store::{ObjectStoreBackend, S3Options, new_commit_id};

use super::jobs::bm25;
use super::levels::vectors;
use super::*;

fn s3_endpoint() -> Option<String> {
    #[expect(
        clippy::disallowed_methods,
        reason = "a test, not the library, chooses its store"
    )]
    std::env::var("COSMOSEARCH_TEST_S3_ENDPOINT").ok()
}

async fn indexed(endpoint: String) -> (Indexer, IndexHandle, MetadataStore) {
    let options = S3Options::builder("cosmosearch-test")
        .endpoint(endpoint)
        .credentials("cosmosearch-local", "cosmosearch-local-only")
        .allow_http(true)
        .build()
        .unwrap();
    let backend = ObjectStoreBackend::s3(&options).unwrap();
    let prefix = format!("indexer/{}", new_commit_id().unwrap());
    let location = |name: &str| {
        Location::new(Scheme::S3, "cosmosearch-test", format!("{prefix}/{name}")).unwrap()
    };
    let data = DataStore::new(backend.clone(), location("data/")).unwrap();
    let metadata = MetadataStore::new(backend.clone(), location("meta")).unwrap();
    let CreateOutcome::Created(index) = metadata
        .create(data.location(), FileSource::HostNamed)
        .await
        .unwrap()
    else {
        panic!("a new prefix is empty");
    };
    let indexer = Indexer::new(index.clone(), data).unwrap();
    for (name, body) in [("a.txt", "alpha words"), ("b.txt", "beta words")] {
        let key = format!("{prefix}/data/{name}");
        backend
            .put_overwrite(&ObjectKey::new(key.as_str()).unwrap(), Bytes::from(body))
            .await
            .unwrap();
        indexer
            .apply(vec![FileChange::Upsert(FileUpsert::new(key))])
            .await
            .unwrap();
    }
    let plan = indexer.plan(10).await.unwrap();
    let files = extract(&indexer, plan.batches().first().unwrap().jobs()).await;
    let first = indexer
        .build(&bm25(), files, vec![])
        .await
        .unwrap()
        .unwrap();
    indexer.publish(&first).await.unwrap();
    (indexer, index, metadata)
}

async fn extract(indexer: &Indexer, jobs: &[FileJob]) -> Vec<ExtractedFile> {
    let extractors = cosmosearch_chunk::Extractors::builtin().unwrap();
    let limits = cosmosearch_core::ExtractLimits::default();
    let mut files = Vec::new();
    for job in jobs {
        files.push(
            indexer
                .extract(job.clone(), &extractors, &limits)
                .await
                .unwrap(),
        );
    }
    files
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn of_two_promotions_racing_on_an_s3_compatible_store_one_commits() {
    let Some(endpoint) = s3_endpoint() else {
        return;
    };
    let (indexer, index, metadata) = indexed(endpoint).await;
    let plugin = vectors("m1");
    let shard = ShardId::new("0").unwrap();
    let mut built = Vec::new();
    for _ in 0..2 {
        let promotion = indexer.plan_promotion(&plugin, &shard).await.unwrap();
        let files = extract(&indexer, promotion.jobs()).await;
        let segment = indexer
            .build_promotion(&plugin, &promotion, files)
            .await
            .unwrap()
            .unwrap();
        built.push(segment);
    }
    let racing: Vec<_> = built
        .into_iter()
        .map(|segment| {
            let indexer = indexer.clone();
            tokio::spawn(async move { indexer.publish(&segment).await.unwrap() })
        })
        .collect();
    let mut outcomes = Vec::new();
    for race in racing {
        outcomes.push(race.await.unwrap());
    }
    let published = outcomes
        .iter()
        .filter(|outcome| matches!(outcome, PublishOutcome::Published(_)))
        .count();
    assert_eq!(published, 1, "{outcomes:?}");
    assert!(
        outcomes.contains(&PublishOutcome::Superseded),
        "{outcomes:?}"
    );
    let head = index.head().await.unwrap().unwrap();
    let promoted = head.shards()[&shard]
        .segments()
        .iter()
        .filter(|segment| segment.level() == Level::L2)
        .count();
    assert_eq!(promoted, 1);
    metadata.destroy().await.unwrap();
}
