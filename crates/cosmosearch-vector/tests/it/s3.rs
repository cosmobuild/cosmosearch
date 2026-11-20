use std::num::NonZeroUsize;

use cosmosearch_core::{Location, Scheme};
use cosmosearch_store::{IndexHandle, ObjectStoreBackend, S3Options};
use cosmosearch_vector::{Ivf, VectorStore};
use uuid::Uuid;

use crate::segments::{ParseEmbedder, Recorder, rounds, search, store_segment, text};
use crate::support::{at, check_found, create, identity, live, put};

fn s3_endpoint() -> Option<String> {
    #[expect(
        clippy::disallowed_methods,
        reason = "a test, not the library, chooses its store"
    )]
    std::env::var("COSMOSEARCH_TEST_S3_ENDPOINT").ok()
}

async fn s3_index(endpoint: String) -> IndexHandle {
    let options = S3Options::builder("cosmosearch-test")
        .endpoint(endpoint)
        .credentials("cosmosearch-local", "cosmosearch-local-only")
        .allow_http(true)
        .build()
        .unwrap();
    let backend = ObjectStoreBackend::s3(&options).unwrap();
    let prefix = format!("vector/{}", Uuid::new_v4().simple());
    create(
        backend,
        Location::new(Scheme::S3, "cosmosearch-test", prefix).unwrap(),
    )
    .await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn vectors_are_reused_and_swept_on_an_s3_compatible_store() {
    let Some(endpoint) = s3_endpoint() else {
        return;
    };
    let index = s3_index(endpoint).await;
    let store = VectorStore::new(&index).unwrap();
    let model = identity("model-a");
    put(&store, &model, &["alpha", "beta"], 1_000).await;
    put(&store, &model, &["gamma"], 2_000).await;

    check_found(&store, &model, &["alpha", "beta", "gamma"], &["delta"]).await;

    let report = store
        .sweep(
            &live(&[(&model, &["alpha", "gamma"])]),
            at(100_000),
            std::time::Duration::from_secs(60),
        )
        .await
        .unwrap();
    assert_eq!((report.written(), report.deleted()), (1, 2));
    check_found(&store, &model, &["alpha", "gamma"], &["beta"]).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_ivf_search_is_two_rounds_of_range_reads_on_an_s3_compatible_store() {
    let Some(endpoint) = s3_endpoint() else {
        return;
    };
    let index = s3_index(endpoint).await;
    let cluster_size = NonZeroUsize::new(8).unwrap();
    let ivf = Ivf::new(ParseEmbedder::shared(crate::segments::identity(
        "parsed", 4,
    )))
    .unwrap()
    .cluster_size(cluster_size);
    let mut segments = Vec::new();
    for (index_in_list, axis) in [0_usize, 1, 2].into_iter().enumerate() {
        let texts: Vec<String> = (0..24_u16)
            .map(|row| {
                let mut vector = [0.1_f32; 4];
                vector[axis] = 1.0 + f32::from(row) * 0.01;
                vector[(axis + 1 + usize::from(row % 3)) % 4] += f32::from(row % 5) * 0.2;
                text(&vector)
            })
            .collect();
        segments.push(store_segment(&index, &ivf, &format!("s{index_in_list}"), &texts).await);
    }
    let recorder = Recorder::new(index.clone());
    let query = text(&[1.0, 0.1, 0.1, 0.1]);

    let cold = search(&ivf, &recorder.source(), &segments, &query, 5).await;
    assert_eq!(cold.len(), 5);
    assert!(cold.iter().all(|hit| hit.segment() == segments[0].id()));
    assert_eq!(rounds(&recorder.take()), 3);

    let warm = search(&ivf, &recorder.source(), &segments, &query, 5).await;
    assert_eq!(warm, cold);
    let reads = recorder.take();
    assert_eq!(rounds(&reads), 2, "{reads:#?}");
}
