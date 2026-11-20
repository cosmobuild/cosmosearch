use std::num::NonZeroUsize;
use std::time::Duration;

use bytes::Bytes;
use cosmosearch_store::WriteStore;

use crate::support::{at, check_found, fixture, identity, live, packs, put};

const GRACE: Duration = Duration::from_secs(60);
const LATER: u64 = 1_000 + 60_000;

#[tokio::test]
async fn a_sweep_keeps_live_vectors_and_drops_dead_ones() {
    let (store, index, backend) = fixture().await;
    let model = identity("model-a");
    put(&store, &model, &["alpha", "beta", "gamma"], 1_000).await;

    let report = store
        .sweep(&live(&[(&model, &["alpha", "gamma"])]), at(LATER), GRACE)
        .await
        .unwrap();

    assert_eq!(
        (report.written(), report.rows_written(), report.deleted()),
        (1, 2, 1)
    );
    assert_eq!(packs(&backend, &index).await.len(), 1);
    check_found(&store, &model, &["alpha", "gamma"], &["beta"]).await;
}

#[tokio::test]
async fn a_sweep_removes_models_no_longer_in_use() {
    let (store, index, backend) = fixture().await;
    let kept = identity("model-a");
    let dropped = identity("model-b");
    put(&store, &kept, &["alpha"], 1_000).await;
    put(&store, &dropped, &["alpha"], 1_000).await;
    put(&store, &dropped, &["beta"], 1_500).await;

    let report = store
        .sweep(&live(&[(&kept, &["alpha"])]), at(LATER + 500), GRACE)
        .await
        .unwrap();

    assert_eq!(report.deleted(), 3);
    let dropped_dir = dropped.model_id().unwrap().to_string();
    let remaining = packs(&backend, &index).await;
    assert_eq!(remaining.len(), 1);
    assert!(!remaining[0].as_str().contains(&dropped_dir));
    check_found(&store, &kept, &["alpha"], &[]).await;
    check_found(&store, &dropped, &[], &["alpha", "beta"]).await;
}

#[tokio::test]
async fn packs_younger_than_the_grace_period_survive_untouched() {
    let (store, index, backend) = fixture().await;
    let active = identity("model-a");
    let unpublished = identity("model-b");
    put(&store, &active, &["alpha"], 1_000).await;
    put(&store, &active, &["beta"], 30_000).await;
    put(&store, &unpublished, &["gamma"], 30_000).await;
    let before = packs(&backend, &index).await;

    let report = store
        .sweep(&live(&[(&active, &[])]), at(LATER), GRACE)
        .await
        .unwrap();

    assert_eq!(
        (report.young(), report.deleted(), report.written()),
        (2, 1, 0)
    );
    let young: Vec<_> = before
        .into_iter()
        .filter(|key| key.file_name().starts_with("30000-"))
        .collect();
    assert_eq!(packs(&backend, &index).await, young);
    check_found(&store, &active, &["beta"], &["alpha"]).await;
    check_found(&store, &unpublished, &["gamma"], &[]).await;
}

#[tokio::test]
async fn many_small_packs_are_merged() {
    let (store, index, backend) = fixture().await;
    let store = store.pack_rows(NonZeroUsize::new(4).unwrap());
    let model = identity("model-a");
    let texts: Vec<String> = (0..10).map(|n| format!("chunk {n}")).collect();
    let texts: Vec<&str> = texts.iter().map(String::as_str).collect();
    for (n, text) in texts.iter().enumerate() {
        put(&store, &model, &[text], 1_000 + u64::try_from(n).unwrap()).await;
    }
    put(&store, &model, &["chunk 0", "chunk 1"], 1_100).await;

    let all_live = live(&[(&model, &texts)]);
    let report = store
        .sweep(&all_live, at(LATER + 100), GRACE)
        .await
        .unwrap();

    assert_eq!(
        (report.written(), report.rows_written(), report.deleted()),
        (3, 10, 11)
    );
    assert_eq!(packs(&backend, &index).await.len(), 3);
    check_found(&store, &model, &texts, &[]).await;

    let report = store.sweep(&all_live, at(2 * LATER), GRACE).await.unwrap();
    assert_eq!(
        (report.kept(), report.written(), report.deleted()),
        (2, 1, 1)
    );
    check_found(&store, &model, &texts, &[]).await;
}

#[tokio::test]
async fn a_damaged_pack_is_swept_and_costs_only_a_re_embed() {
    let (store, index, backend) = fixture().await;
    let model = identity("model-a");
    put(&store, &model, &["alpha"], 1_000).await;
    put(&store, &model, &["beta"], 2_000).await;
    let first = packs(&backend, &index).await.remove(0);
    backend
        .put_overwrite(&first, Bytes::from_static(b"CSVP damaged"))
        .await
        .unwrap();

    let report = store
        .sweep(
            &live(&[(&model, &["alpha", "beta"])]),
            at(LATER + 1_000),
            GRACE,
        )
        .await
        .unwrap();

    assert_eq!(
        (report.damaged(), report.deleted(), report.rows_written()),
        (1, 2, 1)
    );
    check_found(&store, &model, &["beta"], &["alpha"]).await;
}

#[tokio::test]
async fn a_sweep_of_an_empty_store_does_nothing() {
    let (store, _, _) = fixture().await;
    let report = store.sweep(&live(&[]), at(LATER), GRACE).await.unwrap();
    assert_eq!(report, cosmosearch_vector::SweepReport::default());
}
