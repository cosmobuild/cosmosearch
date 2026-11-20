use bytes::Bytes;
use cosmosearch_core::ChunkHash;
use cosmosearch_store::WriteStore;
use cosmosearch_vector::Error;

use crate::support::{at, check_found, fixture, hash, identity, packs, put, vector};

#[tokio::test]
async fn duplicate_chunks_reuse_their_vectors() {
    let (store, _, _) = fixture().await;
    let model = identity("model-a");
    put(&store, &model, &["alpha", "beta", "gamma"], 1_000).await;

    check_found(&store, &model, &["alpha", "gamma"], &["delta"]).await;

    let asked = [hash("beta"), hash("delta"), hash("beta"), hash("delta")];
    let lookup = store.get(&model, &asked).await;
    assert_eq!(lookup.missing(), [hash("delta")]);
    assert_eq!(lookup.found().len(), 1);
    assert_eq!(lookup.usage().object_reads(), 2);
    assert!(lookup.usage().bytes() > 0);
}

#[tokio::test]
async fn vectors_of_another_identity_are_never_returned() {
    let (store, _, _) = fixture().await;
    let model = identity("model-a");
    put(&store, &model, &["alpha"], 1_000).await;

    let others = [
        identity("model-b"),
        model.truncated(2).unwrap(),
        cosmosearch_core::EmbeddingIdentity::builder("model-a", "r2", 4, 512)
            .build()
            .unwrap(),
    ];
    for other in &others {
        check_found(&store, other, &[], &["alpha"]).await;
    }
    check_found(&store, &model, &["alpha"], &[]).await;
}

#[tokio::test]
async fn any_pack_may_answer_and_duplicates_are_harmless() {
    let (store, index, backend) = fixture().await;
    let model = identity("model-a");
    put(&store, &model, &["alpha", "beta"], 1_000).await;
    put(&store, &model, &["beta", "gamma"], 2_000).await;
    put(&store, &model, &[], 3_000).await;

    assert_eq!(packs(&backend, &index).await.len(), 2);
    check_found(&store, &model, &["alpha", "beta", "gamma"], &[]).await;
}

#[tokio::test]
async fn a_put_refuses_vectors_that_do_not_fit_the_identity() {
    let (store, index, backend) = fixture().await;
    let model = identity("model-a");
    let cases: [&[f32]; 3] = [&[1.0, 2.0], &[1.0, 2.0, 3.0, f32::NAN], &[]];
    for vector in cases {
        let err = store
            .put(&model, &[(hash("alpha"), vector)], at(1_000))
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Format(_)), "{err}");
    }
    assert_eq!(packs(&backend, &index).await.len(), 0);
}

#[tokio::test]
async fn deleting_a_pack_costs_only_a_re_embed() {
    let (store, index, backend) = fixture().await;
    let model = identity("model-a");
    put(&store, &model, &["alpha"], 1_000).await;
    put(&store, &model, &["beta"], 2_000).await;

    let first = packs(&backend, &index).await.remove(0);
    backend.delete(&first).await.unwrap();

    check_found(&store, &model, &["beta"], &["alpha"]).await;
}

#[tokio::test]
async fn a_damaged_pack_is_a_miss() {
    let model = identity("model-a");
    let damage: [fn(Vec<u8>) -> Vec<u8>; 5] = [
        |_| b"not a pack".to_vec(),
        |mut bytes| {
            bytes.truncate(bytes.len() - 3);
            bytes
        },
        |mut bytes| {
            bytes.extend([0; 8]);
            bytes
        },
        |mut bytes| {
            bytes[4] = 7;
            bytes
        },
        |mut bytes| {
            let row = bytes.len() - 8;
            bytes[row..row + 4].copy_from_slice(&f32::NAN.to_le_bytes());
            bytes
        },
    ];
    for (case, damage) in damage.iter().enumerate() {
        let (store, index, backend) = fixture().await;
        put(&store, &model, &["alpha"], 1_000).await;
        put(&store, &model, &["beta"], 2_000).await;
        let first = packs(&backend, &index).await.remove(0);
        let bytes = backend.get(&first).await.unwrap().to_vec();
        backend
            .put_overwrite(&first, Bytes::from(damage(bytes)))
            .await
            .unwrap();

        let lookup = store.get(&model, &[hash("alpha"), hash("beta")]).await;
        assert_eq!(lookup.missing(), [hash("alpha")], "case {case}");
        let beta: &[f32] = &lookup.found()[&hash("beta")];
        assert_eq!(beta.len(), vector("beta").len(), "case {case}");
    }
}

#[tokio::test]
async fn an_empty_store_misses_everything() {
    let (store, _, _) = fixture().await;
    let asked: Vec<ChunkHash> = ["alpha", "beta"].iter().map(|text| hash(text)).collect();
    let lookup = store.get(&identity("model-a"), &asked).await;
    assert_eq!(lookup.missing(), asked);
    assert_eq!(lookup.usage().object_reads(), 0);
}
