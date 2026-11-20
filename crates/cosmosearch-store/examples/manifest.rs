//! Commits manifests on the local filesystem and on the local S3-compatible store, with
//! two writers racing for the same version.
//!
//! ```text
//! make store
//! cargo run -p cosmosearch-store --all-features --example manifest
//! ```

#![expect(clippy::print_stdout, reason = "an example reports to the terminal")]

use anyhow::{Context, Result, ensure};
use cosmosearch_core::{FileSource, Location, Manifest, Scheme};
use cosmosearch_store::{
    CommitOutcome, CreateOutcome, IndexHandle, MetadataStore, ObjectStoreBackend, OpenState,
    S3Options, WriteStore, new_commit_id,
};

fn next(parent: &Manifest) -> Result<Manifest> {
    Ok(parent.successor(
        new_commit_id()?,
        parent.shards().clone(),
        parent.model_ids().clone(),
    )?)
}

async fn head(index: &IndexHandle) -> Result<Manifest> {
    index.head().await?.context("the index has no manifest")
}

async fn run(name: &str, backend: impl WriteStore + 'static, location: Location) -> Result<()> {
    println!("{name}: {location}");
    let data: Location = "s3://example-bucket/docs/".parse()?;
    let metadata = MetadataStore::new(backend, location)?;

    let outcome = metadata.create(&data, FileSource::Listing).await?;
    let CreateOutcome::Created(index) = outcome else {
        anyhow::bail!("expected a new index, got {outcome:?}");
    };
    println!("  created index {}", index.id());

    let parent = head(&index).await?;
    let (first, second) = (next(&parent)?, next(&parent)?);
    let (a, b) = tokio::join!(index.commit(&first), index.commit(&second));
    let outcomes = [a?, b?];
    println!("  writer A: {:?}", outcomes[0]);
    println!("  writer B: {:?}", outcomes[1]);
    let winners = outcomes
        .iter()
        .filter(|outcome| matches!(outcome, CommitOutcome::Committed(_)))
        .count();
    let losers = outcomes
        .iter()
        .filter(|outcome| matches!(outcome, CommitOutcome::Conflict))
        .count();
    ensure!(
        winners == 1 && losers == 1,
        "expected one winner and one loser"
    );

    let newest = head(&index).await?;
    ensure!(newest.version().get() == 2, "expected version 2");
    println!("  head is version {}", newest.version());

    metadata.destroy().await?;
    let state = metadata.open(&data, FileSource::Listing).await?;
    ensure!(
        matches!(state, OpenState::NotIndexed),
        "expected an empty location"
    );
    println!("  destroyed; the location is empty again");
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().to_str().context("temporary path is not UTF-8")?;
    let location = Location::new(Scheme::File, "", path.trim_start_matches('/'))?;
    run("local filesystem", ObjectStoreBackend::local(), location).await?;

    let options = S3Options::builder("cosmosearch-test")
        .endpoint("http://127.0.0.1:9000")
        .credentials("cosmosearch-local", "cosmosearch-local-only")
        .allow_http(true)
        .build()?;
    let key = format!("example/{}", std::process::id());
    let location = Location::new(Scheme::S3, "cosmosearch-test", key)?;
    run("RustFS", ObjectStoreBackend::s3(&options)?, location).await
}
