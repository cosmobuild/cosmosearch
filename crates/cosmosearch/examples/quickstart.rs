//! Indexes a folder of text files where it sits and searches it.
//!
//! ```text
//! cargo run -p cosmosearch --features fs,indexer,lexical --example quickstart
//! ```
//!
//! It writes three small files into a temporary folder, keeps the index in a second one, and
//! needs nothing else: no store to start and no model files. With an object store, the same
//! code takes an `S3Options` backend and `s3://` locations.

#![expect(clippy::print_stdout, reason = "an example reports to the terminal")]

use std::num::NonZeroUsize;
use std::path::Path;

use anyhow::{Context, Result, bail};
use cosmosearch::core::{ExtractLimits, SharedIndex};
use cosmosearch::index::{Extractors, Indexer, index_pending};
use cosmosearch::lexical::Bm25;
use cosmosearch::store::ObjectStoreBackend;
use cosmosearch::{
    CreateOutcome, DataStore, Engine, FileSource, Location, MetadataStore, Query, SearchRequest,
};

const FILES: [(&str, &str); 3] = [
    (
        "policies/refunds.md",
        "# Refunds\n\nAnnual plans are refunded in full within 30 days of purchase.\n\n\
         ## After 30 days\n\nA refund after 30 days covers the unused months only.\n",
    ),
    (
        "policies/shipping.md",
        "# Shipping\n\nParcels leave the warehouse on Mondays and Thursdays.\n",
    ),
    (
        "notes/support.txt",
        "Support answers within one working day. Urgent issues go to the on-call rota.\n",
    ),
];

fn location(dir: &Path) -> Result<Location> {
    let path = dir
        .to_str()
        .context("the temporary folder's path is not text")?;
    Ok(format!("file://{path}/").parse()?)
}

#[tokio::main]
async fn main() -> Result<()> {
    let root = tempfile::Builder::new()
        .prefix("cosmosearch-quickstart-")
        .tempdir()?;
    let docs = root.path().join("docs");
    for (name, text) in FILES {
        let path = docs.join(name);
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        tokio::fs::write(&path, text).await?;
    }
    let index = root.path().join("index");
    tokio::fs::create_dir(&index).await?;
    run(&docs, &index).await
}

async fn run(docs: &Path, index: &Path) -> Result<()> {
    let data = DataStore::new(ObjectStoreBackend::local(), location(docs)?)?;
    let metadata = MetadataStore::new(ObjectStoreBackend::local(), location(index)?)?;
    let CreateOutcome::Created(handle) = metadata
        .create(data.location(), FileSource::Listing)
        .await?
    else {
        bail!("an index already exists in {}", index.display());
    };

    let indexer = Indexer::new(handle, data.clone())?;
    indexer.sync_listing().await?;
    let bm25 = SharedIndex::from(Bm25::new()?);
    let batch = NonZeroUsize::new(100).context("a batch holds at least one file")?;
    let extractors = Extractors::builtin()?;
    let report = index_pending(
        &indexer,
        &bm25,
        &extractors,
        &ExtractLimits::default(),
        batch,
    )
    .await?;
    println!(
        "indexed {} files into {} segment(s)",
        report.extracted(),
        report.segments()
    );

    let engine = Engine::builder(data, metadata).index(bm25).open().await?;
    let results = engine
        .search(&SearchRequest::new(Query::new("refund after 30 days")).limit(3))
        .await?;
    println!(
        "{:?}: {} of {} files searched",
        results.status(),
        results.coverage().searched(),
        FILES.len()
    );
    for hit in results.hits() {
        println!("  {:.4}  {}", hit.score(), hit.source());
    }

    let top = results.hits().first().context("the search found nothing")?;
    let reading = engine.read(top.source()).await?;
    let chunk = reading.chunks().first().context("the ref read nothing")?;
    println!("top passage, read again from the file:\n  {}", chunk.text());
    Ok(())
}
