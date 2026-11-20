use std::num::NonZeroUsize;

use cosmosearch_chunk::Extractors;
use cosmosearch_core::{ExtractLimits, SharedIndex};
use cosmosearch_indexer::{ExtractOutcome, Indexer, PublishOutcome, Result};
use futures_util::{StreamExt, TryStreamExt, stream};

const EXTRACT_CONCURRENCY: usize = 4;

/// What [`index_pending`] did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Indexed {
    extracted: usize,
    unread: usize,
    segments: usize,
}

impl Indexed {
    /// Files extracted, including those whose extraction failed.
    pub fn extracted(&self) -> usize {
        self.extracted
    }

    /// Files recorded as failed, unsupported or moved instead of indexed.
    pub fn not_indexed(&self) -> usize {
        self.unread
    }

    /// Segments published.
    pub fn segments(&self) -> usize {
        self.segments
    }
}

/// Indexes every pending file with `index`, an L1 plug-in such as BM25.
///
/// It works in rounds of at most `batch` files: extracts them, records those that failed,
/// then builds and publishes one segment for each shard's files.
///
/// It stops when nothing is pending, or when a round publishes and records nothing, which
/// happens when other writers keep changing the same files. It is one way to run the
/// indexer's jobs in one process; a host that spreads them over workers calls the jobs
/// itself.
///
/// # Errors
///
/// Returns [`cosmosearch_indexer::Error::WrongLevel`] for a plug-in that is not at L1, and
/// the indexer's error when a job fails for a reason that is not the file's own, such as a
/// store that cannot be reached.
pub async fn index_pending(
    indexer: &Indexer,
    index: &SharedIndex,
    extractors: &Extractors,
    limits: &ExtractLimits,
    batch: NonZeroUsize,
) -> Result<Indexed> {
    let mut done = Indexed::default();
    loop {
        let plan = indexer.plan(batch.get()).await?;
        if plan.is_empty() {
            return Ok(done);
        }
        let mut progress = false;
        for shard in plan.batches() {
            let files: Vec<_> = stream::iter(shard.jobs().iter().cloned())
                .map(|job| indexer.extract(job, extractors, limits))
                .buffered(EXTRACT_CONCURRENCY)
                .try_collect()
                .await?;
            done.extracted = done.extracted.saturating_add(files.len());
            let failed = files
                .iter()
                .filter(|file| !matches!(file.outcome(), ExtractOutcome::Chunks { .. }))
                .count();
            if failed > 0 {
                indexer.record(&files).await?;
                done.unread = done.unread.saturating_add(failed);
                progress = true;
            }
            if let Some(built) = indexer.build(index, files, Vec::new()).await?
                && matches!(indexer.publish(&built).await?, PublishOutcome::Published(_))
            {
                done.segments = done.segments.saturating_add(1);
                progress = true;
            }
        }
        if !progress {
            return Ok(done);
        }
    }
}
