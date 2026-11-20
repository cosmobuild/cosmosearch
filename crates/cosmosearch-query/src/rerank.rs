use std::sync::Arc;

use cosmosearch_core::{
    Query, RerankError, RerankScores, SharedReranker, SourceRef, Usage, indexed_text,
};
use thiserror::Error;
use tokio::time::Instant;
use tracing::warn;

use crate::deadline;
use crate::passages::Passages;
use crate::{Error, Result, SearchHit};

/// How reranking went for one search.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Reranking {
    /// Nothing was reranked: the reader has no reranker, its shortlist is zero, or no hit
    /// was asked for.
    Off,
    /// The reranker scored the shortlisted passages it could read.
    Reranked {
        /// Passages the reranker scored.
        scored: usize,
        /// Shortlisted passages that could not be read, so they keep their fused order after
        /// the scored ones.
        unreadable: usize,
    },
    /// The reranker failed, so the shortlist keeps its fused order and the results may be
    /// less relevant than with reranking.
    Failed(RerankFailure),
    /// The search's time ran out before the shortlist was reranked, so it keeps its fused
    /// order. A reranker already scoring finishes on its own thread, and its scores are
    /// dropped.
    OutOfTime,
}

/// Why reranking failed. It never holds query or passage text.
#[derive(Debug, Clone, Error)]
#[non_exhaustive]
pub enum RerankFailure {
    /// The reranker returned an error.
    #[error("the reranker failed")]
    Reranker(#[source] Arc<RerankError>),

    /// The reranker gave a score count other than the passages'.
    #[error("the reranker gave {found} scores for {expected} passages")]
    WrongCount {
        /// Passages given to the reranker.
        expected: usize,
        /// Scores it gave.
        found: usize,
    },

    /// The reranker gave a score that is not finite.
    #[error("the reranker gave a score that is not finite")]
    NotFinite,
}

/// Rescores the first `shortlist` of `hits` with `reranker` and orders them by that score.
/// The reranker reads each passage as an index does, its headings before its text.
///
/// Each file of the shortlist is read once for all of its hits, unless a passage cache
/// already holds their text. A hit whose passage cannot be read is not rescored and follows
/// the rescored ones in its fused order, and the hits after the shortlist keep their place.
/// A reranker that fails, or reading and scoring that do not finish by `deadline`, leaves
/// every hit in fused order. Returns the hits, how reranking went and the tokens the
/// reranker read.
pub(crate) async fn run(
    passages: &mut Passages<'_>,
    reranker: &SharedReranker,
    query: &Query,
    mut hits: Vec<SearchHit>,
    shortlist: usize,
    deadline: Option<Instant>,
) -> Result<(Vec<SearchHit>, Reranking, Usage)> {
    let rest = hits.split_off(shortlist.min(hits.len()));
    let mut shortlisted = hits;
    let references: Vec<&SourceRef> = shortlisted.iter().map(SearchHit::source).collect();
    let Some(texts) = deadline::within(deadline, passages.texts(&references, usize::MAX)).await
    else {
        shortlisted.extend(rest);
        return Ok((shortlisted, Reranking::OutOfTime, Usage::new()));
    };

    let mut readable = Vec::new();
    let mut passages = Vec::new();
    for (index, (hit, text)) in shortlisted.iter().zip(texts).enumerate() {
        if let Some(text) = text {
            readable.push(index);
            passages.push(Arc::from(indexed_text(hit.heading_path(), &text)));
        }
    }
    let unreadable = shortlisted.len().saturating_sub(readable.len());
    if passages.is_empty() {
        shortlisted.extend(rest);
        let reranking = Reranking::Reranked {
            scored: 0,
            unreadable,
        };
        return Ok((shortlisted, reranking, Usage::new()));
    }
    let scoring = score(reranker.clone(), query.text().to_owned(), passages);
    let Some(result) = deadline::within(deadline, scoring).await else {
        shortlisted.extend(rest);
        return Ok((shortlisted, Reranking::OutOfTime, Usage::new()));
    };
    let result = result?;
    let usage = result.as_ref().map_or(Usage::new(), RerankScores::usage);
    let scores = match fitting(result, readable.len()) {
        Ok(fitting) => fitting,
        Err(failure) => {
            warn!(
                reranker = reranker.get().identity().model(),
                error = %failure,
                "the reranker failed, so the shortlist keeps its fused order"
            );
            shortlisted.extend(rest);
            return Ok((shortlisted, Reranking::Failed(failure), usage));
        }
    };
    for (index, score) in readable.iter().zip(scores) {
        if let Some(hit) = shortlisted.get_mut(*index) {
            hit.set_rerank_score(score);
        }
    }
    let (mut ordered, unscored): (Vec<SearchHit>, Vec<SearchHit>) = shortlisted
        .into_iter()
        .partition(|hit| hit.rerank_score().is_some());
    ordered.sort_by(|a, b| {
        let a = a.rerank_score().unwrap_or(f32::NEG_INFINITY);
        let b = b.rerank_score().unwrap_or(f32::NEG_INFINITY);
        b.total_cmp(&a)
    });
    ordered.extend(unscored);
    ordered.extend(rest);
    let reranking = Reranking::Reranked {
        scored: readable.len(),
        unreadable,
    };
    Ok((ordered, reranking, usage))
}

/// The scores of `result`, if it gives one finite score for each of `expected` passages.
fn fitting(
    result: Result<RerankScores, RerankError>,
    expected: usize,
) -> Result<Vec<f32>, RerankFailure> {
    let scores = result
        .map_err(|err| RerankFailure::Reranker(Arc::new(err)))?
        .into_scores();
    if scores.len() != expected {
        return Err(RerankFailure::WrongCount {
            expected,
            found: scores.len(),
        });
    }
    if !scores.iter().all(|score| score.is_finite()) {
        return Err(RerankFailure::NotFinite);
    }
    Ok(scores)
}

/// Runs the reranker on a blocking thread. A worker that stops, because the reranker
/// panicked or the runtime is shutting down, fails the search, since a panic is a bug.
async fn score(
    reranker: SharedReranker,
    query: String,
    passages: Vec<Arc<str>>,
) -> Result<Result<RerankScores, RerankError>> {
    tokio::task::spawn_blocking(move || {
        let passages: Vec<&str> = passages.iter().map(AsRef::as_ref).collect();
        reranker.get().rerank(&query, &passages)
    })
    .await
    .map_err(Error::Worker)
}
