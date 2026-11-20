use futures_util::{StreamExt, TryStreamExt, stream};

use crate::merged::{MERGED, parse_version_key};
use crate::store::CONCURRENCY;
use crate::{HeatStore, Result};

/// What a sweep deleted.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct HeatSweep {
    versions: usize,
    snapshots: usize,
}

impl HeatSweep {
    /// Merged versions deleted.
    pub fn versions(&self) -> usize {
        self.versions
    }

    /// Snapshots deleted.
    pub fn snapshots(&self) -> usize {
        self.snapshots
    }
}

/// Deletes the versions older than the newest one's parent, the snapshots the newest one
/// holds, and the snapshots of writers it dropped or never took before they expired. No
/// merge that could still commit needs any of them: a merge from an older version can only
/// find its slot taken.
pub(crate) async fn run(store: &HeatStore) -> Result<HeatSweep> {
    let Some(head) = store.head().await? else {
        return Ok(HeatSweep::default());
    };
    let mut doomed = Vec::new();
    for info in store.list(MERGED).await? {
        if parse_version_key(info.key().as_str())
            .is_some_and(|version| version.saturating_add(1) < head.version())
        {
            doomed.push(info.key().clone());
        }
    }
    let versions = doomed.len();
    let cutoff = head
        .clock_ms()
        .saturating_sub(store.settings().retention_ms());
    for (snapshot, info) in store.list_snapshots().await? {
        let done = match head.merged_through(snapshot.writer()) {
            Some(through) => snapshot.seq() <= through,
            None => snapshot.writer().born_ms() < cutoff,
        };
        if done {
            doomed.push(info.key().clone());
        }
    }
    let snapshots = doomed.len().saturating_sub(versions);
    let scope = store.scope();
    let deletes: Vec<_> = doomed.iter().map(|key| scope.delete(key)).collect();
    stream::iter(deletes)
        .buffer_unordered(CONCURRENCY)
        .try_collect::<()>()
        .await?;
    Ok(HeatSweep {
        versions,
        snapshots,
    })
}
