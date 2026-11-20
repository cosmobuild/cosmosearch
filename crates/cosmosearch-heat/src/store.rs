use bytes::Bytes;
use cosmosearch_core::ByteRange;
use cosmosearch_store::{Error as StoreError, IndexHandle, ObjectInfo, ObjectKey, ScopedStore};

use crate::codec::{self, DecodeError};
use crate::merged::{MERGED, parse_version_key};
use crate::snapshot::DELTAS;
use crate::{
    Error, HeatSettings, HeatSweep, HeatWriter, MergeOutcome, MergedHeat, Result, Snapshot, merge,
    sweep,
};

const SCOPE: &str = "heat";

/// How often reading the newest merged version starts again after finding it already
/// removed by a sweep, which happens only when two newer versions landed meanwhile.
const HEAD_ATTEMPTS: usize = 4;

/// Reads and deletes at once in a merge or a sweep.
pub(crate) const CONCURRENCY: usize = 8;

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<HeatStore>();
};

/// The heat of one index, kept under its `heat` scope: each writer's snapshots in
/// `deltas/<writer id>/<seq>` and the merged versions in `merged/v<N>`.
#[derive(Debug, Clone)]
pub struct HeatStore {
    index: IndexHandle,
    scope: ScopedStore,
    settings: HeatSettings,
}

impl HeatStore {
    /// The heat of `index`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSettings`] for a writer retention under two milliseconds.
    pub fn new(index: &IndexHandle, settings: HeatSettings) -> Result<Self> {
        settings.check()?;
        Ok(HeatStore {
            index: index.clone(),
            scope: index.scope(SCOPE)?,
            settings,
        })
    }

    /// A writer for this process, under a new writer id. A process holds one and shares its
    /// recorder with its readers.
    ///
    /// # Errors
    ///
    /// Returns an error only if a writer id cannot be formed, which is a bug.
    pub fn writer(&self) -> Result<HeatWriter> {
        HeatWriter::new(self.scope.clone(), self.settings.clone())
    }

    /// The newest merged version, or an empty one before the first merge.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Store`] if the store fails, and [`Error::Malformed`] or
    /// [`Error::UnsupportedFormat`] if the newest version does not decode.
    pub async fn merged(&self) -> Result<MergedHeat> {
        Ok(self
            .head()
            .await?
            .unwrap_or_else(|| MergedHeat::empty(self.settings.half_life)))
    }

    /// Merges every writer's snapshots that follow on from the newest merged version into
    /// the next one, written create-only, and drops the heat of shards no longer in the
    /// index's newest manifest and the writers older than the retention. Any number of
    /// merges may run at once and again: each snapshot counts exactly once.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Store`] if the store fails, and [`Error::Malformed`] or
    /// [`Error::UnsupportedFormat`] if the newest merged version does not decode. A
    /// snapshot that does not decode is not an error: it and its writer's later snapshots
    /// wait, and are dropped with the writer.
    pub async fn merge(&self) -> Result<MergeOutcome> {
        merge::run(self).await
    }

    /// Deletes the snapshots the newest merged version holds or will never take, and the
    /// versions older than its parent.
    ///
    /// # Errors
    ///
    /// As for [`HeatStore::merged`], and [`Error::Store`] if a delete fails.
    pub async fn sweep(&self) -> Result<HeatSweep> {
        sweep::run(self).await
    }

    pub(crate) fn scope(&self) -> &ScopedStore {
        &self.scope
    }

    pub(crate) fn settings(&self) -> &HeatSettings {
        &self.settings
    }

    pub(crate) fn index(&self) -> &IndexHandle {
        &self.index
    }

    /// The newest merged version, if there is one.
    pub(crate) async fn head(&self) -> Result<Option<MergedHeat>> {
        let mut attempt = 0;
        loop {
            let newest = self
                .list(MERGED)
                .await?
                .into_iter()
                .filter_map(|info| Some((parse_version_key(info.key().as_str())?, info)))
                .max_by_key(|(version, _)| *version);
            let Some((_, info)) = newest else {
                return Ok(None);
            };
            match self.read_whole(&info).await {
                Ok(body) => {
                    let parts =
                        codec::decode_merged(&body).map_err(|err| err.at(info.key().as_str()))?;
                    return Ok(Some(MergedHeat::new(parts, self.settings.half_life)));
                }
                Err(Error::Store(StoreError::NotFound { .. })) if attempt < HEAD_ATTEMPTS => {
                    attempt = attempt.saturating_add(1);
                }
                Err(err) => return Err(err),
            }
        }
    }

    pub(crate) async fn list_snapshots(&self) -> Result<Vec<(Snapshot, ObjectInfo)>> {
        Ok(self
            .list(DELTAS)
            .await?
            .into_iter()
            .filter_map(|info| Some((Snapshot::parse(info.key().as_str())?, info)))
            .collect())
    }

    pub(crate) async fn list(&self, prefix: &str) -> Result<Vec<ObjectInfo>> {
        Ok(self.scope.list(Some(&ObjectKey::new(prefix)?)).await?)
    }

    pub(crate) async fn read_whole(&self, info: &ObjectInfo) -> Result<Bytes> {
        let range = ByteRange::new(0, info.size()).map_err(|_empty| {
            DecodeError::Malformed("a heat object is empty").at(info.key().as_str())
        })?;
        Ok(self.scope.get_range(info.key(), range).await?)
    }
}
