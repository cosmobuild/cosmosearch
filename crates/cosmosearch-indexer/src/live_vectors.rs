use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, SystemTime};

use cosmosearch_core::{ChunkHash, ModelId};
use cosmosearch_format::ChunkRecord;
use futures_util::{StreamExt, TryStreamExt, stream};

use crate::{Error, Indexer, Result};

const READ_CONCURRENCY: usize = 4;

impl Indexer {
    /// The vectors still live at `now`, for a vector store's sweep: for each model id of a
    /// retained manifest, the hash of every chunk record in a segment a retained manifest
    /// references. `retention` must be the one garbage collection uses. A segment already
    /// gone holds no chunk records.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NoManifest`] if no manifest resolves, and an error if a retained
    /// manifest cannot be read, the store fails or a chunk table cannot be read.
    pub async fn live_vectors(
        &self,
        retention: Duration,
        now: SystemTime,
    ) -> Result<BTreeMap<ModelId, BTreeSet<ChunkHash>>> {
        let retained = self
            .index()
            .retained(retention, now)
            .await?
            .ok_or(Error::NoManifest)?;
        let mut tables = stream::iter(retained.segments().values())
            .map(|segment| async move {
                match self.load_chunks(segment).await {
                    Err(Error::Store(cosmosearch_store::Error::NotFound { .. })) => Ok(None),
                    found => found,
                }
            })
            .buffer_unordered(READ_CONCURRENCY);
        let mut hashes = BTreeSet::new();
        while let Some(table) = tables.try_next().await? {
            let Some(table) = table else {
                continue;
            };
            let rows = (0..table.len()).filter_map(|row| table.get(row));
            hashes.extend(rows.map(ChunkRecord::hash));
        }
        Ok(retained
            .model_ids()
            .iter()
            .map(|model_id| (model_id.clone(), hashes.clone()))
            .collect())
    }
}
