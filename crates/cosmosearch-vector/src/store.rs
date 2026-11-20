use std::collections::{BTreeMap, BTreeSet};
use std::num::{NonZeroU32, NonZeroUsize};
use std::ops::Range;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::format::{PackTable, QuantizedVector, VectorPack};
use bytes::Bytes;
use cosmosearch_core::{ByteRange, ChunkHash, EmbeddingIdentity, ModelId, Usage};
use cosmosearch_store::{IndexHandle, ObjectInfo, ObjectKey, ScopedStore};
use futures_util::{StreamExt, stream};
use tracing::warn;

use crate::name::PackName;
use crate::{Error, Lookup, Result};

const VECTORS: &str = "vectors";
const DEFAULT_PACK_ROWS: NonZeroUsize = match NonZeroUsize::new(32_768) {
    Some(rows) => rows,
    None => NonZeroUsize::MIN,
};
const READ_CONCURRENCY: usize = 8;
const COALESCE_GAP_BYTES: usize = 256 * 1024;

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<VectorStore>();
};

/// A reuse cache of vectors under an index, keyed by embedding identity and chunk hash, so a
/// chunk seen before is not embedded again.
///
/// It is only a cache: nothing depends on it, and an entry that is missing or cannot be read
/// is a miss. Each [`VectorStore::put`] writes one immutable pack at
/// `<index id>/vectors/<model id>/<created unix ms>-<random>.pack`, and
/// [`VectorStore::sweep`] merges old packs and drops vectors no longer live.
#[derive(Debug, Clone)]
pub struct VectorStore {
    scope: ScopedStore,
    pack_rows: NonZeroUsize,
}

impl VectorStore {
    /// The vector store of `index`.
    ///
    /// # Errors
    ///
    /// Returns an error only if its scope cannot be formed, which is a bug.
    pub fn new(index: &IndexHandle) -> Result<Self> {
        Ok(VectorStore {
            scope: index.scope(VECTORS)?,
            pack_rows: DEFAULT_PACK_ROWS,
        })
    }

    /// Sets the most rows a sweep puts in one pack; 32,768 unless set.
    #[must_use]
    pub fn pack_rows(mut self, rows: NonZeroUsize) -> Self {
        self.pack_rows = rows;
        self
    }

    /// Stores `vectors` of `identity`, each under the hash of its chunk, in one new pack named
    /// for `now`. Of several vectors with one hash, the first is kept.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if a vector does not have the identity's dimensions or holds
    /// a value that is not finite, [`Error::TimeBeforeEpoch`] for such a `now`, and
    /// [`Error::Store`] if the write fails or its outcome is unknown.
    pub async fn put(
        &self,
        identity: &EmbeddingIdentity,
        vectors: &[(ChunkHash, &[f32])],
        now: SystemTime,
    ) -> Result<()> {
        if vectors.is_empty() {
            return Ok(());
        }
        let entries = vectors
            .iter()
            .map(|(hash, vector)| Ok((*hash, QuantizedVector::quantize(vector)?)))
            .collect::<Result<Vec<_>>>()?;
        let pack = VectorPack::new(identity.model_id()?, identity.dimensions(), entries)?;
        self.write_pack(&pack, unix_ms(now)?).await
    }

    /// Looks up the vectors of `hashes` under `identity`. Any pack may answer for a hash.
    /// A pack that is missing, damaged or unreadable answers nothing, and its hashes are
    /// reported missing.
    pub async fn get(&self, identity: &EmbeddingIdentity, hashes: &[ChunkHash]) -> Lookup {
        let mut reads = Reads::default();
        let found = match identity.model_id() {
            Ok(model_id) => {
                self.find(&model_id, identity.dimensions(), hashes, &mut reads)
                    .await
            }
            Err(err) => {
                warn!(error = %err, "an embedding identity has no model id");
                BTreeMap::new()
            }
        };
        let mut reported = BTreeSet::new();
        let missing = hashes
            .iter()
            .filter(|hash| !found.contains_key(*hash) && reported.insert(**hash))
            .copied()
            .collect();
        Lookup::new(found, missing, reads.usage())
    }

    async fn find(
        &self,
        model_id: &ModelId,
        dimensions: NonZeroU32,
        hashes: &[ChunkHash],
        reads: &mut Reads,
    ) -> BTreeMap<ChunkHash, Vec<f32>> {
        let mut wanted: BTreeSet<ChunkHash> = hashes.iter().copied().collect();
        let mut found = BTreeMap::new();
        if wanted.is_empty() {
            return found;
        }
        let tables: Vec<_> = stream::iter(self.list_model(model_id).await)
            .map(|pack| async move {
                let table = self.read_table(&pack, model_id, dimensions).await;
                (pack, table)
            })
            .buffer_unordered(READ_CONCURRENCY)
            .collect()
            .await;
        let mut assigned = Vec::new();
        for (pack, table) in tables {
            let Some((table, bytes)) = table else {
                continue;
            };
            reads.add(bytes);
            let rows: Vec<(usize, ChunkHash)> = wanted
                .iter()
                .filter_map(|hash| Some((table.find(hash)?, *hash)))
                .collect();
            for (_, hash) in &rows {
                wanted.remove(hash);
            }
            if !rows.is_empty() {
                assigned.push((pack, table, rows));
            }
        }
        let answers: Vec<_> = stream::iter(assigned)
            .map(|(pack, table, rows)| async move { self.read_rows(&pack, &table, rows).await })
            .buffer_unordered(READ_CONCURRENCY)
            .collect()
            .await;
        for (bytes, rows) in answers.into_iter().flatten() {
            reads.add(bytes);
            for (hash, vector) in rows {
                if let Some(vector) = vector {
                    found.insert(hash, vector.dequantize());
                }
            }
        }
        found
    }

    pub(crate) fn scope(&self) -> &ScopedStore {
        &self.scope
    }

    pub(crate) fn rows_per_pack(&self) -> NonZeroUsize {
        self.pack_rows
    }

    pub(crate) async fn write_pack(&self, pack: &VectorPack, created_ms: u64) -> Result<()> {
        let model_dir = ObjectKey::new(pack.table().model_id().as_str())?;
        let key = model_dir.child(&PackName::new(created_ms).to_string());
        let body = Bytes::from(pack.encode()?);
        Ok(self.scope.put_create(&key, body).await?)
    }

    async fn list_model(&self, model_id: &ModelId) -> Vec<StoredPack> {
        let listing = match ObjectKey::new(model_id.as_str()) {
            Ok(dir) => self.scope.list(Some(&dir)).await,
            Err(err) => Err(err),
        };
        match listing {
            Ok(objects) => objects
                .iter()
                .filter_map(StoredPack::parse)
                .filter(|pack| &pack.model_id == model_id)
                .collect(),
            Err(err) => {
                warn!(model = %model_id, error = %err, "the vector packs could not be listed");
                Vec::new()
            }
        }
    }

    pub(crate) async fn read_table(
        &self,
        pack: &StoredPack,
        model_id: &ModelId,
        dimensions: NonZeroU32,
    ) -> Option<(PackTable, usize)> {
        let read = async {
            let prefix_len = PackTable::prefix_len(model_id, dimensions, pack.size)?;
            let bytes = self
                .scope
                .get_range(&pack.key, ByteRange::new(0, prefix_len)?)
                .await?;
            let table = PackTable::decode(&bytes, pack.size)?;
            if table.model_id() != model_id || table.dimensions() != dimensions {
                return Err(crate::format::Error::Malformed {
                    reason: "a vector pack holds another model's vectors",
                }
                .into());
            }
            Ok::<_, Error>((table, bytes.len()))
        };
        match read.await {
            Ok(table) => Some(table),
            Err(err) => {
                warn!(key = %pack.key, error = %err, "a vector pack is skipped");
                None
            }
        }
    }

    async fn read_rows(
        &self,
        pack: &StoredPack,
        table: &PackTable,
        mut rows: Vec<(usize, ChunkHash)>,
    ) -> Vec<(usize, Vec<(ChunkHash, Option<QuantizedVector>)>)> {
        rows.sort_unstable();
        let stride = usize::try_from(table.dimensions().get())
            .unwrap_or(usize::MAX)
            .saturating_add(4);
        let mut answers = Vec::new();
        for run in runs(&rows, COALESCE_GAP_BYTES.checked_div(stride).unwrap_or(0)) {
            let Some(first) = run.first().map(|(row, _)| *row) else {
                continue;
            };
            let last = run.last().map_or(first, |(row, _)| *row);
            match self
                .read_run(pack, table, first..last.saturating_add(1))
                .await
            {
                Ok((vectors, bytes)) => {
                    let picked = run
                        .iter()
                        .map(|(row, hash)| {
                            let vector = vectors.get(row.saturating_sub(first)).cloned();
                            (*hash, vector)
                        })
                        .collect();
                    answers.push((bytes, picked));
                }
                Err(err) => {
                    warn!(key = %pack.key, error = %err, "vector rows could not be read");
                    answers.push((0, run.iter().map(|(_, hash)| (*hash, None)).collect()));
                }
            }
        }
        answers
    }

    pub(crate) async fn read_run(
        &self,
        pack: &StoredPack,
        table: &PackTable,
        rows: Range<usize>,
    ) -> Result<(Vec<QuantizedVector>, usize)> {
        let range = table.rows_range(rows)?;
        let bytes = self.scope.get_range(&pack.key, range).await?;
        Ok((table.decode_rows(&bytes)?, bytes.len()))
    }
}

/// One pack found by listing.
#[derive(Debug, Clone)]
pub(crate) struct StoredPack {
    pub(crate) key: ObjectKey,
    pub(crate) model_id: ModelId,
    pub(crate) name: PackName,
    pub(crate) size: u64,
}

impl StoredPack {
    pub(crate) fn parse(object: &ObjectInfo) -> Option<Self> {
        let (model, name) = object.key().as_str().split_once('/')?;
        Some(StoredPack {
            key: object.key().clone(),
            model_id: ModelId::new(model).ok()?,
            name: PackName::parse(name)?,
            size: object.size(),
        })
    }
}

#[derive(Debug, Default)]
pub(crate) struct Reads {
    count: u64,
    bytes: u64,
}

impl Reads {
    pub(crate) fn add(&mut self, bytes: usize) {
        if bytes == 0 {
            return;
        }
        self.count = self.count.saturating_add(1);
        self.bytes = self
            .bytes
            .saturating_add(u64::try_from(bytes).unwrap_or(u64::MAX));
    }

    pub(crate) fn merge(&mut self, other: &Reads) {
        self.count = self.count.saturating_add(other.count);
        self.bytes = self.bytes.saturating_add(other.bytes);
    }

    pub(crate) fn usage(&self) -> Usage {
        Usage::of_reads(self.count, self.bytes)
    }
}

pub(crate) fn unix_ms(time: SystemTime) -> Result<u64> {
    let since = time
        .duration_since(UNIX_EPOCH)
        .map_err(|_before| Error::TimeBeforeEpoch)?;
    Ok(u64::try_from(since.as_millis()).unwrap_or(u64::MAX))
}

fn runs<T>(rows: &[(usize, T)], max_gap: usize) -> Vec<&[(usize, T)]> {
    let mut runs = Vec::new();
    let mut start = 0;
    for (index, pair) in rows.windows(2).enumerate() {
        if let [(previous, _), (next, _)] = pair
            && next.saturating_sub(*previous) > max_gap.saturating_add(1)
        {
            let end = index.saturating_add(1);
            runs.extend(rows.get(start..end));
            start = end;
        }
    }
    runs.extend(rows.get(start..));
    runs.retain(|run| !run.is_empty());
    runs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grouped(rows: &[usize], max_gap: usize) -> Vec<Vec<usize>> {
        let pairs: Vec<(usize, ())> = rows.iter().map(|row| (*row, ())).collect();
        runs(&pairs, max_gap)
            .iter()
            .map(|run| run.iter().map(|(row, ())| *row).collect())
            .collect()
    }

    #[test]
    fn rows_close_together_are_read_in_one_run() {
        assert_eq!(grouped(&[], 4), Vec::<Vec<usize>>::new());
        assert_eq!(grouped(&[3], 0), [vec![3]]);
        assert_eq!(grouped(&[1, 2, 4, 9], 1), [vec![1, 2, 4], vec![9]]);
        assert_eq!(grouped(&[1, 2, 4, 9], 0), [vec![1, 2], vec![4], vec![9]]);
    }

    #[test]
    fn a_time_before_the_epoch_is_refused() {
        let before = UNIX_EPOCH - std::time::Duration::from_secs(1);
        assert!(matches!(unix_ms(before), Err(Error::TimeBeforeEpoch)));
        assert_eq!(unix_ms(UNIX_EPOCH).unwrap(), 0);
    }
}
