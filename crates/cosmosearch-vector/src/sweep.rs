use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::mem;
use std::num::NonZeroU32;
use std::time::{Duration, SystemTime};

use crate::format::{PackTable, QuantizedVector, VectorPack};
use cosmosearch_core::{ByteRange, ChunkHash, ModelId, Usage};
use cosmosearch_store::ObjectKey;
use futures_util::{StreamExt, TryStreamExt, stream};
use tracing::warn;

use crate::store::{Reads, StoredPack, unix_ms};
use crate::{Error, Result, VectorStore};

const READ_CONCURRENCY: usize = 4;
const DELETE_CONCURRENCY: usize = 16;

/// What a sweep did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SweepReport {
    young: u64,
    kept: u64,
    written: u64,
    rows_written: u64,
    deleted: u64,
    damaged: u64,
    unreadable: u64,
    usage: Usage,
}

impl SweepReport {
    /// Packs left alone because they are younger than the grace period.
    pub fn young(&self) -> u64 {
        self.young
    }

    /// Old packs kept as they are, because they are full and every row is live.
    pub fn kept(&self) -> u64 {
        self.kept
    }

    /// New packs written with the live rows of old ones.
    pub fn written(&self) -> u64 {
        self.written
    }

    /// Rows in the new packs.
    pub fn rows_written(&self) -> u64 {
        self.rows_written
    }

    /// Packs deleted, including damaged ones.
    pub fn deleted(&self) -> u64 {
        self.deleted
    }

    /// Packs deleted because they could not be decoded.
    pub fn damaged(&self) -> u64 {
        self.damaged
    }

    /// Packs left in place because the store failed to read them.
    pub fn unreadable(&self) -> u64 {
        self.unreadable
    }

    /// The reads the sweep made.
    pub fn usage(&self) -> Usage {
        self.usage
    }
}

enum Examined {
    Gone,
    Unreadable,
    Damaged,
    Dead,
    Kept(Vec<ChunkHash>),
    Merge {
        dimensions: NonZeroU32,
        rows: Vec<(ChunkHash, QuantizedVector)>,
    },
}

struct Merge {
    model_id: ModelId,
    dimensions: Option<NonZeroU32>,
    rows: Vec<(ChunkHash, QuantizedVector)>,
    taken: BTreeSet<ChunkHash>,
    sources: VecDeque<(ObjectKey, usize)>,
}

impl VectorStore {
    /// Removes vectors that are no longer live. `live` holds, per model id still in use, the
    /// hashes of every chunk in a retained manifest.
    ///
    /// Only packs older than `grace` at `now`, by the time in their name, are touched, so a
    /// sweep never removes vectors written for a segment not yet published. Old packs of a
    /// model not in `live` are deleted. The live rows of the other old packs are written to
    /// new packs, of at most the set row count, before those packs are deleted; a pack that is
    /// full and wholly live stays as it is. A damaged pack is deleted, and one the store fails
    /// to read is left for a later sweep. Sweeps running at once, or a sweep and a lookup, lose
    /// at most cache entries.
    ///
    /// # Errors
    ///
    /// Returns [`Error::TimeBeforeEpoch`] for such a `now`, and [`Error::Store`] if listing,
    /// writing or deleting fails. Packs written before a failure only duplicate rows that are
    /// still stored.
    pub async fn sweep(
        &self,
        live: &BTreeMap<ModelId, BTreeSet<ChunkHash>>,
        now: SystemTime,
        grace: Duration,
    ) -> Result<SweepReport> {
        let now_ms = unix_ms(now)?;
        let cutoff = now_ms.saturating_sub(u64::try_from(grace.as_millis()).unwrap_or(u64::MAX));
        let objects = self.scope().list(None).await?;
        let mut models: BTreeMap<ModelId, Vec<StoredPack>> = BTreeMap::new();
        for object in &objects {
            if let Some(pack) = StoredPack::parse(object) {
                models.entry(pack.model_id.clone()).or_default().push(pack);
            }
        }
        let mut report = SweepReport::default();
        let mut reads = Reads::default();
        for (model_id, mut packs) in models {
            packs.sort_by(|a, b| a.name.cmp(&b.name));
            let (old, young): (Vec<_>, Vec<_>) = packs
                .into_iter()
                .partition(|pack| pack.name.created_ms() <= cutoff);
            report.young = report.young.saturating_add(count(young.len()));
            let Some(hashes) = live.get(&model_id) else {
                let keys: Vec<ObjectKey> = old.into_iter().map(|pack| pack.key).collect();
                self.delete_all(&keys, &mut report).await?;
                continue;
            };
            let mut merge = Merge {
                model_id,
                dimensions: None,
                rows: Vec::new(),
                taken: BTreeSet::new(),
                sources: VecDeque::new(),
            };
            self.merge(&mut merge, old, hashes, now_ms, &mut report, &mut reads)
                .await?;
        }
        report.usage = reads.usage();
        Ok(report)
    }

    async fn merge(
        &self,
        merge: &mut Merge,
        old: Vec<StoredPack>,
        live: &BTreeSet<ChunkHash>,
        now_ms: u64,
        report: &mut SweepReport,
        reads: &mut Reads,
    ) -> Result<()> {
        let pack_rows = self.rows_per_pack().get();
        let mut examined = stream::iter(old)
            .map(|pack| async move {
                let mut reads = Reads::default();
                let outcome = self.examine(&pack, live, pack_rows, &mut reads).await;
                (pack, outcome, reads)
            })
            .buffered(READ_CONCURRENCY);
        while let Some((pack, mut outcome, pack_reads)) = examined.next().await {
            reads.merge(&pack_reads);
            if let Examined::Merge { dimensions, .. } = &outcome
                && *merge.dimensions.get_or_insert(*dimensions) != *dimensions
            {
                warn!(key = %pack.key, "a vector pack's dimensions differ from its model's");
                outcome = Examined::Damaged;
            }
            match outcome {
                Examined::Gone => {}
                Examined::Unreadable => report.unreadable = report.unreadable.saturating_add(1),
                Examined::Damaged => {
                    report.damaged = report.damaged.saturating_add(1);
                    self.delete_all(&[pack.key], report).await?;
                }
                Examined::Dead => self.delete_all(&[pack.key], report).await?,
                Examined::Kept(hashes) => {
                    report.kept = report.kept.saturating_add(1);
                    merge.taken.extend(hashes);
                }
                Examined::Merge { rows, .. } => {
                    let before = merge.rows.len();
                    for (hash, vector) in rows {
                        if merge.taken.insert(hash) {
                            merge.rows.push((hash, vector));
                        }
                    }
                    let added = merge.rows.len().saturating_sub(before);
                    merge.sources.push_back((pack.key, added));
                    while merge.rows.len() >= pack_rows {
                        let rest = merge.rows.split_off(pack_rows);
                        let full = mem::replace(&mut merge.rows, rest);
                        self.flush(merge, full, now_ms, report).await?;
                    }
                }
            }
        }
        let rest = mem::take(&mut merge.rows);
        self.flush(merge, rest, now_ms, report).await
    }

    async fn flush(
        &self,
        merge: &mut Merge,
        rows: Vec<(ChunkHash, QuantizedVector)>,
        now_ms: u64,
        report: &mut SweepReport,
    ) -> Result<()> {
        let mut written = rows.len();
        if let Some(dimensions) = merge.dimensions
            && !rows.is_empty()
        {
            let pack = VectorPack::new(merge.model_id.clone(), dimensions, rows)?;
            self.write_pack(&pack, now_ms).await?;
            report.written = report.written.saturating_add(1);
            report.rows_written = report.rows_written.saturating_add(count(written));
        }
        let mut released = Vec::new();
        while let Some((key, remaining)) = merge.sources.front_mut() {
            if *remaining > written {
                *remaining = remaining.saturating_sub(written);
                break;
            }
            written = written.saturating_sub(*remaining);
            released.push(key.clone());
            merge.sources.pop_front();
        }
        self.delete_all(&released, report).await
    }

    async fn examine(
        &self,
        pack: &StoredPack,
        live: &BTreeSet<ChunkHash>,
        pack_rows: usize,
        reads: &mut Reads,
    ) -> Examined {
        match self.read_pack(pack, live, pack_rows, reads).await {
            Ok(examined) => examined,
            Err(Error::Store(cosmosearch_store::Error::NotFound { .. })) => Examined::Gone,
            Err(err @ (Error::Format(_) | Error::Core(_))) => {
                warn!(key = %pack.key, error = %err, "a damaged vector pack is deleted");
                Examined::Damaged
            }
            Err(
                err @ (Error::Store(_)
                | Error::Clustering { .. }
                | Error::UnsupportedMetric { .. }
                | Error::TimeBeforeEpoch),
            ) => {
                warn!(key = %pack.key, error = %err, "a vector pack could not be read");
                Examined::Unreadable
            }
        }
    }

    async fn read_pack(
        &self,
        pack: &StoredPack,
        live: &BTreeSet<ChunkHash>,
        pack_rows: usize,
        reads: &mut Reads,
    ) -> Result<Examined> {
        let header_end = pack.size.min(PackTable::MAX_HEADER_LEN);
        let start = self
            .scope()
            .get_range(&pack.key, ByteRange::new(0, header_end)?)
            .await?;
        reads.add(start.len());
        let (model_id, dimensions) = PackTable::peek(&start)?;
        if model_id != pack.model_id {
            return Err(crate::format::Error::Malformed {
                reason: "a vector pack holds another model's vectors",
            }
            .into());
        }
        let prefix_len = PackTable::prefix_len(&model_id, dimensions, pack.size)?;
        let prefix = self
            .scope()
            .get_range(&pack.key, ByteRange::new(0, prefix_len)?)
            .await?;
        reads.add(prefix.len());
        let table = PackTable::decode(&prefix, pack.size)?;
        let live_rows = table
            .hashes()
            .iter()
            .filter(|hash| live.contains(*hash))
            .count();
        if live_rows == table.len() && table.len() >= pack_rows {
            return Ok(Examined::Kept(table.hashes().to_vec()));
        }
        if live_rows == 0 {
            return Ok(Examined::Dead);
        }
        let (vectors, bytes) = self.read_run(pack, &table, 0..table.len()).await?;
        reads.add(bytes);
        if vectors.len() != table.len() {
            return Err(crate::format::Error::Malformed {
                reason: "a vector pack has fewer rows than hashes",
            }
            .into());
        }
        let rows = table
            .hashes()
            .iter()
            .copied()
            .zip(vectors)
            .filter(|(hash, _)| live.contains(hash))
            .collect();
        Ok(Examined::Merge {
            dimensions: table.dimensions(),
            rows,
        })
    }

    async fn delete_all(&self, keys: &[ObjectKey], report: &mut SweepReport) -> Result<()> {
        stream::iter(keys)
            .map(|key| self.scope().delete(key))
            .buffer_unordered(DELETE_CONCURRENCY)
            .try_collect::<Vec<()>>()
            .await?;
        report.deleted = report.deleted.saturating_add(count(keys.len()));
        Ok(())
    }
}

fn count(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}
