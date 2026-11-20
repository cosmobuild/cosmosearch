use std::collections::{BTreeMap, BTreeSet};

use cosmosearch_core::{
    Index, IndexId, IndexKind, Level, Manifest, ModelId, SegmentEntry, SegmentId, ShardCost,
    ShardEntry, ShardId, SharedIndex,
};
use cosmosearch_format::FileStatus;
use serde::{Deserialize, Serialize};

use crate::job::{chunk_segments, live_file};
use crate::{BuiltSegment, Error, ExtractedFile, FileJob, Indexer, Result};

/// Work that builds one segment of an index above L1, such as IVF at L2, for ready files of a
/// shard.
///
/// A promotion holds every ready file of the shard and replaces every segment the same
/// plug-in built for it, possibly none. An eager embedding holds only the named files and
/// replaces nothing. Either commits only while no other segment of that plug-in has appeared
/// in the shard since it was planned.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Promotion {
    index_id: IndexId,
    shard: ShardId,
    kind: IndexKind,
    model_id: Option<ModelId>,
    replaces: Vec<SegmentId>,
    keeps: Vec<SegmentId>,
    jobs: Vec<FileJob>,
}

impl Promotion {
    /// The shard promoted.
    pub fn shard(&self) -> &ShardId {
        &self.shard
    }

    /// The kind of the index plug-in that builds it.
    pub fn kind(&self) -> &IndexKind {
        &self.kind
    }

    /// The embedding identity of that plug-in, for an index with vectors.
    pub fn model_id(&self) -> Option<&ModelId> {
        self.model_id.as_ref()
    }

    /// The segments it replaces.
    pub fn replaces(&self) -> &[SegmentId] {
        &self.replaces
    }

    /// The files to extract, possibly none.
    pub fn jobs(&self) -> &[FileJob] {
        &self.jobs
    }
}

/// What a promotion's commit checks: no segment of its plug-in in the shards it touches
/// other than those it saw when planned.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Guard {
    shard: ShardId,
    kind: IndexKind,
    model_id: Option<ModelId>,
    seen: BTreeSet<SegmentId>,
}

fn built_by(segment: &SegmentEntry, kind: &IndexKind, model_id: Option<&ModelId>) -> bool {
    segment.built_by().index() == Some(kind) && segment.model_id() == model_id
}

/// Whether a segment of the guarded plug-in that the promotion did not see when planned is
/// in a shard it touches: the shards of its files, of the segments it replaces, or its own.
pub(crate) fn appeared<'a>(
    head: &Manifest,
    guard: Option<&Guard>,
    keys: impl Iterator<Item = &'a str>,
    replaces: &[SegmentId],
) -> bool {
    let Some(guard) = guard else {
        return false;
    };
    let mut shards: BTreeSet<&ShardId> = keys
        .filter_map(|key| head.shard_for(key).map(|(id, _)| id))
        .collect();
    for (id, shard) in head.shards() {
        if *id == guard.shard || shard.segments().iter().any(|s| replaces.contains(s.id())) {
            shards.insert(id);
        }
    }
    shards
        .iter()
        .filter_map(|id| head.shards().get(*id))
        .flat_map(ShardEntry::segments)
        .any(|segment| {
            built_by(segment, &guard.kind, guard.model_id.as_ref())
                && !guard.seen.contains(segment.id())
        })
}

fn promotable(index: &dyn Index) -> Result<()> {
    let level = index.level();
    if level > Level::L1 {
        Ok(())
    } else {
        Err(Error::WrongLevel { level })
    }
}

impl Indexer {
    /// Plans the promotion of `shard` with `index`: one segment holding every ready file of
    /// the shard, in place of every segment that plug-in built for it. A file still pending
    /// is promoted by a later promotion, once its first segment is published.
    ///
    /// # Errors
    ///
    /// Returns [`Error::WrongLevel`] for a plug-in at L1 or below, [`Error::UnknownShard`] for
    /// a shard the index does not have, and an error if the store fails or a stored table
    /// cannot be read.
    pub async fn plan_promotion(&self, index: &SharedIndex, shard: &ShardId) -> Result<Promotion> {
        let plugin = index.get();
        promotable(plugin)?;
        let head = self.head().await?;
        let index_id = head.index_id().clone();
        let entry = head
            .shards()
            .get(shard)
            .ok_or_else(|| Error::UnknownShard {
                shard: shard.clone(),
            })?;
        let replaces: Vec<SegmentId> = entry
            .segments()
            .iter()
            .filter(|segment| built_by(segment, plugin.kind(), plugin.model_id()))
            .map(|segment| segment.id().clone())
            .collect();
        let mut jobs: BTreeMap<String, FileJob> = self
            .load_table(entry)
            .await?
            .iter()
            .filter(|file| file.status() == FileStatus::Ready)
            .map(|file| (file.key().to_owned(), FileJob::of(&index_id, shard, file)))
            .collect();
        for (key, job) in self.shared_live_files(&head, &replaces).await? {
            jobs.entry(key).or_insert(job);
        }
        Ok(Promotion {
            index_id,
            shard: shard.clone(),
            kind: plugin.kind().clone(),
            model_id: plugin.model_id().cloned(),
            replaces,
            keeps: Vec::new(),
            jobs: jobs.into_values().collect(),
        })
    }

    /// The live files of other shards held by those of `replaces` that split shards share,
    /// which a promotion must hold too, since replacing a segment takes it from every shard.
    async fn shared_live_files(
        &self,
        head: &Manifest,
        replaces: &[SegmentId],
    ) -> Result<Vec<(String, FileJob)>> {
        let segments = chunk_segments(head);
        let shared: Vec<_> = replaces
            .iter()
            .filter_map(|id| segments.get(id))
            .filter(|(_, shards)| shards.len() > 1)
            .collect();
        let tables = self.load_holders(head, shared.iter().copied()).await?;
        let mut found = Vec::new();
        for (segment, shards) in shared {
            let Some(chunks) = self.load_chunks(segment).await? else {
                continue;
            };
            for (ordinal, version) in chunks.file_versions() {
                if let Some((holder, file)) = live_file(&tables, shards, ordinal, version) {
                    let job = FileJob::of(head.index_id(), holder, file);
                    found.push((file.key().to_owned(), job));
                }
            }
        }
        Ok(found)
    }

    /// Plans the eager embedding of the files `keys` names with `index`: per shard, one
    /// segment holding those that are ready and in no segment of that plug-in yet, added
    /// beside that plug-in's other segments. Files not in the index, not ready or already
    /// held are left out, and a shard with none left gets no work.
    ///
    /// # Errors
    ///
    /// Returns [`Error::WrongLevel`] for a plug-in at L1 or below, and an error if the store
    /// fails or a stored table cannot be read.
    pub async fn plan_embedding(
        &self,
        index: &SharedIndex,
        keys: &[String],
    ) -> Result<Vec<Promotion>> {
        let plugin = index.get();
        promotable(plugin)?;
        let head = self.head().await?;
        let mut wanted: BTreeMap<&ShardId, BTreeSet<&str>> = BTreeMap::new();
        for key in keys {
            let (id, _) = head
                .shard_for(key)
                .ok_or_else(|| Error::NoShard { key: key.clone() })?;
            wanted.entry(id).or_default().insert(key);
        }
        let mut promotions = Vec::new();
        for (id, keys) in wanted {
            let Some(shard) = head.shards().get(id) else {
                continue;
            };
            let keeps: Vec<&SegmentEntry> = shard
                .segments()
                .iter()
                .filter(|segment| built_by(segment, plugin.kind(), plugin.model_id()))
                .collect();
            let mut embedded = BTreeSet::new();
            for segment in &keeps {
                if let Some(chunks) = self.load_chunks(segment).await? {
                    embedded.extend(chunks.file_versions());
                }
            }
            let table = self.load_table(shard).await?;
            let jobs: Vec<FileJob> = keys
                .iter()
                .filter_map(|key| table.get(key))
                .filter(|file| {
                    file.status() == FileStatus::Ready
                        && !embedded.contains(&(file.ordinal(), file.content_version()))
                })
                .map(|file| FileJob::of(head.index_id(), id, file))
                .collect();
            if jobs.is_empty() {
                continue;
            }
            promotions.push(Promotion {
                index_id: head.index_id().clone(),
                shard: id.clone(),
                kind: plugin.kind().clone(),
                model_id: plugin.model_id().cloned(),
                replaces: Vec::new(),
                keeps: keeps.iter().map(|segment| segment.id().clone()).collect(),
                jobs,
            });
        }
        Ok(promotions)
    }

    /// What a heat policy needs to know of each shard that holds a ready file, for `index`: the
    /// plug-in's level if its segments hold every ready file of the shard, with the bytes of
    /// those files as the cost of building it again, and otherwise L1, with the bytes of the
    /// ready files its segments lack as the cost of promoting it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::WrongLevel`] for a plug-in at L1 or below, and an error if the store
    /// fails or a stored table cannot be read.
    pub async fn shard_costs(&self, index: &SharedIndex) -> Result<Vec<ShardCost>> {
        let plugin = index.get();
        promotable(plugin)?;
        let head = self.head().await?;
        let mut costs = Vec::new();
        for (id, shard) in head.shards() {
            let mut embedded = BTreeSet::new();
            for segment in shard.segments() {
                if built_by(segment, plugin.kind(), plugin.model_id())
                    && let Some(chunks) = self.load_chunks(segment).await?
                {
                    embedded.extend(chunks.file_versions());
                }
            }
            let (mut ready, mut lacking, mut files) = (0u64, 0u64, 0usize);
            for file in self.load_table(shard).await?.iter() {
                if file.status() != FileStatus::Ready {
                    continue;
                }
                files = files.saturating_add(1);
                ready = ready.saturating_add(file.size());
                if !embedded.contains(&(file.ordinal(), file.content_version())) {
                    lacking = lacking.saturating_add(file.size());
                }
            }
            if files == 0 {
                continue;
            }
            costs.push(if lacking == 0 {
                ShardCost::new(id.clone(), plugin.level(), ready)
            } else {
                ShardCost::new(id.clone(), Level::L1, lacking)
            });
        }
        Ok(costs)
    }

    /// Plans the demotion of `shard` for `index`: taking away every segment that plug-in
    /// built for it, from every shard that shares them, and leaving its files searchable at
    /// L1. Publish it with [`Indexer::publish`]; it commits only while no other segment of
    /// that plug-in has appeared in the shard. Returns `None` if the plug-in has no segment
    /// there.
    ///
    /// # Errors
    ///
    /// Returns [`Error::WrongLevel`] for a plug-in at L1 or below, [`Error::UnknownShard`] for
    /// a shard the index does not have, and an error if the store fails.
    pub async fn plan_demotion(
        &self,
        index: &SharedIndex,
        shard: &ShardId,
    ) -> Result<Option<BuiltSegment>> {
        let plugin = index.get();
        promotable(plugin)?;
        let head = self.head().await?;
        let entry = head
            .shards()
            .get(shard)
            .ok_or_else(|| Error::UnknownShard {
                shard: shard.clone(),
            })?;
        let replaces: Vec<SegmentId> = entry
            .segments()
            .iter()
            .filter(|segment| built_by(segment, plugin.kind(), plugin.model_id()))
            .map(|segment| segment.id().clone())
            .collect();
        if replaces.is_empty() {
            return Ok(None);
        }
        let guard = Guard {
            shard: shard.clone(),
            kind: plugin.kind().clone(),
            model_id: plugin.model_id().cloned(),
            seen: replaces.iter().cloned().collect(),
        };
        Ok(Some(BuiltSegment::dropping(
            head.index_id().clone(),
            replaces,
            guard,
        )))
    }

    /// Builds the segment of `promotion` from its extracted files with `index`, which must be
    /// the plug-in it was planned with, and writes its files. Publish it with
    /// [`Indexer::publish`]. With no chunks it only takes away the segments it replaces, or
    /// returns `None` if it replaces none.
    ///
    /// # Errors
    ///
    /// Returns [`Error::WrongPlugin`] for another plug-in, [`Error::WrongIndex`] for a
    /// promotion of another index, and otherwise as for [`Indexer::build`].
    pub async fn build_promotion(
        &self,
        index: &SharedIndex,
        promotion: &Promotion,
        files: Vec<ExtractedFile>,
    ) -> Result<Option<BuiltSegment>> {
        self.check_index(&promotion.index_id)?;
        let plugin = index.get();
        if plugin.kind() != &promotion.kind || plugin.model_id() != promotion.model_id.as_ref() {
            return Err(Error::WrongPlugin {
                kind: promotion.kind.clone(),
            });
        }
        let guard = Guard {
            shard: promotion.shard.clone(),
            kind: promotion.kind.clone(),
            model_id: promotion.model_id.clone(),
            seen: promotion
                .replaces
                .iter()
                .chain(&promotion.keeps)
                .cloned()
                .collect(),
        };
        self.build_segment(index, files, promotion.replaces.clone(), Some(guard))
            .await
    }
}
