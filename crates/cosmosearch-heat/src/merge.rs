use std::collections::{BTreeMap, BTreeSet};

use cosmosearch_store::{Error as StoreError, ObjectInfo};
use futures_util::{StreamExt, stream};
use tracing::{debug, warn};

use crate::codec::{self, MergedParts};
use crate::merged::version_key;
use crate::store::CONCURRENCY;
use crate::{Error, HeatStore, MergedHeat, Result, WriterId};

/// How a merge ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MergeOutcome {
    /// The merge wrote this version.
    Committed(u64),
    /// No snapshot, writer or shard changed since the newest version, so nothing was written.
    NothingNew,
    /// Another merge wrote first; merging again takes what is left.
    Conflict,
    /// The version may still land; merging again finds out.
    Unknown,
}

pub(crate) enum Planned {
    Next(MergedParts),
    NothingNew,
    Stale,
}

pub(crate) async fn run(store: &HeatStore) -> Result<MergeOutcome> {
    match plan(store).await? {
        Planned::Next(next) => commit(store, &next).await,
        Planned::NothingNew => Ok(MergeOutcome::NothingNew),
        Planned::Stale => Ok(MergeOutcome::Conflict),
    }
}

/// Reads the newest version and every writer's snapshots that follow on from it, then the
/// newest manifest, and forms the next version.
pub(crate) async fn plan(store: &HeatStore) -> Result<Planned> {
    let settings = store.settings();
    let half_life = settings.half_life;
    let parent = store
        .head()
        .await?
        .map_or_else(MergedParts::default, MergedHeat::into_parts);
    let clock = settings.now_ms().max(parent.clock);
    let cutoff = clock.saturating_sub(settings.retention_ms());

    let mut listed: BTreeMap<WriterId, BTreeMap<u64, ObjectInfo>> = BTreeMap::new();
    for (snapshot, info) in store.list_snapshots().await? {
        listed
            .entry(snapshot.writer().clone())
            .or_default()
            .insert(snapshot.seq(), info);
    }
    let mut wanted: Vec<(&WriterId, u64, &ObjectInfo)> = Vec::new();
    for (writer, seqs) in &listed {
        let through = parent.writers.get(writer).copied();
        if through.is_none() && writer.born_ms() < cutoff {
            continue;
        }
        let mut seq = through.unwrap_or(0);
        while let Some(next) = seq.checked_add(1)
            && let Some(info) = seqs.get(&next)
        {
            wanted.push((writer, next, info));
            seq = next;
        }
    }
    let reads: Vec<_> = wanted
        .iter()
        .map(|(_, _, info)| store.read_whole(info))
        .collect();
    let mut bodies = stream::iter(&wanted).zip(stream::iter(reads).buffered(CONCURRENCY));

    let mut next = MergedParts {
        version: parent.version.saturating_add(1),
        clock,
        writers: parent.writers,
        table: parent.table,
    };
    let mut taken = 0usize;
    let mut stopped = BTreeSet::new();
    while let Some(((writer, seq, info), body)) = bodies.next().await {
        if stopped.contains(writer) {
            continue;
        }
        let body = match body {
            Ok(body) => body,
            Err(Error::Store(StoreError::NotFound { .. })) => return Ok(Planned::Stale),
            Err(err) => return Err(err),
        };
        match codec::decode_snapshot(&body) {
            Ok(table) => {
                next.table.add(&table, half_life);
                next.writers.insert((*writer).clone(), *seq);
                taken = taken.saturating_add(1);
            }
            Err(err) => {
                warn!(key = %info.key(), error = %err.at(info.key().as_str()), "a heat snapshot does not decode, so it and its writer's later snapshots wait");
                stopped.insert(writer);
            }
        }
    }
    let tracked = next.writers.len();
    next.writers.retain(|writer, _| writer.born_ms() >= cutoff);
    let expired = tracked.saturating_sub(next.writers.len());
    let removed = match store.index().head().await? {
        Some(manifest) => next
            .table
            .retain(|shard| manifest.shards().contains_key(shard)),
        None => 0,
    };
    if taken == 0 && expired == 0 && removed == 0 {
        return Ok(Planned::NothingNew);
    }
    debug!(
        version = next.version,
        snapshots = taken,
        expired,
        removed,
        "heat merge planned"
    );
    let as_of = next.table.as_of().max(clock);
    next.table.decay_to(as_of, half_life);
    Ok(Planned::Next(next))
}

pub(crate) async fn commit(store: &HeatStore, next: &MergedParts) -> Result<MergeOutcome> {
    let key = version_key(next.version)?;
    match store
        .scope()
        .put_create(&key, codec::encode_merged(next))
        .await
    {
        Ok(()) => Ok(MergeOutcome::Committed(next.version)),
        Err(StoreError::AlreadyExists { .. }) => Ok(MergeOutcome::Conflict),
        Err(StoreError::OutcomeUnknown { .. }) => Ok(MergeOutcome::Unknown),
        Err(err) => Err(err.into()),
    }
}
