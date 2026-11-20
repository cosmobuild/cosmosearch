use std::collections::VecDeque;

use bytes::Bytes;
use cosmosearch_store::{Error as StoreError, ObjectKey, ScopedStore};

use crate::table::Table;
use crate::{Heat, HeatRecorder, HeatSettings, MergedHeat, Result, Snapshot, WriterId, codec};

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<HeatWriter>();
    assert_send_sync::<HeatRecorder>();
};

/// Writes one process's heat to the store as numbered snapshots, each the heat recorded since
/// the one before.
///
/// A snapshot whose write fails, or whose outcome is unknown, is written again with the same
/// key and body by the next flush, before anything newer, so a writer's snapshots have no
/// gaps. A process that stops loses only the heat no flush has written. The writer keeps
/// the snapshots it wrote for [`HeatWriter::unmerged`], at most
/// [`HeatSettings::kept_snapshots`], dropping the oldest beyond that. The library runs
/// nothing in the background: the host calls [`HeatWriter::flush`], every minute or so.
#[derive(Debug)]
pub struct HeatWriter {
    scope: ScopedStore,
    settings: HeatSettings,
    recorder: HeatRecorder,
    id: WriterId,
    next_seq: u64,
    in_flight: Option<Sealed>,
    unmerged: VecDeque<(Snapshot, Table)>,
}

#[derive(Debug)]
struct Sealed {
    snapshot: Snapshot,
    key: ObjectKey,
    body: Bytes,
    table: Table,
}

impl HeatWriter {
    pub(crate) fn new(scope: ScopedStore, settings: HeatSettings) -> Result<Self> {
        let id = WriterId::new(settings.now_ms())?;
        let recorder = HeatRecorder::new(settings.clock.clone(), settings.half_life);
        Ok(HeatWriter {
            scope,
            settings,
            recorder,
            id,
            next_seq: 1,
            in_flight: None,
            unmerged: VecDeque::new(),
        })
    }

    /// The recorder whose heat this writer writes, to hand to readers.
    pub fn recorder(&self) -> HeatRecorder {
        self.recorder.clone()
    }

    /// The id the next snapshot is written under. It changes when it is half the writer
    /// retention old.
    pub fn id(&self) -> &WriterId {
        &self.id
    }

    /// The snapshot whose write failed or has an unknown outcome, written again by the next
    /// flush.
    pub fn in_flight(&self) -> Option<&Snapshot> {
        self.in_flight.as_ref().map(|sealed| &sealed.snapshot)
    }

    /// Writes the snapshot in flight, if any, then the heat recorded since the last snapshot,
    /// if any, and returns the newest snapshot written.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Store`](crate::Error::Store) if a write fails or its outcome is
    /// unknown; that snapshot stays in flight and recording goes on.
    pub async fn flush(&mut self) -> Result<Option<Snapshot>> {
        let mut written = None;
        if let Some(sealed) = &self.in_flight {
            create(&self.scope, sealed).await?;
            written = self.confirm();
        }
        let now = self.settings.now_ms();
        let rotate_at = self
            .id
            .born_ms()
            .saturating_add(self.settings.retention_ms() / 2);
        if now >= rotate_at {
            self.id = WriterId::new(now)?;
            self.next_seq = 1;
        }
        let snapshot = Snapshot::new(self.id.clone(), self.next_seq);
        let key = snapshot.key()?;
        let mut table = self.recorder.take();
        if table.is_empty() {
            return Ok(written);
        }
        table.decay_to(now.max(table.as_of()), self.settings.half_life);
        self.next_seq = self.next_seq.saturating_add(1);
        let sealed = self.in_flight.insert(Sealed {
            snapshot,
            key,
            body: codec::encode_snapshot(&table),
            table,
        });
        create(&self.scope, sealed).await?;
        Ok(self.confirm())
    }

    /// This process's heat that `merged` does not hold: snapshots written but not merged, up
    /// to the kept ones, the one in flight and what was recorded since. Added to `merged`'s
    /// heat it gives the freshest heat this process can know, never more than it recorded.
    pub fn unmerged(&mut self, merged: &MergedHeat) -> Heat {
        let half_life = self.settings.half_life;
        let cutoff = merged
            .clock_ms()
            .saturating_sub(self.settings.retention_ms());
        let waiting = |snapshot: &Snapshot| match merged.merged_through(snapshot.writer()) {
            Some(through) => snapshot.seq() > through,
            None => snapshot.writer().born_ms() >= cutoff,
        };
        self.unmerged.retain(|(snapshot, _)| waiting(snapshot));
        if self
            .in_flight
            .as_ref()
            .is_some_and(|sealed| !waiting(&sealed.snapshot))
        {
            self.in_flight = None;
        }
        let mut table = self.recorder.pending_table();
        let written = self.unmerged.iter().map(|(_, table)| table);
        for snapshot in written.chain(self.in_flight.as_ref().map(|sealed| &sealed.table)) {
            table.add(snapshot, half_life);
        }
        Heat::from_table(table, half_life)
    }

    fn confirm(&mut self) -> Option<Snapshot> {
        let sealed = self.in_flight.take()?;
        let cutoff = self
            .settings
            .now_ms()
            .saturating_sub(self.settings.retention_ms());
        self.unmerged
            .retain(|(snapshot, _)| snapshot.writer().born_ms() >= cutoff);
        self.unmerged
            .push_back((sealed.snapshot.clone(), sealed.table));
        while self.unmerged.len() > self.settings.kept_snapshots {
            self.unmerged.pop_front();
        }
        Some(sealed.snapshot)
    }
}

async fn create(scope: &ScopedStore, sealed: &Sealed) -> Result<()> {
    match scope.put_create(&sealed.key, sealed.body.clone()).await {
        Ok(()) | Err(StoreError::AlreadyExists { .. }) => Ok(()),
        Err(err) => Err(err.into()),
    }
}
