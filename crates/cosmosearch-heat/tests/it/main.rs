//! Contract tests for `cosmosearch-heat`.

#![expect(
    clippy::panic,
    clippy::unwrap_used,
    reason = "helpers in a test binary fail the test by panicking"
)]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use cosmosearch_core::{
    Action, FileSource, HeatPolicy, HeatSink, KeyRange, Level, ShardCost, ShardEntry, ShardId,
    Signal,
};
use cosmosearch_heat::{
    Clock, Error, FakeClock, HalfLife, HeatSettings, HeatStore, MergeOutcome, MergedHeat,
    PlannedPromotion, PolicySettings, Snapshot, SpendPlan, ThresholdPolicy,
};
use cosmosearch_store::{
    CreateOutcome, Fault, FaultStore, IndexHandle, MemoryStore, MetadataStore, ObjectKey, Op,
    new_commit_id,
};

const MINUTE: Duration = Duration::from_secs(60);
const DAY: Duration = Duration::from_hours(24);

struct Fixture {
    clock: FakeClock,
    faults: FaultStore,
    index: IndexHandle,
    heat: HeatStore,
}

fn shard(index: usize) -> ShardId {
    ShardId::new(format!("s{index}")).unwrap()
}

fn start() -> SystemTime {
    UNIX_EPOCH + Duration::from_hours(500_000)
}

async fn set_shards(index: &IndexHandle, count: usize) {
    let head = index.head().await.unwrap().unwrap();
    let mut shards = BTreeMap::new();
    for position in 0..count {
        let start = if position == 0 {
            String::new()
        } else {
            format!("k{position}")
        };
        let end = (position + 1 < count).then(|| format!("k{}", position + 1));
        let range = KeyRange::new(start, end).unwrap();
        shards.insert(shard(position), ShardEntry::new(range, Vec::new()));
    }
    let next = head
        .successor(new_commit_id().unwrap(), shards, BTreeSet::new())
        .unwrap();
    index.commit(&next).await.unwrap();
}

impl Fixture {
    async fn new() -> Fixture {
        Fixture::with(HeatSettings::new()).await
    }

    async fn with(settings: HeatSettings) -> Fixture {
        let clock = FakeClock::new(start());
        let faults = FaultStore::new(MemoryStore::new());
        let metadata =
            MetadataStore::new(faults.clone(), "s3://example-bucket/index".parse().unwrap())
                .unwrap();
        let data = "s3://example-bucket/docs/".parse().unwrap();
        let outcome = metadata.create(&data, FileSource::Listing).await.unwrap();
        let CreateOutcome::Created(index) = outcome else {
            panic!("expected created, got {outcome:?}");
        };
        set_shards(&index, 5).await;
        let heat = HeatStore::new(&index, settings.clock(clock.clone())).unwrap();
        Fixture {
            clock,
            faults,
            index,
            heat,
        }
    }

    fn process(&self, now: SystemTime) -> (FakeClock, HeatStore) {
        let clock = FakeClock::new(now);
        let settings = HeatSettings::new().clock(clock.clone());
        (clock, HeatStore::new(&self.index, settings).unwrap())
    }

    async fn merge_until_quiet(&self) -> MergedHeat {
        merge_until_quiet(&self.heat).await
    }
}

async fn merge_until_quiet(heat: &HeatStore) -> MergedHeat {
    for _ in 0..50 {
        if heat.merge().await.unwrap() == MergeOutcome::NothingNew {
            return heat.merged().await.unwrap();
        }
    }
    panic!("merging never settled");
}

fn worth(age: Duration) -> f64 {
    (-age.as_secs_f64() / HalfLife::default().get().as_secs_f64()).exp2()
}

fn value(merged: &MergedHeat, shard: &ShardId, signal: Signal, at: SystemTime) -> f64 {
    merged.heat().shard(shard, at).get(signal)
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0)
}

#[tokio::test]
async fn restart_keeps_heat_and_a_crash_loses_only_what_was_not_flushed() {
    let f = Fixture::new().await;
    let s1 = shard(1);
    let mut first = f.heat.writer().unwrap();
    first.recorder().record(Signal::Read, &[&s1]);
    f.clock.advance(MINUTE);
    first.flush().await.unwrap().unwrap();
    first.recorder().record(Signal::Read, &[&s1]);
    drop(first);

    let (clock, restarted) = f.process(start() + 2 * MINUTE);
    let mut second = restarted.writer().unwrap();
    second.recorder().record(Signal::Read, &[&s1]);
    clock.advance(MINUTE);
    second.flush().await.unwrap().unwrap();
    f.clock.set(start() + 3 * MINUTE);

    let merged = f.merge_until_quiet().await;
    let at = start() + 3 * MINUTE;
    let want = worth(3 * MINUTE) + worth(MINUTE);
    assert!(close(value(&merged, &s1, Signal::Read, at), want));
    let read_after_restart = restarted.merged().await.unwrap();
    assert_eq!(read_after_restart.version(), merged.version());
    assert!(close(
        value(&read_after_restart, &s1, Signal::Read, at),
        want
    ));

    assert_eq!(f.heat.merge().await.unwrap(), MergeOutcome::NothingNew);
    let swept = f.heat.sweep().await.unwrap();
    assert_eq!(swept.snapshots(), 2);
    f.clock.advance(DAY);
    assert_eq!(f.heat.merge().await.unwrap(), MergeOutcome::NothingNew);
    assert_eq!(f.heat.merged().await.unwrap().version(), merged.version());
}

#[derive(Debug, Default)]
struct Recorded {
    events: BTreeMap<Snapshot, Vec<(ShardId, Signal, SystemTime)>>,
}

impl Recorded {
    fn expected(&self, shard: &ShardId, signal: Signal, at: SystemTime) -> f64 {
        self.events
            .values()
            .flatten()
            .filter(|(s, g, _)| s == shard && *g == signal)
            .map(|(_, _, time)| worth(at.duration_since(*time).unwrap()))
            .sum()
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_writers_and_merges_lose_no_counts() {
    const WRITERS: usize = 8;
    const EVENTS: usize = 600;
    const FLUSH_EVERY: usize = 20;
    let f = Fixture::new().await;
    let done = Arc::new(AtomicBool::new(false));
    let commits = Arc::new(AtomicUsize::new(0));

    let mut writers = Vec::new();
    for w in 0..WRITERS {
        let (clock, heat) = f.process(start());
        writers.push(tokio::spawn(async move {
            let mut writer = heat.writer().unwrap();
            let recorder = writer.recorder();
            let mut written = Recorded::default();
            let mut pending = Vec::new();
            for i in 0..EVENTS {
                clock.advance(Duration::from_secs(1));
                let id = shard((i * 7 + w * 3) % 5);
                let signal = Signal::ALL[(i + w) % Signal::ALL.len()];
                recorder.record(signal, &[&id]);
                pending.push((id, signal, clock.now()));
                if i % FLUSH_EVERY == FLUSH_EVERY - 1 {
                    let snapshot = writer.flush().await.unwrap().unwrap();
                    written
                        .events
                        .insert(snapshot, std::mem::take(&mut pending));
                    tokio::task::yield_now().await;
                }
            }
            written
        }));
    }
    let mut mergers = Vec::new();
    for _ in 0..3 {
        let (clock, heat) = f.process(start());
        let done = Arc::clone(&done);
        let commits = Arc::clone(&commits);
        mergers.push(tokio::spawn(async move {
            while !done.load(Ordering::SeqCst) {
                clock.advance(Duration::from_secs(1));
                if let MergeOutcome::Committed(_) = heat.merge().await.unwrap() {
                    commits.fetch_add(1, Ordering::SeqCst);
                }
                heat.sweep().await.unwrap();
                tokio::task::yield_now().await;
            }
        }));
    }
    let mut recorded = Recorded::default();
    for handle in writers {
        recorded.events.extend(handle.await.unwrap().events);
    }
    done.store(true, Ordering::SeqCst);
    for handle in mergers {
        handle.await.unwrap();
    }

    let merged = f.merge_until_quiet().await;
    assert_eq!(recorded.events.len(), WRITERS * EVENTS / FLUSH_EVERY);
    assert!(
        commits.load(Ordering::SeqCst) > 1,
        "merges ran during the writes"
    );
    let at = merged.heat().as_of();
    for position in 0..5 {
        for signal in Signal::ALL {
            let id = shard(position);
            let want = recorded.expected(&id, signal, at);
            let got = value(&merged, &id, signal, at);
            assert!(
                close(got, want),
                "{id} {signal:?}: merged {got}, recorded {want}"
            );
        }
    }
}

#[tokio::test]
async fn merges_run_twice_or_at_once_count_each_snapshot_once() {
    let f = Fixture::new().await;
    let s2 = shard(2);
    for _ in 0..2 {
        let mut writer = f.heat.writer().unwrap();
        writer.recorder().record(Signal::Search, &[&s2]);
        writer.flush().await.unwrap().unwrap();
    }
    let racing = tokio::join!(
        f.heat.merge(),
        f.heat.merge(),
        f.heat.merge(),
        f.heat.merge()
    );
    let outcomes = [
        racing.0.unwrap(),
        racing.1.unwrap(),
        racing.2.unwrap(),
        racing.3.unwrap(),
    ];
    let committed = outcomes
        .iter()
        .filter(|outcome| **outcome == MergeOutcome::Committed(1))
        .count();
    assert_eq!(committed, 1, "{outcomes:?}");
    assert!(outcomes.iter().all(|outcome| matches!(
        outcome,
        MergeOutcome::Committed(1) | MergeOutcome::Conflict | MergeOutcome::NothingNew
    )));
    assert_eq!(f.heat.merge().await.unwrap(), MergeOutcome::NothingNew);
    let merged = f.heat.merged().await.unwrap();
    assert!(close(value(&merged, &s2, Signal::Search, start()), 2.0));
}

#[tokio::test]
async fn a_failed_or_unknown_write_is_retried_with_the_same_key() {
    let f = Fixture::new().await;
    let s0 = shard(0);
    let mut writer = f.heat.writer().unwrap();
    let recorder = writer.recorder();

    for fault in [
        Fault::Timeout,
        Fault::LostResponse,
        Fault::Delayed,
        Fault::Crash,
    ] {
        recorder.record(Signal::Warm, &[&s0]);
        f.faults.inject(Op::PutCreate, "/heat/deltas/", fault);
        writer.flush().await.unwrap_err();
        let stuck = writer.in_flight().unwrap().clone();
        recorder.record(Signal::Warm, &[&s0]);
        if fault == Fault::Delayed {
            f.faults.release_delayed().await.unwrap();
        }
        let written = writer.flush().await.unwrap().unwrap();
        assert_eq!(written.writer(), stuck.writer());
        assert_eq!(written.seq(), stuck.seq() + 1, "{fault:?}");
        assert!(writer.in_flight().is_none());
    }
    let merged = f.merge_until_quiet().await;
    assert!(close(value(&merged, &s0, Signal::Warm, start()), 8.0));
}

#[tokio::test]
async fn writer_ids_rotate_at_half_the_retention_and_old_writers_are_dropped() {
    let f = Fixture::with(HeatSettings::new().writer_retention(2 * DAY)).await;
    let s3 = shard(3);
    let mut writer = f.heat.writer().unwrap();
    let first = writer.id().clone();
    writer.recorder().record(Signal::Read, &[&s3]);
    writer.flush().await.unwrap().unwrap();
    f.merge_until_quiet().await;

    f.clock.advance(DAY);
    writer.recorder().record(Signal::Read, &[&s3]);
    let written = writer.flush().await.unwrap().unwrap();
    assert_ne!(written.writer(), &first);
    assert_eq!(written.seq(), 1);
    f.merge_until_quiet().await;

    f.clock.advance(DAY + MINUTE);
    let merged = f.merge_until_quiet().await;
    let at = f.clock.now();
    let want = worth(2 * DAY + MINUTE) + worth(DAY + MINUTE);
    assert!(close(value(&merged, &s3, Signal::Read, at), want));
    let swept = f.heat.sweep().await.unwrap();
    assert_eq!(swept.snapshots(), 2);
    assert_eq!(f.heat.sweep().await.unwrap().snapshots(), 0);
}

#[tokio::test]
async fn heat_of_shards_no_longer_in_the_manifest_is_dropped() {
    let f = Fixture::new().await;
    let (s1, s4) = (shard(1), shard(4));
    let mut writer = f.heat.writer().unwrap();
    writer.recorder().record(Signal::Search, &[&s1, &s4]);
    writer.flush().await.unwrap().unwrap();
    let merged = f.merge_until_quiet().await;
    assert_eq!(merged.heat().len(), 2);

    set_shards(&f.index, 4).await;
    assert_eq!(f.heat.merge().await.unwrap(), MergeOutcome::Committed(2));
    let merged = f.heat.merged().await.unwrap();
    assert_eq!(merged.heat().len(), 1);
    assert!(close(value(&merged, &s1, Signal::Search, start()), 1.0));
    assert!(close(value(&merged, &s4, Signal::Search, start()), 0.0));
}

#[tokio::test]
async fn a_process_sees_its_own_unmerged_heat_without_counting_it_twice() {
    let f = Fixture::new().await;
    let s1 = shard(1);
    let mut writer = f.heat.writer().unwrap();
    let total = |merged: &MergedHeat, writer: &mut cosmosearch_heat::HeatWriter| {
        let mut heat = merged.heat().clone();
        heat.add(&writer.unmerged(merged));
        heat.shard(&s1, start()).get(Signal::Read)
    };
    writer.recorder().record(Signal::Read, &[&s1]);
    let merged = f.heat.merged().await.unwrap();
    assert!(close(total(&merged, &mut writer), 1.0));

    writer.flush().await.unwrap().unwrap();
    writer.recorder().record(Signal::Read, &[&s1]);
    assert!(close(total(&merged, &mut writer), 2.0));

    let merged = f.merge_until_quiet().await;
    assert!(close(value(&merged, &s1, Signal::Read, start()), 1.0));
    assert!(close(total(&merged, &mut writer), 2.0));
    assert!(close(
        writer
            .unmerged(&merged)
            .shard(&s1, start())
            .get(Signal::Read),
        1.0
    ));
}

#[tokio::test]
async fn a_writer_keeps_a_bounded_number_of_snapshots_and_never_counts_twice() {
    let f = Fixture::with(HeatSettings::new().kept_snapshots(10)).await;
    let s1 = shard(1);
    let mut writer = f.heat.writer().unwrap();
    let total = |merged: &MergedHeat, writer: &mut cosmosearch_heat::HeatWriter| {
        let mut heat = merged.heat().clone();
        heat.add(&writer.unmerged(merged));
        heat.shard(&s1, start()).get(Signal::Read)
    };
    let flush_reads = async |writer: &mut cosmosearch_heat::HeatWriter, count: usize| {
        for _ in 0..count {
            writer.recorder().record(Signal::Read, &[&s1]);
            writer.flush().await.unwrap().unwrap();
        }
    };

    flush_reads(&mut writer, 500).await;
    let unmerged = f.heat.merged().await.unwrap();
    assert!(close(total(&unmerged, &mut writer), 10.0));

    f.faults
        .inject(Op::PutCreate, "/heat/deltas/", Fault::Timeout);
    writer.recorder().record(Signal::Read, &[&s1]);
    writer.flush().await.unwrap_err();
    assert!(writer.in_flight().is_some());
    assert!(close(total(&unmerged, &mut writer), 11.0));

    let merged = f.merge_until_quiet().await;
    assert!(close(total(&merged, &mut writer), 501.0));
    flush_reads(&mut writer, 30).await;
    assert!(writer.in_flight().is_none());
    assert!(close(total(&merged, &mut writer), 510.0));

    let merged = f.merge_until_quiet().await;
    assert!(close(value(&merged, &s1, Signal::Read, start()), 531.0));
    assert!(close(total(&merged, &mut writer), 531.0));
}

#[tokio::test]
async fn a_damaged_snapshot_holds_back_only_its_writer() {
    let f = Fixture::new().await;
    let (s0, s1) = (shard(0), shard(1));
    let mut damaged = f.heat.writer().unwrap();
    let mut healthy = f.heat.writer().unwrap();
    damaged.recorder().record(Signal::Read, &[&s0]);
    damaged.flush().await.unwrap().unwrap();
    let scope = f.index.scope("heat").unwrap();
    let key = ObjectKey::new(format!("deltas/{}/000000000002", damaged.id())).unwrap();
    scope
        .put_create(&key, Bytes::from_static(b"not heat"))
        .await
        .unwrap();
    for _ in 0..2 {
        damaged.recorder().record(Signal::Read, &[&s0]);
        damaged.flush().await.unwrap().unwrap();
    }
    healthy.recorder().record(Signal::Read, &[&s1]);
    healthy.flush().await.unwrap().unwrap();

    let merged = f.merge_until_quiet().await;
    assert!(close(value(&merged, &s0, Signal::Read, start()), 1.0));
    assert!(close(value(&merged, &s1, Signal::Read, start()), 1.0));
}

#[tokio::test]
async fn a_merged_version_of_another_format_is_reported() {
    let f = Fixture::new().await;
    let scope = f.index.scope("heat").unwrap();
    let mut body = b"CSHT".to_vec();
    body.extend_from_slice(&9u32.to_le_bytes());
    body.push(2);
    let key = ObjectKey::new("merged/v000000000001").unwrap();
    scope.put_create(&key, Bytes::from(body)).await.unwrap();
    let err = f.heat.merged().await.unwrap_err();
    assert!(
        matches!(err, Error::UnsupportedFormat { format: 9, .. }),
        "{err}"
    );
    assert!(matches!(
        f.heat.merge().await,
        Err(Error::UnsupportedFormat { .. })
    ));
}

#[tokio::test]
async fn settings_out_of_range_are_refused() {
    let f = Fixture::new().await;
    let settings = HeatSettings::new().writer_retention(Duration::ZERO);
    assert!(matches!(
        HeatStore::new(&f.index, settings),
        Err(Error::InvalidSettings { .. })
    ));
    assert!(matches!(
        HalfLife::new(Duration::ZERO),
        Err(Error::InvalidSettings { .. })
    ));
}

#[tokio::test]
async fn merged_heat_decides_promotions_within_a_spend_limit() {
    const MEBIBYTE: u64 = 1 << 20;
    let fixture = Fixture::new().await;
    let mut writer = fixture.heat.writer().unwrap();
    let recorder = writer.recorder();
    for _ in 0..40 {
        recorder.record(Signal::Read, &[&shard(1)]);
    }
    for _ in 0..12 {
        recorder.record(Signal::Read, &[&shard(3)]);
    }
    recorder.record(Signal::Search, &[&shard(4)]);
    writer.flush().await.unwrap();
    let merged = fixture.merge_until_quiet().await;
    let manifest = fixture.index.head().await.unwrap().unwrap();
    let settings = PolicySettings::new().thresholds(1.0, 0.25);
    let policy =
        ThresholdPolicy::new(merged.heat(), fixture.clock.now(), &manifest, &settings).unwrap();

    assert!(close(policy.value(&shard(1)), 120.0));
    assert!(close(policy.value(&shard(2)), 0.1 * (120.0 + 36.0)));
    assert_eq!(
        policy.decide(&shard(2), Level::L1, 8 * MEBIBYTE),
        Action::Promote {
            level: Level::L2,
            value: policy.value(&shard(2)),
        }
    );

    let shards: Vec<ShardCost> = (0..5)
        .map(|position| ShardCost::new(shard(position), Level::L1, 30 * MEBIBYTE))
        .collect();
    let plan = SpendPlan::fill(&policy, &shards, 70 * MEBIBYTE);
    let promoted: Vec<&ShardId> = plan
        .promotions()
        .iter()
        .map(PlannedPromotion::shard)
        .collect();
    assert_eq!(promoted, [&shard(1), &shard(3)]);
    assert_eq!(plan.spent(), 60 * MEBIBYTE);
    assert_eq!(plan.deferred(), []);

    fixture.clock.advance(60 * DAY);
    let later =
        ThresholdPolicy::new(merged.heat(), fixture.clock.now(), &manifest, &settings).unwrap();
    assert_eq!(
        later.decide(&shard(1), Level::L2, 30 * MEBIBYTE),
        Action::Demote { level: Level::L1 }
    );
    assert_eq!(
        later.decide(&shard(1), Level::L1, 30 * MEBIBYTE),
        Action::Keep
    );
}
