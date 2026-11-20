use std::collections::{BTreeMap, BTreeSet};
use std::mem;
use std::time::Duration;

use cosmosearch_core::{FileSource, KeyRange, ShardEntry, ShardId, Signal};
use cosmosearch_store::{
    CreateOutcome, Fault, FaultStore, MemoryStore, MetadataStore, Op, new_commit_id,
};
use proptest::prelude::*;

use crate::clock::{system_time, unix_ms};
use crate::merge::{self, Planned};
use crate::table::{Counts, SIGNALS};
use crate::{
    Clock, FakeClock, HalfLife, HeatSettings, HeatStore, HeatWriter, MergeOutcome, MergedHeat,
    Snapshot, sweep,
};

const WRITERS: usize = 3;
const MERGERS: usize = 2;
const SHARDS: usize = 5;
const HOUR: u64 = 3_600_000;
const DAY: u64 = 24 * HOUR;
const START: u64 = 1_800_000_000_000;

#[derive(Debug, Clone)]
enum Step {
    Record {
        writer: usize,
        shard: usize,
        slot: usize,
        amount: f64,
    },
    Advance(u64),
    Flush {
        writer: usize,
        fault: Option<Fault>,
    },
    Crash {
        writer: usize,
    },
    Plan {
        merger: usize,
    },
    Commit {
        merger: usize,
        fault: Option<Fault>,
    },
    Merge,
    Sweep,
    Release,
}

fn fault() -> impl Strategy<Value = Option<Fault>> {
    prop_oneof![
        4 => Just(None),
        1 => Just(Some(Fault::Timeout)),
        1 => Just(Some(Fault::LostResponse)),
        1 => Just(Some(Fault::Delayed)),
        1 => Just(Some(Fault::Crash)),
    ]
}

fn step(max_advance: u64) -> impl Strategy<Value = Step> {
    prop_oneof![
        6 => (0..WRITERS, 0..SHARDS, 0..SIGNALS, 0.1f64..10.0).prop_map(
            |(writer, shard, slot, amount)| Step::Record { writer, shard, slot, amount }
        ),
        2 => (1..max_advance).prop_map(Step::Advance),
        3 => (0..WRITERS, fault()).prop_map(|(writer, fault)| Step::Flush { writer, fault }),
        1 => (0..WRITERS).prop_map(|writer| Step::Crash { writer }),
        2 => (0..MERGERS).prop_map(|merger| Step::Plan { merger }),
        2 => (0..MERGERS, fault()).prop_map(|(merger, fault)| Step::Commit { merger, fault }),
        1 => Just(Step::Merge),
        1 => Just(Step::Sweep),
        1 => Just(Step::Release),
    ]
}

fn shard(index: usize) -> ShardId {
    ShardId::new(format!("s{index}")).unwrap()
}

async fn setup(clock: &FakeClock) -> (FaultStore, HeatStore) {
    let faults = FaultStore::new(MemoryStore::new());
    let metadata =
        MetadataStore::new(faults.clone(), "s3://example-bucket/index".parse().unwrap()).unwrap();
    let data = "s3://example-bucket/docs/".parse().unwrap();
    let CreateOutcome::Created(index) = metadata.create(&data, FileSource::Listing).await.unwrap()
    else {
        panic!("the index was not created");
    };
    let head = index.head().await.unwrap().unwrap();
    let starts = ["", "b", "c", "d", "e"];
    let ends = ["b", "c", "d", "e"];
    let mut shards = BTreeMap::new();
    for (position, start) in starts.iter().enumerate() {
        let end = ends.get(position).map(|end| (*end).to_owned());
        let range = KeyRange::new(*start, end).unwrap();
        shards.insert(shard(position), ShardEntry::new(range, Vec::new()));
    }
    let next = head
        .successor(new_commit_id().unwrap(), shards, BTreeSet::new())
        .unwrap();
    index.commit(&next).await.unwrap();
    let settings = HeatSettings::new().clock(clock.clone());
    (faults, HeatStore::new(&index, settings).unwrap())
}

#[derive(Debug, Clone)]
struct Event {
    shard: ShardId,
    slot: usize,
    amount: f64,
    at: u64,
}

#[derive(Debug, Default)]
struct Oracle {
    sealed: BTreeMap<Snapshot, Vec<Event>>,
    landed: BTreeSet<Snapshot>,
}

impl Oracle {
    async fn observe(&mut self, store: &HeatStore) {
        for (snapshot, _) in store.list_snapshots().await.unwrap() {
            if self.sealed.contains_key(&snapshot) {
                self.landed.insert(snapshot);
            }
        }
    }

    fn expected(&self, at: u64, half_life: HalfLife) -> BTreeMap<ShardId, Counts> {
        let mut out: BTreeMap<ShardId, Counts> = BTreeMap::new();
        for snapshot in &self.landed {
            for event in &self.sealed[snapshot] {
                let counts = out.entry(event.shard.clone()).or_default();
                counts[event.slot] += event.amount * half_life.factor(event.at, at);
            }
        }
        out
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Bound {
    Exact,
    AtMost,
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0)
}

fn compare(version: &MergedHeat, oracle: &Oracle, bound: Bound) -> Result<(), String> {
    let heat = version.heat();
    let at = heat.as_of();
    let expected = oracle.expected(unix_ms(at), heat.half_life());
    let merged = heat.shards(at);
    let shards: BTreeSet<&ShardId> = expected.keys().chain(merged.keys()).collect();
    for shard in shards {
        for (slot, signal) in Signal::ALL.iter().enumerate() {
            let got = merged.get(shard).map_or(0.0, |heat| heat.get(*signal));
            let want = expected.get(shard).map_or(0.0, |counts| counts[slot]);
            if !(close(got, want) || (bound == Bound::AtMost && got < want)) {
                return Err(format!("{shard} {signal:?}: merged {got}, recorded {want}"));
            }
        }
    }
    Ok(())
}

async fn merge_until_quiet(store: &HeatStore) -> MergedHeat {
    for _ in 0..50 {
        if merge::run(store).await.unwrap_or(MergeOutcome::Unknown) == MergeOutcome::NothingNew {
            break;
        }
    }
    store.merged().await.unwrap()
}

struct Slot {
    writer: HeatWriter,
    pending: Vec<Event>,
}

impl Slot {
    fn new(store: &HeatStore) -> Slot {
        Slot {
            writer: store.writer().unwrap(),
            pending: Vec::new(),
        }
    }

    async fn flush(&mut self, oracle: &mut Oracle) {
        let before = self.writer.in_flight().cloned();
        let result = self.writer.flush().await;
        let sealed = match (&result, self.writer.in_flight()) {
            (_, Some(now)) if Some(now) != before.as_ref() => Some(now.clone()),
            (Ok(Some(written)), _) if Some(written) != before.as_ref() => Some(written.clone()),
            _ => None,
        };
        if let Some(snapshot) = sealed {
            oracle.sealed.insert(snapshot, mem::take(&mut self.pending));
        }
    }

    fn settled(&self) -> bool {
        self.writer.in_flight().is_none() && self.pending.is_empty()
    }
}

async fn run(steps: Vec<Step>, bound: Bound) -> Result<(), String> {
    let clock = FakeClock::new(system_time(START));
    let (faults, store) = setup(&clock).await;
    let mut slots: Vec<Slot> = (0..WRITERS).map(|_| Slot::new(&store)).collect();
    let mut plans: Vec<Option<Planned>> = (0..MERGERS).map(|_| None).collect();
    let mut oracle = Oracle::default();

    for step in steps {
        match step {
            Step::Record {
                writer,
                shard: index,
                slot,
                amount,
            } => {
                let slot_state = &mut slots[writer];
                let id = shard(index);
                slot_state
                    .writer
                    .recorder()
                    .add(&id, Signal::ALL[slot], amount)
                    .unwrap();
                slot_state.pending.push(Event {
                    shard: id,
                    slot,
                    amount,
                    at: unix_ms(clock.now()),
                });
            }
            Step::Advance(ms) => clock.advance(Duration::from_millis(ms)),
            Step::Flush { writer, fault } => {
                if let Some(fault) = fault {
                    faults.inject(Op::PutCreate, "/heat/deltas/", fault);
                }
                slots[writer].flush(&mut oracle).await;
                oracle.observe(&store).await;
            }
            Step::Crash { writer } => slots[writer] = Slot::new(&store),
            Step::Plan { merger } => plans[merger] = Some(merge::plan(&store).await.unwrap()),
            Step::Commit { merger, fault } => {
                if let Some(Planned::Next(next)) = plans[merger].take() {
                    if let Some(fault) = fault {
                        faults.inject(Op::PutCreate, "/heat/merged/", fault);
                    }
                    merge::commit(&store, &next)
                        .await
                        .unwrap_or(MergeOutcome::Unknown);
                }
            }
            Step::Merge => {
                merge::run(&store).await.unwrap_or(MergeOutcome::Unknown);
            }
            Step::Sweep => {
                sweep::run(&store).await.unwrap();
            }
            Step::Release => {
                faults.release_delayed().await.unwrap();
                oracle.observe(&store).await;
            }
        }
        if let Some(head) = store.head().await.unwrap() {
            compare(&head, &oracle, Bound::AtMost)
                .map_err(|err| format!("double count mid-run: {err}"))?;
        }
    }

    faults.release_delayed().await.unwrap();
    oracle.observe(&store).await;
    for slot in &mut slots {
        for _ in 0..10 {
            if slot.settled() {
                break;
            }
            slot.flush(&mut oracle).await;
        }
        assert!(slot.settled(), "a writer could not flush");
    }
    faults.release_delayed().await.unwrap();
    oracle.observe(&store).await;
    let head = merge_until_quiet(&store).await;
    compare(&head, &oracle, bound)?;
    sweep::run(&store).await.unwrap();
    clock.advance(Duration::from_millis(HOUR));
    let again = merge_until_quiet(&store).await;
    compare(&again, &oracle, bound)?;
    Ok(())
}

fn check(steps: Vec<Step>, bound: Bound) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    runtime.block_on(run(steps, bound)).unwrap();
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn merged_heat_equals_the_heat_whose_snapshots_landed(
        steps in prop::collection::vec(step(HOUR), 1..100),
    ) {
        check(steps, Bound::Exact);
    }

    #[test]
    fn expiring_writers_never_count_twice(
        steps in prop::collection::vec(step(5 * DAY), 1..100),
    ) {
        check(steps, Bound::AtMost);
    }
}
