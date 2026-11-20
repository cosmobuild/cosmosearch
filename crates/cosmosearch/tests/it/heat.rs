use std::time::{Duration, UNIX_EPOCH};

use cosmosearch::{
    core::Signal,
    heat::{Clock, FakeClock, HeatSettings, HeatStore, MergeOutcome},
};

use crate::reading::{Fixture, search};

#[tokio::test]
async fn engines_sharing_a_recorder_write_heat_that_a_merge_counts() {
    let f = Fixture::new().await;
    f.index(&[("docs/a.txt", b"refund policy text")]).await;
    let clock = FakeClock::new(UNIX_EPOCH + Duration::from_hours(500_000));
    let heat = HeatStore::new(&f.index, HeatSettings::new().clock(clock.clone())).unwrap();
    let mut writer = heat.writer().unwrap();
    let first = f.builder().heat(writer.recorder()).open().await.unwrap();
    let second = f.builder().heat(writer.recorder()).open().await.unwrap();
    let shard = first.manifest().shards().keys().next().unwrap().clone();

    let results = search(&first, "refund").await;
    search(&second, "refund").await;
    second.read(results.hits()[0].source()).await.unwrap();
    writer.flush().await.unwrap().unwrap();
    assert_eq!(heat.merge().await.unwrap(), MergeOutcome::Committed(1));

    let merged = heat.merged().await.unwrap();
    let counts = merged.heat().shard(&shard, clock.now());
    assert!((counts.get(Signal::Search) - 2.0).abs() < 1e-9);
    assert!((counts.get(Signal::Hit) - 2.0).abs() < 1e-9);
    assert!((counts.get(Signal::Read) - 1.0).abs() < 1e-9);
    assert!(counts.get(Signal::Unmet).abs() < f64::EPSILON);
}
