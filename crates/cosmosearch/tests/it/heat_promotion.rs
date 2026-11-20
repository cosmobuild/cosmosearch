use std::sync::Arc;
use std::time::{Duration, UNIX_EPOCH};

use cosmosearch::{
    Engine, ShardId,
    core::{Level, Manifest, ShardCost, Signal},
    heat::{
        Clock, FakeClock, Heat, HeatSettings, HeatStore, PlannedPromotion, PolicySettings,
        SpendPlan, ThresholdPolicy,
    },
    index::PublishOutcome,
};
use cosmosearch_core::SharedIndex;

use crate::promotion::Vectors;
use crate::reading::{keys, search};

const DAY: Duration = Duration::from_hours(24);

fn settings() -> PolicySettings {
    PolicySettings::new()
        .weight(Signal::Search, 0.0)
        .weight(Signal::Hit, 0.0)
        .neighbour_share(0.0)
        .thresholds(1_000.0, 100.0)
}

fn shard(name: &str) -> ShardId {
    ShardId::new(name).unwrap()
}

fn cost_of<'a>(costs: &'a [ShardCost], name: &str) -> &'a ShardCost {
    costs
        .iter()
        .find(|cost| cost.shard().as_str() == name)
        .unwrap()
}

struct Traffic {
    v: Vectors,
    clock: FakeClock,
    heat: Heat,
    manifest: Arc<Manifest>,
    engine: Engine,
    ivf: SharedIndex,
}

impl Traffic {
    async fn new() -> Traffic {
        let v = Vectors::new().await;
        let f = &v.fixture;
        f.split_at("docs/m").await;
        f.index(&[
            ("docs/a.txt", b"refund policy text"),
            ("docs/b.txt", b"refund rules for returns"),
            ("docs/z.txt", b"shipping rules"),
        ])
        .await;
        let clock = FakeClock::new(UNIX_EPOCH + Duration::from_hours(500_000));
        let store = HeatStore::new(&f.index, HeatSettings::new().clock(clock.clone())).unwrap();
        let mut writer = store.writer().unwrap();
        let engine = f
            .builder()
            .index(v.ivf())
            .heat(writer.recorder())
            .open()
            .await
            .unwrap();
        let results = search(&engine, "refund").await;
        assert_eq!(keys(&results).len(), 2, "both files of shard a match");
        for hit in results.hits() {
            engine.read(hit.source()).await.unwrap();
            engine.read(hit.source()).await.unwrap();
        }
        writer.flush().await.unwrap().unwrap();
        store.merge().await.unwrap();
        Traffic {
            ivf: SharedIndex::new(v.ivf()),
            heat: store.merged().await.unwrap().into_heat(),
            manifest: engine.manifest(),
            v,
            clock,
            engine,
        }
    }

    fn policy(&self) -> ThresholdPolicy {
        ThresholdPolicy::new(&self.heat, self.clock.now(), &self.manifest, &settings()).unwrap()
    }

    async fn costs(&self) -> Vec<ShardCost> {
        self.v.fixture.indexer.shard_costs(&self.ivf).await.unwrap()
    }

    async fn promote(&self, plan: &SpendPlan) {
        let indexer = &self.v.fixture.indexer;
        for promotion in plan.promotions() {
            let planned = indexer
                .plan_promotion(&self.ivf, promotion.shard())
                .await
                .unwrap();
            let built = self.v.build(&self.ivf, &planned).await;
            let outcome = indexer.publish(&built).await.unwrap();
            assert!(
                matches!(outcome, PublishOutcome::Published(_)),
                "the promotion was not published: {outcome:?}"
            );
        }
    }
}

#[tokio::test]
async fn spend_follows_traffic_within_the_limit() {
    let t = Traffic::new().await;
    let policy = t.policy();
    let costs = t.costs().await;
    assert_eq!(costs.len(), 2);
    let hot = cost_of(&costs, "a");
    assert_eq!((hot.level(), hot.cost()), (Level::L1, 18 + 24));
    assert_eq!(cost_of(&costs, "b").cost(), 14);

    let none = SpendPlan::fill(&policy, &costs, 41);
    assert_eq!(none.promotions(), []);
    assert_eq!(
        none.deferred().len(),
        1,
        "a limit too small defers the shard"
    );
    let plan = SpendPlan::fill(&policy, &costs, 42);
    let promoted: Vec<&ShardId> = plan
        .promotions()
        .iter()
        .map(PlannedPromotion::shard)
        .collect();
    assert_eq!(promoted, [&shard("a")], "only the shard that was read");
    assert_eq!(plan.spent(), 42);

    t.v.embedded();
    t.promote(&plan).await;
    assert_eq!(
        t.v.embedded(),
        2,
        "only the hot shard's chunks are embedded"
    );
    assert_eq!(t.v.promoted().await.len(), 1);

    let costs = t.costs().await;
    let hot = cost_of(&costs, "a");
    assert_eq!((hot.level(), hot.cost()), (Level::L2, 42));
    assert_eq!(SpendPlan::fill(&policy, &costs, u64::MAX).promotions(), []);

    t.v.fixture.index(&[("docs/c.txt", b"refund forms")]).await;
    let costs = t.costs().await;
    let hot = cost_of(&costs, "a");
    assert_eq!(
        (hot.level(), hot.cost()),
        (Level::L1, 12),
        "a new file makes the shard cost its bytes to promote again"
    );
}

#[tokio::test]
async fn a_shard_gone_cold_is_demoted_and_stays_searchable() {
    let t = Traffic::new().await;
    let indexer = &t.v.fixture.indexer;
    t.promote(&SpendPlan::fill(&t.policy(), &t.costs().await, u64::MAX))
        .await;
    assert_eq!(t.v.promoted().await.len(), 1);

    t.clock.advance(40 * DAY);
    let costs = t.costs().await;
    assert_eq!(
        SpendPlan::fill(&t.policy(), &costs, 0).demotions(),
        [],
        "between the thresholds a shard stays"
    );
    t.clock.advance(80 * DAY);
    let plan = SpendPlan::fill(&t.policy(), &costs, 0);
    assert_eq!(plan.demotions().len(), 1);

    let demotion = indexer
        .plan_demotion(&t.ivf, plan.demotions()[0].shard())
        .await
        .unwrap()
        .unwrap();
    let outcome = indexer.publish(&demotion).await.unwrap();
    assert!(matches!(outcome, PublishOutcome::Published(_)));
    assert_eq!(t.v.promoted().await, []);
    assert_eq!(
        indexer.publish(&demotion).await.unwrap(),
        PublishOutcome::Superseded,
        "publishing a demotion twice does nothing"
    );
    let again = indexer.plan_demotion(&t.ivf, &shard("a")).await.unwrap();
    assert!(again.is_none());
    assert_eq!(cost_of(&t.costs().await, "a").level(), Level::L1);

    t.engine.refresh().await.unwrap();
    let results = search(&t.engine, "refund").await;
    assert_eq!(keys(&results).len(), 2, "its files stay searchable at L1");
}
