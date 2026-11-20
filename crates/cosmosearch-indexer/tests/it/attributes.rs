use cosmosearch_core::{Attributes, Filter, Level, SegmentEntry};
use cosmosearch_indexer::{AttributeChange, PublishOutcome};

use super::*;

fn label(key: &str, value: &str) -> FileChange {
    FileChange::Attributes(
        AttributeChange::new(key).attribute("label", AttributeValue::One(value.to_owned())),
    )
}

fn labelled(key: &str, value: AttributeValue) -> FileChange {
    FileChange::Upsert(FileUpsert::new(key).attribute("label", value))
}

fn one(value: &str) -> AttributeValue {
    AttributeValue::One(value.to_owned())
}

fn legal() -> Filter {
    Filter::new().equals("label", "legal")
}

impl Fixture {
    async fn l1_segments(&self) -> Vec<SegmentEntry> {
        let head = self.index.head().await.unwrap().unwrap();
        head.shards()
            .values()
            .flat_map(|shard| shard.segments().iter().cloned())
            .filter(|segment| segment.level() == Level::L1)
            .collect()
    }
}

#[tokio::test]
async fn an_attribute_change_reads_nothing_and_leaves_the_content_indexed() {
    let f = fixture(FileSource::HostNamed).await;
    f.named_file("docs/a.txt", b"refund policy").await;
    f.index_everything().await;
    let before = f.file("docs/a.txt").await.unwrap();
    let segments = f.l1_segments().await;
    let (heads, reads) = (
        f.heads.load(Ordering::SeqCst),
        f.reads.load(Ordering::SeqCst),
    );

    let change = AttributeChange::new("docs/a.txt")
        .display_name("Refund policy")
        .attribute("label", one("legal"));
    assert_eq!(
        f.send(vec![FileChange::Attributes(change)]).await,
        [ChangeOutcome::Applied]
    );

    assert_eq!(f.heads.load(Ordering::SeqCst), heads);
    assert_eq!(f.reads.load(Ordering::SeqCst), reads);
    let after = f.file("docs/a.txt").await.unwrap();
    assert_eq!(after.content_version(), before.content_version());
    assert_eq!(after.etag(), before.etag());
    assert_eq!(after.status(), FileStatus::Ready);
    assert_eq!(after.display_name(), Some("Refund policy"));
    assert_eq!(after.attributes().get("label"), Some(&one("legal")));
    assert_eq!(f.l1_segments().await, segments);
    assert!(f.indexer.plan(10).await.unwrap().is_empty());
    assert_eq!(f.keys_for("refund", &legal(), 10).await, ["docs/a.txt"]);
}

#[tokio::test]
async fn an_attribute_change_replaces_what_was_there_and_checks_its_version() {
    let f = fixture(FileSource::HostNamed).await;
    f.send(vec![FileChange::Upsert(
        upsert("docs/a.txt", "e1")
            .display_name("Old name")
            .attribute("owner", one("sales")),
    )])
    .await;
    let at = f.file("docs/a.txt").await.unwrap().content_version();
    let wrong = ManifestVersion::new(9).unwrap();

    let outcomes = f
        .send(vec![
            FileChange::Attributes(AttributeChange::new("docs/a.txt").expected(wrong)),
            FileChange::Attributes(AttributeChange::new("docs/unknown.txt")),
            label("docs/a.txt", "legal"),
        ])
        .await;
    assert_eq!(
        outcomes,
        [
            ChangeOutcome::Stale,
            ChangeOutcome::NotFound,
            ChangeOutcome::Applied
        ]
    );
    let entry = f.file("docs/a.txt").await.unwrap();
    assert_eq!(entry.display_name(), None);
    assert_eq!(
        entry.attributes(),
        &Attributes::from([("label".to_owned(), one("legal"))])
    );
    assert_eq!(entry.content_version(), at);

    let version = f.version().await;
    let again = FileChange::Attributes(
        AttributeChange::new("docs/a.txt")
            .attribute("label", one("legal"))
            .expected(at),
    );
    assert_eq!(f.send(vec![again]).await, [ChangeOutcome::Unchanged]);
    assert_eq!(f.version().await, version);
}

#[tokio::test]
async fn an_attribute_change_is_for_host_named_files() {
    let f = fixture(FileSource::Listing).await;
    let err = f
        .indexer
        .apply(vec![label("docs/a.txt", "legal")])
        .await
        .unwrap_err();
    assert!(matches!(err, Error::WrongFileSource { .. }), "{err}");
}

#[tokio::test]
async fn an_attribute_change_rewrites_only_the_file_table_of_its_shard() {
    let f = fixture(FileSource::HostNamed).await;
    f.named_file("docs/a.txt", b"alpha").await;
    f.named_file("docs/z.txt", b"omega").await;
    f.index_everything().await;
    let before = f.index.head().await.unwrap().unwrap();
    f.send(vec![label("docs/a.txt", "legal")]).await;
    let after = f.index.head().await.unwrap().unwrap();
    let only = |manifest: &cosmosearch_core::Manifest, level: Level| -> Vec<SegmentEntry> {
        manifest
            .shards()
            .values()
            .flat_map(|shard| shard.segments().iter().cloned())
            .filter(|segment| segment.level() == level)
            .collect()
    };
    assert_eq!(only(&before, Level::L1), only(&after, Level::L1));
    assert_ne!(only(&before, Level::L0), only(&after, Level::L0));
}

#[tokio::test]
async fn a_job_planned_before_an_attribute_change_publishes_with_the_newest_attributes() {
    let f = fixture(FileSource::HostNamed).await;
    f.named_file("docs/a.txt", b"refund policy").await;
    let plan = f.indexer.plan(10).await.unwrap();
    let files = f.extract_all(&plan).await;
    f.send(vec![label("docs/a.txt", "legal")]).await;

    assert!(matches!(
        f.indexer.publish(&f.build(files).await).await.unwrap(),
        PublishOutcome::Published(_)
    ));
    let entry = f.file("docs/a.txt").await.unwrap();
    assert_eq!(entry.status(), FileStatus::Ready);
    assert_eq!(entry.attributes().get("label"), Some(&one("legal")));
    assert_eq!(f.keys_for("refund", &legal(), 10).await, ["docs/a.txt"]);
}

#[tokio::test]
async fn an_attribute_change_after_a_job_published_keeps_the_file_ready_and_findable() {
    let f = fixture(FileSource::HostNamed).await;
    f.put("docs/a.txt", b"refund policy").await;
    f.send(vec![labelled("docs/a.txt", one("hr"))]).await;
    f.index_everything().await;
    assert_eq!(
        f.keys_for("refund", &legal(), 10).await,
        Vec::<String>::new()
    );

    f.send(vec![label("docs/a.txt", "legal")]).await;
    let entry = f.file("docs/a.txt").await.unwrap();
    assert_eq!(entry.status(), FileStatus::Ready);
    assert_eq!(entry.attributes().get("label"), Some(&one("legal")));
    assert_eq!(f.keys_for("refund", &legal(), 10).await, ["docs/a.txt"]);
    let hr = Filter::new().equals("label", "hr");
    assert_eq!(f.keys_for("refund", &hr, 10).await, Vec::<String>::new());
}

#[tokio::test]
async fn a_filter_takes_any_of_a_list_and_matches_list_attributes() {
    let f = fixture(FileSource::HostNamed).await;
    let files: [(&str, &[u8], AttributeValue); 4] = [
        ("docs/a.txt", b"refund for legal", one("legal")),
        ("docs/b.txt", b"refund for finance", one("finance")),
        (
            "docs/c.txt",
            b"refund for drafts",
            AttributeValue::Many(vec!["draft".to_owned(), "finance".to_owned()]),
        ),
        ("docs/d.txt", b"refund for hr", one("hr")),
    ];
    for (key, body, value) in files {
        f.put(key, body).await;
        f.send(vec![labelled(key, value)]).await;
    }
    f.index_everything().await;

    let sorted = |mut keys: Vec<String>| {
        keys.sort();
        keys
    };
    let cases = [
        (
            Filter::new(),
            vec!["docs/a.txt", "docs/b.txt", "docs/c.txt", "docs/d.txt"],
        ),
        (
            Filter::new().any_of("label", &["legal", "finance"]),
            vec!["docs/a.txt", "docs/b.txt", "docs/c.txt"],
        ),
        (Filter::new().equals("label", "draft"), vec!["docs/c.txt"]),
        (Filter::new().any_of("label", &["ops"]), vec![]),
        (Filter::new().equals("owner", "legal"), vec![]),
    ];
    for (filter, expected) in cases {
        assert_eq!(
            sorted(f.keys_for("refund", &filter, 10).await),
            expected,
            "{filter:?}"
        );
    }
}

#[tokio::test]
async fn a_filtered_out_file_that_would_rank_first_never_takes_a_place() {
    let f = fixture(FileSource::HostNamed).await;
    let files: [(&str, &[u8], &str); 3] = [
        ("docs/best.txt", b"refund refund refund refund", "hr"),
        ("docs/good.txt", b"refund policy", "legal"),
        (
            "docs/fair.txt",
            b"refund window among many other words",
            "legal",
        ),
    ];
    for (key, body, value) in files {
        f.put(key, body).await;
        f.send(vec![labelled(key, one(value))]).await;
    }
    f.index_everything().await;
    assert_eq!(
        f.keys_for("refund", &Filter::new(), 1).await,
        ["docs/best.txt"]
    );

    assert_eq!(f.keys_for("refund", &legal(), 1).await, ["docs/good.txt"]);
    assert_eq!(
        f.keys_for("refund", &legal(), 2).await,
        ["docs/good.txt", "docs/fair.txt"]
    );
}
