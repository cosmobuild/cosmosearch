use cosmosearch_chunk::Extractors;
use cosmosearch_core::{ExtractFailure, ExtractLimits, Level, LimitKind};
use cosmosearch_format::FileReason;
use cosmosearch_indexer::{Batch, ExtractOutcome, ExtractedFile, PublishOutcome};

use super::*;

const NO_TEXT: FileReason = FileReason::Extraction(ExtractFailure::NoText);
const TOO_MANY_CHARS: FileReason =
    FileReason::Extraction(ExtractFailure::LimitExceeded(LimitKind::Chars));

fn retry(key: &str) -> FileChange {
    FileChange::Retry {
        key: key.to_owned(),
        expected: None,
    }
}

impl Fixture {
    async fn extract_with(&self, limits: &ExtractLimits) -> Vec<ExtractedFile> {
        let extractors = Extractors::builtin().unwrap();
        let plan = self.indexer.plan(1_000).await.unwrap();
        let mut found = Vec::new();
        for job in plan.batches().iter().flat_map(Batch::jobs) {
            found.push(
                self.indexer
                    .extract(job.clone(), &extractors, limits)
                    .await
                    .unwrap(),
            );
        }
        found
    }

    async fn record(&self, files: &[ExtractedFile]) -> Vec<ChangeOutcome> {
        self.indexer
            .record(files)
            .await
            .unwrap()
            .outcomes()
            .to_vec()
    }

    async fn state(&self, key: &str) -> (FileStatus, Option<FileReason>, Option<Level>) {
        let file = self.indexer.file(key).await.unwrap().unwrap();
        let entry = file.entry();
        (entry.status(), entry.reason(), file.level())
    }

    async fn planned_keys(&self) -> Vec<String> {
        let plan = self.indexer.plan(1_000).await.unwrap();
        plan.batches()
            .iter()
            .flat_map(Batch::jobs)
            .map(|job| job.key().to_owned())
            .collect()
    }
}

#[tokio::test]
async fn a_bad_file_never_blocks_the_others() {
    let f = fixture(FileSource::HostNamed).await;
    f.named_file("docs/blank.txt", b"   ").await;
    f.named_file("docs/good.txt", b"refund policy").await;
    f.named_file("docs/long.txt", b"a file with too many characters")
        .await;
    let tight = ExtractLimits::builder().chars(20).build().unwrap();
    let files = f.extract_with(&tight).await;

    assert_eq!(
        f.record(&files).await,
        [
            ChangeOutcome::Applied,
            ChangeOutcome::Unchanged,
            ChangeOutcome::Applied
        ]
    );
    let built = f.build(files).await;
    assert!(matches!(
        f.indexer.publish(&built).await.unwrap(),
        PublishOutcome::Published(_)
    ));

    assert_eq!(
        f.state("docs/blank.txt").await,
        (FileStatus::Unsupported, Some(NO_TEXT), None)
    );
    assert_eq!(
        f.state("docs/long.txt").await,
        (FileStatus::Failed, Some(TOO_MANY_CHARS), None)
    );
    assert_eq!(
        f.state("docs/good.txt").await,
        (FileStatus::Ready, None, Some(Level::L1))
    );
    assert_eq!(f.live_keys_for("refund").await, ["docs/good.txt"]);
    assert_eq!(f.planned_keys().await, Vec::<String>::new());
    assert_eq!(f.indexer.file("docs/unknown.txt").await.unwrap(), None);
}

#[tokio::test]
async fn a_type_no_extractor_reads_is_unsupported_without_reading_it() {
    let f = fixture(FileSource::HostNamed).await;
    f.named_file("docs/scan.tiff", b"image bytes").await;
    let reads = f.reads.load(Ordering::SeqCst);
    let files = f.extract_with(&ExtractLimits::default()).await;
    assert_eq!(f.reads.load(Ordering::SeqCst), reads);
    assert_eq!(f.record(&files).await, [ChangeOutcome::Applied]);
    assert_eq!(
        f.state("docs/scan.tiff").await,
        (
            FileStatus::Unsupported,
            Some(FileReason::Extraction(ExtractFailure::NoExtractor)),
            None
        )
    );

    f.send(vec![retry("docs/scan.tiff")]).await;
    assert_eq!(f.planned_keys().await, ["docs/scan.tiff"]);
}

#[tokio::test]
async fn a_file_over_the_size_limit_fails_without_being_read() {
    let f = fixture(FileSource::HostNamed).await;
    f.named_file("docs/large.txt", b"twelve bytes").await;
    let limits = ExtractLimits::builder().file_bytes(11).build().unwrap();
    let reads = f.reads.load(Ordering::SeqCst);
    let files = f.extract_with(&limits).await;
    assert_eq!(f.reads.load(Ordering::SeqCst), reads);
    assert_eq!(f.record(&files).await, [ChangeOutcome::Applied]);
    assert_eq!(
        f.state("docs/large.txt").await,
        (
            FileStatus::Failed,
            Some(FileReason::Extraction(ExtractFailure::LimitExceeded(
                LimitKind::FileBytes
            ))),
            None
        )
    );
}

#[tokio::test]
async fn a_failed_file_can_be_retried() {
    let f = fixture(FileSource::HostNamed).await;
    f.named_file("docs/long.txt", b"a file with too many characters")
        .await;
    let tight = ExtractLimits::builder().chars(20).build().unwrap();
    let files = f.extract_with(&tight).await;
    f.record(&files).await;
    let failed_at = f.file("docs/long.txt").await.unwrap().content_version();

    assert_eq!(
        f.send(vec![retry("docs/long.txt")]).await,
        [ChangeOutcome::Applied]
    );
    let again = f.file("docs/long.txt").await.unwrap();
    assert!(again.content_version() > failed_at);
    assert_eq!(
        f.state("docs/long.txt").await,
        (FileStatus::Pending, None, None)
    );

    assert_eq!(f.record(&files).await, [ChangeOutcome::Stale]);
    assert!(matches!(
        f.index_everything().await,
        PublishOutcome::Published(_)
    ));
    assert_eq!(f.status("docs/long.txt").await, FileStatus::Ready);
    assert_eq!(f.live_keys_for("characters").await, ["docs/long.txt"]);
}

#[tokio::test]
async fn a_read_error_leaves_the_file_pending_and_can_be_retried() {
    let f = fixture(FileSource::HostNamed).await;
    f.named_file("docs/a.txt", b"refund").await;
    let version = f.version().await;
    let plan = f.indexer.plan(10).await.unwrap();
    let job = plan.batches()[0].jobs()[0].clone();
    let extractors = Extractors::builtin().unwrap();
    let limits = ExtractLimits::default();

    f.reads_fail.store(true, Ordering::SeqCst);
    let err = f
        .indexer
        .extract(job.clone(), &extractors, &limits)
        .await
        .unwrap_err();
    assert!(err.is_retryable(), "{err}");
    assert_eq!(
        f.state("docs/a.txt").await,
        (FileStatus::Pending, None, None)
    );
    assert_eq!(f.version().await, version);

    f.reads_fail.store(false, Ordering::SeqCst);
    let file = f.indexer.extract(job, &extractors, &limits).await.unwrap();
    assert!(matches!(file.outcome(), ExtractOutcome::Chunks { .. }));
}

#[tokio::test]
async fn only_passing_trouble_is_retryable() {
    let f = fixture(FileSource::Listing).await;
    let err = f
        .indexer
        .apply(vec![add("docs/a.txt", "v1")])
        .await
        .unwrap_err();
    assert!(!err.is_retryable(), "{err}");
    assert!(Error::CommitUnknown.is_retryable());
    assert!(Error::TooManyConflicts { attempts: 8 }.is_retryable());
}

#[tokio::test]
async fn a_recorded_failure_is_checked_against_its_version_and_repeats_safely() {
    let f = fixture(FileSource::HostNamed).await;
    f.named_file("docs/a.txt", b"   ").await;
    let files = f.extract_with(&ExtractLimits::default()).await;

    assert_eq!(f.record(&files).await, [ChangeOutcome::Applied]);
    let version = f.version().await;
    assert_eq!(f.record(&files).await, [ChangeOutcome::Unchanged]);
    assert_eq!(f.version().await, version);

    f.named_file("docs/a.txt", b"now it has text").await;
    assert_eq!(f.record(&files).await, [ChangeOutcome::Stale]);
    assert_eq!(
        f.state("docs/a.txt").await,
        (FileStatus::Pending, None, None)
    );

    f.send(vec![remove("docs/a.txt")]).await;
    assert_eq!(f.record(&files).await, [ChangeOutcome::Stale]);
    assert_eq!(f.file("docs/a.txt").await, None);
}

#[tokio::test]
async fn a_host_records_a_failure_for_the_one_file_that_caused_it() {
    let f = fixture(FileSource::HostNamed).await;
    f.named_file("docs/a.txt", b"refund").await;
    f.named_file("docs/b.txt", b"refund").await;
    let plan = f.indexer.plan(10).await.unwrap();
    let job = plan.batches()[0].jobs()[1].clone();

    let crashed = ExtractedFile::failed(job, ExtractFailure::Panicked);
    assert_eq!(f.record(&[crashed]).await, [ChangeOutcome::Applied]);

    assert_eq!(
        f.state("docs/b.txt").await,
        (
            FileStatus::Failed,
            Some(FileReason::Extraction(ExtractFailure::Panicked)),
            None
        )
    );
    assert_eq!(f.planned_keys().await, ["docs/a.txt"]);
}

#[tokio::test]
async fn a_moved_source_waits_with_that_reason_until_a_newer_change() {
    let f = fixture(FileSource::HostNamed).await;
    f.named_file("docs/a.txt", b"first").await;
    let plan = f.indexer.plan(10).await.unwrap();
    f.put("docs/a.txt", b"second").await;
    let files = f.extract_all(&plan).await;
    assert_eq!(files[0].outcome(), &ExtractOutcome::SourceMoved);

    assert_eq!(f.record(&files).await, [ChangeOutcome::Applied]);
    assert_eq!(
        f.state("docs/a.txt").await,
        (FileStatus::Pending, Some(FileReason::SourceMoved), None)
    );
    assert_eq!(f.planned_keys().await, Vec::<String>::new());

    f.send(vec![FileChange::Upsert(FileUpsert::new("docs/a.txt"))])
        .await;
    assert_eq!(
        f.state("docs/a.txt").await,
        (FileStatus::Pending, None, None)
    );
    assert_eq!(f.planned_keys().await, ["docs/a.txt"]);
}

#[tokio::test]
async fn a_retry_takes_a_ready_file_out_of_results_until_it_is_indexed_again() {
    let f = fixture(FileSource::HostNamed).await;
    f.named_file("docs/a.txt", b"refund").await;
    f.index_everything().await;
    assert_eq!(f.live_keys_for("refund").await, ["docs/a.txt"]);

    f.send(vec![retry("docs/a.txt")]).await;
    assert_eq!(f.live_keys_for("refund").await, Vec::<String>::new());

    f.index_everything().await;
    assert_eq!(f.live_keys_for("refund").await, ["docs/a.txt"]);
}

#[tokio::test]
async fn a_retry_needs_a_known_file_at_the_expected_version() {
    let f = fixture(FileSource::HostNamed).await;
    f.send(vec![add("docs/a.txt", "v1")]).await;
    let version = f.version().await;
    let stale = FileChange::Retry {
        key: "docs/a.txt".to_owned(),
        expected: Some(ManifestVersion::new(1).unwrap()),
    };

    assert_eq!(
        f.send(vec![retry("docs/unknown.txt"), stale]).await,
        [ChangeOutcome::NotFound, ChangeOutcome::Stale]
    );
    assert_eq!(f.version().await, version);
}

#[tokio::test]
async fn a_listed_index_takes_retries_from_the_host() {
    let f = fixture(FileSource::Listing).await;
    f.put("docs/a.txt", b"refund").await;
    f.indexer.sync_listing().await.unwrap();
    let before = f.file("docs/a.txt").await.unwrap().content_version();

    assert_eq!(
        f.send(vec![retry("docs/a.txt")]).await,
        [ChangeOutcome::Applied]
    );
    assert!(f.file("docs/a.txt").await.unwrap().content_version() > before);
    assert_eq!(f.lists.load(Ordering::SeqCst), 1);
}
