use std::assert_matches;

use bytes::Bytes;
use cosmosearch_core::{
    ByteRange, IndexError, ManifestPointer, ManifestVersion, SegmentFileName, SegmentFiles,
    SegmentId,
};
use cosmosearch_store::{
    CommitOutcome, Error, Fault, IndexHandle, MemoryStore, ObjectKey, Op, ReadStore, WriteStore,
};

use crate::support::{commit_next, create, fail_every_attempt, fixture, head, next};

fn object(index: &IndexHandle, name: &str) -> ObjectKey {
    ObjectKey::new(format!("meta/{}/{name}", index.id())).unwrap()
}

fn slot(index: &IndexHandle, version: u64) -> ObjectKey {
    let version = ManifestVersion::new(version).unwrap();
    object(index, &format!("manifests/{}", version.file_name()))
}

async fn set_pointer(store: &MemoryStore, index: &IndexHandle, version: u64) {
    let pointer = ManifestPointer::new(ManifestVersion::new(version).unwrap());
    store
        .put_overwrite(&object(index, "MANIFEST"), pointer.encode().unwrap().into())
        .await
        .unwrap();
}

#[tokio::test]
async fn commits_advance_the_head_one_version_at_a_time() {
    let (metadata, _, _) = fixture();
    let index = create(&metadata).await;
    let second = commit_next(&index).await;
    let third = commit_next(&index).await;
    assert_eq!(second.version().get(), 2);
    assert_eq!(head(&index).await, third);
}

#[tokio::test]
async fn of_two_writers_from_one_version_the_second_conflicts() {
    let (metadata, _, _) = fixture();
    let index = create(&metadata).await;
    let parent = head(&index).await;
    let (winner, loser) = (next(&parent), next(&parent));

    let won = index.commit(&winner).await.unwrap();
    assert_eq!(won, CommitOutcome::Committed(winner.version()));
    assert_eq!(index.commit(&loser).await.unwrap(), CommitOutcome::Conflict);
    assert_eq!(head(&index).await, winner);
}

#[tokio::test]
async fn a_manifest_of_another_index_is_refused() {
    let (first, _, _) = fixture();
    let (second, _, _) = fixture();
    let (a, b) = (create(&first).await, create(&second).await);
    let foreign = next(&head(&b).await);
    let err = a.commit(&foreign).await.unwrap_err();
    assert_matches!(err, Error::WrongIndex { .. });
}
#[tokio::test]
async fn a_lost_response_is_recognised_as_committed() {
    let (metadata, faults, _) = fixture();
    let index = create(&metadata).await;
    let manifest = next(&head(&index).await);
    faults.inject(Op::PutCreate, "manifests", Fault::LostResponse);

    let outcome = index.commit(&manifest).await.unwrap();
    assert_eq!(outcome, CommitOutcome::Committed(manifest.version()));
    assert_eq!(head(&index).await, manifest);
}

#[tokio::test]
async fn a_timed_out_create_is_retried_with_the_same_body() {
    let (metadata, faults, _) = fixture();
    let index = create(&metadata).await;
    let manifest = next(&head(&index).await);
    faults.inject(Op::PutCreate, "manifests", Fault::Timeout);

    let outcome = index.commit(&manifest).await.unwrap();
    assert_eq!(outcome, CommitOutcome::Committed(manifest.version()));
}

#[tokio::test]
async fn a_conflict_with_nothing_visible_is_not_taken_at_its_word() {
    let (metadata, faults, _) = fixture();
    let index = create(&metadata).await;
    let manifest = next(&head(&index).await);
    faults.inject(Op::PutCreate, "manifests", Fault::InvisibleConflict);

    let outcome = index.commit(&manifest).await.unwrap();
    assert_eq!(outcome, CommitOutcome::Committed(manifest.version()));
}

#[tokio::test]
async fn when_nothing_can_be_established_the_outcome_is_unknown() {
    let (metadata, faults, _) = fixture();
    let index = create(&metadata).await;
    let manifest = next(&head(&index).await);
    fail_every_attempt(&faults, "manifests", Fault::Timeout);
    assert_eq!(
        index.commit(&manifest).await.unwrap(),
        CommitOutcome::Unknown
    );
}
#[tokio::test]
async fn a_late_landing_commit_becomes_the_head_if_its_slot_is_free() {
    let (metadata, faults, _) = fixture();
    let index = create(&metadata).await;
    let manifest = next(&head(&index).await);
    fail_every_attempt(&faults, "manifests", Fault::Delayed);
    assert_eq!(
        index.commit(&manifest).await.unwrap(),
        CommitOutcome::Unknown
    );
    assert_eq!(head(&index).await.version().get(), 1);

    faults.release_delayed().await.unwrap();
    assert_eq!(head(&index).await, manifest);
}

#[tokio::test]
async fn a_late_landing_commit_loses_to_a_writer_that_got_there_first() {
    let (metadata, faults, _) = fixture();
    let index = create(&metadata).await;
    let parent = head(&index).await;
    let (late, first) = (next(&parent), next(&parent));
    fail_every_attempt(&faults, "manifests", Fault::Delayed);
    assert_eq!(index.commit(&late).await.unwrap(), CommitOutcome::Unknown);
    index.commit(&first).await.unwrap();

    faults.release_delayed().await.unwrap();
    assert_eq!(head(&index).await, first);
}

#[tokio::test]
async fn a_crash_at_each_step_of_a_commit_recovers() {
    for writes in 0..2 {
        let (metadata, faults, _) = fixture();
        let index = create(&metadata).await;
        let manifest = next(&head(&index).await);
        faults.stop_after_writes(writes);
        let outcome = index.commit(&manifest).await;
        faults.resume_writes();

        let expected = u64::try_from(writes).unwrap() + 1;
        assert_eq!(head(&index).await.version().get(), expected, "{outcome:?}");
        commit_next(&index).await;
    }
}
#[tokio::test]
async fn a_delayed_create_into_an_expired_slot_fails() {
    let (metadata, faults, inner) = fixture();
    let index = create(&metadata).await;
    let second = next(&head(&index).await);
    fail_every_attempt(&faults, "manifests", Fault::Delayed);
    assert_eq!(index.commit(&second).await.unwrap(), CommitOutcome::Unknown);
    let winner = next(&head(&index).await);
    index.commit(&winner).await.unwrap();
    let third = commit_next(&index).await;

    inner
        .put_overwrite(&slot(&index, 2), winner.stub().encode().unwrap().into())
        .await
        .unwrap();
    faults.release_delayed().await.unwrap();

    assert_eq!(head(&index).await, third);
    let held = inner.get(&slot(&index, 2)).await.unwrap();
    assert_eq!(held, Bytes::from(winner.stub().encode().unwrap()));
}

#[tokio::test]
async fn a_stale_or_missing_pointer_still_finds_the_head() {
    let (metadata, _, inner) = fixture();
    let index = create(&metadata).await;
    commit_next(&index).await;
    let newest = commit_next(&index).await;

    set_pointer(&inner, &index, 1).await;
    assert_eq!(head(&index).await, newest, "regressed");

    inner.delete(&object(&index, "MANIFEST")).await.unwrap();
    assert_eq!(head(&index).await, newest, "missing");

    inner
        .put_overwrite(&object(&index, "MANIFEST"), Bytes::from_static(b"garbage"))
        .await
        .unwrap();
    assert_eq!(head(&index).await, newest, "unreadable");

    set_pointer(&inner, &index, 999).await;
    assert_eq!(head(&index).await, newest, "ahead of every manifest");
}

#[tokio::test]
async fn a_pointer_far_behind_falls_back_to_listing() {
    let (metadata, _, inner) = fixture();
    let index = create(&metadata).await;
    let mut newest = head(&index).await;
    for _ in 0..40 {
        newest = commit_next(&index).await;
    }
    set_pointer(&inner, &index, 1).await;
    assert_eq!(head(&index).await, newest);
}

#[tokio::test]
async fn a_pointer_at_a_stub_rolls_forward() {
    let (metadata, _, inner) = fixture();
    let index = create(&metadata).await;
    let second = commit_next(&index).await;
    let third = commit_next(&index).await;
    inner
        .put_overwrite(&slot(&index, 2), second.stub().encode().unwrap().into())
        .await
        .unwrap();
    set_pointer(&inner, &index, 2).await;
    assert_eq!(head(&index).await, third);
}
#[tokio::test]
async fn late_writes_from_an_old_index_cannot_reach_a_new_one() {
    let (metadata, faults, inner) = fixture();
    let old = create(&metadata).await;
    let late_manifest = next(&head(&old).await);
    fail_every_attempt(&faults, "manifests", Fault::Delayed);
    assert_eq!(
        old.commit(&late_manifest).await.unwrap(),
        CommitOutcome::Unknown
    );
    faults.inject(Op::PutOverwrite, "MANIFEST", Fault::Delayed);
    let late_pointer = ManifestPointer::new(ManifestVersion::new(9).unwrap());
    faults
        .put_overwrite(
            &object(&old, "MANIFEST"),
            late_pointer.encode().unwrap().into(),
        )
        .await
        .unwrap_err();
    let segment = object(&old, "shards/0/lex-old.bin");
    faults.inject(Op::PutCreate, "lex-old", Fault::Delayed);
    faults
        .put_create(&segment, Bytes::from_static(b"old"))
        .await
        .unwrap_err();

    metadata.destroy().await.unwrap();
    let new = create(&metadata).await;
    assert_ne!(old.id(), new.id());

    let parent = head(&new).await;
    let (first, second) = (next(&parent), next(&parent));
    let (a, b) = tokio::join!(new.commit(&first), new.commit(&second));
    faults.release_delayed().await.unwrap();

    let outcomes = [a.unwrap(), b.unwrap()];
    let wins = outcomes
        .iter()
        .filter(|o| matches!(o, CommitOutcome::Committed(_)))
        .count();
    assert_eq!(wins, 1, "{outcomes:?}");
    let winner = if outcomes[0] == CommitOutcome::Conflict {
        &second
    } else {
        &first
    };
    assert_eq!(&head(&new).await, winner);
    inner.head(&segment).await.unwrap();
    assert_eq!(head(&new).await.index_id(), new.id());
}

#[tokio::test]
async fn a_commit_cannot_skip_a_version() {
    let (metadata, _, _) = fixture();
    let index = create(&metadata).await;
    let never_landed = next(&head(&index).await);
    let after_it = next(&never_landed);
    assert_matches!(
        index.commit(&after_it).await,
        Err(Error::ParentMissing { .. })
    );
    assert_eq!(head(&index).await.version().get(), 1);
}

#[tokio::test]
async fn a_refusal_after_an_unknown_write_stays_unknown() {
    let (metadata, faults, _) = fixture();
    let index = create(&metadata).await;
    let manifest = next(&head(&index).await);
    faults.inject(Op::PutCreate, "manifests", Fault::Delayed);
    faults.inject(Op::PutCreate, "manifests", Fault::Refuse);

    assert_eq!(
        index.commit(&manifest).await.unwrap(),
        CommitOutcome::Unknown
    );

    faults.release_delayed().await.unwrap();
    assert_eq!(head(&index).await, manifest);
}

#[tokio::test]
async fn a_refusal_before_anything_was_sent_stays_an_error() {
    let (metadata, faults, _) = fixture();
    let index = create(&metadata).await;
    let manifest = next(&head(&index).await);
    faults.inject(Op::PutCreate, "manifests", Fault::Refuse);
    assert_matches!(index.commit(&manifest).await, Err(Error::Refused { .. }));
}

#[tokio::test]
async fn a_segment_file_that_is_gone_reads_as_missing() {
    let (metadata, _, inner) = fixture();
    let index = create(&metadata).await;
    let segment = SegmentId::new("s1").unwrap();
    let name = SegmentFileName::new("lex-s1.bin").unwrap();
    index
        .write_segment_file(&segment, &name, Bytes::from_static(b"postings"))
        .await
        .unwrap();
    let range = ByteRange::new(0, 4).unwrap();
    assert_eq!(
        index.read_range(&segment, &name, range).await.unwrap(),
        b"post"
    );

    inner
        .delete(&object(&index, "segments/s1/lex-s1.bin"))
        .await
        .unwrap();
    let err = index.read_range(&segment, &name, range).await.unwrap_err();
    assert_matches!(err, IndexError::MissingSegment { segment: found } if found == segment);
}

#[tokio::test]
async fn a_plug_in_keeps_its_objects_in_its_scope_under_the_index_id() {
    let (metadata, _, inner) = fixture();
    let index = create(&metadata).await;
    let scope = index.scope("vectors").unwrap();
    let pack = ObjectKey::new("model/a.pack").unwrap();
    scope
        .put_create(&pack, Bytes::from_static(b"cached"))
        .await
        .unwrap();

    let stored = object(&index, "vectors/model/a.pack");
    assert_eq!(
        inner.get(&stored).await.unwrap(),
        Bytes::from_static(b"cached")
    );
    let listed = scope.list(None).await.unwrap();
    let keys: Vec<_> = listed.iter().map(|object| object.key().as_str()).collect();
    assert_eq!(keys, ["model/a.pack"]);
    let range = ByteRange::new(1, 3).unwrap();
    assert_eq!(
        scope.get_range(&pack, range).await.unwrap(),
        Bytes::from_static(b"ac")
    );

    metadata.destroy().await.unwrap();
    assert_matches!(inner.get(&stored).await, Err(Error::NotFound { .. }));
}

#[tokio::test]
async fn a_scope_lists_and_deletes_only_its_own_objects() {
    let (metadata, _, _) = fixture();
    let index = create(&metadata).await;
    let (vectors, other) = (
        index.scope("vectors").unwrap(),
        index.scope("vectors-2").unwrap(),
    );
    for (scope, name) in [(&vectors, "m1/a"), (&vectors, "m2/b"), (&other, "m1/c")] {
        let key = ObjectKey::new(name).unwrap();
        scope
            .put_create(&key, Bytes::from_static(b"x"))
            .await
            .unwrap();
    }

    let m1 = ObjectKey::new("m1").unwrap();
    let listed = vectors.list(Some(&m1)).await.unwrap();
    let keys: Vec<_> = listed.iter().map(|object| object.key().as_str()).collect();
    assert_eq!(keys, ["m1/a"]);
    vectors
        .delete(&ObjectKey::new("m1/c").unwrap())
        .await
        .unwrap();
    assert_eq!(other.list(None).await.unwrap().len(), 1);
}

#[tokio::test]
async fn a_scope_name_must_be_plain_and_not_one_the_index_uses() {
    let (metadata, _, _) = fixture();
    let index = create(&metadata).await;
    for name in [
        "",
        "manifests",
        "segments",
        "MANIFEST",
        "a/b",
        "Vectors",
        "a.b",
    ] {
        assert_matches!(index.scope(name), Err(Error::InvalidKey { .. }), "{name}");
    }
    index.scope("vectors-2").unwrap();
}
