use cosmosearch_core::{
    FileSource, IndexOptions, Locator, Query, SegmentEntry, ShardEntry, SourceRef,
};
use cosmosearch_query::{Error, Reader, ReaderOptions, SearchRequest, TableCache};
use cosmosearch_store::{ObjectKey, WriteStore};

use super::{Fixture, plugins};

const PAGES: &[u8] =
    b"Refund terms apply.\x0cLate refunds need approval.\n\nApproval takes a week.";

async fn stored() -> Fixture {
    let f = Fixture::with_options(IndexOptions::new().stored_text(true)).await;
    f.index(&[
        ("docs/a.pages", PAGES),
        ("docs/b.txt", b"refund rules in plain text"),
    ])
    .await;
    f
}

async fn page_two(reader: &Reader) -> SourceRef {
    let results = reader
        .search(&SearchRequest::new(Query::new("approval week")))
        .await
        .unwrap();
    let hit = results
        .hits()
        .iter()
        .find(|hit| hit.source().locator().to_string() == "page=2")
        .unwrap();
    hit.source().clone()
}

async fn segment_files(f: &Fixture) -> Vec<String> {
    let head = f.handle.head().await.unwrap().unwrap();
    head.shards()
        .values()
        .flat_map(ShardEntry::segments)
        .flat_map(SegmentEntry::files)
        .map(|file| file.name().as_str().to_owned())
        .collect()
}

#[tokio::test]
async fn an_index_stores_text_only_when_created_to() {
    let plain = Fixture::new().await;
    plain.index(&[("docs/a.pages", PAGES)]).await;
    assert!(
        !segment_files(&plain)
            .await
            .iter()
            .any(|name| name.starts_with("text-"))
    );
    let f = stored().await;
    assert!(
        f.handle
            .head()
            .await
            .unwrap()
            .unwrap()
            .options()
            .has_stored_text()
    );
    assert!(
        segment_files(&f)
            .await
            .iter()
            .any(|name| name.starts_with("text-"))
    );
}

#[tokio::test]
async fn a_page_is_read_from_stored_text_without_reading_its_file() {
    let f = stored().await;
    let plain = Fixture::new().await;
    plain.index(&[("docs/a.pages", PAGES)]).await;
    let reference = page_two(&f.reader).await;
    let reading = f.reader.read(&reference).await.unwrap();
    let expected = plain
        .reader
        .read(&page_two(&plain.reader).await)
        .await
        .unwrap();
    assert_eq!(
        reading.chunks().first().unwrap().text(),
        expected.chunks().first().unwrap().text()
    );
    let again = f.reader.read(&reference).await.unwrap();
    assert_eq!(
        (again.usage().object_reads(), again.usage().bytes()),
        (1, 0),
        "one check of the file's version, and no bytes of the file"
    );
    let page = f.reader.chunks("docs/a.pages", 0, 5).await.unwrap();
    let locators: Vec<String> = page
        .chunks()
        .iter()
        .map(|chunk| chunk.source().locator().to_string())
        .collect();
    assert_eq!(locators, ["page=1", "page=2"]);
}

#[tokio::test]
async fn a_changed_or_deleted_file_is_refused_as_stale() {
    let f = stored().await;
    let reference = page_two(&f.reader).await;
    f.put("docs/a.pages", b"Something else entirely.").await;
    let err = f.reader.read(&reference).await.unwrap_err();
    assert!(matches!(err, Error::StaleRef { .. }), "{err}");
    f.store
        .delete(&ObjectKey::new("docs/a.pages").unwrap())
        .await
        .unwrap();
    let err = f.reader.read(&reference).await.unwrap_err();
    assert!(matches!(err, Error::StaleRef { .. }), "{err}");
}

#[tokio::test]
async fn a_snippet_shows_the_hits_own_chunk() {
    let f = stored().await;
    let reader = Reader::open(
        f.data.clone(),
        f.metadata.clone(),
        FileSource::HostNamed,
        plugins(),
        ReaderOptions::new().snippets(10),
    )
    .await
    .unwrap();
    let results = reader
        .search(&SearchRequest::new(Query::new("week")))
        .await
        .unwrap();
    let hit = results.hits().first().unwrap();
    assert_eq!(hit.source().locator(), Locator::Page(2.try_into().unwrap()));
    assert_eq!(hit.snippet().unwrap().text(), "Approval takes a week.");
    let text = results
        .hits()
        .iter()
        .find(|hit| hit.source().location().key() == "docs/b.txt");
    if let Some(text) = text {
        assert_eq!(text.snippet().unwrap().text(), "refund rules in plain text");
    }
}

#[tokio::test]
async fn a_page_of_several_chunks_reads_its_text_group_once_when_nothing_is_cached() {
    let f = stored().await;
    let reader = Reader::open(
        f.data.clone(),
        f.metadata.clone(),
        FileSource::HostNamed,
        plugins(),
        ReaderOptions::new().tables(TableCache::new(1)),
    )
    .await
    .unwrap();
    let two = page_two(&reader).await;
    let one = f
        .reader
        .chunks("docs/a.pages", 0, 1)
        .await
        .unwrap()
        .chunks()
        .first()
        .unwrap()
        .source()
        .clone();
    reader.read(&two).await.unwrap();
    reader.read(&one).await.unwrap();
    let two = reader.read(&two).await.unwrap();
    let one = reader.read(&one).await.unwrap();
    assert_eq!(
        two.usage().object_reads(),
        one.usage().object_reads(),
        "a page of two chunks costs what a page of one does"
    );
}

#[tokio::test]
async fn a_stored_text_index_with_only_plain_text_keeps_no_text_and_reads_the_file() {
    let f = Fixture::with_options(IndexOptions::new().stored_text(true)).await;
    f.index(&[("docs/b.txt", b"refund rules in plain text")])
        .await;
    assert!(
        !segment_files(&f)
            .await
            .iter()
            .any(|name| name.starts_with("text-"))
    );
    let page = f.reader.chunks("docs/b.txt", 0, 5).await.unwrap();
    assert_eq!(
        page.chunks().first().unwrap().text(),
        "refund rules in plain text"
    );
}
