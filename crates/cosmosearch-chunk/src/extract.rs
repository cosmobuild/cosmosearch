use std::panic::{AssertUnwindSafe, catch_unwind};

use cosmosearch_core::{
    Chunk, ETag, ExtractFailure, ExtractLimits, Extraction, Extractor, LimitKind, Location,
    Locator, Passage,
};

use crate::Result;

/// The chunks of one file, and how many parts of it the extractor could not read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileChunks {
    chunks: Vec<Chunk>,
    unread_parts: u32,
}

impl FileChunks {
    /// The chunks, in reading order.
    pub fn chunks(&self) -> &[Chunk] {
        &self.chunks
    }

    /// The chunks, in reading order.
    pub fn into_chunks(self) -> Vec<Chunk> {
        self.chunks
    }

    /// How many parts of the file, such as PDF pages that hold only an image, could not be
    /// read.
    pub fn unread_parts(&self) -> u32 {
        self.unread_parts
    }
}

/// Extracts one file into chunks, so a limit hit or a panic fails only this file.
///
/// Blocks for as long as the extractor runs. It cannot survive an abort, a stack overflow or
/// running out of memory, so a host indexing untrusted files runs it in a separate process.
///
/// # Errors
///
/// Returns [`Error::File`](crate::Error::File) if the file cannot be extracted, including when
/// it has no text or is larger than the limit on file bytes, and
/// [`Error::Core`](crate::Error::Core) if `location` names no file.
pub fn extract_chunks(
    extractor: &dyn Extractor,
    location: &Location,
    etag: &ETag,
    file: &[u8],
    limits: &ExtractLimits,
) -> Result<FileChunks> {
    if u64::try_from(file.len()).map_or(true, |len| len > limits.file_bytes()) {
        return Err(ExtractFailure::LimitExceeded(LimitKind::FileBytes).into());
    }
    #[expect(
        clippy::disallowed_methods,
        reason = "an extractor wraps a third-party parser, and its panic fails only this file"
    )]
    let extraction: Extraction = catch_unwind(AssertUnwindSafe(|| extractor.extract(file, limits)))
        .map_err(|_payload| ExtractFailure::Panicked)??;
    let unread_parts = extraction.unread_parts();

    let mut chunks = Vec::new();
    let mut chars = 0usize;
    for passage in extraction.into_passages() {
        if passage.is_blank() {
            continue;
        }
        check_locator(&passage, file)?;
        chars = chars.saturating_add(passage.text().chars().count());
        if chars > limits.chars() {
            return Err(ExtractFailure::LimitExceeded(LimitKind::Chars).into());
        }
        if chunks.len() >= limits.chunks() {
            return Err(ExtractFailure::LimitExceeded(LimitKind::Chunks).into());
        }
        chunks.push(passage.into_chunk(location, etag)?);
    }
    if chunks.is_empty() {
        let failure = if unread_parts > 0 {
            ExtractFailure::Unreadable
        } else {
            ExtractFailure::NoText
        };
        return Err(failure.into());
    }
    Ok(FileChunks {
        chunks,
        unread_parts,
    })
}

fn check_locator(passage: &Passage, file: &[u8]) -> Result<(), ExtractFailure> {
    let Locator::Bytes(range) = passage.locator() else {
        return Ok(());
    };
    match u64::try_from(file.len()) {
        Ok(len) if range.end() <= len => Ok(()),
        Ok(_) | Err(_) => Err(ExtractFailure::BadLocator),
    }
}

#[cfg(test)]
mod tests {
    use cosmosearch_core::{ByteRange, ExtractorProfile};

    use super::*;
    use crate::Error;

    #[derive(Debug, Clone, Copy)]
    enum Behaviour {
        Passages(&'static [&'static str]),
        Scanned(u32),
        OutOfBounds,
        Fail(ExtractFailure),
        Panic,
    }

    #[derive(Debug)]
    struct Fake(ExtractorProfile, Behaviour);

    fn fake(behaviour: Behaviour) -> Fake {
        Fake(ExtractorProfile::new("fake-1").unwrap(), behaviour)
    }

    fn at(start: u64, end: u64, text: &str) -> Passage {
        let range = ByteRange::new(start, end).unwrap();
        Passage::new(Locator::Bytes(range), vec!["Heading".to_owned()], text)
    }

    impl Extractor for Fake {
        fn profile(&self) -> &ExtractorProfile {
            &self.0
        }

        fn extensions(&self) -> &[&str] {
            &["txt"]
        }

        fn extract(&self, _: &[u8], _: &ExtractLimits) -> Result<Extraction, ExtractFailure> {
            match self.1 {
                Behaviour::Passages(texts) => Ok(Extraction::new(
                    texts.iter().map(|text| at(0, 1, text)).collect(),
                )),
                Behaviour::Scanned(pages) => {
                    Ok(Extraction::new(Vec::new()).with_unread_parts(pages))
                }
                Behaviour::OutOfBounds => Ok(Extraction::new(vec![at(0, 1_000, "text")])),
                Behaviour::Fail(failure) => Err(failure),
                Behaviour::Panic => panic!("parser exploded on secret text"),
            }
        }
    }

    fn location() -> Location {
        "s3://example-bucket/a.txt".parse().unwrap()
    }

    fn etag() -> ETag {
        ETag::new("9f3a").unwrap()
    }

    fn run(behaviour: Behaviour, limits: &ExtractLimits) -> Result<Vec<Chunk>> {
        extract_chunks(
            &fake(behaviour),
            &location(),
            &etag(),
            b"file bytes",
            limits,
        )
        .map(FileChunks::into_chunks)
    }

    fn failure(behaviour: Behaviour, limits: &ExtractLimits) -> ExtractFailure {
        match run(behaviour, limits).unwrap_err() {
            Error::File(failure) => failure,
            Error::Core(err) => panic!("expected a file failure, got {err}"),
        }
    }

    #[test]
    fn keeps_passages_in_order_with_refs_and_headings() {
        let texts = Behaviour::Passages(&["first", "second"]);
        let chunks = run(texts, &ExtractLimits::default()).unwrap();
        let found: Vec<_> = chunks.iter().map(Chunk::text).collect();
        assert_eq!(found, ["first", "second"]);
        assert_eq!(
            chunks[0].source().to_string(),
            "s3://example-bucket/a.txt#bytes=0-1@etag=9f3a"
        );
        assert_eq!(chunks[0].heading_path(), ["Heading"]);
    }

    #[test]
    fn drops_blank_passages() {
        let chunks = run(
            Behaviour::Passages(&[" ", "kept", "\n"]),
            &ExtractLimits::default(),
        );
        assert_eq!(chunks.unwrap().len(), 1);
    }

    #[test]
    fn each_way_of_failing_is_reported() {
        let default = ExtractLimits::default();
        let two_chars = ExtractLimits::builder().chars(2).build().unwrap();
        let one_chunk = ExtractLimits::builder().chunks(1).build().unwrap();
        let chars = ExtractFailure::LimitExceeded(LimitKind::Chars);
        let chunks = ExtractFailure::LimitExceeded(LimitKind::Chunks);
        let pages = ExtractFailure::LimitExceeded(LimitKind::Pages);
        let cases = [
            (Behaviour::Panic, default, ExtractFailure::Panicked),
            (Behaviour::Fail(pages), default, pages),
            (
                Behaviour::Fail(ExtractFailure::Malformed),
                default,
                ExtractFailure::Malformed,
            ),
            (Behaviour::OutOfBounds, default, ExtractFailure::BadLocator),
            (Behaviour::Passages(&[]), default, ExtractFailure::NoText),
            (
                Behaviour::Passages(&[" \n"]),
                default,
                ExtractFailure::NoText,
            ),
            (Behaviour::Scanned(3), default, ExtractFailure::Unreadable),
            (Behaviour::Passages(&["abc"]), two_chars, chars),
            (Behaviour::Passages(&["a", "b", "c"]), two_chars, chars),
            (Behaviour::Passages(&["a", "b"]), one_chunk, chunks),
        ];
        for (behaviour, limits, expected) in cases {
            assert_eq!(failure(behaviour, &limits), expected, "{behaviour:?}");
        }
    }

    #[test]
    fn counts_characters_not_bytes() {
        let limits = ExtractLimits::builder().chars(2).build().unwrap();
        assert_eq!(run(Behaviour::Passages(&["éé"]), &limits).unwrap().len(), 1);
    }

    #[test]
    fn a_file_over_the_size_limit_fails_before_its_extractor_runs() {
        let limits = ExtractLimits::builder().file_bytes(3).build().unwrap();
        let found = run(Behaviour::Panic, &limits);
        assert!(matches!(
            found,
            Err(Error::File(ExtractFailure::LimitExceeded(
                LimitKind::FileBytes
            )))
        ));
    }

    #[test]
    fn a_bad_file_fails_only_that_file() {
        let good = Behaviour::Passages(&["fine"]);
        let batch = [
            good,
            Behaviour::Panic,
            good,
            Behaviour::Fail(ExtractFailure::LimitExceeded(LimitKind::FileBytes)),
            good,
        ];
        let limits = ExtractLimits::default();
        let outcomes: Vec<bool> = batch.iter().map(|b| run(*b, &limits).is_ok()).collect();
        assert_eq!(outcomes, [true, false, true, false, true]);
    }

    #[test]
    fn a_panic_message_is_not_passed_on() {
        let message = run(Behaviour::Panic, &ExtractLimits::default())
            .unwrap_err()
            .to_string();
        assert_eq!(message, "the extractor panicked");
    }

    #[test]
    fn a_location_without_a_file_is_a_core_error() {
        let err = extract_chunks(
            &fake(Behaviour::Passages(&["text"])),
            &"s3://example-bucket/".parse().unwrap(),
            &ETag::new("9f3a").unwrap(),
            b"file bytes",
            &ExtractLimits::default(),
        )
        .unwrap_err();
        assert!(matches!(err, Error::Core(_)), "{err}");
    }
}
