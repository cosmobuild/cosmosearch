use cosmosearch_core::{ExtractFailure, LimitKind};
use lopdf::{DecompressError, Document, LoadOptions, Object};

/// The parser's own bound on each stream it decodes while loading; using the same one loads
/// the same objects.
const LOAD_BOUND: usize = 8 * 1024 * 1024;

/// Fails a document whose streams decode to more than `limit` bytes in all.
///
/// The parser decodes page contents, forms and fonts without a bound, so every compressed
/// stream is decoded here first, one at a time and within what is left of the limit,
/// whatever kind the stream claims to be. Streams in formats nothing decodes, such as JPEG
/// images, do not count. A document this check cannot load is malformed, so no file reaches
/// the parser unchecked.
pub(crate) fn check(file: &[u8], limit: u64) -> Result<(), ExtractFailure> {
    let document = load(file).ok_or(ExtractFailure::Malformed)?;
    let mut left = usize::try_from(limit).unwrap_or(usize::MAX);
    let over = ExtractFailure::LimitExceeded(LimitKind::DecodedBytes);
    for object in document.objects.values() {
        let Object::Stream(stream) = object else {
            continue;
        };
        if stream.filters().map_or(true, |filters| filters.is_empty()) {
            continue;
        }
        match stream.decompressed_content_with_limit(left) {
            Ok(content) => left = left.checked_sub(content.len()).ok_or(over)?,
            Err(lopdf::Error::Decompress(DecompressError::MemoryLimitExceeded { .. })) => {
                return Err(over);
            }
            Err(_) => {}
        }
    }
    Ok(())
}

fn load(file: &[u8]) -> Option<Document> {
    match Document::load_mem_with_options(file, options(None)) {
        Ok(document) if !document.is_encrypted() => Some(document),
        Ok(_) | Err(_) => Document::load_mem_with_options(file, options(Some(""))).ok(),
    }
}

fn options(password: Option<&str>) -> LoadOptions {
    LoadOptions {
        password: password.map(str::to_owned),
        max_decompressed_size: Some(LOAD_BOUND),
        ..LoadOptions::default()
    }
}
