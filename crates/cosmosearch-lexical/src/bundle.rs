use std::collections::{BTreeMap, HashMap};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use std::{io, mem};

use async_trait::async_trait;
use cosmosearch_core::{
    ByteRange, IndexCause, IndexError, SegmentFile, SegmentFileName, SegmentId, SegmentSource,
};
use serde::{Deserialize, Serialize};
use tantivy::HasLen;
use tantivy::directory::error::{DeleteError, LockError, OpenReadError, OpenWriteError};
use tantivy::directory::{
    Directory, DirectoryLock, FileHandle, Lock, OwnedBytes, WatchCallback, WatchHandle, WritePtr,
};

const MAGIC: &[u8; 8] = b"CSLEXv01";
const TRAILER_BYTES: u64 = 16;
const TAIL_BYTES: u64 = 64 * 1024;

#[derive(Serialize, Deserialize)]
struct Footer {
    files: BTreeMap<String, (u64, u64)>,
    hot_start: u64,
    hot: Vec<(u64, u64)>,
}

fn malformed(reason: &'static str) -> IndexError {
    IndexError::Malformed { reason }
}

fn engine(source: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> IndexError {
    IndexError::Engine(IndexCause::new(source))
}

#[expect(
    clippy::expect_used,
    reason = "a poisoned lock means an earlier panic, and its state is not trusted"
)]
fn locked<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .expect("no thread panicked while holding the lock")
}

fn not_warmed() -> io::Error {
    io::Error::other("this part of the segment has not been fetched")
}

fn read_only() -> io::Error {
    io::Error::other("a stored segment cannot be written")
}

#[derive(Debug)]
struct Remote {
    source: SegmentSource,
    segment: SegmentId,
    name: SegmentFileName,
}

#[derive(Debug)]
struct Inner {
    remote: Option<Remote>,
    files: HashMap<PathBuf, (u64, u64)>,
    cache: Mutex<BTreeMap<u64, OwnedBytes>>,
    recorded: Option<Mutex<Vec<(u64, u64)>>>,
}

/// A read-only tantivy directory over one bundled segment file.
#[derive(Debug, Clone)]
pub(crate) struct Bundle(Arc<Inner>);

/// Bundles `files`, working out the hot section by opening the result with `open`.
pub(crate) fn write_bundle(
    files: Vec<(PathBuf, Vec<u8>)>,
    always_hot: impl Fn(&Path) -> bool,
    open: impl Fn(Bundle) -> Result<(), IndexError>,
) -> Result<Vec<u8>, IndexError> {
    let mut body = Vec::new();
    let mut table = BTreeMap::new();
    let mut forced = Vec::new();
    for (path, bytes) in files {
        let start = len_u64(body.len())?;
        let len = len_u64(bytes.len())?;
        body.extend_from_slice(&bytes);
        if always_hot(&path) && len > 0 {
            forced.push((start, len));
        }
        let name = path.to_string_lossy().into_owned();
        table.insert(name, (start, len));
    }
    let recorder = Bundle::recording(body.clone(), &table);
    open(recorder.clone())?;
    let mut hot = recorder.take_recorded();
    hot.extend(forced);
    let hot = merge_ranges(hot);
    let hot_start = len_u64(body.len())?;
    for (start, len) in &hot {
        let begin = to_usize(*start)?;
        let end = to_usize(start.saturating_add(*len))?;
        let part = body
            .get(begin..end)
            .ok_or_else(|| malformed("a hot range is outside the file"))?
            .to_vec();
        body.extend_from_slice(&part);
    }
    let footer = Footer {
        files: table,
        hot_start,
        hot,
    };
    let footer = serde_json::to_vec(&footer).map_err(engine)?;
    let footer_len = len_u64(footer.len())?;
    body.extend_from_slice(&footer);
    body.extend_from_slice(&footer_len.to_le_bytes());
    body.extend_from_slice(MAGIC);
    Ok(body)
}

fn len_u64(len: usize) -> Result<u64, IndexError> {
    u64::try_from(len).map_err(|_cause| malformed("a length does not fit"))
}

fn to_usize(value: u64) -> Result<usize, IndexError> {
    usize::try_from(value).map_err(|_cause| malformed("an offset does not fit"))
}

fn merge_ranges(mut ranges: Vec<(u64, u64)>) -> Vec<(u64, u64)> {
    ranges.sort_unstable();
    let mut merged: Vec<(u64, u64)> = Vec::with_capacity(ranges.len());
    for (start, len) in ranges {
        let end = start.saturating_add(len);
        match merged.last_mut() {
            Some((open, open_len)) if open.saturating_add(*open_len) >= start => {
                *open_len = end
                    .max(open.saturating_add(*open_len))
                    .saturating_sub(*open);
            }
            _ => merged.push((start, len)),
        }
    }
    merged
}

impl Bundle {
    fn recording(body: Vec<u8>, table: &BTreeMap<String, (u64, u64)>) -> Bundle {
        let mut cache = BTreeMap::new();
        cache.insert(0, OwnedBytes::new(body));
        Bundle(Arc::new(Inner {
            remote: None,
            files: table
                .iter()
                .map(|(name, range)| (PathBuf::from(name), *range))
                .collect(),
            cache: Mutex::new(cache),
            recorded: Some(Mutex::new(Vec::new())),
        }))
    }

    /// Opens a stored bundle, reading only the end of the file.
    pub(crate) async fn open(
        source: SegmentSource,
        segment: SegmentId,
        file: &SegmentFile,
    ) -> Result<Bundle, IndexError> {
        Bundle::open_reading_tail(source, segment, file, TAIL_BYTES).await
    }

    async fn open_reading_tail(
        source: SegmentSource,
        segment: SegmentId,
        file: &SegmentFile,
        tail_bytes: u64,
    ) -> Result<Bundle, IndexError> {
        let size = file.size();
        let tail_start = size.saturating_sub(tail_bytes);
        let range = ByteRange::new(tail_start, size)
            .map_err(|_cause| malformed("the segment file is empty"))?;
        let mut tail = source.read_range(&segment, file.name(), range).await?;
        let mut tail_at = tail_start;
        let footer = parse_footer(&tail, tail_at, size)?;
        if footer.hot_start < tail_at && !footer.hot.is_empty() {
            let range = ByteRange::new(footer.hot_start, tail_at)
                .map_err(|_cause| malformed("the hot section is misplaced"))?;
            let mut front = source.read_range(&segment, file.name(), range).await?;
            front.extend_from_slice(&tail);
            tail = front;
            tail_at = footer.hot_start;
        }
        let tail = OwnedBytes::new(tail);
        let mut cache = BTreeMap::new();
        let mut at = to_usize(footer.hot_start.saturating_sub(tail_at))?;
        for (start, len) in &footer.hot {
            let end = at.saturating_add(to_usize(*len)?);
            if end > tail.len() {
                return Err(malformed("the hot section is shorter than its table"));
            }
            cache.insert(*start, tail.slice(at..end));
            at = end;
        }
        Ok(Bundle(Arc::new(Inner {
            remote: Some(Remote {
                source,
                segment,
                name: file.name().clone(),
            }),
            files: footer
                .files
                .into_iter()
                .map(|(name, range)| (PathBuf::from(name), range))
                .collect(),
            cache: Mutex::new(cache),
            recorded: None,
        })))
    }

    fn cache(&self) -> MutexGuard<'_, BTreeMap<u64, OwnedBytes>> {
        locked(&self.0.cache)
    }

    fn take_recorded(&self) -> Vec<(u64, u64)> {
        self.0
            .recorded
            .as_ref()
            .map_or_else(Vec::new, |recorded| mem::take(&mut *locked(recorded)))
    }

    fn lookup(&self, start: u64, len: usize) -> Option<OwnedBytes> {
        let cache = self.cache();
        let (at, bytes) = cache.range(..=start).next_back()?;
        let from = usize::try_from(start.checked_sub(*at)?).ok()?;
        let to = from.checked_add(len)?;
        (to <= bytes.len()).then(|| bytes.slice(from..to))
    }

    fn store(&self, start: u64, bytes: OwnedBytes) {
        let mut cache = self.cache();
        if cache
            .get(&start)
            .is_none_or(|held| held.len() < bytes.len())
        {
            cache.insert(start, bytes);
        }
    }

    fn read_now(&self, start: u64, len: usize) -> io::Result<OwnedBytes> {
        if let Some(recorded) = &self.0.recorded
            && let Ok(len) = u64::try_from(len)
        {
            locked(recorded).push((start, len));
        }
        self.lookup(start, len).ok_or_else(not_warmed)
    }

    async fn fetch(&self, start: u64, len: usize) -> io::Result<OwnedBytes> {
        if let Some(bytes) = self.lookup(start, len) {
            return Ok(bytes);
        }
        let remote = self.0.remote.as_ref().ok_or_else(not_warmed)?;
        let end = u64::try_from(len)
            .ok()
            .and_then(|len| start.checked_add(len))
            .ok_or_else(not_warmed)?;
        let range = ByteRange::new(start, end).map_err(io::Error::other)?;
        let body = remote
            .source
            .read_range(&remote.segment, &remote.name, range)
            .await
            .map_err(io::Error::other)?;
        let bytes = OwnedBytes::new(body);
        self.store(start, bytes.clone());
        Ok(bytes)
    }

    fn entry(&self, path: &Path) -> Result<(u64, u64), OpenReadError> {
        self.0
            .files
            .get(path)
            .copied()
            .ok_or_else(|| OpenReadError::FileDoesNotExist(path.to_path_buf()))
    }
}

fn parse_footer(tail: &[u8], tail_at: u64, size: u64) -> Result<Footer, IndexError> {
    let trailer_start = size
        .checked_sub(TRAILER_BYTES)
        .and_then(|at| at.checked_sub(tail_at))
        .ok_or_else(|| malformed("the file is too short"))?;
    let trailer_start = to_usize(trailer_start)?;
    let trailer = tail
        .get(trailer_start..)
        .ok_or_else(|| malformed("the file is too short"))?;
    let (length, magic) = trailer
        .split_first_chunk::<8>()
        .ok_or_else(|| malformed("the file is too short"))?;
    if magic != MAGIC {
        return Err(malformed("the segment file has the wrong format"));
    }
    let length = to_usize(u64::from_le_bytes(*length))?;
    let start = trailer_start
        .checked_sub(length)
        .ok_or_else(|| malformed("the footer does not fit in the part of the file read"))?;
    let footer = tail
        .get(start..trailer_start)
        .ok_or_else(|| malformed("the footer is outside the file"))?;
    serde_json::from_slice(footer).map_err(|_cause| malformed("the footer does not decode"))
}

#[derive(Debug)]
struct Handle {
    bundle: Bundle,
    start: u64,
    len: u64,
}

impl Handle {
    fn locate(&self, range: &Range<usize>) -> io::Result<(u64, usize)> {
        let end = u64::try_from(range.end).map_err(io::Error::other)?;
        if end > self.len || range.start > range.end {
            return Err(io::Error::other("a read went past the end of a file"));
        }
        let offset = u64::try_from(range.start).map_err(io::Error::other)?;
        let start = self.start.checked_add(offset).ok_or_else(not_warmed)?;
        Ok((start, range.len()))
    }
}

impl HasLen for Handle {
    fn len(&self) -> usize {
        usize::try_from(self.len).unwrap_or(usize::MAX)
    }
}

#[async_trait]
impl FileHandle for Handle {
    fn read_bytes(&self, range: Range<usize>) -> io::Result<OwnedBytes> {
        let (start, len) = self.locate(&range)?;
        if len == 0 {
            return Ok(OwnedBytes::empty());
        }
        self.bundle.read_now(start, len)
    }

    async fn read_bytes_async(&self, range: Range<usize>) -> io::Result<OwnedBytes> {
        let (start, len) = self.locate(&range)?;
        if len == 0 {
            return Ok(OwnedBytes::empty());
        }
        self.bundle.fetch(start, len).await
    }
}

impl Directory for Bundle {
    fn get_file_handle(&self, path: &Path) -> Result<Arc<dyn FileHandle>, OpenReadError> {
        let (start, len) = self.entry(path)?;
        Ok(Arc::new(Handle {
            bundle: self.clone(),
            start,
            len,
        }))
    }

    fn delete(&self, _: &Path) -> Result<(), DeleteError> {
        Ok(())
    }

    fn exists(&self, path: &Path) -> Result<bool, OpenReadError> {
        Ok(self.0.files.contains_key(path))
    }

    fn open_write(&self, path: &Path) -> Result<WritePtr, OpenWriteError> {
        Err(OpenWriteError::wrap_io_error(
            read_only(),
            path.to_path_buf(),
        ))
    }

    fn atomic_read(&self, path: &Path) -> Result<Vec<u8>, OpenReadError> {
        let (start, len) = self.entry(path)?;
        let len = usize::try_from(len).unwrap_or(usize::MAX);
        let bytes = self
            .read_now(start, len)
            .map_err(|err| OpenReadError::wrap_io_error(err, path.to_path_buf()))?;
        Ok(bytes.as_slice().to_vec())
    }

    fn atomic_write(&self, _: &Path, _: &[u8]) -> io::Result<()> {
        Err(read_only())
    }

    fn sync_directory(&self) -> io::Result<()> {
        Ok(())
    }

    fn acquire_lock(&self, _: &Lock) -> Result<DirectoryLock, LockError> {
        Ok(DirectoryLock::from(Box::new(())))
    }

    fn watch(&self, _: WatchCallback) -> tantivy::Result<WatchHandle> {
        Ok(WatchHandle::empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merges_touching_and_overlapping_ranges() {
        let merged = merge_ranges(vec![(10, 5), (0, 4), (15, 5), (12, 1), (30, 1)]);
        assert_eq!(merged, [(0, 4), (10, 10), (30, 1)]);
    }

    #[derive(Debug)]
    struct InMemory(Vec<u8>);

    #[async_trait]
    impl cosmosearch_core::SegmentFiles for InMemory {
        async fn read_range(
            &self,
            _: &SegmentId,
            _: &SegmentFileName,
            range: ByteRange,
        ) -> Result<Vec<u8>, IndexError> {
            let (start, end) = (to_usize(range.start())?, to_usize(range.end())?);
            self.0
                .get(start..end)
                .map(<[u8]>::to_vec)
                .ok_or_else(|| malformed("a read went past the end of the object"))
        }
    }

    fn sample(hot_file: &[u8]) -> Vec<u8> {
        let files = vec![
            (PathBuf::from("cold.idx"), vec![1u8; 100]),
            (PathBuf::from("hot.fast"), hot_file.to_vec()),
        ];
        write_bundle(
            files,
            |path| path.extension().is_some_and(|e| e == "fast"),
            |_| Ok(()),
        )
        .unwrap()
    }

    async fn open(bundle: Vec<u8>, tail_bytes: u64) -> Result<Bundle, IndexError> {
        let name = SegmentFileName::new("lex-a.bin").unwrap();
        let file = SegmentFile::new(name, u64::try_from(bundle.len()).unwrap());
        Bundle::open_reading_tail(
            SegmentSource::new(InMemory(bundle)),
            SegmentId::new("a").unwrap(),
            &file,
            tail_bytes,
        )
        .await
    }

    #[tokio::test]
    async fn opens_when_the_hot_section_is_larger_than_the_tail_read() {
        let hot_file = vec![7u8; 4_096];
        let opened = open(sample(&hot_file), 1_024).await.unwrap();
        let read = opened.atomic_read(Path::new("hot.fast")).unwrap();
        assert_eq!(read, hot_file);
        opened.atomic_read(Path::new("cold.idx")).unwrap_err();
    }

    #[tokio::test]
    async fn damaged_or_cut_bundles_give_an_error_and_never_a_panic() {
        let good = sample(&[7u8; 300]);
        let mut damaged: Vec<Vec<u8>> = (0..good.len())
            .map(|at| {
                let mut bad = good.clone();
                bad[at] ^= 0xff;
                bad
            })
            .collect();
        damaged.extend((0..good.len()).map(|len| good[..len].to_vec()));
        for bad in damaged {
            if bad.is_empty() {
                continue;
            }
            if let Ok(opened) = open(bad, TAIL_BYTES).await {
                for name in ["cold.idx", "hot.fast", "missing"] {
                    let _outcome = opened.atomic_read(Path::new(name));
                }
            }
        }
    }

    #[test]
    fn refuses_a_tail_that_is_not_a_bundle() {
        for tail in [&b"short"[..], &[0u8; 64]] {
            let size = u64::try_from(tail.len()).unwrap();
            assert!(matches!(
                parse_footer(tail, 0, size),
                Err(IndexError::Malformed { .. })
            ));
        }
    }
}
