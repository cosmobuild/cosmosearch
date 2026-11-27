use std::collections::BTreeMap;
use std::ops::Range;

use bytes::Bytes;

use crate::Result;
use crate::error::malformed;

/// The bytes at the end of a grouped table's file that say where its index is.
pub const TRAILER_BYTES: u64 = 16;

pub(crate) const GROUP_ROWS: usize = 512;

const GROUP_ROWS_KEY: &str = "cosmosearch.group_rows";
const GROUP_BYTES_KEY: &str = "cosmosearch.group_bytes";

/// The bytes of a table kept as `groups` of rows, then the index `index` makes from the
/// metadata it is given, then a trailer of [`TRAILER_BYTES`] with the index's length and
/// `magic`.
pub(crate) fn assemble(
    groups: &[Vec<u8>],
    index: impl FnOnce(Vec<(&'static str, String)>) -> Result<Vec<u8>>,
    magic: [u8; 8],
) -> Result<Vec<u8>> {
    let lengths: Vec<String> = groups.iter().map(|group| group.len().to_string()).collect();
    let metadata = vec![
        (GROUP_ROWS_KEY, GROUP_ROWS.to_string()),
        (GROUP_BYTES_KEY, lengths.join(",")),
    ];
    let mut bytes = groups.concat();
    let index = index(metadata)?;
    bytes.extend_from_slice(&index);
    bytes.extend_from_slice(&as_u64(index.len()).to_le_bytes());
    bytes.extend_from_slice(&magic);
    Ok(bytes)
}

/// The byte range of the index in a file of `size` bytes, from its last [`TRAILER_BYTES`]
/// bytes.
pub(crate) fn locate(trailer: &[u8], size: u64, magic: [u8; 8]) -> Result<Range<u64>> {
    let (length, found) = trailer
        .split_first_chunk::<8>()
        .ok_or_else(|| malformed("a grouped table is too short"))?;
    if *found != magic {
        return Err(malformed("a grouped table has no trailer"));
    }
    let end = size
        .checked_sub(TRAILER_BYTES)
        .ok_or_else(|| malformed("a grouped table is too short"))?;
    let start = end
        .checked_sub(u64::from_le_bytes(*length))
        .ok_or_else(|| malformed("a grouped table's index is longer than its file"))?;
    Ok(start..end)
}

/// The index of a whole file's bytes.
pub(crate) fn index_of(bytes: &Bytes, magic: [u8; 8]) -> Result<Bytes> {
    let size = as_u64(bytes.len());
    let trailer = slice(bytes, size.saturating_sub(TRAILER_BYTES)..size)?;
    slice(bytes, locate(&trailer, size, magic)?)
}

/// Where each group of rows is in a grouped table's file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Groups {
    rows: usize,
    ranges: Vec<Range<u64>>,
}

impl Groups {
    /// The groups an index's metadata names, which must hold `rows` rows.
    pub(crate) fn of(metadata: &BTreeMap<String, String>, rows: usize) -> Result<Self> {
        let size = metadata
            .get(GROUP_ROWS_KEY)
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|size| *size == GROUP_ROWS)
            .ok_or_else(|| malformed("a grouped table has another group size"))?;
        let lengths = metadata
            .get(GROUP_BYTES_KEY)
            .ok_or_else(|| malformed("a grouped table has no group lengths"))?;
        let mut ranges = Vec::new();
        let mut at = 0u64;
        for length in lengths.split(',').filter(|length| !length.is_empty()) {
            let length: u64 = length
                .parse()
                .map_err(|_cause| malformed("a group length is not a number"))?;
            let end = at
                .checked_add(length)
                .ok_or_else(|| malformed("a group length is too large"))?;
            ranges.push(at..end);
            at = end;
        }
        if rows.div_ceil(size) != ranges.len() {
            return Err(malformed("a grouped table has the wrong number of groups"));
        }
        Ok(Groups { rows: size, ranges })
    }

    /// How many groups there are.
    pub(crate) fn len(&self) -> usize {
        self.ranges.len()
    }

    /// The group holding row `row` of a table of `total` rows.
    pub(crate) fn of_row(&self, row: usize, total: usize) -> Option<usize> {
        (row < total).then(|| row / self.rows)
    }

    /// The rows of group `group` of a table of `total` rows, and its byte range.
    pub(crate) fn get(&self, group: usize, total: usize) -> Option<(Range<usize>, Range<u64>)> {
        let at = self.ranges.get(group)?.clone();
        let start = group.checked_mul(self.rows)?;
        let end = start.saturating_add(self.rows).min(total);
        Some((start..end, at))
    }

    pub(crate) fn heap_bytes(&self) -> usize {
        self.ranges
            .len()
            .saturating_mul(size_of::<Range<u64>>())
            .saturating_add(size_of::<Self>())
    }
}

pub(crate) fn slice(bytes: &Bytes, range: Range<u64>) -> Result<Bytes> {
    let start = as_usize(range.start)?;
    let end = as_usize(range.end)?;
    if start > end || end > bytes.len() {
        return Err(malformed("a range lies outside the table's file"));
    }
    Ok(bytes.slice(start..end))
}

pub(crate) fn as_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

pub(crate) fn as_usize(value: u64) -> Result<usize> {
    usize::try_from(value).map_err(|_cause| malformed("a number is too large"))
}
