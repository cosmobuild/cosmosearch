use std::ops::Range;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, BooleanArray, StringArray};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use cosmosearch_core::RowSet;

use crate::Result;
use crate::error::malformed;
use crate::grouped::{self, Groups, as_u64};
use crate::stored::{self, optional, required};

const FORMAT: u32 = 1;
const MAGIC: [u8; 8] = *b"CSTEXTS1";
const GROUP_ROWS: usize = 64;
const TEXT: &str = "text";
const STORED: &str = "stored";

/// The stored text of a segment's chunks, by row. A row without text is a chunk read back from
/// its file by a byte range.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TextTable {
    texts: Vec<Option<String>>,
}

impl TextTable {
    /// A table of `texts`, one for each row of the segment.
    pub fn new(texts: Vec<Option<String>>) -> Self {
        TextTable { texts }
    }

    /// Whether any row has stored text.
    pub fn holds_text(&self) -> bool {
        self.texts.iter().any(Option::is_some)
    }

    /// The bytes stored for this table: its texts in Parquet groups of up to 64 rows, then an
    /// index of which rows have text and where each group is, then a trailer of
    /// [`TRAILER_BYTES`](crate::TRAILER_BYTES) giving the index's length.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Encoding`](crate::Error::Encoding) if encoding fails.
    pub fn encode(&self) -> Result<Vec<u8>> {
        let groups = self
            .texts
            .chunks(GROUP_ROWS)
            .map(|group| {
                let schema = Schema::new(vec![Field::new(TEXT, DataType::Utf8, true)]);
                let column: ArrayRef = Arc::new(StringArray::from(
                    group.iter().map(Option::as_deref).collect::<Vec<_>>(),
                ));
                stored::write(schema, vec![column], FORMAT, vec![])
            })
            .collect::<Result<Vec<_>>>()?;
        grouped::assemble(
            &groups,
            GROUP_ROWS,
            |metadata| {
                let schema = Schema::new(vec![Field::new(STORED, DataType::Boolean, false)]);
                let column: ArrayRef = Arc::new(BooleanArray::from(
                    self.texts.iter().map(Option::is_some).collect::<Vec<_>>(),
                ));
                stored::write(schema, vec![column], FORMAT, metadata)
            },
            MAGIC,
        )
    }
}

/// Which rows of a segment have stored text and where each group of texts is: what a reader
/// keeps to read stored text by row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextIndex {
    rows: usize,
    stored: RowSet,
    groups: Groups,
}

impl TextIndex {
    /// The byte range of the index in a text file of `size` bytes, from its last
    /// [`TRAILER_BYTES`](crate::TRAILER_BYTES) bytes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Malformed`](crate::Error::Malformed) if the trailer is not one.
    pub fn locate(trailer: &[u8], size: u64) -> Result<Range<u64>> {
        grouped::locate(trailer, size, MAGIC)
    }

    /// Reads the index of a text file, the bytes [`TextIndex::locate`] gave.
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnsupportedFormat`](crate::Error::UnsupportedFormat) for another
    /// format, and an error for bytes that do not decode or an index that is inconsistent.
    pub fn decode(bytes: Bytes) -> Result<Self> {
        let stored = stored::read(bytes, FORMAT)?;
        let mut rows = 0usize;
        let mut with_text = RowSet::new();
        for batch in &stored.batches {
            let flags = required::<BooleanArray>(batch, STORED)?;
            for at in 0..batch.num_rows() {
                if flags.value(at) {
                    let row = as_u64(rows);
                    with_text.insert(row..row.saturating_add(1));
                }
                rows = rows.saturating_add(1);
            }
        }
        let groups = Groups::of(&stored.metadata, rows, GROUP_ROWS)?;
        Ok(TextIndex {
            rows,
            stored: with_text,
            groups,
        })
    }

    /// How many rows the segment has.
    pub fn len(&self) -> usize {
        self.rows
    }

    /// Whether the segment has no rows.
    pub fn is_empty(&self) -> bool {
        self.rows == 0
    }

    /// Whether this row has stored text.
    pub fn has_text(&self, row: usize) -> bool {
        row < self.rows && self.stored.contains(as_u64(row))
    }

    /// The group holding `row`, if the segment has that row.
    pub fn group_of(&self, row: usize) -> Option<usize> {
        self.groups.of_row(row, self.rows)
    }

    /// The rows of group `group` and its byte range in the text file.
    pub fn group(&self, group: usize) -> Option<(Range<usize>, Range<u64>)> {
        self.groups.get(group, self.rows)
    }

    /// About how many bytes the index holds in memory.
    pub fn heap_bytes(&self) -> usize {
        self.stored
            .heap_bytes()
            .saturating_add(self.groups.heap_bytes())
            .saturating_add(size_of::<Self>())
    }
}

/// The stored texts of one group of rows of a segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextGroup {
    first: usize,
    texts: Vec<Option<Arc<str>>>,
}

impl TextGroup {
    /// Reads group `group` of the segment `index` describes, from the bytes
    /// [`TextIndex::group`] located.
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnsupportedFormat`](crate::Error::UnsupportedFormat) for another
    /// format, and an error for bytes that do not decode or that hold other rows than the
    /// index says.
    pub fn decode(bytes: Bytes, index: &TextIndex, group: usize) -> Result<Self> {
        let (rows, _) = index
            .group(group)
            .ok_or_else(|| malformed("a group is missing"))?;
        let stored = stored::read(bytes, FORMAT)?;
        let mut texts: Vec<Option<Arc<str>>> = Vec::new();
        for batch in &stored.batches {
            let column = optional::<StringArray>(batch, TEXT)?;
            for at in 0..batch.num_rows() {
                texts.push((!column.is_null(at)).then(|| Arc::from(column.value(at))));
            }
        }
        let matches = texts.len() == rows.len()
            && rows
                .clone()
                .zip(&texts)
                .all(|(row, text)| index.has_text(row) == text.is_some());
        if !matches {
            return Err(malformed("a group holds other rows than the index says"));
        }
        Ok(TextGroup {
            first: rows.start,
            texts,
        })
    }

    /// The text stored for this row of the segment, if the group holds it.
    pub fn get(&self, row: usize) -> Option<Arc<str>> {
        self.texts.get(row.checked_sub(self.first)?)?.clone()
    }

    /// About how many bytes the group holds in memory.
    pub fn heap_bytes(&self) -> usize {
        self.texts
            .iter()
            .map(|text| {
                text.as_ref()
                    .map_or(0, |text| text.len())
                    .saturating_add(size_of::<Option<Arc<str>>>())
            })
            .sum::<usize>()
            .saturating_add(size_of::<Self>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TRAILER_BYTES;
    use crate::grouped::slice;

    fn texts(count: usize) -> Vec<Option<String>> {
        (0..count)
            .map(|row| (row % 3 != 0).then(|| format!("passage {row} é")))
            .collect()
    }

    fn encoded(texts: Vec<Option<String>>) -> Bytes {
        Bytes::from(TextTable::new(texts).encode().unwrap())
    }

    fn index_of(encoded: &Bytes) -> TextIndex {
        let size = as_u64(encoded.len());
        let trailer = slice(encoded, size - TRAILER_BYTES..size).unwrap();
        let at = TextIndex::locate(&trailer, size).unwrap();
        TextIndex::decode(slice(encoded, at).unwrap()).unwrap()
    }

    fn read_all(bytes: &Bytes) -> Result<Vec<Option<String>>> {
        let index = TextIndex::decode(grouped::index_of(bytes, MAGIC)?)?;
        let mut found = Vec::new();
        for row in 0..index.len() {
            let group = index.group_of(row).ok_or_else(|| malformed("no group"))?;
            let (_, at) = index.group(group).ok_or_else(|| malformed("no group"))?;
            let texts = TextGroup::decode(slice(bytes, at)?, &index, group)?;
            found.push(texts.get(row).map(|text| text.to_string()));
        }
        Ok(found)
    }

    #[test]
    fn a_table_round_trips_and_each_group_reads_alone() {
        let expected = texts(150);
        let encoded = encoded(expected.clone());
        assert_eq!(read_all(&encoded).unwrap(), expected);
        let index = index_of(&encoded);
        assert_eq!(index.len(), 150);
        assert!(!index.has_text(0) && index.has_text(1) && !index.has_text(150));
        assert_eq!(index.group_of(150), None);
    }

    #[test]
    fn an_empty_table_round_trips() {
        let encoded = encoded(Vec::new());
        assert!(index_of(&encoded).is_empty());
        assert_eq!(read_all(&encoded).unwrap(), Vec::<Option<String>>::new());
        assert!(!TextTable::new(vec![None, None]).holds_text());
    }

    #[test]
    fn a_group_that_does_not_match_its_index_is_refused() {
        let encoded = encoded(texts(150));
        let index = index_of(&encoded);
        let (_, at) = index.group(0).unwrap();
        TextGroup::decode(slice(&encoded, at).unwrap(), &index, 1).unwrap_err();
        let other = self::encoded(vec![Some("x".to_owned()); 150]);
        let (_, theirs) = index_of(&other).group(0).unwrap();
        TextGroup::decode(slice(&other, theirs).unwrap(), &index, 0).unwrap_err();
    }

    #[test]
    fn damaged_bytes_give_an_error_and_never_a_panic() {
        read_all(&Bytes::from_static(b"garbage")).unwrap_err();
        let good = encoded(texts(5)).to_vec();
        for at in 0..good.len() {
            let mut bad = good.clone();
            if let Some(byte) = bad.get_mut(at) {
                *byte ^= 0xff;
            }
            let _outcome = read_all(&Bytes::from(bad));
        }
    }
}
