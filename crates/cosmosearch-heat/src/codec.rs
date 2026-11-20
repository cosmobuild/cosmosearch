use std::collections::BTreeMap;

use bytes::Bytes;
use cosmosearch_core::ShardId;

use crate::table::{Counts, SIGNALS, Table};
use crate::{Error, WriterId};

const MAGIC: [u8; 4] = *b"CSHT";
const FORMAT: u32 = 2;
const SNAPSHOT: u8 = 1;
const MERGED: u8 = 2;
const COUNT_BYTES: usize = 8;

const _: () = assert!(SIGNALS == 5, "format 2 stores five counts per shard");

/// A merged version as stored: its number, the merge clock, the highest snapshot it took of
/// each writer, and the heat.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct MergedParts {
    pub(crate) version: u64,
    pub(crate) clock: u64,
    pub(crate) writers: BTreeMap<WriterId, u64>,
    pub(crate) table: Table,
}

/// Why stored bytes did not decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DecodeError {
    Malformed(&'static str),
    Format(u32),
}

impl DecodeError {
    pub(crate) fn at(self, key: &str) -> Error {
        match self {
            DecodeError::Malformed(reason) => Error::Malformed {
                key: key.to_owned(),
                reason,
            },
            DecodeError::Format(format) => Error::UnsupportedFormat {
                key: key.to_owned(),
                format,
            },
        }
    }
}

pub(crate) fn encode_snapshot(table: &Table) -> Bytes {
    let mut out = header(SNAPSHOT);
    put_table(&mut out, table);
    Bytes::from(out)
}

pub(crate) fn decode_snapshot(bytes: &[u8]) -> Result<Table, DecodeError> {
    let mut reader = Reader::open(bytes, SNAPSHOT)?;
    let table = reader.table()?;
    reader.finish()?;
    Ok(table)
}

pub(crate) fn encode_merged(merged: &MergedParts) -> Bytes {
    let mut out = header(MERGED);
    out.extend_from_slice(&merged.version.to_le_bytes());
    out.extend_from_slice(&merged.clock.to_le_bytes());
    put_len(&mut out, merged.writers.len());
    for (writer, seq) in &merged.writers {
        put_id(&mut out, &writer.to_string());
        out.extend_from_slice(&seq.to_le_bytes());
    }
    put_table(&mut out, &merged.table);
    Bytes::from(out)
}

pub(crate) fn decode_merged(bytes: &[u8]) -> Result<MergedParts, DecodeError> {
    let mut reader = Reader::open(bytes, MERGED)?;
    let version = reader.u64()?;
    let clock = reader.u64()?;
    let mut writers = BTreeMap::new();
    let mut previous: Option<WriterId> = None;
    for _ in 0..reader.u64()? {
        let writer = WriterId::parse(reader.id()?)
            .ok_or(DecodeError::Malformed("a writer id is invalid"))?;
        if previous
            .as_ref()
            .is_some_and(|previous| *previous >= writer)
        {
            return Err(DecodeError::Malformed("writer ids are not in order"));
        }
        previous = Some(writer.clone());
        writers.insert(writer, reader.u64()?);
    }
    let table = reader.table()?;
    reader.finish()?;
    Ok(MergedParts {
        version,
        clock,
        writers,
        table,
    })
}

fn header(kind: u8) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&FORMAT.to_le_bytes());
    out.push(kind);
    out
}

fn put_table(out: &mut Vec<u8>, table: &Table) {
    out.extend_from_slice(&table.as_of().to_le_bytes());
    put_len(out, table.len());
    for (shard, counts) in table.shards() {
        put_id(out, shard.as_str());
        for count in counts {
            out.extend_from_slice(&count.to_le_bytes());
        }
    }
}

fn put_len(out: &mut Vec<u8>, len: usize) {
    out.extend_from_slice(&u64::try_from(len).unwrap_or(u64::MAX).to_le_bytes());
}

#[expect(
    clippy::expect_used,
    reason = "shard ids are at most 128 bytes and writer ids at most 85"
)]
fn put_id(out: &mut Vec<u8>, id: &str) {
    let len = u8::try_from(id.len()).expect("an id is shorter than 256 bytes");
    out.push(len);
    out.extend_from_slice(id.as_bytes());
}

struct Reader<'bytes>(&'bytes [u8]);

impl<'bytes> Reader<'bytes> {
    fn open(bytes: &'bytes [u8], kind: u8) -> Result<Self, DecodeError> {
        let mut reader = Reader(bytes);
        if reader.array::<4>()? != MAGIC {
            return Err(DecodeError::Malformed("not a heat object"));
        }
        let format = u32::from_le_bytes(reader.array()?);
        if format != FORMAT {
            return Err(DecodeError::Format(format));
        }
        let [found] = reader.array::<1>()?;
        if found != kind {
            return Err(DecodeError::Malformed("a heat object of another kind"));
        }
        Ok(reader)
    }

    fn table(&mut self) -> Result<Table, DecodeError> {
        let as_of = self.u64()?;
        let mut shards = BTreeMap::new();
        let mut previous: Option<ShardId> = None;
        for _ in 0..self.u64()? {
            let shard = ShardId::new(self.id()?)
                .map_err(|_invalid| DecodeError::Malformed("a shard id is invalid"))?;
            if previous.as_ref().is_some_and(|previous| *previous >= shard) {
                return Err(DecodeError::Malformed("shard ids are not in order"));
            }
            previous = Some(shard.clone());
            let mut counts: Counts = [0.0; SIGNALS];
            for count in &mut counts {
                let value = f64::from_le_bytes(self.array::<COUNT_BYTES>()?);
                if !value.is_finite() || value < 0.0 {
                    return Err(DecodeError::Malformed("a count is negative or not finite"));
                }
                *count = value;
            }
            shards.insert(shard, counts);
        }
        Ok(Table::from_parts(as_of, shards))
    }

    fn id(&mut self) -> Result<&'bytes str, DecodeError> {
        let [len] = self.array::<1>()?;
        let bytes = self.take(usize::from(len))?;
        std::str::from_utf8(bytes).map_err(|_invalid| DecodeError::Malformed("an id is not UTF-8"))
    }

    fn u64(&mut self) -> Result<u64, DecodeError> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    fn take(&mut self, count: usize) -> Result<&'bytes [u8], DecodeError> {
        let (taken, rest) = self
            .0
            .split_at_checked(count)
            .ok_or(DecodeError::Malformed("a heat object is truncated"))?;
        self.0 = rest;
        Ok(taken)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], DecodeError> {
        let (taken, rest) = self
            .0
            .split_first_chunk::<N>()
            .ok_or(DecodeError::Malformed("a heat object is truncated"))?;
        self.0 = rest;
        Ok(*taken)
    }

    fn finish(self) -> Result<(), DecodeError> {
        if self.0.is_empty() {
            Ok(())
        } else {
            Err(DecodeError::Malformed(
                "a heat object has bytes after its end",
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fmt::Write;

    use proptest::prelude::*;

    use super::*;

    fn table(as_of: u64, shards: &[(&str, Counts)]) -> Table {
        let shards = shards
            .iter()
            .map(|(id, counts)| (ShardId::new(*id).unwrap(), *counts))
            .collect();
        Table::from_parts(as_of, shards)
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().fold(String::new(), |mut out, byte| {
            write!(out, "{byte:02x}").unwrap();
            out
        })
    }

    fn merged() -> MergedParts {
        let writer = WriterId::parse("1800000000000-ab").unwrap();
        MergedParts {
            version: 3,
            clock: 1_800_000_060_000,
            writers: BTreeMap::from([(writer, 7)]),
            table: table(1_800_000_060_000, &[("s1", [1.0, 0.5, 0.0, 2.0, 0.125])]),
        }
    }

    #[test]
    fn the_format_stays_as_written() {
        let snapshot = encode_snapshot(&table(
            1_800_000_000_000,
            &[("s1", [1.0, 0.0, 0.25, 0.0, 3.0])],
        ));
        insta::assert_snapshot!(hex(&snapshot), @"43534854020000000100505c18a30100000100000000000000027331000000000000f03f0000000000000000000000000000d03f00000000000000000000000000000840");
        insta::assert_snapshot!(hex(&encode_merged(&merged())), @"4353485402000000020300000000000000603a5d18a3010000010000000000000010313830303030303030303030302d61620700000000000000603a5d18a30100000100000000000000027331000000000000f03f000000000000e03f00000000000000000000000000000040000000000000c03f");
    }

    #[test]
    fn both_kinds_round_trip() {
        let parts = merged();
        assert_eq!(decode_merged(&encode_merged(&parts)).unwrap(), parts);
        let empty = MergedParts::default();
        assert_eq!(decode_merged(&encode_merged(&empty)).unwrap(), empty);
        assert_eq!(
            decode_snapshot(&encode_snapshot(&parts.table)).unwrap(),
            parts.table
        );
    }

    #[test]
    fn damaged_objects_are_refused() {
        let good = encode_snapshot(&table(5, &[("a", [1.0; 5]), ("b", [2.0; 5])])).to_vec();
        let mut wrong_format = good.clone();
        wrong_format[4] = 1;
        assert_eq!(decode_snapshot(&wrong_format), Err(DecodeError::Format(1)));
        assert!(matches!(
            decode_merged(&good),
            Err(DecodeError::Malformed("a heat object of another kind"))
        ));
        let mut trailing = good.clone();
        trailing.push(0);
        decode_snapshot(&trailing).unwrap_err();
        for cut in 0..good.len() {
            decode_snapshot(&good[..cut]).unwrap_err();
        }
        let unordered = encode_snapshot(&table(5, &[("a", [1.0; 5])])).to_vec();
        let mut twice = unordered.clone();
        twice[17] = 2;
        twice.extend_from_slice(&unordered[25..]);
        assert!(matches!(
            decode_snapshot(&twice),
            Err(DecodeError::Malformed("shard ids are not in order"))
        ));
        let mut negative = good;
        let first_count = 25 + 2;
        negative[first_count..first_count + 8].copy_from_slice(&(-1.0f64).to_le_bytes());
        assert!(matches!(
            decode_snapshot(&negative),
            Err(DecodeError::Malformed("a count is negative or not finite"))
        ));
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn decoding_never_panics(bytes in prop::collection::vec(any::<u8>(), 0..96)) {
            let _snapshot = decode_snapshot(&bytes);
            let _merged = decode_merged(&bytes);
        }

        #[test]
        fn any_table_round_trips(
            as_of in any::<u64>(),
            shards in prop::collection::btree_map("[a-z0-9]{1,12}", prop::array::uniform5(0.0f64..1e12), 0..8),
        ) {
            let shards = shards
                .into_iter()
                .map(|(id, counts)| (ShardId::new(id).unwrap(), counts))
                .collect();
            let table = Table::from_parts(as_of, shards);
            prop_assert_eq!(decode_snapshot(&encode_snapshot(&table)).unwrap(), table);
        }
    }
}
