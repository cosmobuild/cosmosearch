#![warn(clippy::arithmetic_side_effects)]

use std::num::{NonZeroU32, TryFromIntError};
use std::ops::Range;
use std::str;

use cosmosearch_core::{ByteRange, ChunkHash, ModelId};

use super::error::malformed;
use super::{Error, QuantizedVector, Result};

const MAGIC: [u8; 4] = *b"CSVP";
const FORMAT: u32 = 1;
const FIXED_HEADER: usize = 17;
const HASH_BYTES: usize = 32;
const SCALE_BYTES: usize = 4;

/// The start of a vector pack: its header and the sorted hashes of its rows. A reader that
/// has this reads only the rows it needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackTable {
    model_id: ModelId,
    dimensions: NonZeroU32,
    hashes: Vec<ChunkHash>,
}

impl PackTable {
    /// The most bytes a pack's header can take.
    pub const MAX_HEADER_LEN: u64 = FIXED_HEADER as u64 + u8::MAX as u64;

    /// How many bytes at the start of a pack of `object_size` bytes hold the header and
    /// hashes, for vectors of `dimensions` under `model_id`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Malformed`] if no pack of these vectors has that size.
    pub fn prefix_len(model_id: &ModelId, dimensions: NonZeroU32, object_size: u64) -> Result<u64> {
        let header = to_u64(header_len(model_id))?;
        let hash_bytes = to_u64(HASH_BYTES)?;
        let per_row = hash_bytes.saturating_add(to_u64(stride(dimensions)?)?);
        let body = object_size
            .checked_sub(header)
            .ok_or_else(|| malformed("a vector pack is shorter than its header"))?;
        if !body.is_multiple_of(per_row) {
            return Err(malformed("a vector pack has a partial row"));
        }
        body.checked_div(per_row)
            .and_then(|rows| rows.checked_mul(hash_bytes))
            .and_then(|hashes| hashes.checked_add(header))
            .ok_or_else(too_many)
    }

    /// Reads the header and hashes from the first bytes of a pack of `object_size` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnsupportedFormat`] for another format version, and
    /// [`Error::Malformed`] if the bytes are not a consistent pack of that size.
    pub fn decode(prefix: &[u8], object_size: u64) -> Result<Self> {
        let (table, rest) = Self::decode_start(prefix)?;
        if !rest.is_empty() {
            return Err(malformed("a vector pack prefix has bytes after its hashes"));
        }
        let rows_bytes = table
            .stride()?
            .checked_mul(table.len())
            .ok_or_else(too_many)?;
        let expected = to_u64(prefix.len())?.checked_add(to_u64(rows_bytes)?);
        if expected != Some(object_size) {
            return Err(malformed(
                "a vector pack's size does not match its row count",
            ));
        }
        Ok(table)
    }

    /// Reads the model id and dimensions from the start of a pack, which must hold its
    /// header: [`PackTable::MAX_HEADER_LEN`] bytes, or the whole pack if it is shorter.
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnsupportedFormat`] for another format version, and
    /// [`Error::Malformed`] if the bytes are not the start of a pack.
    pub fn peek(start: &[u8]) -> Result<(ModelId, NonZeroU32)> {
        let header = Header::decode(&mut Reader(start))?;
        Ok((header.model_id, header.dimensions))
    }

    fn decode_start(bytes: &[u8]) -> Result<(Self, &[u8])> {
        let mut reader = Reader(bytes);
        let Header {
            model_id,
            dimensions,
            rows,
        } = Header::decode(&mut reader)?;
        let hash_bytes = rows.checked_mul(HASH_BYTES).ok_or_else(too_many)?;
        let hashes: Vec<ChunkHash> = reader
            .take(hash_bytes)?
            .as_chunks::<HASH_BYTES>()
            .0
            .iter()
            .map(|hash| ChunkHash::from_bytes(*hash))
            .collect();
        if !hashes.is_sorted_by(|a, b| a < b) {
            return Err(malformed(
                "a vector pack's hashes are not strictly ascending",
            ));
        }
        let table = PackTable {
            model_id,
            dimensions,
            hashes,
        };
        Ok((table, reader.0))
    }

    /// The model id of every vector in the pack.
    pub fn model_id(&self) -> &ModelId {
        &self.model_id
    }

    /// The length of every vector in the pack.
    pub fn dimensions(&self) -> NonZeroU32 {
        self.dimensions
    }

    /// The hashes of the rows, ascending.
    pub fn hashes(&self) -> &[ChunkHash] {
        &self.hashes
    }

    /// The number of rows.
    pub fn len(&self) -> usize {
        self.hashes.len()
    }

    /// Whether the pack has no rows.
    pub fn is_empty(&self) -> bool {
        self.hashes.is_empty()
    }

    /// The row holding `hash`, if any.
    pub fn find(&self, hash: &ChunkHash) -> Option<usize> {
        self.hashes.binary_search(hash).ok()
    }

    /// Where `rows` sit in the pack.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Malformed`] if `rows` is empty or reaches past the last row.
    pub fn rows_range(&self, rows: Range<usize>) -> Result<ByteRange> {
        if rows.is_empty() || rows.end > self.len() {
            return Err(malformed("the rows asked for are not in the vector pack"));
        }
        let stride = self.stride()?;
        let start = self.rows_start()?;
        let offset = |row: usize| {
            row.checked_mul(stride)
                .and_then(|bytes| bytes.checked_add(start))
                .ok_or_else(too_many)
                .and_then(to_u64)
        };
        Ok(ByteRange::new(offset(rows.start)?, offset(rows.end)?)?)
    }

    /// Decodes consecutive rows read from the range [`PackTable::rows_range`] gave.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Malformed`] if `bytes` is not a whole number of rows, and
    /// [`Error::InvalidVector`] for a row with a bad scale.
    pub fn decode_rows(&self, bytes: &[u8]) -> Result<Vec<QuantizedVector>> {
        let stride = self.stride()?;
        if !bytes.len().is_multiple_of(stride) {
            return Err(malformed("a vector pack has a partial row"));
        }
        bytes.chunks_exact(stride).map(decode_row).collect()
    }

    fn stride(&self) -> Result<usize> {
        stride(self.dimensions)
    }

    fn rows_start(&self) -> Result<usize> {
        self.len()
            .checked_mul(HASH_BYTES)
            .and_then(|hashes| hashes.checked_add(header_len(&self.model_id)))
            .ok_or_else(too_many)
    }
}

/// An immutable file of vectors of one model, each keyed by the hash of its chunk.
///
/// The header names the format, model id, dimensions and row count; the hashes follow in
/// ascending order, then the rows in the same order, each a scale and one byte per dimension.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorPack {
    table: PackTable,
    rows: Vec<QuantizedVector>,
}

impl VectorPack {
    /// A pack of `entries` under `model_id`. Entries are sorted by hash, and of several with
    /// one hash only the first is kept.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidVector`] if a vector does not have `dimensions` values.
    pub fn new(
        model_id: ModelId,
        dimensions: NonZeroU32,
        mut entries: Vec<(ChunkHash, QuantizedVector)>,
    ) -> Result<Self> {
        let width = usize::try_from(dimensions.get()).map_err(too_large)?;
        if entries.iter().any(|(_, vector)| vector.len() != width) {
            return Err(Error::InvalidVector {
                reason: "a vector does not have the pack's dimensions",
            });
        }
        entries.sort_by_key(|(hash, _)| *hash);
        entries.dedup_by_key(|(hash, _)| *hash);
        let (hashes, rows) = entries.into_iter().unzip();
        Ok(VectorPack {
            table: PackTable {
                model_id,
                dimensions,
                hashes,
            },
            rows,
        })
    }

    /// The header and hashes.
    pub fn table(&self) -> &PackTable {
        &self.table
    }

    /// The vectors, in the order of [`PackTable::hashes`].
    pub fn rows(&self) -> &[QuantizedVector] {
        &self.rows
    }

    /// The hashes and vectors, ascending by hash.
    pub fn into_entries(self) -> Vec<(ChunkHash, QuantizedVector)> {
        self.table.hashes.into_iter().zip(self.rows).collect()
    }

    /// Encodes the pack.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Malformed`] if the pack has more rows than the format can count.
    pub fn encode(&self) -> Result<Vec<u8>> {
        let table = &self.table;
        let rows = u32::try_from(table.len()).map_err(too_large)?;
        let model = table.model_id.as_str().as_bytes();
        let model_len = u8::try_from(model.len()).map_err(too_large)?;
        let size = table
            .stride()?
            .checked_mul(table.len())
            .and_then(|bytes| bytes.checked_add(header_len(&table.model_id)))
            .and_then(|bytes| bytes.checked_add(table.len().checked_mul(HASH_BYTES)?))
            .ok_or_else(too_many)?;
        let mut bytes = Vec::with_capacity(size);
        bytes.extend_from_slice(&MAGIC);
        bytes.extend_from_slice(&FORMAT.to_le_bytes());
        bytes.extend_from_slice(&table.dimensions.get().to_le_bytes());
        bytes.extend_from_slice(&rows.to_le_bytes());
        bytes.push(model_len);
        bytes.extend_from_slice(model);
        for hash in &table.hashes {
            bytes.extend_from_slice(hash.as_bytes());
        }
        for row in &self.rows {
            bytes.extend_from_slice(&row.scale().to_le_bytes());
            bytes.extend(row.values().iter().map(|value| value.to_le_bytes()[0]));
        }
        Ok(bytes)
    }

    /// Decodes a whole pack.
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnsupportedFormat`] for another format version, and
    /// [`Error::Malformed`] or [`Error::InvalidVector`] if the bytes are not a consistent
    /// pack.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let (table, rest) = PackTable::decode_start(bytes)?;
        let expected = table.stride()?.checked_mul(table.len());
        if expected != Some(rest.len()) {
            return Err(malformed(
                "a vector pack's size does not match its row count",
            ));
        }
        let rows = table.decode_rows(rest)?;
        Ok(VectorPack { table, rows })
    }
}

struct Header {
    model_id: ModelId,
    dimensions: NonZeroU32,
    rows: usize,
}

impl Header {
    fn decode(reader: &mut Reader<'_>) -> Result<Self> {
        if reader.take(MAGIC.len())? != MAGIC {
            return Err(malformed(
                "a vector pack does not start with its magic bytes",
            ));
        }
        let format = reader.u32()?;
        if format != FORMAT {
            return Err(Error::UnsupportedFormat { format });
        }
        let dimensions = NonZeroU32::new(reader.u32()?)
            .ok_or_else(|| malformed("a vector pack has zero dimensions"))?;
        let rows = usize::try_from(reader.u32()?).map_err(too_large)?;
        let model_len = usize::from(reader.u8()?);
        let model = str::from_utf8(reader.take(model_len)?)
            .map_err(|_utf8| malformed("a vector pack's model id is not text"))?;
        Ok(Header {
            model_id: ModelId::new(model)?,
            dimensions,
            rows,
        })
    }
}

fn decode_row(row: &[u8]) -> Result<QuantizedVector> {
    let (scale, values) = row
        .split_first_chunk::<SCALE_BYTES>()
        .ok_or_else(|| malformed("a vector pack has a partial row"))?;
    let values = values.iter().map(|value| i8::from_le_bytes([*value]));
    QuantizedVector::from_parts(f32::from_le_bytes(*scale), values.collect())
}

fn header_len(model_id: &ModelId) -> usize {
    FIXED_HEADER.saturating_add(model_id.as_str().len())
}

fn stride(dimensions: NonZeroU32) -> Result<usize> {
    usize::try_from(dimensions.get())
        .ok()
        .and_then(|width| width.checked_add(SCALE_BYTES))
        .ok_or_else(|| malformed("a vector pack's rows are too wide"))
}

fn to_u64(value: usize) -> Result<u64> {
    u64::try_from(value).map_err(too_large)
}

fn too_large(_overflow: TryFromIntError) -> Error {
    too_many()
}

fn too_many() -> Error {
    malformed("a vector pack is too large")
}

struct Reader<'bytes>(&'bytes [u8]);

impl<'bytes> Reader<'bytes> {
    fn take(&mut self, count: usize) -> Result<&'bytes [u8]> {
        let (taken, rest) = self
            .0
            .split_at_checked(count)
            .ok_or_else(|| malformed("a vector pack is truncated"))?;
        self.0 = rest;
        Ok(taken)
    }

    fn u8(&mut self) -> Result<u8> {
        let [byte] = self.array::<1>()?;
        Ok(byte)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        let (taken, rest) = self
            .0
            .split_first_chunk::<N>()
            .ok_or_else(|| malformed("a vector pack is truncated"))?;
        self.0 = rest;
        Ok(*taken)
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    fn model() -> ModelId {
        ModelId::new("emb-0123456789abcdef0123456789abcdef").unwrap()
    }

    fn dims(value: u32) -> NonZeroU32 {
        NonZeroU32::new(value).unwrap()
    }

    fn vector(seed: f32) -> QuantizedVector {
        QuantizedVector::quantize(&[seed, -seed / 2.0, 1.0]).unwrap()
    }

    fn sample() -> VectorPack {
        let entries = vec![
            (ChunkHash::of("b"), vector(2.0)),
            (ChunkHash::of("a"), vector(3.0)),
            (ChunkHash::of("c"), vector(0.5)),
        ];
        VectorPack::new(model(), dims(3), entries).unwrap()
    }

    fn check_rejected(bytes: &[u8], is_expected: fn(&Error) -> bool) {
        let err = VectorPack::decode(bytes).unwrap_err();
        assert!(is_expected(&err), "unexpected error: {err}");
    }

    #[test]
    fn a_pack_round_trips_sorted_by_hash() {
        let pack = sample();
        let decoded = VectorPack::decode(&pack.encode().unwrap()).unwrap();
        assert_eq!(decoded, pack);
        assert!(decoded.table().hashes().is_sorted());
    }

    #[test]
    fn an_empty_pack_round_trips() {
        let pack = VectorPack::new(model(), dims(3), vec![]).unwrap();
        assert_eq!(VectorPack::decode(&pack.encode().unwrap()).unwrap(), pack);
    }

    #[test]
    fn keeps_the_first_of_duplicate_hashes() {
        let hash = ChunkHash::of("a");
        let pack = VectorPack::new(
            model(),
            dims(3),
            vec![(hash, vector(1.0)), (hash, vector(2.0))],
        )
        .unwrap();
        assert_eq!(pack.into_entries(), [(hash, vector(1.0))]);
    }

    #[test]
    fn refuses_a_vector_of_other_dimensions() {
        let err =
            VectorPack::new(model(), dims(4), vec![(ChunkHash::of("a"), vector(1.0))]).unwrap_err();
        assert!(matches!(err, Error::InvalidVector { .. }), "{err}");
    }

    #[test]
    fn reads_rows_through_the_table_alone() {
        let pack = sample();
        let bytes = pack.encode().unwrap();
        let size = u64::try_from(bytes.len()).unwrap();
        let prefix_len = PackTable::prefix_len(&model(), dims(3), size).unwrap();
        let prefix = &bytes[..usize::try_from(prefix_len).unwrap()];
        let table = PackTable::decode(prefix, size).unwrap();
        assert_eq!(&table, pack.table());

        let row = table.find(&ChunkHash::of("b")).unwrap();
        let range = table.rows_range(row..row + 1).unwrap();
        let slice =
            &bytes[usize::try_from(range.start()).unwrap()..usize::try_from(range.end()).unwrap()];
        assert_eq!(table.decode_rows(slice).unwrap(), [vector(2.0)]);
        assert_eq!(table.find(&ChunkHash::of("z")), None);
    }

    #[test]
    fn the_header_alone_names_model_and_dimensions() {
        let bytes = sample().encode().unwrap();
        let start = &bytes[..bytes
            .len()
            .min(usize::try_from(PackTable::MAX_HEADER_LEN).unwrap())];
        assert_eq!(PackTable::peek(start).unwrap(), (model(), dims(3)));
        PackTable::peek(&bytes[..12]).unwrap_err();
    }

    #[test]
    fn rejects_a_wrong_version() {
        let mut bytes = sample().encode().unwrap();
        bytes[4] = 9;
        check_rejected(&bytes, |err| {
            matches!(err, Error::UnsupportedFormat { format: 9 })
        });
    }

    #[test]
    fn rejects_damaged_packs() {
        let bytes = sample().encode().unwrap();
        let mut wrong_magic = bytes.clone();
        wrong_magic[0] = b'X';
        let mut more_rows = bytes.clone();
        more_rows[12] = 4;
        let mut fewer_rows = bytes.clone();
        fewer_rows[12] = 2;
        let mut unsorted = bytes.clone();
        let hashes = header_len(&model());
        unsorted[hashes..hashes + 64].rotate_left(32);
        let mut extra = bytes.clone();
        extra.push(0);
        let cases = [
            &bytes[..10],
            &bytes[..bytes.len() - 1],
            &wrong_magic,
            &more_rows,
            &fewer_rows,
            &unsorted,
            &extra,
        ];
        for case in cases {
            check_rejected(case, |err| matches!(err, Error::Malformed { .. }));
        }
    }

    #[test]
    fn rejects_a_negative_scale() {
        let mut bytes = sample().encode().unwrap();
        let first_row = header_len(&model()) + 3 * HASH_BYTES;
        bytes[first_row..first_row + 4].copy_from_slice(&(-1.0_f32).to_le_bytes());
        check_rejected(&bytes, |err| matches!(err, Error::InvalidVector { .. }));
    }

    #[test]
    fn a_table_must_match_the_object_size() {
        let bytes = sample().encode().unwrap();
        let size = u64::try_from(bytes.len()).unwrap();
        for wrong in [size - 1, size + 7, 3] {
            PackTable::prefix_len(&model(), dims(3), wrong).unwrap_err();
        }
        let prefix_len =
            usize::try_from(PackTable::prefix_len(&model(), dims(3), size).unwrap()).unwrap();
        let err = PackTable::decode(&bytes[..prefix_len], size + 7).unwrap_err();
        assert!(matches!(err, Error::Malformed { .. }), "{err}");
        let err = PackTable::decode(&bytes[..prefix_len], size - 7).unwrap_err();
        assert!(matches!(err, Error::Malformed { .. }), "{err}");
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        #[test]
        fn damaged_bytes_give_an_error_and_never_a_panic(
            position in any::<prop::sample::Index>(),
            byte in any::<u8>(),
            cut in any::<prop::sample::Index>(),
        ) {
            let mut bytes = sample().encode().unwrap();
            let at = position.index(bytes.len());
            bytes[at] ^= byte;
            let length = cut.index(bytes.len().saturating_add(1));
            if let Ok(pack) = VectorPack::decode(&bytes[..length]) {
                prop_assert_eq!(pack.table().len(), 3);
            }
        }

        #[test]
        fn any_pack_round_trips(
            dimensions in 1_usize..12,
            seeds in prop::collection::vec((any::<[u8; 32]>(), -10.0_f32..10.0), 0..20),
        ) {
            let entries: Vec<(ChunkHash, QuantizedVector)> = seeds
                .iter()
                .map(|(hash, seed)| {
                    let values = vec![*seed; dimensions];
                    (ChunkHash::from_bytes(*hash), QuantizedVector::quantize(&values).unwrap())
                })
                .collect();
            let width = dims(u32::try_from(dimensions).unwrap());
            let pack = VectorPack::new(model(), width, entries).unwrap();
            prop_assert_eq!(VectorPack::decode(&pack.encode().unwrap()).unwrap(), pack);
        }
    }
}
