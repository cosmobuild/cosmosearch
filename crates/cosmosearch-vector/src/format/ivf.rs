#![warn(clippy::arithmetic_side_effects)]

use std::collections::BTreeSet;
use std::num::{NonZeroU32, NonZeroUsize, TryFromIntError};
use std::ops::Range;
use std::slice::ChunksExact;

use cosmosearch_core::ByteRange;

use super::error::malformed;
use super::{BinaryVector, Error, QuantizedVector, Result};

const MAGIC: [u8; 4] = *b"CSIV";
const FORMAT: u32 = 1;
const HEADER_LEN: usize = 20;
const FLOAT_BYTES: usize = 4;
const CLUSTER_ENTRY_BYTES: usize = 8;
const ROW_BYTES: usize = 4;
const SCALE_BYTES: usize = 4;
const BITS_PER_BYTE: usize = 8;

/// One vector of an IVF file: the chunk row it belongs to and its binary and int8 forms.
#[derive(Debug, Clone, PartialEq)]
pub struct IvfVector {
    row: u32,
    binary: BinaryVector,
    int8: QuantizedVector,
}

impl IvfVector {
    /// Quantizes `vector`, the vector of chunk `row`, both ways.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidVector`] if `vector` is empty or holds a value that is not
    /// finite.
    pub fn quantize(row: u32, vector: &[f32]) -> Result<Self> {
        Ok(IvfVector {
            row,
            binary: BinaryVector::quantize(vector)?,
            int8: QuantizedVector::quantize(vector)?,
        })
    }

    /// The chunk row of the segment this vector belongs to.
    pub fn row(&self) -> u32 {
        self.row
    }

    /// The binary form, scanned first.
    pub fn binary(&self) -> &BinaryVector {
        &self.binary
    }

    /// The int8 form, used to rescore candidates.
    pub fn int8(&self) -> &QuantizedVector {
        &self.int8
    }
}

/// The start of an IVF file: everything a search keeps in memory to choose clusters and
/// filter rows, and the positions of every other section.
///
/// The file holds a header (format, dimensions, cluster count, row count), the centroids,
/// each cluster's offset and count, the chunk row of every vector in cluster order, then
/// all binary vectors and then all int8 vectors (a scale and one byte per dimension), both
/// in that same order. A vector's position is its index in that order.
#[derive(Debug, Clone, PartialEq)]
pub struct IvfHead {
    dimensions: NonZeroU32,
    centroids: Vec<f32>,
    clusters: Vec<Range<usize>>,
    rows: Vec<u32>,
}

impl IvfHead {
    /// Bytes of the fixed header, the least a file can hold.
    pub const HEADER_LEN: u64 = HEADER_LEN as u64;

    /// The head length the header at the start of a file names.
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnsupportedFormat`] for another format version, and
    /// [`Error::Malformed`] if `start` is shorter than the header or is not one.
    pub fn peek_len(start: &[u8]) -> Result<u64> {
        let header = Header::decode(&mut Reader(start))?;
        to_u64(header.head_len()?)
    }

    /// An upper bound on the head length of a file of `object_size` bytes and `dimensions`
    /// whose clusters average at least `cluster_size` vectors; exact enough to read the head
    /// of such a file in one request.
    pub fn likely_len(dimensions: NonZeroU32, object_size: u64, cluster_size: NonZeroUsize) -> u64 {
        let estimate = || -> Option<u64> {
            let layout = Layout::new(dimensions).ok()?;
            let per_row = to_u64(layout.per_row()).ok()?;
            let rows = object_size
                .checked_sub(Self::HEADER_LEN)?
                .checked_div(per_row)?;
            let cluster_size = to_u64(cluster_size.get()).ok()?;
            let clusters = rows.div_ceil(cluster_size).max(1);
            let per_cluster = to_u64(layout.per_cluster()).ok()?;
            let rows_bytes = rows.checked_mul(to_u64(ROW_BYTES).ok()?)?;
            clusters
                .checked_mul(per_cluster)?
                .checked_add(rows_bytes)?
                .checked_add(Self::HEADER_LEN)
        };
        estimate().unwrap_or(object_size).min(object_size)
    }

    /// Reads the head, exactly [`IvfHead::peek_len`] bytes, of a file of `object_size` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnsupportedFormat`] for another format version, and
    /// [`Error::Malformed`] if the bytes are not a consistent head of a file of that size.
    pub fn decode(head: &[u8], object_size: u64) -> Result<Self> {
        let mut reader = Reader(head);
        let header = Header::decode(&mut reader)?;
        let layout = Layout::new(header.dimensions)?;
        if to_u64(header.file_len()?)? != object_size {
            return Err(malformed("an IVF file's size does not match its counts"));
        }
        let width = layout.width;
        let centroid_bytes = header
            .clusters
            .checked_mul(width)
            .and_then(|values| values.checked_mul(FLOAT_BYTES))
            .ok_or_else(too_large)?;
        let centroids: Vec<f32> = reader
            .take(centroid_bytes)?
            .as_chunks::<FLOAT_BYTES>()
            .0
            .iter()
            .map(|bytes| f32::from_le_bytes(*bytes))
            .collect();
        if !centroids.iter().all(|value| value.is_finite()) {
            return Err(malformed("an IVF file has a centroid that is not finite"));
        }
        let table_bytes = header
            .clusters
            .checked_mul(CLUSTER_ENTRY_BYTES)
            .ok_or_else(too_large)?;
        let mut clusters = Vec::with_capacity(header.clusters);
        let mut next = 0_usize;
        for entry in reader
            .take(table_bytes)?
            .as_chunks::<CLUSTER_ENTRY_BYTES>()
            .0
        {
            let [offset, count] = entry.as_chunks::<4>().0 else {
                return Err(malformed("an IVF file has a damaged cluster table"));
            };
            let offset = to_usize(u32::from_le_bytes(*offset))?;
            let count = to_usize(u32::from_le_bytes(*count))?;
            if offset != next {
                return Err(malformed("an IVF file's clusters are not contiguous"));
            }
            next = offset.checked_add(count).ok_or_else(too_large)?;
            clusters.push(offset..next);
        }
        if next != header.rows {
            return Err(malformed("an IVF file's clusters do not hold every row"));
        }
        let row_bytes = header.rows.checked_mul(ROW_BYTES).ok_or_else(too_large)?;
        let rows: Vec<u32> = reader
            .take(row_bytes)?
            .as_chunks::<ROW_BYTES>()
            .0
            .iter()
            .map(|bytes| u32::from_le_bytes(*bytes))
            .collect();
        if !reader.0.is_empty() {
            return Err(malformed("an IVF head has bytes after its rows"));
        }
        let distinct: BTreeSet<u32> = rows.iter().copied().collect();
        if distinct.len() != rows.len() {
            return Err(malformed("an IVF file holds a row twice"));
        }
        Ok(IvfHead {
            dimensions: header.dimensions,
            centroids,
            clusters,
            rows,
        })
    }

    /// The length of every vector.
    pub fn dimensions(&self) -> NonZeroU32 {
        self.dimensions
    }

    /// The number of clusters.
    pub fn clusters(&self) -> usize {
        self.clusters.len()
    }

    /// The number of vectors.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether the file holds no vectors.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// The centroid of `cluster`.
    pub fn centroid(&self, cluster: usize) -> Option<&[f32]> {
        let width = usize::try_from(self.dimensions.get()).ok()?;
        self.centroids.chunks_exact(width).nth(cluster)
    }

    /// The positions of the vectors in `cluster`.
    pub fn cluster(&self, cluster: usize) -> Option<Range<usize>> {
        self.clusters.get(cluster).cloned()
    }

    /// The chunk rows of the vectors, by position.
    pub fn rows(&self) -> &[u32] {
        &self.rows
    }

    /// Bytes of one binary vector.
    pub fn binary_stride(&self) -> usize {
        self.layout().map_or(0, |layout| layout.binary)
    }

    /// Bytes of one int8 vector with its scale.
    pub fn int8_stride(&self) -> usize {
        self.layout().map_or(0, |layout| layout.int8)
    }

    /// Where the binary vectors at `positions` sit in the file.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Malformed`] if `positions` is empty or reaches past the last vector.
    pub fn binary_range(&self, positions: Range<usize>) -> Result<ByteRange> {
        let start = self.head_len()?;
        self.section_range(start, self.binary_stride(), positions)
    }

    /// Where the int8 vectors at `positions` sit in the file.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Malformed`] if `positions` is empty or reaches past the last vector.
    pub fn int8_range(&self, positions: Range<usize>) -> Result<ByteRange> {
        let start = self
            .len()
            .checked_mul(self.binary_stride())
            .and_then(|bytes| bytes.checked_add(self.head_len().ok()?))
            .ok_or_else(too_large)?;
        self.section_range(start, self.int8_stride(), positions)
    }

    /// Splits bytes read from a [`IvfHead::binary_range`] into one packed vector each.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Malformed`] if `bytes` is not a whole number of vectors.
    pub fn binary_vectors<'bytes>(&self, bytes: &'bytes [u8]) -> Result<ChunksExact<'bytes, u8>> {
        let stride = self.binary_stride();
        if stride == 0 || !bytes.len().is_multiple_of(stride) {
            return Err(malformed("an IVF file has a partial binary vector"));
        }
        Ok(bytes.chunks_exact(stride))
    }

    /// Decodes bytes read from a [`IvfHead::int8_range`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Malformed`] if `bytes` is not a whole number of vectors, and
    /// [`Error::InvalidVector`] for one with a bad scale.
    pub fn int8_vectors(&self, bytes: &[u8]) -> Result<Vec<QuantizedVector>> {
        let stride = self.int8_stride();
        if stride == 0 || !bytes.len().is_multiple_of(stride) {
            return Err(malformed("an IVF file has a partial int8 vector"));
        }
        bytes.chunks_exact(stride).map(decode_int8).collect()
    }

    fn layout(&self) -> Result<Layout> {
        Layout::new(self.dimensions)
    }

    fn head_len(&self) -> Result<usize> {
        self.header().head_len()
    }

    fn header(&self) -> Header {
        Header {
            dimensions: self.dimensions,
            clusters: self.clusters.len(),
            rows: self.rows.len(),
        }
    }

    fn section_range(
        &self,
        start: usize,
        stride: usize,
        positions: Range<usize>,
    ) -> Result<ByteRange> {
        if positions.is_empty() || positions.end > self.len() {
            return Err(malformed("the positions asked for are not in the IVF file"));
        }
        let offset = |position: usize| {
            position
                .checked_mul(stride)
                .and_then(|bytes| bytes.checked_add(start))
                .ok_or_else(too_large)
                .and_then(to_u64)
        };
        Ok(ByteRange::new(
            offset(positions.start)?,
            offset(positions.end)?,
        )?)
    }
}

/// The IVF file of one segment: its clusters, near ones next to each other, and every vector
/// in cluster order.
#[derive(Debug, Clone, PartialEq)]
pub struct IvfFile {
    head: IvfHead,
    vectors: Vec<IvfVector>,
}

impl IvfFile {
    /// A file of `clusters` in the order given, each a centroid and its vectors. A file with
    /// no vectors may have no clusters.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidVector`] if a centroid or vector does not have `dimensions`
    /// values or a centroid holds a value that is not finite, and [`Error::Malformed`] if
    /// a row appears twice or the counts are too large for the format.
    pub fn new(dimensions: NonZeroU32, clusters: Vec<(Vec<f32>, Vec<IvfVector>)>) -> Result<Self> {
        let layout = Layout::new(dimensions)?;
        let mut centroids = Vec::with_capacity(clusters.len().saturating_mul(layout.width));
        let mut ranges = Vec::with_capacity(clusters.len());
        let mut vectors = Vec::new();
        for (centroid, members) in clusters {
            if centroid.len() != layout.width || !centroid.iter().all(|value| value.is_finite()) {
                return Err(Error::InvalidVector {
                    reason: "a centroid does not have the file's dimensions or is not finite",
                });
            }
            if members.iter().any(|vector| {
                vector.binary.dimensions() != layout.width || vector.int8.len() != layout.width
            }) {
                return Err(Error::InvalidVector {
                    reason: "a vector does not have the file's dimensions",
                });
            }
            centroids.extend(centroid);
            let start = vectors.len();
            vectors.extend(members);
            ranges.push(start..vectors.len());
        }
        let rows: Vec<u32> = vectors.iter().map(IvfVector::row).collect();
        let distinct: BTreeSet<u32> = rows.iter().copied().collect();
        if distinct.len() != rows.len() {
            return Err(malformed("an IVF file holds a row twice"));
        }
        let head = IvfHead {
            dimensions,
            centroids,
            clusters: ranges,
            rows,
        };
        head.header().counts()?;
        Ok(IvfFile { head, vectors })
    }

    /// The head of the file.
    pub fn head(&self) -> &IvfHead {
        &self.head
    }

    /// The vectors, in cluster order.
    pub fn vectors(&self) -> &[IvfVector] {
        &self.vectors
    }

    /// Encodes the file.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Malformed`] if the file is too large for the format.
    pub fn encode(&self) -> Result<Vec<u8>> {
        let header = self.head.header();
        let (clusters, rows) = header.counts()?;
        let mut bytes = Vec::with_capacity(header.file_len()?);
        bytes.extend_from_slice(&MAGIC);
        bytes.extend_from_slice(&FORMAT.to_le_bytes());
        bytes.extend_from_slice(&self.head.dimensions.get().to_le_bytes());
        bytes.extend_from_slice(&clusters.to_le_bytes());
        bytes.extend_from_slice(&rows.to_le_bytes());
        for value in &self.head.centroids {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for cluster in &self.head.clusters {
            let offset = u32::try_from(cluster.start).map_err(too_large_from)?;
            let count = u32::try_from(cluster.len()).map_err(too_large_from)?;
            bytes.extend_from_slice(&offset.to_le_bytes());
            bytes.extend_from_slice(&count.to_le_bytes());
        }
        for row in &self.head.rows {
            bytes.extend_from_slice(&row.to_le_bytes());
        }
        for vector in &self.vectors {
            bytes.extend_from_slice(vector.binary.as_bytes());
        }
        for vector in &self.vectors {
            bytes.extend_from_slice(&vector.int8.scale().to_le_bytes());
            bytes.extend(
                vector
                    .int8
                    .values()
                    .iter()
                    .map(|value| value.to_le_bytes()[0]),
            );
        }
        Ok(bytes)
    }

    /// Decodes a whole file.
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnsupportedFormat`] for another format version, and
    /// [`Error::Malformed`] or [`Error::InvalidVector`] if the bytes are not a consistent
    /// file.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let size = to_u64(bytes.len())?;
        let head_len = to_usize_u64(IvfHead::peek_len(bytes)?)?;
        let (head_bytes, body) = bytes
            .split_at_checked(head_len)
            .ok_or_else(|| malformed("an IVF file is truncated"))?;
        let head = IvfHead::decode(head_bytes, size)?;
        let binary_bytes = head
            .len()
            .checked_mul(head.binary_stride())
            .ok_or_else(too_large)?;
        let (binary, int8) = body
            .split_at_checked(binary_bytes)
            .ok_or_else(|| malformed("an IVF file is truncated"))?;
        let width = usize::try_from(head.dimensions.get()).map_err(too_large_from)?;
        let binary = head
            .binary_vectors(binary)?
            .map(|bytes| BinaryVector::from_parts(width, bytes.to_vec()));
        let int8 = head.int8_vectors(int8)?;
        let vectors = head
            .rows
            .iter()
            .zip(binary)
            .zip(int8)
            .map(|((row, binary), int8)| {
                Ok(IvfVector {
                    row: *row,
                    binary: binary?,
                    int8,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(IvfFile { head, vectors })
    }
}

#[derive(Debug, Clone, Copy)]
struct Layout {
    width: usize,
    binary: usize,
    int8: usize,
}

impl Layout {
    fn new(dimensions: NonZeroU32) -> Result<Self> {
        let width = usize::try_from(dimensions.get()).map_err(too_large_from)?;
        let int8 = width.checked_add(SCALE_BYTES).ok_or_else(too_large)?;
        Ok(Layout {
            width,
            binary: width.div_ceil(BITS_PER_BYTE),
            int8,
        })
    }

    fn per_cluster(self) -> usize {
        self.width
            .saturating_mul(FLOAT_BYTES)
            .saturating_add(CLUSTER_ENTRY_BYTES)
    }

    fn per_row(self) -> usize {
        ROW_BYTES
            .saturating_add(self.binary)
            .saturating_add(self.int8)
    }
}

#[derive(Debug, Clone, Copy)]
struct Header {
    dimensions: NonZeroU32,
    clusters: usize,
    rows: usize,
}

impl Header {
    fn decode(reader: &mut Reader<'_>) -> Result<Self> {
        if reader.take(MAGIC.len())? != MAGIC {
            return Err(malformed("an IVF file does not start with its magic bytes"));
        }
        let format = reader.u32()?;
        if format != FORMAT {
            return Err(Error::UnsupportedFormat { format });
        }
        let dimensions = NonZeroU32::new(reader.u32()?)
            .ok_or_else(|| malformed("an IVF file has zero dimensions"))?;
        let clusters = to_usize(reader.u32()?)?;
        let rows = to_usize(reader.u32()?)?;
        if clusters == 0 && rows > 0 {
            return Err(malformed("an IVF file with vectors has no cluster"));
        }
        let header = Header {
            dimensions,
            clusters,
            rows,
        };
        header.file_len()?;
        Ok(header)
    }

    fn counts(self) -> Result<(u32, u32)> {
        let clusters = u32::try_from(self.clusters).map_err(too_large_from)?;
        let rows = u32::try_from(self.rows).map_err(too_large_from)?;
        Ok((clusters, rows))
    }

    fn head_len(self) -> Result<usize> {
        let layout = Layout::new(self.dimensions)?;
        let clusters = self.clusters.checked_mul(layout.per_cluster());
        let rows = self.rows.checked_mul(ROW_BYTES);
        clusters
            .zip(rows)
            .and_then(|(clusters, rows)| clusters.checked_add(rows)?.checked_add(HEADER_LEN))
            .ok_or_else(too_large)
    }

    fn file_len(self) -> Result<usize> {
        let layout = Layout::new(self.dimensions)?;
        let body = layout
            .binary
            .checked_add(layout.int8)
            .and_then(|per_row| per_row.checked_mul(self.rows));
        body.zip(self.head_len().ok())
            .and_then(|(body, head)| body.checked_add(head))
            .ok_or_else(too_large)
    }
}

fn decode_int8(bytes: &[u8]) -> Result<QuantizedVector> {
    let (scale, values) = bytes
        .split_first_chunk::<SCALE_BYTES>()
        .ok_or_else(|| malformed("an IVF file has a partial int8 vector"))?;
    let values = values.iter().map(|value| i8::from_le_bytes([*value]));
    QuantizedVector::from_parts(f32::from_le_bytes(*scale), values.collect())
}

fn to_u64(value: usize) -> Result<u64> {
    u64::try_from(value).map_err(too_large_from)
}

fn to_usize(value: u32) -> Result<usize> {
    usize::try_from(value).map_err(too_large_from)
}

fn to_usize_u64(value: u64) -> Result<usize> {
    usize::try_from(value).map_err(too_large_from)
}

fn too_large_from(_overflow: TryFromIntError) -> Error {
    too_large()
}

fn too_large() -> Error {
    malformed("an IVF file is too large")
}

struct Reader<'bytes>(&'bytes [u8]);

impl<'bytes> Reader<'bytes> {
    fn take(&mut self, count: usize) -> Result<&'bytes [u8]> {
        let (taken, rest) = self
            .0
            .split_at_checked(count)
            .ok_or_else(|| malformed("an IVF file is truncated"))?;
        self.0 = rest;
        Ok(taken)
    }

    fn u32(&mut self) -> Result<u32> {
        let (taken, rest) = self
            .0
            .split_first_chunk::<4>()
            .ok_or_else(|| malformed("an IVF file is truncated"))?;
        self.0 = rest;
        Ok(u32::from_le_bytes(*taken))
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    fn dims(value: u32) -> NonZeroU32 {
        NonZeroU32::new(value).unwrap()
    }

    fn vector(row: u32, values: &[f32]) -> IvfVector {
        IvfVector::quantize(row, values).unwrap()
    }

    fn sample() -> IvfFile {
        IvfFile::new(
            dims(3),
            vec![
                (
                    vec![1.0, 0.0, 0.0],
                    vec![vector(4, &[0.9, 0.1, -0.2]), vector(0, &[1.0, -0.3, 0.0])],
                ),
                (vec![0.0, 1.0, 0.0], vec![]),
                (vec![0.0, 0.0, -1.0], vec![vector(2, &[0.1, 0.2, -0.9])]),
            ],
        )
        .unwrap()
    }

    fn slice(bytes: &[u8], range: ByteRange) -> &[u8] {
        &bytes[usize::try_from(range.start()).unwrap()..usize::try_from(range.end()).unwrap()]
    }

    #[test]
    fn a_file_round_trips_in_cluster_order() {
        let file = sample();
        let decoded = IvfFile::decode(&file.encode().unwrap()).unwrap();
        assert_eq!(decoded, file);
        assert_eq!(decoded.head().rows(), [4, 0, 2]);
        assert_eq!(decoded.head().clusters(), 3);
        assert_eq!(decoded.head().cluster(1), Some(2..2));
        assert_eq!(decoded.head().cluster(2), Some(2..3));
        assert_eq!(decoded.head().centroid(2), Some(&[0.0, 0.0, -1.0][..]));
    }

    #[test]
    fn an_empty_file_round_trips() {
        let file = IvfFile::new(dims(5), vec![]).unwrap();
        let bytes = file.encode().unwrap();
        assert_eq!(bytes.len(), HEADER_LEN);
        assert_eq!(IvfFile::decode(&bytes).unwrap(), file);
        assert!(file.head().is_empty());
    }

    #[test]
    fn the_head_alone_locates_every_vector() {
        let file = sample();
        let bytes = file.encode().unwrap();
        let size = u64::try_from(bytes.len()).unwrap();
        let head_len = IvfHead::peek_len(&bytes[..HEADER_LEN]).unwrap();
        let head = IvfHead::decode(&bytes[..usize::try_from(head_len).unwrap()], size).unwrap();
        assert_eq!(&head, file.head());

        let binary = slice(&bytes, head.binary_range(1..3).unwrap());
        let found: Vec<&[u8]> = head.binary_vectors(binary).unwrap().collect();
        assert_eq!(found[0], file.vectors()[1].binary().as_bytes());
        assert_eq!(found[1], file.vectors()[2].binary().as_bytes());

        let int8 = slice(&bytes, head.int8_range(0..1).unwrap());
        assert_eq!(
            head.int8_vectors(int8).unwrap(),
            [file.vectors()[0].int8().clone()]
        );
        let last = head.int8_range(2..3).unwrap();
        assert_eq!(last.end(), size);

        head.binary_range(0..0).unwrap_err();
        head.int8_range(2..4).unwrap_err();
        head.int8_vectors(&int8[1..]).unwrap_err();
    }

    #[test]
    fn the_likely_head_length_covers_a_file_with_clusters_that_large() {
        let width = 16;
        let members = |cluster: u32| -> Vec<IvfVector> {
            (0..64)
                .map(|row| vector(cluster * 64 + row, &[1.0; 16]))
                .collect()
        };
        let file = IvfFile::new(
            dims(width),
            (0..8)
                .map(|cluster| (vec![0.25; 16], members(cluster)))
                .collect(),
        )
        .unwrap();
        let bytes = file.encode().unwrap();
        let size = u64::try_from(bytes.len()).unwrap();
        let exact = IvfHead::peek_len(&bytes).unwrap();
        let cluster_size = |rows| NonZeroUsize::new(rows).unwrap();
        let likely = IvfHead::likely_len(dims(width), size, cluster_size(64));
        assert!(likely >= exact, "{likely} < {exact}");
        assert!(likely - exact < exact / 10, "{likely} is far above {exact}");
        assert!(IvfHead::likely_len(dims(width), size, cluster_size(8)) > likely);
        assert_eq!(IvfHead::likely_len(dims(width), 3, cluster_size(64)), 3);
    }

    #[test]
    fn refuses_inconsistent_parts() {
        let cases = [
            IvfFile::new(dims(3), vec![(vec![1.0, 0.0], vec![])]),
            IvfFile::new(dims(3), vec![(vec![1.0, 0.0, f32::NAN], vec![])]),
            IvfFile::new(dims(2), vec![(vec![1.0, 0.0], vec![vector(0, &[1.0])])]),
            IvfFile::new(
                dims(1),
                vec![
                    (vec![1.0], vec![vector(3, &[1.0])]),
                    (vec![-1.0], vec![vector(3, &[-1.0])]),
                ],
            ),
        ];
        for case in cases {
            case.unwrap_err();
        }
    }

    #[test]
    fn rejects_a_wrong_version() {
        let mut bytes = sample().encode().unwrap();
        bytes[4] = 9;
        let err = IvfFile::decode(&bytes).unwrap_err();
        assert!(
            matches!(err, Error::UnsupportedFormat { format: 9 }),
            "{err}"
        );
    }

    #[test]
    fn rejects_damaged_files() {
        let bytes = sample().encode().unwrap();
        let head_len = usize::try_from(IvfHead::peek_len(&bytes).unwrap()).unwrap();
        let table = HEADER_LEN + 3 * 3 * FLOAT_BYTES;
        let rows = table + 3 * CLUSTER_ENTRY_BYTES;
        let with = |position: usize, value: u32| {
            let mut changed = bytes.clone();
            changed[position..position + 4].copy_from_slice(&value.to_le_bytes());
            changed
        };
        let mut wrong_magic = bytes.clone();
        wrong_magic[0] = b'X';
        let mut extra = bytes.clone();
        extra.push(0);
        let cases = [
            bytes[..10].to_vec(),
            bytes[..bytes.len() - 1].to_vec(),
            wrong_magic,
            extra,
            with(8, 0),
            with(12, 0),
            with(12, 4),
            with(16, 2),
            with(16, 4),
            with(16, u32::MAX),
            with(HEADER_LEN, f32::NAN.to_bits()),
            with(table, 1),
            with(table + 4, 3),
            with(rows + 4, 2),
        ];
        for (index, case) in cases.iter().enumerate() {
            let err = IvfFile::decode(case).unwrap_err();
            assert!(
                matches!(err, Error::Malformed { .. }),
                "case {index}: {err}"
            );
        }
        let size = u64::try_from(bytes.len()).unwrap();
        IvfHead::decode(&bytes[..head_len], size + 1).unwrap_err();
        IvfHead::decode(&bytes[..head_len - 1], size).unwrap_err();
        IvfHead::decode(&bytes[..=head_len], size).unwrap_err();
    }

    #[test]
    fn rejects_a_bad_scale_or_padding_bits() {
        let bytes = sample().encode().unwrap();
        let head_len = usize::try_from(IvfHead::peek_len(&bytes).unwrap()).unwrap();
        let mut padding = bytes.clone();
        padding[head_len] |= 0b1000_0000;
        let int8_start = head_len + 3;
        let mut scale = bytes;
        scale[int8_start..int8_start + 4].copy_from_slice(&(-1.0_f32).to_le_bytes());
        for case in [padding, scale] {
            let err = IvfFile::decode(&case).unwrap_err();
            assert!(matches!(err, Error::InvalidVector { .. }), "{err}");
        }
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
            let damaged = &bytes[..length];
            if let Ok(file) = IvfFile::decode(damaged) {
                prop_assert_eq!(file.vectors().len(), 3);
            }
            if let Ok(head_len) = IvfHead::peek_len(damaged) {
                let size = u64::try_from(length).unwrap();
                let end = usize::try_from(head_len).unwrap();
                if let Some(Ok(head)) = damaged.get(..end).map(|head| IvfHead::decode(head, size)) {
                    prop_assert!(head.clusters() <= head.len().max(3));
                }
            }
        }

        #[test]
        fn any_file_round_trips(
            dimensions in 1_u32..20,
            sizes in prop::collection::vec(0_usize..6, 1..5),
            seed in -10.0_f32..10.0,
        ) {
            let width = usize::try_from(dimensions).unwrap();
            let mut rows = (1_u16..).map(|row| (u32::from(row).checked_mul(7).unwrap(), f32::from(row)));
            let clusters: Vec<(Vec<f32>, Vec<IvfVector>)> = sizes
                .iter()
                .map(|size| {
                    let members = rows
                        .by_ref()
                        .take(*size)
                        .map(|(row, offset)| {
                            let values: Vec<f32> = (0..width)
                                .map(|i| seed * f32::from(u16::try_from(i).unwrap()) - offset)
                                .collect();
                            vector(row, &values)
                        })
                        .collect();
                    (vec![0.5; width], members)
                })
                .collect();
            let file = IvfFile::new(dims(dimensions), clusters).unwrap();
            let bytes = file.encode().unwrap();
            prop_assert_eq!(IvfFile::decode(&bytes).unwrap(), file);
        }
    }
}
