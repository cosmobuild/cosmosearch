use super::{Error, Result};

const BITS_PER_BYTE: usize = 8;

/// A vector kept as one bit per dimension, set when the original value is above zero.
///
/// Bits are packed least significant first, so dimension `i` is bit `i % 8` of byte `i / 8`,
/// and the unused bits of the last byte are zero.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BinaryVector {
    dimensions: usize,
    bytes: Vec<u8>,
}

impl BinaryVector {
    /// Quantizes `vector`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidVector`] if `vector` is empty or holds a value that is not
    /// finite.
    pub fn quantize(vector: &[f32]) -> Result<Self> {
        if vector.is_empty() {
            return Err(invalid("a vector must not be empty"));
        }
        if !vector.iter().all(|value| value.is_finite()) {
            return Err(invalid("a vector holds a value that is not finite"));
        }
        let bytes = vector
            .chunks(BITS_PER_BYTE)
            .map(|chunk| {
                chunk
                    .iter()
                    .enumerate()
                    .filter(|(_, value)| **value > 0.0)
                    .fold(0_u8, |byte, (bit, _)| byte | (1 << bit))
            })
            .collect();
        Ok(BinaryVector {
            dimensions: vector.len(),
            bytes,
        })
    }

    /// A vector read back from storage.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidVector`] if `dimensions` is zero, `bytes` is not exactly
    /// `ceil(dimensions / 8)` long, or a bit beyond `dimensions` is set.
    pub fn from_parts(dimensions: usize, bytes: Vec<u8>) -> Result<Self> {
        if dimensions == 0 {
            return Err(invalid("a vector must not be empty"));
        }
        if bytes.len() != dimensions.div_ceil(BITS_PER_BYTE) {
            return Err(invalid("the bytes do not match the dimensions"));
        }
        let used_in_last = dimensions % BITS_PER_BYTE;
        if used_in_last != 0 {
            let unused_mask = u8::MAX << used_in_last;
            if bytes.last().is_some_and(|last| last & unused_mask != 0) {
                return Err(invalid("a bit beyond the dimensions is set"));
            }
        }
        Ok(BinaryVector { dimensions, bytes })
    }

    /// The number of dimensions.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// The packed bits.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The number of dimensions in which the two vectors differ.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidVector`] if the vectors have different dimensions.
    pub fn hamming(&self, other: &BinaryVector) -> Result<u32> {
        if self.dimensions != other.dimensions {
            return Err(invalid("the vectors have different dimensions"));
        }
        Ok(hamming_bytes(&self.bytes, &other.bytes))
    }
}

/// The number of differing bits between two packed byte slices, over the shorter of the two.
pub fn hamming_bytes(a: &[u8], b: &[u8]) -> u32 {
    a.iter().zip(b).fold(0_u32, |sum, (x, y)| {
        sum.saturating_add((x ^ y).count_ones())
    })
}

fn invalid(reason: &'static str) -> Error {
    Error::InvalidVector { reason }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    fn unit_noise(seed: u64, dimensions: usize) -> Vec<f32> {
        let mut state = seed;
        (0..dimensions)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let high = u32::try_from(state >> 40).unwrap();
                f32::from(u16::try_from(high & 0xffff).unwrap()) / 32768.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn a_bit_is_set_for_a_value_above_zero() {
        let vector = BinaryVector::quantize(&[1.0, -1.0, 0.0, 0.5, -0.0, 2.0, 3.0, -4.0, 1.0]);
        let vector = vector.unwrap();
        assert_eq!(vector.dimensions(), 9);
        assert_eq!(vector.as_bytes(), [0b0110_1001, 0b0000_0001]);
    }

    #[test]
    fn refuses_empty_and_non_finite_vectors() {
        for vector in [vec![], vec![1.0, f32::NAN], vec![f32::NEG_INFINITY]] {
            let err = BinaryVector::quantize(&vector).unwrap_err();
            assert!(matches!(err, Error::InvalidVector { .. }), "{err}");
        }
    }

    #[test]
    fn refuses_bad_stored_parts() {
        for (dimensions, bytes) in [
            (0, vec![]),
            (9, vec![0]),
            (8, vec![0, 0]),
            (9, vec![0, 0b10]),
            (3, vec![0b1000]),
        ] {
            let err = BinaryVector::from_parts(dimensions, bytes).unwrap_err();
            assert!(matches!(err, Error::InvalidVector { .. }), "{err}");
        }
    }

    #[test]
    fn refuses_to_compare_different_dimensions() {
        let a = BinaryVector::quantize(&[1.0; 8]).unwrap();
        let b = BinaryVector::quantize(&[1.0; 9]).unwrap();
        a.hamming(&b).unwrap_err();
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn packed_bytes_round_trip(
            vector in prop::collection::vec(-10.0_f32..10.0, 1..200),
        ) {
            let quantized = BinaryVector::quantize(&vector).unwrap();
            let back = BinaryVector::from_parts(
                quantized.dimensions(),
                quantized.as_bytes().to_vec(),
            ).unwrap();
            prop_assert_eq!(&back, &quantized);
            prop_assert_eq!(quantized.as_bytes().len(), vector.len().div_ceil(8));
            for (index, value) in vector.iter().enumerate() {
                let byte = quantized.as_bytes()[index / 8];
                prop_assert_eq!(byte >> (index % 8) & 1 == 1, *value > 0.0);
            }
        }

        #[test]
        fn hamming_is_a_symmetric_bounded_distance(
            a in prop::collection::vec(-1.0_f32..1.0, 1..200),
            seed in any::<u64>(),
        ) {
            let b = unit_noise(seed, a.len());
            let a = BinaryVector::quantize(&a).unwrap();
            let b = BinaryVector::quantize(&b).unwrap();
            let dimensions = u32::try_from(a.dimensions()).unwrap();
            prop_assert_eq!(a.hamming(&a).unwrap(), 0);
            prop_assert_eq!(a.hamming(&b).unwrap(), b.hamming(&a).unwrap());
            prop_assert!(a.hamming(&b).unwrap() <= dimensions);
        }

        #[test]
        fn a_nearby_vector_is_closer_than_an_unrelated_one(
            seed in any::<u64>(),
            dimensions in 128_usize..256,
        ) {
            let base = unit_noise(seed, dimensions);
            let unrelated = unit_noise(seed.wrapping_add(1), dimensions);
            let nearby: Vec<f32> = base
                .iter()
                .zip(unit_noise(seed.wrapping_add(2), dimensions))
                .map(|(x, noise)| x + 0.3 * noise)
                .collect();
            let base = BinaryVector::quantize(&base).unwrap();
            let near = base.hamming(&BinaryVector::quantize(&nearby).unwrap()).unwrap();
            let far = base.hamming(&BinaryVector::quantize(&unrelated).unwrap()).unwrap();
            prop_assert!(near < far, "near {near} far {far}");
        }
    }
}
