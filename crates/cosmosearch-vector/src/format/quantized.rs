use super::{Error, Result};

const INT8_LIMIT: f32 = 127.0;

/// A vector kept as one signed byte per dimension and one scale, the largest magnitude of
/// the original values, so each value is `round(x / scale * 127)`.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedVector {
    scale: f32,
    values: Vec<i8>,
}

impl QuantizedVector {
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
        let scale = vector
            .iter()
            .fold(0.0_f32, |max, value| max.max(value.abs()));
        let values = vector.iter().map(|value| to_int8(*value, scale)).collect();
        Ok(QuantizedVector { scale, values })
    }

    /// A vector read back from storage.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidVector`] if `values` is empty or `scale` is negative or not
    /// finite.
    pub fn from_parts(scale: f32, values: Vec<i8>) -> Result<Self> {
        if values.is_empty() {
            return Err(invalid("a vector must not be empty"));
        }
        if !scale.is_finite() || scale.is_sign_negative() {
            return Err(invalid("a scale must be finite and not negative"));
        }
        Ok(QuantizedVector { scale, values })
    }

    /// The largest magnitude of the original values.
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// One signed byte per dimension.
    pub fn values(&self) -> &[i8] {
        &self.values
    }

    /// The number of dimensions.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Always `false`: a vector has at least one dimension.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// The dot product of `query` with the vector this approximates, computed without
    /// building that vector. Each stored value is off by at most half a step,
    /// `scale / 254`, so the result is off by at most that times the sum of the absolute
    /// values of `query`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidVector`] if `query` has a different number of dimensions.
    pub fn dot(&self, query: &[f32]) -> Result<f32> {
        if query.len() != self.values.len() {
            return Err(invalid("the vectors have different dimensions"));
        }
        let sum: f32 = self
            .values
            .iter()
            .zip(query)
            .map(|(value, q)| f32::from(*value) * q)
            .sum();
        Ok(sum * (self.scale / INT8_LIMIT))
    }

    /// The vector this approximates.
    pub fn dequantize(&self) -> Vec<f32> {
        let step = self.scale / INT8_LIMIT;
        self.values
            .iter()
            .map(|value| f32::from(*value) * step)
            .collect()
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "the rounded value lies within -127..=127 because no value exceeds the scale"
)]
fn to_int8(value: f32, scale: f32) -> i8 {
    if scale == 0.0 {
        return 0;
    }
    (value / scale * INT8_LIMIT)
        .round()
        .clamp(-INT8_LIMIT, INT8_LIMIT) as i8
}

fn invalid(reason: &'static str) -> Error {
    Error::InvalidVector { reason }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    fn max_error(original: &[f32], vector: &QuantizedVector) -> f32 {
        original
            .iter()
            .zip(vector.dequantize())
            .fold(0.0, |max, (a, b)| max.max((a - b).abs()))
    }

    #[test]
    fn the_largest_value_keeps_its_exact_magnitude() {
        let vector = QuantizedVector::quantize(&[0.5, -2.0, 1.0]).unwrap();
        assert_eq!(vector.values(), [32, -127, 64]);
        assert!((vector.dequantize()[1] + 2.0).abs() < 1e-6);
    }

    #[test]
    fn a_zero_vector_stays_zero() {
        let vector = QuantizedVector::quantize(&[0.0, 0.0]).unwrap();
        assert_eq!(vector.values(), [0, 0]);
        assert_eq!(vector.dequantize(), [0.0, 0.0]);
    }

    #[test]
    fn refuses_empty_and_non_finite_vectors() {
        for vector in [vec![], vec![1.0, f32::NAN], vec![f32::INFINITY]] {
            let err = QuantizedVector::quantize(&vector).unwrap_err();
            assert!(matches!(err, Error::InvalidVector { .. }), "{err}");
        }
    }

    #[test]
    fn refuses_bad_stored_parts() {
        for (scale, values) in [(1.0, vec![]), (-1.0, vec![1]), (f32::NAN, vec![1])] {
            let err = QuantizedVector::from_parts(scale, values).unwrap_err();
            assert!(matches!(err, Error::InvalidVector { .. }), "{err}");
        }
    }

    #[test]
    fn refuses_to_score_a_query_of_other_dimensions() {
        let vector = QuantizedVector::quantize(&[1.0, 2.0]).unwrap();
        vector.dot(&[1.0]).unwrap_err();
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn the_asymmetric_score_stays_within_its_error_bound(
            pair in (1_usize..128).prop_flat_map(|len| (
                prop::collection::vec(-1.0_f32..1.0, len),
                prop::collection::vec(-1.0_f32..1.0, len),
            )),
        ) {
            let (vector, query) = pair;
            let quantized = QuantizedVector::quantize(&vector).unwrap();
            let exact: f32 = vector.iter().zip(&query).map(|(a, b)| a * b).sum();
            let l1: f32 = query.iter().map(|q| q.abs()).sum();
            let bound = quantized.scale() / 254.0 * l1 + 1e-4;
            let score = quantized.dot(&query).unwrap();
            prop_assert!((score - exact).abs() <= bound, "{score} vs {exact}, bound {bound}");
        }

        #[test]
        fn round_trips_within_half_a_step(
            vector in prop::collection::vec(-100.0_f32..100.0, 1..64),
        ) {
            let quantized = QuantizedVector::quantize(&vector).unwrap();
            let step = quantized.scale() / INT8_LIMIT;
            prop_assert_eq!(quantized.len(), vector.len());
            prop_assert!(max_error(&vector, &quantized) <= step * 0.5 + 1e-5);
        }
    }
}
