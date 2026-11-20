use std::cmp::Ordering;

use crate::{Error, Result};

/// The centres of a set of clusters, as unit vectors, kept in row-major order.
#[derive(Debug, Clone, PartialEq)]
pub struct Centroids {
    dimensions: usize,
    values: Vec<f32>,
}

impl Centroids {
    /// Centroids read back from storage: `values` holds `values.len() / dimensions` rows.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Clustering`] if `dimensions` is zero, `values` is empty or does not
    /// hold a whole number of rows, or a value is not finite.
    pub fn from_parts(dimensions: usize, values: Vec<f32>) -> Result<Self> {
        if dimensions == 0 || values.is_empty() {
            return Err(invalid(
                "there must be at least one centroid of one dimension",
            ));
        }
        if values.len().checked_rem(dimensions) != Some(0) {
            return Err(invalid("the values are not a whole number of rows"));
        }
        if !values.iter().all(|value| value.is_finite()) {
            return Err(invalid("a value is not finite"));
        }
        Ok(Centroids { dimensions, values })
    }

    pub(crate) fn from_trusted(dimensions: usize, values: Vec<f32>) -> Self {
        Centroids { dimensions, values }
    }

    /// The number of dimensions of each centroid.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// The number of centroids.
    pub fn len(&self) -> usize {
        self.values.chunks_exact(self.dimensions).len()
    }

    /// Always `false`: there is at least one centroid.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// All values, one row of `dimensions` per centroid.
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// The centroid at `index`.
    pub fn get(&self, index: usize) -> Option<&[f32]> {
        self.values.chunks_exact(self.dimensions).nth(index)
    }

    pub(crate) fn get_mut(&mut self, index: usize) -> Option<&mut [f32]> {
        self.values.chunks_exact_mut(self.dimensions).nth(index)
    }

    /// The index of the centroid with the largest dot product with `vector`, the nearest by
    /// angle. Ties go to the lowest index.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Clustering`] if `vector` has other dimensions or a value that is not
    /// finite.
    pub fn assign(&self, vector: &[f32]) -> Result<usize> {
        self.check(vector)?;
        Ok(self.best(vector).0)
    }

    /// The indices of the `count` nearest centroids, nearest first, fewer if there are fewer
    /// centroids.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Clustering`] if `vector` has other dimensions or a value that is not
    /// finite.
    pub fn nearest(&self, vector: &[f32], count: usize) -> Result<Vec<usize>> {
        self.check(vector)?;
        let mut scored: Vec<(usize, f32)> = self
            .values
            .chunks_exact(self.dimensions)
            .map(|row| dot(row, vector))
            .enumerate()
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        Ok(scored
            .into_iter()
            .take(count)
            .map(|(index, _)| index)
            .collect())
    }

    /// An order of all centroids in which near ones sit next to each other: start at the
    /// first and always step to the nearest centroid not yet visited.
    pub fn adjacent_order(&self) -> Vec<usize> {
        let mut remaining: Vec<usize> = (1..self.len()).collect();
        let mut order = vec![0];
        let mut current = self.get(0);
        while let Some(from) = current {
            let next = remaining
                .iter()
                .enumerate()
                .filter_map(|(position, index)| {
                    self.get(*index).map(|row| (position, dot(from, row)))
                })
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(position, _)| remaining.swap_remove(position));
            let Some(index) = next else {
                break;
            };
            order.push(index);
            current = self.get(index);
        }
        order
    }

    pub(crate) fn best(&self, vector: &[f32]) -> (usize, f32) {
        let mut best = (0, f32::NEG_INFINITY);
        for (index, row) in self.values.chunks_exact(self.dimensions).enumerate() {
            let score = dot(row, vector);
            if score.total_cmp(&best.1) == Ordering::Greater {
                best = (index, score);
            }
        }
        best
    }

    fn check(&self, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dimensions {
            return Err(invalid(
                "the vector has other dimensions than the centroids",
            ));
        }
        if !vector.iter().all(|value| value.is_finite()) {
            return Err(invalid("a value is not finite"));
        }
        Ok(())
    }
}

pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn invalid(reason: &'static str) -> Error {
    Error::Clustering { reason }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn axes() -> Centroids {
        Centroids::from_parts(2, vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0]).unwrap()
    }

    #[test]
    fn refuses_bad_stored_parts() {
        for (dimensions, values) in [
            (0, vec![1.0]),
            (2, vec![]),
            (2, vec![1.0, 0.0, 1.0]),
            (1, vec![f32::NAN]),
        ] {
            let err = Centroids::from_parts(dimensions, values).unwrap_err();
            assert!(matches!(err, Error::Clustering { .. }), "{err}");
        }
    }

    #[test]
    fn assigns_the_nearest_by_angle_and_ranks_the_rest() {
        let centroids = axes();
        assert_eq!(centroids.len(), 3);
        assert_eq!(centroids.assign(&[0.2, 5.0]).unwrap(), 1);
        assert_eq!(centroids.assign(&[-3.0, 1.0]).unwrap(), 2);
        assert_eq!(centroids.nearest(&[0.9, 0.5], 2).unwrap(), [0, 1]);
        assert_eq!(centroids.nearest(&[0.9, 0.5], 9).unwrap().len(), 3);
        centroids.assign(&[1.0]).unwrap_err();
        centroids.assign(&[1.0, f32::NAN]).unwrap_err();
    }

    #[test]
    fn a_tie_goes_to_the_lowest_index() {
        let centroids = Centroids::from_parts(2, vec![1.0, 0.0, 1.0, 0.0]).unwrap();
        assert_eq!(centroids.assign(&[1.0, 0.0]).unwrap(), 0);
    }

    #[test]
    fn near_centroids_are_placed_next_to_each_other() {
        let centroids =
            Centroids::from_parts(2, vec![1.0, 0.0, -1.0, 0.0, 0.9, 0.1, -0.9, 0.1]).unwrap();
        assert_eq!(centroids.adjacent_order(), [0, 2, 3, 1]);
    }
}
