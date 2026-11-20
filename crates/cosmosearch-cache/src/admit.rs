use cosmosearch_core::{Admission, CachedFile, Placement};

/// Keeps every block in memory, so it moves to disk when evicted.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct AdmitAll;

impl AdmitAll {
    /// Admission that keeps every block.
    pub fn new() -> Self {
        AdmitAll
    }
}

impl Admission for AdmitAll {
    fn place(&self, _: &CachedFile, _: u64) -> Placement {
        Placement::Memory
    }
}
