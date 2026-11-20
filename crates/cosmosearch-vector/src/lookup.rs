use std::collections::BTreeMap;

use cosmosearch_core::{ChunkHash, Usage};

/// What a lookup found: vectors for some hashes, and the hashes to embed.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Lookup {
    found: BTreeMap<ChunkHash, Vec<f32>>,
    missing: Vec<ChunkHash>,
    usage: Usage,
}

impl Lookup {
    pub(crate) fn new(
        found: BTreeMap<ChunkHash, Vec<f32>>,
        missing: Vec<ChunkHash>,
        usage: Usage,
    ) -> Self {
        Lookup {
            found,
            missing,
            usage,
        }
    }

    /// The vectors found, by hash, as close to the stored ones as int8 allows.
    pub fn found(&self) -> &BTreeMap<ChunkHash, Vec<f32>> {
        &self.found
    }

    /// Takes the vectors found.
    pub fn into_found(self) -> BTreeMap<ChunkHash, Vec<f32>> {
        self.found
    }

    /// The hashes asked for that have no vector, each once, in the order asked.
    pub fn missing(&self) -> &[ChunkHash] {
        &self.missing
    }

    /// The reads the lookup made.
    pub fn usage(&self) -> Usage {
        self.usage
    }
}
