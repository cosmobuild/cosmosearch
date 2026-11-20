use std::fmt;
use std::sync::Arc;

use crate::{IndexId, SourceRef};

/// A cache of passage text read back from source files, shared by the readers of any number
/// of indexes.
///
/// A ref holds its file's ETag, so text kept under it is the text of that version of the
/// file. It is called on the search path, so it answers from memory and never waits on
/// storage.
pub trait PassageCache: Send + Sync + fmt::Debug {
    /// The text kept for the passage at `reference` in index `index`, if it is still held.
    fn get(&self, index: &IndexId, reference: &SourceRef) -> Option<Arc<str>>;

    /// Keeps `text` as the passage at `reference` in index `index`.
    fn put(&self, index: &IndexId, reference: &SourceRef, text: Arc<str>);
}

/// A shared handle on a passage cache.
#[derive(Debug, Clone)]
pub struct SharedPassages(Arc<dyn PassageCache>);

impl SharedPassages {
    /// Shares `cache`.
    pub fn new(cache: impl PassageCache + 'static) -> Self {
        SharedPassages(Arc::new(cache))
    }

    /// The cache.
    pub fn get(&self) -> &dyn PassageCache {
        self.0.as_ref()
    }
}
