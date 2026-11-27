use std::collections::BTreeMap;
use std::sync::Arc;

use cosmosearch_core::{IndexId, Manifest, ShardEntry, SharedPassages, SourceRef};

use crate::SearchHit;
use crate::search::ChunkAt;
use futures_util::{StreamExt, stream};
use tracing::warn;

use crate::read::{self, Sources};

/// Files or groups of stored text read at once; a search reads few, so this rarely binds.
const READ_CONCURRENCY: usize = 8;

/// The passage text one search has read, and where it finds more: a passage cache, if the
/// reader has one, and then the source files, each read and extracted once for all of its
/// passages asked for together.
pub(crate) struct Passages<'a> {
    sources: Sources<'a>,
    manifest: &'a Manifest,
    index: &'a IndexId,
    shared: Option<&'a SharedPassages>,
    held: BTreeMap<String, Arc<str>>,
}

impl<'a> Passages<'a> {
    pub(crate) fn new(
        sources: Sources<'a>,
        manifest: &'a Manifest,
        index: &'a IndexId,
        shared: Option<&'a SharedPassages>,
    ) -> Self {
        Passages {
            sources,
            manifest,
            index,
            shared,
            held: BTreeMap::new(),
        }
    }

    /// The text of each of `hits`, or none where it could not be had. A hit whose chunk has
    /// stored text takes it; passages this search or the cache already holds cost nothing;
    /// for the rest at most `max_files` files are read, those of the first hits first.
    pub(crate) async fn texts(
        &mut self,
        hits: &[&SearchHit],
        max_files: usize,
    ) -> Vec<Option<Arc<str>>> {
        let references: Vec<&SourceRef> = hits.iter().map(|hit| hit.source()).collect();
        let references = references.as_slice();
        let mut texts: Vec<Option<Arc<str>>> = Vec::with_capacity(references.len());
        let mut files: Vec<Vec<usize>> = Vec::new();
        let looks: Vec<_> = hits.iter().map(|hit| self.own(hit)).collect();
        let stored: Vec<Option<Arc<str>>> = stream::iter(looks)
            .buffered(READ_CONCURRENCY)
            .collect()
            .await;
        for (position, (reference, own)) in references.iter().zip(stored).enumerate() {
            if own.is_some() {
                texts.push(own);
                continue;
            }
            let known = self.known(reference);
            if known.is_none() {
                let same_file = |positions: &&mut Vec<usize>| {
                    positions
                        .first()
                        .and_then(|first| references.get(*first))
                        .is_some_and(|first| {
                            first.location().key() == reference.location().key()
                                && first.etag() == reference.etag()
                        })
                };
                match files.iter_mut().find(same_file) {
                    Some(positions) => positions.push(position),
                    None => files.push(vec![position]),
                }
            }
            texts.push(known);
        }
        files.truncate(max_files);
        let (sources, manifest) = (self.sources, self.manifest);
        let reading = files.into_iter().map(|positions| async move {
            let of_file: Vec<&SourceRef> = positions
                .iter()
                .filter_map(|position| references.get(*position).copied())
                .collect();
            let found = read::passages(sources, manifest, &of_file).await;
            (positions, found)
        });
        let mut reads = stream::iter(reading).buffer_unordered(READ_CONCURRENCY);
        while let Some((positions, found)) = reads.next().await {
            let first = positions.first().and_then(|at| references.get(*at));
            let found = match found {
                Ok(found) => found,
                Err(error) => {
                    let file = first.map_or("", |reference| reference.location().key());
                    warn!(file, %error, "a file could not be read for its passages");
                    continue;
                }
            };
            for (position, text) in positions.into_iter().zip(found) {
                let (Some(text), Some(reference)) = (text, references.get(position)) else {
                    continue;
                };
                let text: Arc<str> = Arc::from(text);
                if let Some(shared) = self.shared {
                    shared.get().put(self.index, reference, Arc::clone(&text));
                }
                self.held.insert(reference.to_string(), Arc::clone(&text));
                if let Some(slot) = texts.get_mut(position) {
                    *slot = Some(text);
                }
            }
        }
        texts
    }

    /// The stored text of `hit`'s own chunk, if its segment stores one.
    async fn own(&self, hit: &SearchHit) -> Option<Arc<str>> {
        match hit.chunk() {
            Some(chunk) => self.stored(chunk).await,
            None => None,
        }
    }

    /// The stored text of the chunk at `chunk`, if its segment stores one.
    async fn stored(&self, chunk: &ChunkAt) -> Option<Arc<str>> {
        let segment = self
            .manifest
            .shards()
            .values()
            .flat_map(ShardEntry::segments)
            .find(|segment| segment.id() == &chunk.segment)?;
        let loader = self.sources.loader;
        let found = async {
            let Some(index) = loader.text_index(segment).await? else {
                return Ok(None);
            };
            loader.stored_text(segment, &index, chunk.row).await
        };
        match found.await {
            Ok(text) => text,
            Err(error) => {
                let error: crate::Error = error;
                warn!(segment = %chunk.segment, %error, "a chunk's stored text could not be read");
                None
            }
        }
    }

    fn known(&mut self, reference: &SourceRef) -> Option<Arc<str>> {
        let key = reference.to_string();
        if let Some(text) = self.held.get(&key) {
            return Some(Arc::clone(text));
        }
        let text = self.shared?.get().get(self.index, reference)?;
        self.held.insert(key, Arc::clone(&text));
        Some(text)
    }
}
