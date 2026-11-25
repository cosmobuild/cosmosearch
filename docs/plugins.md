# Writing a plug-in

Everything outside cosmosearch's kernel is a plug-in: the file formats beyond text and
Markdown, BM25, vector search, the embedders and the rerankers. The built-in ones use the same
contracts you do, so you can add a format, swap a model or replace an index without forking
the library.

The contracts live in [`cosmosearch-core`](https://docs.rs/cosmosearch-core). A plug-in
crate depends on it, and on `cosmosearch-format` or `cosmosearch-store` only if it needs
them. If you're writing a plug-in inside your own program, the facade re-exports the same
types as `cosmosearch::core`.

| Contract | What it does | Runs |
| --- | --- | --- |
| `Extractor` | Turns one file into passages | Blocking |
| `Index` | Builds and searches one kind of index | Async |
| `Embedder` | Turns text into vectors | Blocking |
| `Reranker` | Scores passages against a query | Blocking |

All contracts work as trait objects, take and return only cosmosearch's own types, and
require `Send + Sync + Debug`. Blocking calls are run off the async runtime by their caller;
async calls run on it, so an `Index` moves heavy work to `spawn_blocking` itself.

## Extractors

An extractor reads the bytes of one file and returns its text as passages, each with a
locator that says where in the file it came from.

```rust
use cosmosearch::core::{
    ByteRange, ExtractFailure, ExtractLimits, Extraction, Extractor, ExtractorProfile, LimitKind,
    Locator, Passage,
};

/// Reads `.log` files, one passage per line.
#[derive(Debug)]
struct LogLines {
    profile: ExtractorProfile,
}

impl LogLines {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(LogLines { profile: ExtractorProfile::new("log-lines-1")? })
    }
}

impl Extractor for LogLines {
    fn profile(&self) -> &ExtractorProfile {
        &self.profile
    }

    fn extensions(&self) -> &[&str] {
        &["log"]
    }

    fn extract(&self, file: &[u8], limits: &ExtractLimits) -> Result<Extraction, ExtractFailure> {
        let text = std::str::from_utf8(file).map_err(|_| ExtractFailure::Malformed)?;
        let mut passages = Vec::new();
        let mut start = 0;
        for line in text.split_inclusive('\n') {
            let end = start + line.len();
            if !line.trim().is_empty() {
                if passages.len() == limits.chunks() {
                    return Err(ExtractFailure::LimitExceeded(LimitKind::Chunks));
                }
                let range = ByteRange::new(start as u64, end as u64)
                    .map_err(|_| ExtractFailure::BadLocator)?;
                passages.push(Passage::new(Locator::Bytes(range), Vec::new(), line.trim_end()));
            }
            start = end;
        }
        Ok(Extraction::new(passages))
    }
}
```

Register it in two places: with the indexer's extractors, so files get indexed, and with the
engine, so `read` and snippets can extract the file again.

```rust
let mut extractors = Extractors::builtin()?;
extractors.insert(SharedExtractor::new(LogLines::new()?));
// pass `extractors` to index_pending or to the extract jobs

let engine = Engine::builder(data, metadata)
    .extractor(LogLines::new()?)
    .index(bm25)
    .open()
    .await?;
```

An extractor added for an extension replaces any built-in one for that extension, so you can
also wrap a built-in extractor, for example to send PDFs without text to an OCR service.

An extractor can also name the media types it reads, such as `text/x-log`, by implementing
`media_types`. A host that knows a file's type passes it with `FileUpsert::media_type`, and
the extractor that reads that type is chosen ahead of the key's extension. Without a media
type, or with one no extractor reads, the extension decides.

Things to get right:

- **The profile id** is recorded with every segment the extractor builds. Change it (for
  example from `log-lines-1` to `log-lines-2`) whenever the passages it produces change,
  so indexes know which segments to rebuild.
- **Locators must come from the file itself:** byte ranges, pages, elements, rows or values.
  The same file must always give the same locators, because saved refs depend on them.
- **Respect the limits** you're given, and fail with `LimitExceeded` rather than doing
  unbounded work. Panics are caught and reported as a failed file, but out-of-memory is not.
- **Count what you couldn't read.** If parts of the file have no usable text (a scanned page,
  say), return `Extraction::new(passages).with_unread_parts(n)`. Coverage then reports the
  file as only partly searched. If no part could be read, the file fails as unreadable. A
  file with no text at all and nothing unread should fail with `NoText`, which marks it
  unsupported rather than failed.
- **Never put document text in errors or `Debug` output.** They end up in logs.

## Indexes

An index plug-in builds index files for a segment and searches them later. BM25
(`cosmosearch-lexical`) and IVF (`cosmosearch-vector`) are both built this way.

```rust
#[async_trait]
pub trait Index: Send + Sync + fmt::Debug {
    fn kind(&self) -> &IndexKind;
    fn level(&self) -> Level;
    fn model_id(&self) -> Option<&ModelId> { None }
    fn search_settings(&self) -> Vec<u64> { Vec::new() }

    async fn build(&self, segment: &SegmentId, chunks: &[Chunk]) -> Result<Built, IndexError>;

    async fn search(
        &self,
        source: &SegmentSource,
        segments: &[SegmentEntry],
        allowed: &AllowedRows,
        query: &Query,
        limit: usize,
    ) -> Result<Hits, IndexError>;
}
```

- **`build`** gets every chunk of a new segment and returns the files to store, as names and
  bytes. A chunk's row is its position in the slice; hits refer to chunks by row. The
  indexer writes the files and records them in the manifest; your plug-in never writes to
  the store itself.
- **`search`** gets the segments it built that the search may touch, and `AllowedRows`, the
  rows the scope and filters allow. Score only those rows, so filtered-out chunks can never
  take a place in the results. Read your files through `source`, by byte range; it goes
  through the shared cache when there is one. A keyword index should also honour the
  query's `exact_phrases` (text in double quotes is a phrase) and `every_word` (a passage
  must hold every word); other kinds may ignore them.
- **`kind`, `level` and `model_id`** are recorded with every segment. Opening an index whose
  segments need a plug-in the engine wasn't given fails with an error naming it.
- **`search_settings`** names every setting that changes which hits a search returns, such
  as IVF's probes. A shared result cache keeps results apart by them, so leave out settings
  that only change speed.
- **The level** decides how the plug-in is used. An L1 plug-in indexes new files, so every
  file has a keyword segment. A plug-in above L1 builds only through promotion, and the
  indexer refuses it for new files.

Register an index with both the indexer (through the jobs, `index_pending` or a promotion)
and the engine (`EngineBuilder::index`). Wrap it in a `SharedIndex` to use one instance, and
its caches, for both.

## Embedders

```rust
pub trait Embedder: Send + Sync + fmt::Debug {
    fn identity(&self) -> &EmbeddingIdentity;
    fn embed_documents(&self, texts: &[&str]) -> Result<Embeddings, EmbedError>;
    fn embed_query(&self, text: &str) -> Result<Embeddings, EmbedError>;
}
```

The **identity** describes the vector space: model, revision, dimensions, normalisation,
metric and preprocessing. Vectors are stored and reused under it, so two embedders with the
same identity must produce interchangeable vectors. Change any part of it and the vectors
are treated as new.

`embed_documents` is called with large batches during promotion; `embed_query` once per
search, so it should be fast. Models that want a prefix on queries (or on documents) apply
it here. Return token counts in the `Embeddings` usage so that searches and promotions can
report what they spent.

Pass an embedder to the vector index: `Ivf::new(SharedEmbedder::new(embedder))`.

## Rerankers

```rust
pub trait Reranker: Send + Sync + fmt::Debug {
    fn identity(&self) -> &RerankerIdentity;
    fn rerank(&self, query: &str, passages: &[&str]) -> Result<RerankScores, RerankError>;
}
```

Return one score per passage, in the order given; higher is better. If reranking fails, the
search keeps the fused order and reports the failure, so an error here never fails a search.

Set it on the engine with `EngineBuilder::reranker`. It rescores the top 20 hits unless you
set another shortlist size with `rerank_shortlist`.

## Testing a plug-in

`cosmosearch-core` has a `test-util` feature with a fake embedder and reranker, and helpers
that feed an extractor damaged input. The built-in plug-ins' tests are good examples: every
extractor is tested with each byte of a sample file replaced and with every prefix of it,
and must fail cleanly rather than panic or hang.
