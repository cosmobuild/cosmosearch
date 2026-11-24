# cosmosearch

Rust library for hybrid keyword and semantic search over large collections of files in
S3-compatible storage, built for RAG, with no database dependency. Layered indexes and
usage-based promotion keep search fast without the cost of indexing everything up front.

cosmosearch works with S3, any S3-compatible store, or a local folder. It keeps a small index
next to your data and reads it straight from the store at query time, and every result
points back to the exact bytes it came from in the original file.

> **Early release.** Expect the API to change before 1.0.

Maintained by [cosmobuild](https://cosmobuild.dev).

## How it works

You give cosmosearch two locations: where your files are, and where it may keep its index.
It lists the files, extracts their text, and writes a keyword index (BM25) for all of them.
Parts of the corpus that get searched a lot can also get vector embeddings, within a
spending limit you set, so semantic search costs nothing until it is actually needed.

A search reads only the index pieces it needs, using range requests against the store, and
caches them in memory and on disk. Results from keyword and vector search are merged and can
be reranked. Every search also reports its coverage: how many files it searched, and which
ones it could not search and why (not indexed yet, failed to extract, changed since).

If the index is lost or damaged, you delete it and build it again from the files.

## Features

- Keyword search everywhere, vector search where it pays off, merged and optionally reranked.
  Keyword search stems English words by default, or those of another of 18 languages, so
  `refund` also finds `refunds`.
- Exact phrases on request: with `Query::with_exact_phrases`, text in double quotes, such as
  `"ERR_CONN_42"` or `"refund policy"`, must appear in keyword results. With
  `Query::with_every_word`, keyword results hold every word instead of any.
- Text, Markdown, HTML, CSV, JSON, Word (`.docx`) and PDF.
- Embedding and reranking models that run locally on the CPU, or any OpenAI-compatible
  embedding API and `/rerank` API over HTTP.
- Results that cite a file, a byte range or page, and the file version they were read from.
  Reading a result back fails if the file has changed since.
- Search scopes (only these prefixes), filters on your own attributes (values, numbers and
  times, by value or range) and on modified time, boosts for matching or recent files, and
  budgets (this many shards, this much time).
- Safe concurrent writers: index updates are atomic commits, and two writers never
  overwrite each other.
- Everything beyond the core is an optional feature, so you compile only what you use.

## Quickstart

Add the library with the features you need. This example uses local files, indexing and
keyword search:

```sh
cargo add cosmosearch --features fs,indexer,lexical
cargo add tokio --features macros,rt-multi-thread
```

The same steps against a bucket need the `s3` feature in place of `fs`:

```sh
cargo add cosmosearch --features s3,indexer,lexical
```

```rust
use std::num::NonZeroUsize;

use cosmosearch::core::{ExtractLimits, SharedIndex};
use cosmosearch::index::{Extractors, Indexer, index_pending};
use cosmosearch::lexical::Bm25;
use cosmosearch::store::{ObjectStoreBackend, S3Options};
use cosmosearch::{
    CreateOutcome, DataStore, Engine, FileSource, MetadataStore, Query, SearchRequest,
};

// Where the files are, and where the index goes.
let backend = ObjectStoreBackend::s3(&S3Options::builder("example-bucket").build()?)?;
let data = DataStore::new(backend.clone(), "s3://example-bucket/docs/".parse()?)?;
let metadata = MetadataStore::new(backend, "s3://example-bucket/index/".parse()?)?;
let CreateOutcome::Created(handle) = metadata.create(data.location(), FileSource::Listing).await?
else {
    return Err("an index already exists there".into());
};

// Find the files and index them.
let indexer = Indexer::new(handle, data.clone())?;
indexer.sync_listing().await?;
let bm25 = SharedIndex::from(Bm25::new()?);
let batch = NonZeroUsize::new(100).ok_or("a batch holds at least one file")?;
index_pending(&indexer, &bm25, &Extractors::builtin()?, &ExtractLimits::default(), batch).await?;

// Search.
let engine = Engine::builder(data, metadata).index(bm25).open().await?;
let results = engine.search(&SearchRequest::new(Query::new("refund policy"))).await?;
for hit in results.hits() {
    println!("{:.3} {}", hit.score(), hit.source());
}
```

To run a complete example on your machine, with no bucket or Docker needed, clone this
repository and run:

```sh
cargo run -p cosmosearch --features fs,indexer,lexical --example quickstart
```

## Cargo features

None are on by default.

| Feature | Adds |
| --- | --- |
| `s3`, `fs` | S3-compatible storage, local folders |
| `indexer` | Indexing files (the `index` module) |
| `lexical` | Keyword search with BM25 |
| `vector` | Vector search |
| `pdf`, `docx`, `html`, `csv`, `json` | File formats beyond text and Markdown |
| `embed-local`, `rerank-local` | Models that run on the CPU; you provide the model files |
| `embed-http`, `rerank-http` | Models behind an HTTP API |
| `cache` | Memory and disk caches shared by every index in a process |
| `heat` | Usage tracking that decides which parts get vectors |
| `test-util` | For your tests: a fake embedder, reranker and clock, an in-memory store and a store that injects faults |

## Limits

- **Size.** A search without a scope covers every segment of the index and keeps each
  segment's file and chunk tables in memory. For large corpora, use search scopes.
- **Untrusted files.** Text extraction runs inside your process, both when indexing and when
  a search reads passages back (`read`, `expand`, snippets and reranking), and a crafted
  PDF can use a lot of memory and time. If you index files you don't trust, run both in
  separate processes; [SECURITY.md](https://github.com/cosmobuild/cosmosearch/blob/main/SECURITY.md)
  has the details.
- **Text encoding and scans.** Text must be UTF-8. Scanned PDF pages are not read; results
  report a file with some scanned pages as only partly searched, and a fully scanned file as
  failed.
- **Stores.** The store must support conditional writes and reads (`If-None-Match`,
  `If-Match`) and return strong ETags. cosmosearch checks for conditional writes when it
  creates an index and refuses a store without them. It is tested against RustFS and the
  local filesystem. In a local folder, a file's ETag comes from its inode, size and
  modification time, so change files by writing a new file and renaming it over the old
  one, not by rewriting them in place.
- **Reading is not scoped.** `read`, `expand`, `file` and `chunks` return any indexed
  passage or file whose ref or key they are given, whatever scope or filter the search had.
  If refs or keys come from your users, use `read_within`, `expand_within`, `file_within`
  and `chunks_within` with the scope and filter of that user's searches.
- **No "not found" answer.** A search with no results ends `inconclusive`, never "absent":
  it tells you what it searched, but it does not claim that no answer exists.

See [SECURITY.md](https://github.com/cosmobuild/cosmosearch/blob/main/SECURITY.md) for the
full list.

## Documentation

- [Architecture](https://github.com/cosmobuild/cosmosearch/blob/main/docs/architecture.md):
  how cosmosearch stores, indexes and searches.
- [Writing a plug-in](https://github.com/cosmobuild/cosmosearch/blob/main/docs/plugins.md):
  your own extractor, index, embedder or reranker.
- [API reference](https://docs.rs/cosmosearch) on docs.rs.
- [Contributing](https://github.com/cosmobuild/cosmosearch/blob/main/CONTRIBUTING.md) and the
  [changelog](https://github.com/cosmobuild/cosmosearch/blob/main/CHANGELOG.md).

## License

Licensed under either of [Apache License, Version 2.0](https://github.com/cosmobuild/cosmosearch/blob/main/LICENSE-APACHE)
or [MIT license](https://github.com/cosmobuild/cosmosearch/blob/main/LICENSE-MIT), at your
option.

Unless you explicitly state otherwise, any contribution intentionally submitted for
inclusion in this work by you, as defined in the Apache-2.0 license, shall be dual licensed
as above, without any additional terms or conditions.
