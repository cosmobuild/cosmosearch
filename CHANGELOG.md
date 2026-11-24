# Changelog

All notable changes are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/); before 1.0, a minor version may break the API.

## [Unreleased]

### Added

- `ExtractLimits::decoded_bytes` and `LimitKind::DecodedBytes`: the most bytes the
  compressed streams of a document decode to, in all, 128 MiB by default. The PDF extractor
  checks it before parsing, so a PDF that inflates far beyond its size fails with this limit
  instead of exhausting memory.
- `Engine::read_within` and `Engine::expand_within` (and the same on `Reader`): reads that
  take a scope and a filter, and refuse a file outside them as not in the index, so the
  rules that limit a caller's searches also limit what the caller reads.
- `FileUpsert::media_type` names a file's format, which chooses its extractor ahead of the
  key's extension, so files without an extension, or with a misleading one, can be indexed.
  Extractors name the media types they read with `Extractor::media_types`, and the built-in
  ones name theirs. Changing a file's media type indexes it again.
- Attributes can be numbers (`AttributeValue::Number`) and times (`AttributeValue::Time`,
  a `Timestamp`) as well as strings. `Filter::numbers` and `Filter::times` keep files whose
  attribute is within a range, and `Filter::modified` keeps files modified within a range.
  The file table records each file's modified time from the listing or a HEAD, or from
  `FileUpsert::modified`.
- `Bm25::stemmed(Language)` builds a keyword index that reduces words to their stems in one
  of 18 languages, so a search for `refund` also finds `refunds` and `refunded`. Its kind,
  such as `bm25-stem-en`, names the language, so a plug-in searches only the segments built
  the same way.
- Keyword search matches a file's name: its display name and the last part of its key
  without the extension, on the first chunk of the file and with the same weight as the
  text, unless `Bm25::name_weight` sets another weight. Words in quotes still match only
  the text. A display name changed through `FileChange::Attributes` is matched once the
  file is indexed again, for example after a retry.
- `SearchHit::display_name` gives the name the host gave the hit's file, and
  `SearchHit::list_ranks` gives the hit's rank in each index kind's list, beside
  `list_scores`, so a host can see why fusion placed it.
- `Engine::file` gives a file as the index holds it, with its status and why it is not
  ready, without the indexer, and `Engine::file_within` does so only within a scope and
  filter. `Engine::chunks` reads a file's chunks in reading order, page by page, without a
  search hit, and `Engine::chunks_within` does so only within a scope and filter. The same
  calls are on `Reader`. `FileEntry`, `FileStatus` and `FileReason` are exported at the
  crate root.
- `Query::with_every_word` asks keyword search for passages that hold every word outside
  quotes, instead of any of them; exact phrases are required either way. The result cache
  keeps the two modes apart.
- `SearchRequest::boost` with `Boost::matching(filter, weight)` or
  `Boost::recency(now, half_life, weight)` moves up the candidates whose file passes a
  filter, or by how recently it was modified, with a reranker or without one. With a weight
  of 1, a qualifying candidate counts as ranked twice as high, and `SearchHit::boost` gives
  each hit's multiplier. A boost never adds a candidate, and the result cache keys on it.
  A weight below zero or not finite fails with `Error::InvalidBoost`.

### Changed

- The file table format changed, so indexes built by 0.1 must be built again.
- `Bm25::new` stems English words, with the kind `bm25-stem-en`. `Bm25::plain` keeps the
  behaviour of 0.1, for text such as code or logs.
- `Filter::matches` also takes the file's modified time.
- `Extractors::for_file` replaces `Extractors::for_key` and takes the file's media type.
- A PDF that the decoded-bytes check cannot load fails as malformed, even where the parser
  could have repaired it.
- A search whose scope names a prefix outside the data location fails with
  `Error::ScopeOutsideData` instead of finding nothing. `Indexer::plan_embedding` and
  `Indexer::file` refuse keys outside the data location, as `Indexer::apply` already did.
  Keys are whole object keys in the store; the docs of `Scope` and `FileUpsert::new` say so.
- The README lists the `test-util` feature, and the architecture guide says how key layout
  and shard size decide what a promotion embeds.

## [0.1.0] - 2026-11-21

The first release.

### Added

- Hybrid keyword and semantic search over files in S3, S3-compatible stores and local folders.
- Index creation, opening and deletion, with atomic commits that are safe with several
  writers.
- Indexing from a listing or from files you name, with attributes, filters, file removal and
  garbage collection.
- Extractors for text, Markdown, HTML, CSV, JSON, Word (`.docx`) and PDF, with limits on
  every file.
- Keyword search with BM25, including exact phrases in double quotes on request, vector
  search with IVF, result fusion and optional reranking.
- Embedders and rerankers that run locally on the CPU or call HTTP APIs.
- Search scopes, time and shard budgets, and a coverage report on every search.
- Usage tracking (heat) that decides which shards get vectors, within a spending limit.
- Memory and disk caches for index files, results and passages, and snippets.
- `index::index_pending` to run all indexing steps in one process, and a `quickstart`
  example.

Requires Rust 1.99 or later.

[Unreleased]: https://github.com/cosmobuild/cosmosearch/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/cosmobuild/cosmosearch/releases/tag/v0.1.0
