# Changelog

All notable changes are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/); before 1.0, a minor version may break the API.

## [Unreleased]

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
