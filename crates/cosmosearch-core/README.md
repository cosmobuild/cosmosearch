# cosmosearch-core

Shared types and plug-in contracts for cosmosearch.

Part of [cosmosearch](https://github.com/cosmobuild/cosmosearch). The types every cosmosearch crate shares (refs, ids, the manifest, queries) and the contracts plug-ins implement: `Extractor`, `Index`, `Embedder`, `Reranker` and the cache and heat traits.

Depend on this crate to write a plug-in; see [Writing a plug-in](https://github.com/cosmobuild/cosmosearch/blob/main/docs/plugins.md). Most programs should depend on [`cosmosearch`](https://crates.io/crates/cosmosearch) instead, which re-exports it as `cosmosearch::core`.

## License

Licensed under either of [Apache License, Version 2.0](https://github.com/cosmobuild/cosmosearch/blob/main/LICENSE-APACHE)
or [MIT license](https://github.com/cosmobuild/cosmosearch/blob/main/LICENSE-MIT), at your
option.
