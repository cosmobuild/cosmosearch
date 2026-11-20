# cosmosearch-store

Storage for cosmosearch: backends, index lifecycle and atomic commits.

Part of [cosmosearch](https://github.com/cosmobuild/cosmosearch). Storage backends (S3-compatible, local filesystem, in-memory), creating and destroying indexes, committing manifests, and garbage collection.

Most programs should depend on [`cosmosearch`](https://crates.io/crates/cosmosearch) instead, which re-exports it as `cosmosearch::store`. The S3 and filesystem backends are behind the `s3` and `fs` features.

## License

Licensed under either of [Apache License, Version 2.0](https://github.com/cosmobuild/cosmosearch/blob/main/LICENSE-APACHE)
or [MIT license](https://github.com/cosmobuild/cosmosearch/blob/main/LICENSE-MIT), at your
option.
