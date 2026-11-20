# cosmosearch-http

Embeddings and reranking over HTTP APIs for cosmosearch.

A plug-in for [cosmosearch](https://github.com/cosmobuild/cosmosearch). To use it, turn on the `embed-http` or `rerank-http` feature of [`cosmosearch`](https://crates.io/crates/cosmosearch); its types are in `cosmosearch::http`.

Embeddings use any OpenAI-compatible API; reranking uses the common `POST /rerank` shape. Requests are retried with backoff.

## License

Licensed under either of [Apache License, Version 2.0](https://github.com/cosmobuild/cosmosearch/blob/main/LICENSE-APACHE)
or [MIT license](https://github.com/cosmobuild/cosmosearch/blob/main/LICENSE-MIT), at your
option.
