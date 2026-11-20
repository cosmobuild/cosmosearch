# cosmosearch-extract-pdf

PDF extractor for cosmosearch.

A plug-in for [cosmosearch](https://github.com/cosmobuild/cosmosearch). To use it, turn on the `pdf` feature of [`cosmosearch`](https://crates.io/crates/cosmosearch); its types are in `cosmosearch::extract`.

The PDF parser runs in your process, and a crafted file can make it use unbounded memory. If you index PDFs you don't trust, run extraction in a separate process.

## License

Licensed under either of [Apache License, Version 2.0](https://github.com/cosmobuild/cosmosearch/blob/main/LICENSE-APACHE)
or [MIT license](https://github.com/cosmobuild/cosmosearch/blob/main/LICENSE-MIT), at your
option.
