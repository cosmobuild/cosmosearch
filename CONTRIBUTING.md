# Contributing

Thanks for helping with cosmosearch. This page covers how to build and test it, how to
propose a change, and the conventions the code follows.

## Getting set up

The toolchain is pinned in `rust-toolchain.toml`, so `rustup` installs the right version
on first build. The full check also needs:

- [cargo-nextest](https://nexte.st/), [cargo-deny](https://embarkstudios.github.io/cargo-deny/),
  [cargo-machete](https://github.com/bnjbvr/cargo-machete) and
  [cargo-hack](https://github.com/taiki-e/cargo-hack)
- [gitleaks](https://github.com/gitleaks/gitleaks), for the commit hook
- Docker, for the local S3-compatible store the integration tests use

Turn on the git hooks once. They scan every commit for secrets and run `fmt` and `clippy`:

```sh
git config core.hooksPath .githooks
```

## Building and testing

```sh
make check
```

This runs everything CI runs: formatting, clippy with warnings as errors, all tests, doc
tests, the docs build, the licence and advisory check, unused dependencies, every feature on
its own, and a check that the minimal build pulls in no plug-ins. A change is ready when it
passes.

Each step is also a target of its own: `make fmt`, `make lint`, `make test`, `make doc`,
`make deny`, `make unused`, `make features` and `make minimal`. `make test` starts the local
store (RustFS in Docker, on localhost only, with fixed test credentials); `make store-down`
stops it.

Tests that need the local model files are ignored by default. To run them, put the model
folders (`mdbr-leaf-ir`, `ettin-reranker-17m-v1`) in one folder, set
`COSMOSEARCH_TEST_MODELS` to it, and run `cargo nextest run --run-ignored all`.

To see the library work without Docker:

```sh
cargo run -p cosmosearch --features fs,indexer,lexical --example quickstart
```

## Proposing a change

- For anything bigger than a fix, open an issue first so we can agree on the approach.
- A new dependency, or a new feature of an existing one, needs agreement in the issue
  before the code.
- Keep a pull request to one purpose, with tests that show it works.
- Add user-visible changes to the Unreleased section of `CHANGELOG.md`.
- Update `docs/` in the same pull request if your change makes them wrong.
- Commits are signed, with a short subject line and a plain body.

Open the pull request against `main`. CI runs `make check`; a change can merge only after it
passes and a maintainer has reviewed it.

## How the code is written

[docs/architecture.md](docs/architecture.md) explains how the pieces fit together. Most of
the rules below are enforced by the lints in `Cargo.toml` and by `make check`; the rest are
checked in review.

**Structure**

- The kernel (`query`, `indexer` and what they use) never depends on a plug-in. Plug-ins
  depend on `core`, and on `format` and `store` only if they need them, never on each other.
- Plug-in contracts live in `core`, work as trait objects and use only our own types.
  Third-party types stay out of public APIs.
- Features are additive: turning one on never changes behaviour elsewhere.
- Public enums and option structs are `#[non_exhaustive]`, and structs keep their fields
  private.

**Errors and panics**

- Each crate has one public error enum, built with `thiserror`, that keeps the cause.
- A panic means a bug in cosmosearch, never bad input, a bad file or a failed request.
  `unwrap`, `expect` and indexing that can panic are linted.
- `unsafe` is forbidden.
- Every public function that can fail documents its errors under `# Errors`.

**Async and resources**

- Tokio is the runtime, and the library never starts one. Blocking work goes through
  `spawn_blocking`.
- Concurrency is bounded, and so is every in-memory structure. If something can grow with
  the input, it needs a limit.
- Every future must stay correct if it is dropped at any `.await`.

**Privacy**

- Logs, traces, errors and `Debug` output never contain document text, query text or
  credentials. Keep no third-party error whose message can quote its input; keep what it
  says about where it failed instead.

**Tests**

- Prefer unit tests, with I/O kept out of the logic so most of it runs without a store.
- Use the in-memory backend for contract tests. Use the local store only for what a real
  store shows: conditional writes, range reads and listing.
- Use property tests (`proptest`) for parsers and format round trips.
- Tests are independent: each uses its own key prefix, with no `sleep` and no shared state.

**Style**

- Plain `cargo fmt`, no configuration.
- No comments in code unless they say something important the code can't. Public items have
  short docs.

**AI-assisted contributions** are welcome and follow the same rules. You are responsible
for every line you submit: read it, test it, and make sure it does what the pull request
says.

## Minimum Rust version

`rust-version` in `Cargo.toml` states the oldest Rust that builds cosmosearch. It can rise in
any release.

## Releases

Maintainers release all crates together at the same version. A release moves the
changelog's Unreleased entries under the new version, passes `cargo publish --workspace
--dry-run`, is published to crates.io and is marked with a signed tag.

## Licence

cosmosearch is licensed under either of the Apache License, Version 2.0 or the MIT licence,
at your option. Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0 licence, shall be
dual licensed as above, without any additional terms or conditions.
