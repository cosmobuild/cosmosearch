.PHONY: check fmt lint test doc deny unused features minimal store store-down

S3_ENDPOINT := http://127.0.0.1:9000
STORE_ACCESS_KEY := cosmosearch-local
STORE_SECRET_KEY := cosmosearch-local-only
PLUGINS := cosmosearch-(extract-.*|lexical|vector|embed-.*|rerank-.*|cache|heat)
LIBRARIES := object_store|reqwest|tantivy|pdf-inspector|lopdf|zip|quick-xml|html5gum|html5ever|scraper|csv|ort|candle-core|foyer

check: fmt lint test doc deny unused features minimal
	@echo "check: all passed"

fmt:
	cargo fmt --all --check

lint:
	cargo clippy --workspace --all-targets --all-features --locked -- -D warnings

test: store
	COSMOSEARCH_TEST_S3_ENDPOINT=$(S3_ENDPOINT) cargo nextest run --workspace --all-features --locked --no-tests=warn
	cargo test --workspace --doc --all-features --locked

doc:
	RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features --locked

deny:
	cargo deny --locked check

unused:
	cargo machete

features:
	cargo hack check --workspace --each-feature --no-dev-deps

minimal:
	@tree=$$(cargo tree -p cosmosearch --edges normal --no-default-features --locked --prefix none --format "{p}") \
		|| { echo "minimal: cargo tree failed"; exit 1; }; \
	found=$$(printf '%s\n' "$$tree" | grep -E "^($(PLUGINS)|$(LIBRARIES)) " | sort -u); \
	if [ -n "$$found" ]; then echo "minimal: the minimal build pulls in plug-in dependencies:"; echo "$$found"; exit 1; fi; \
	echo "minimal: the minimal build has no plug-in dependencies"

store:
	docker compose up -d --wait
	@for i in $$(seq 30); do \
		code=$$(curl -s -o /dev/null -w "%{http_code}" -X PUT --aws-sigv4 "aws:amz:us-east-1:s3" \
			--user "$(STORE_ACCESS_KEY):$(STORE_SECRET_KEY)" "$(S3_ENDPOINT)/cosmosearch-test" || true); \
		case $$code in 200|409) exit 0;; esac; sleep 1; \
	done; echo "store: the local store did not accept requests (last status $$code)"; exit 1

store-down:
	docker compose down
