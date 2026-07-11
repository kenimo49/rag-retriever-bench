# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — 2026-07-11

Initial release. Benchmark harness for RAG retrieval backends with seven
in-tree implementations and a fixed methodology (deterministic metrics,
aligned HNSW parameters, silent-full-scan defense via `self_check`).

### Added

- Pluggable `BaseRetriever` interface: `setup` → `load` → `build_index` →
  `search` → `close` lifecycle.
- Seven backend implementations:
  - **pgvector** (PostgreSQL), HNSW, EXPLAIN-based self_check
  - **ClickHouse**, `vector_similarity` HNSW (two `index_granularity`
    variants) + brute-force full scan, EXPLAIN-based self_check
  - **Qdrant**, HNSW with gRPC loading, server-reported self_check
  - **Weaviate**, HNSW, config-based self_check
  - **Milvus** (standalone), HNSW, `describe_index`-based self_check
  - **Chroma**, embedded HNSW, config-only self_check
  - **LanceDB**, embedded IVF_HNSW_SQ (8-bit scalar quantization),
    `list_indices`-based self_check
- Deterministic metrics: recall@k, hit@k, MRR@k, nDCG@k (binary qrels),
  duplicate-docid deduplication, standard uncapped recall definition.
- Client-side latency measurement with p50 / p95 / p99 percentiles and
  configurable warmup queries.
- Server / embedded backend separation in the report (two tables) to
  avoid comparing in-process latency against network-hop latency.
- MIRACL-ja dataset loader with corpus sampling that always includes
  every positive passage for the eval query set.
- `--corpus-size` CLI flag for scale sweeps (10k → 100k → …).
- OpenAI `text-embedding-3-small` embedding with local cache — a 100k
  corpus costs ~$0.30 to embed, once.
- Markdown + JSONL report writer, one row per backend, per-backend
  `describe()` block, per-backend `self_check` block.
- `docker-compose.yml` for the five server backends (pgvector,
  ClickHouse, Qdrant, Weaviate, Milvus).
- Integration tests under a dedicated `rrb_it_*` namespace so live tests
  don't touch bench data; `RRB_IT_HOST` env var to point at a remote
  benchmarking machine.
- End-to-end pipeline test that runs the full harness against a
  fake-backend fixture.
- CI on GitHub Actions (ruff + pytest).
- Documentation:
  [`docs/adding-a-backend.md`](docs/adding-a-backend.md) (interface
  walkthrough + fair-comparison conventions),
  [`docs/methodology.md`](docs/methodology.md) (design notes and
  citations).
- Backend-request issue template
  ([`.github/ISSUE_TEMPLATE/backend-request.yml`](.github/ISSUE_TEMPLATE/backend-request.yml)).

### Fixed (during pre-0.1 development)

Three backends shipped a silent full-scan degradation path. All three
were caught by `self_check` before the harness was published:

- **ClickHouse** would silently fall back to brute force if the HNSW
  index build was skipped — the fix defers the build to `OPTIMIZE FINAL`
  and extends the client timeout so it actually completes.
- **Qdrant** could ignore an HNSW build request when
  `indexing_threshold_kb` was above the corpus size. `build_index` now
  forces the build regardless.
- **Milvus** loaded a stale snapshot that missed sealed segments after
  flush, so queries returned partial results. `refresh_load` was added
  to force a full re-load after ingestion.

Also fixed during the pre-release run: pgvector `shm_size` raised for
parallel HNSW build at 100k scale; Milvus explicit schema to avoid an
AUTOINDEX trap; Weaviate port offset; LanceDB migration to the unified
index API (deprecated in 0.34); pgvector `self_check` false-negative
trimmed; standard recall definition (was capped-in-a-buggy-way);
synthetic-warmup and dedupe added to the scorer.

[Unreleased]: https://github.com/kenimo49/rag-retriever-bench/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/kenimo49/rag-retriever-bench/releases/tag/v0.1.0
