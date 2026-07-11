# Methodology

Design notes for people who want to interpret, cite, or extend
`rag-retriever-bench`. If you're just running it, the README is enough.

The overriding constraint is simple: **every backend should be trying to
make the same trade-off, so the differences that show up in the report
reflect the engine rather than the tuning or the measurement setup.**
Everything below follows from that.

## 1. Why deterministic metrics (no LLM judge)

Recall@k, hit@k, MRR@k, and nDCG@k are computed against binary qrels
(positive docids from the MIRACL-ja dev split). There is no LLM-as-judge
anywhere in the pipeline.

Concretely, this buys three things:

- **Reproducibility.** A run pinned to a corpus size and seed gives the
  same numbers on any machine. Nothing depends on a hosted model's version
  or a temperature setting.
- **Cost.** A full 100k-passage MIRACL-ja run costs about $0.30 to embed,
  once, and $0 to score. A judge-based eval at the same scale runs into
  three digits and requires re-embedding whenever the judge changes.
- **Separation of concerns.** LLM-as-judge conflates retrieval quality
  with generator quality. If you care about the retrieval layer
  specifically, you don't want your numbers to move because GPT-5 tuned
  its refusal rate.

The trade-off: you can't measure semantic near-misses (retrieved a passage
that answers the question but wasn't marked positive). MIRACL's human
qrels mitigate this — they're annotated for relevance, not lexical
overlap — but the effect is real and worth stating.

## 2. recall@k — the standard uncapped definition

```python
def recall_at_k(retrieved, positives, k):
    hits = sum(1 for d in retrieved[:k] if d in positives)
    return hits / len(positives) if positives else 0.0
```

This is the definition used by BEIR and `pytrec_eval`. It matches capped
recall (`hits / min(len(positives), k)`) to within about 1e-4 on
MIRACL-ja dev — only a single query has more than 10 positives — but we
report the textbook definition so scores compare directly across
published RAG papers.

Duplicate docids returned by a backend are deduplicated before scoring
(see `metrics.dedupe`). A backend that returns the same id twice must
not inflate recall.

## 3. Latency measurement

Latency is measured **client-side, per query, in wall time**, including
serialization. Concretely:

```python
t0 = time.perf_counter()
result = backend.search(query_embedding, top_k)
elapsed_ms = (time.perf_counter() - t0) * 1000
```

The harness runs a fixed number of warmup queries (default 20) before
timing to let JIT paths, caches, and connection pools stabilize. Warmup
results are discarded.

Because every server backend pays the same localhost network hop, their
p50/p95/p99 numbers are directly comparable. Embedded backends (Chroma,
LanceDB) run in-process and don't pay that hop, so they're reported in a
separate table — do not compare the two classes directly.

We report p50 / p95 / p99 rather than mean because vector-search latency
distributions are skewed enough that the mean hides everything
interesting.

## 4. HNSW parameter alignment

All backends that support HNSW are configured with the same three
parameters:

- `m = 16`
- `ef_construction = 64`
- `ef_search = 100`
- distance = cosine

These aren't tuned. The point isn't to find each backend's peak; it's to
hold the algorithm and its knobs fixed so the difference in the report
attributes to the engine (memory layout, index build strategy, query
executor). A backend that can't hit the exact parameter values declares
so in `describe()`.

If you want to sweep parameters, add multiple config entries with
distinct `label`s. The `ClickHouse (HNSW, g=128)` row in the sample
report is an example: same HNSW knobs, different `index_granularity`.

## 5. `self_check` — the silent-full-scan defense

The most important optional hook on `BaseRetriever`. It runs once after
warmup and its output lands in the report next to each backend row:

```python
def self_check(self, query_embedding) -> dict[str, Any]:
    return {
        "ann_index_used": True | False,
        "method": "EXPLAIN" | "server-reported" | "config-only",
        # backend-specific evidence: plan excerpt, index stats, etc.
    }
```

The reason this exists: **three of the in-tree backends shipped a way to
silently degrade to full scan with zero errors.** The harness caught all
three during development:

- **ClickHouse HNSW** would degrade to brute force if the `OPTIMIZE
  FINAL` step was skipped. The self_check EXPLAIN caught the degradation
  before it could show up as a "ClickHouse is slow" result. (See commit
  `de42013`.)
- **Qdrant** could ignore a build request if
  `indexing_threshold_kb` was above the corpus size — collections came
  back with `indexed_vectors_count = 0` and searches ran on the raw
  segment. Explicit force-build in `build_index` was added. (See commit
  `1de5a04`.)
- **Milvus** would load a stale snapshot that missed sealed segments
  after flush, so queries returned partial results with no error.
  `refresh_load` was added to force a re-load. (See commit `680c9d6`.)

In every one of those cases, the silent-degradation path produced
plausible-looking numbers. The report would have said "backend X is
slower / lower recall than backend Y" — but the difference was a
misconfiguration, not the engine. `self_check` closed that gap.

Verification methods, in order of preference:

1. **`EXPLAIN` on the actual query** — pgvector, ClickHouse. Strongest
   evidence: you're inspecting the plan the executor used.
2. **Server-side stats** — Qdrant `indexed_vectors_count`, Milvus
   `describe_index`. Confirms the index was built and loaded but doesn't
   prove this query used it.
3. **Config-only** — Chroma, Weaviate. Reports what was configured; the
   report labels this as weaker evidence.

When a backend falls back to config-only, the harness marks the row
accordingly. Don't compare config-only self_checks against EXPLAIN-based
ones without noting the difference.

## 6. Server vs embedded backends

Two separate tables in every report. The reason:

| | server (pgvector, ClickHouse, Qdrant, Weaviate, Milvus) | embedded (Chroma, LanceDB) |
|---|---|---|
| Process model | separate process, network hop | in-process function call |
| Serialization | protobuf / SQL / gRPC over TCP | Python object → C++ / Rust |
| Concurrency | server-controlled | single-threaded per call |

An in-process HNSW search on a 100k corpus can hit sub-2ms because it
never leaves the process. Comparing that against a server backend's
2ms — which includes localhost TCP setup, protobuf encode/decode, and
executor overhead — would be misleading. Both are fair comparisons
inside their own class.

## 7. Corpus sampling

`corpus_size` controls how many passages are sampled from the full
MIRACL-ja corpus, but the sampler always includes every positive passage
for the evaluation query set. This means:

- Recall is measured against complete ground truth even at 10k. A backend
  that misses a positive missed a passage that was actually in its
  index — never one that got sampled out.
- Sweeping `corpus_size` (10k → 100k → 1M) tests how latency and recall
  degrade with scale, on your own hardware, without changing the ground
  truth.

Seed is fixed (`seed: 42` in the default config) so the sample is
deterministic across runs.

## 8. What this benchmark deliberately doesn't measure

Stating the omissions is more useful than pretending they don't exist.

- **Embedding quality.** Every backend receives identical embeddings from
  the same OpenAI model. If you want to benchmark embedding models, use
  MTEB / JMTEB.
- **RAG answer quality.** No generator runs. If you want to score the
  full pipeline, use ragas / DeepEval on top of the retriever this
  benchmark told you to use.
- **Cost per query.** Backend cost is a function of your infrastructure,
  not the engine. The report gives you the ingredients (recall × latency
  × index size) so you can compute cost against your own pricing.
- **Multi-tenant / partitioned setups.** Every backend runs single-tenant
  with the full corpus in one collection. Partitioned or sharded setups
  are backend-specific and would defeat parameter alignment.
- **Hybrid search.** Vector-only in v0.1. Hybrid (vector + BM25 / dense +
  sparse) is on the roadmap for v0.2 because the ranking and metric
  semantics change enough that it deserves its own report layout.
- **Filtered search.** Also roadmap. Filter selectivity interacts with
  HNSW in engine-specific ways and needs a curated filter set to be
  meaningful.
- **Recovery / failover / durability.** Out of scope. This is a
  microbenchmark, not a production readiness suite.

## 9. Citing this benchmark

If you use numbers from this repo in a paper, blog post, or vendor
comparison, please cite the specific run:

```
rag-retriever-bench, MIRACL-ja, 100k passages,
text-embedding-3-small, HNSW m=16 / ef_c=64 / ef_s=100,
report: results/published/miracl-ja-100000-20260711T053605Z.md
```

The pinned report file, timestamp, and config hash together let a reader
reproduce the numbers exactly — or notice that a later run moved them.
