# rag-retriever-bench

Benchmark harness for RAG retrieval backends: **same corpus, same queries, same metrics — swap the database.**

Most RAG evaluation tools score the *answers* (ragas, DeepEval) or the *embedding models* (MTEB, JMTEB). This one scores the layer in between: the **retrieval backend**. It answers questions like *"at my corpus size and query pattern, is pgvector enough, or do I need something else?"* — with measured numbers instead of vendor benchmarks.

## What it measures

| Dimension | Metrics |
|---|---|
| Retrieval quality | recall@k, hit@k, MRR@k, nDCG@k (binary qrels, no LLM judge) |
| Query latency | p50 / p95 / p99 / mean (ms, client-side) |
| Ingestion | bulk load seconds, index build seconds |

All metrics are deterministic. No LLM-as-judge anywhere, so runs are cheap and reproducible.

## Backends (v0.1)

- **pgvector** (PostgreSQL, HNSW)
- **ClickHouse** (`vector_similarity` HNSW index, and brute-force full scan mode)

Backends implement a small interface (`retrievers/base.py`); adding one is a single file.

## Quick start

```bash
git clone https://github.com/kenimo49/rag-retriever-bench
cd rag-retriever-bench
pip install -e .

docker compose up -d          # pgvector + ClickHouse
cp .env.example .env          # set OPENAI_API_KEY (used for embeddings)

# 10k-passage smoke run (MIRACL-ja downloads on first use)
rag-retriever-bench run -c configs/miracl-ja.yaml --corpus-size 10000

# full 100k run
rag-retriever-bench run -c configs/miracl-ja.yaml
```

Reports land in `results/` as Markdown + JSONL.

## Dataset

Default config uses [MIRACL](https://huggingface.co/datasets/miracl/miracl) (ja): Japanese Wikipedia passages with human-annotated relevance judgments. The sampled corpus always contains every positive passage for the query set, so recall is measured against complete ground truth. Embeddings (`text-embedding-3-small`) are cached locally — a 100k-passage corpus costs roughly $0.30 to embed, once.

## Design notes

- The harness never assumes a winner. Index parameters (HNSW m / ef) are aligned across backends so differences reflect the engine, not the tuning.
- Latency is measured client-side per query, including serialization — the same overhead for every backend.
- `corpus_size` is a CLI flag so you can sweep scale (10k → 100k → …) and find where the trade-offs actually flip, on your own hardware.

## Roadmap

- v0.2: more backends (Qdrant, Weaviate, Milvus, Chroma, LanceDB), metadata-filtered search, hybrid (vector + full-text) mode
- Synthetic QA generation for bring-your-own-corpus evaluation

## License

MIT
