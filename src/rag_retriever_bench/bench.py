from __future__ import annotations

import time
from typing import Any

import numpy as np
from tqdm import tqdm

from . import dataset, embed, metrics
from .config import Config
from .retrievers import create_retriever


def run(cfg: Config) -> list[dict[str, Any]]:
    corpus = dataset.load_corpus(cfg)
    queries = dataset.load_queries(cfg)
    docids, doc_vecs = embed.embed_corpus(cfg, corpus)
    _, query_vecs = embed.embed_queries(cfg, queries)
    texts = [r["text"] for r in corpus]

    results = []
    for backend_options in cfg.backends:
        results.append(
            _run_backend(cfg, backend_options, docids, texts, doc_vecs, queries, query_vecs)
        )
    return results


def _run_backend(
    cfg: Config,
    backend_options: dict[str, Any],
    docids: list[str],
    texts: list[str],
    doc_vecs: np.ndarray,
    queries: list[dict],
    query_vecs: np.ndarray,
) -> dict[str, Any]:
    retriever = create_retriever(backend_options)
    label = retriever.label
    k = cfg.bench.top_k
    print(f"\n=== backend: {label} ===")

    try:
        retriever.setup(dim=doc_vecs.shape[1])

        load_s = retriever.load(docids, texts, doc_vecs)
        print(f"load: {load_s:.1f}s ({len(docids)} docs)")
        index_s = retriever.build_index()
        print(f"index build: {index_s:.1f}s")

        for vec in query_vecs[: cfg.bench.warmup_queries]:
            retriever.search(vec, k)

        self_check = retriever.self_check(query_vecs[0])
        if self_check:
            print(f"self_check: {self_check}")

        latencies_ms: list[float] = []
        per_query: list[dict[str, float]] = []
        for query, vec in zip(tqdm(queries, desc=f"search {label}"), query_vecs):
            t0 = time.perf_counter()
            retrieved = retriever.search(vec, k)
            latencies_ms.append((time.perf_counter() - t0) * 1000)
            positives = set(query["positives"])
            per_query.append(
                {
                    "recall": metrics.recall_at_k(retrieved, positives, k),
                    "hit": metrics.hit_at_k(retrieved, positives, k),
                    "mrr": metrics.mrr_at_k(retrieved, positives, k),
                    "ndcg": metrics.ndcg_at_k(retrieved, positives, k),
                }
            )

        latencies_ms.sort()
        n = len(per_query)
        return {
            "backend": retriever.describe(),
            "corpus_size": len(docids),
            "num_queries": n,
            "top_k": k,
            "quality": {
                f"recall@{k}": sum(q["recall"] for q in per_query) / n,
                f"hit@{k}": sum(q["hit"] for q in per_query) / n,
                f"mrr@{k}": sum(q["mrr"] for q in per_query) / n,
                f"ndcg@{k}": sum(q["ndcg"] for q in per_query) / n,
            },
            "latency_ms": {
                "p50": metrics.percentile(latencies_ms, 50),
                "p95": metrics.percentile(latencies_ms, 95),
                "p99": metrics.percentile(latencies_ms, 99),
                "mean": sum(latencies_ms) / n,
            },
            "build": {"load_seconds": load_s, "index_seconds": index_s},
            "self_check": self_check,
        }
    finally:
        retriever.close()
