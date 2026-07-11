"""End-to-end bench.run() on a tiny synthetic dataset with Chroma embedded.

Covers the full pipeline — dataset load, embeddings cache hit, backend
lifecycle, metrics, report — without OpenAI or Docker. The embeddings cache
is pre-seeded so embed_corpus/embed_queries never call the API.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("chromadb")

from rag_retriever_bench import bench, report
from rag_retriever_bench.config import BenchConfig, Config, DatasetConfig

N_DOCS = 16
DIM = 16
TOP_K = 10


@pytest.fixture()
def cfg(tmp_path):
    cfg = Config(
        dataset=DatasetConfig(name="tiny", corpus_size=N_DOCS, data_dir=tmp_path / "data"),
        bench=BenchConfig(top_k=TOP_K, warmup_queries=2),
        backends=[
            {
                "type": "chroma",
                "label": "chroma-e2e",
                "path": str(tmp_path / "chroma"),
                "collection": "rrb_e2e_docs",
            }
        ],
    )
    out = cfg.corpus_path.parent
    out.mkdir(parents=True)

    # One-hot corpus: each query is its positive's vector plus tiny noise, so
    # every metric has an exact expected value of 1.0.
    rng = np.random.default_rng(3)
    doc_vecs = np.eye(N_DOCS, DIM, dtype=np.float32)
    docids = [f"doc{i}" for i in range(N_DOCS)]
    corpus = [{"docid": d, "title": "", "text": f"passage {d}"} for d in docids]
    queries = [
        {"qid": f"q{i}", "text": f"query {i}", "positives": [docids[i]]} for i in range(4)
    ]
    qvecs = doc_vecs[:4] + 0.01 * rng.standard_normal((4, DIM)).astype(np.float32)

    with open(cfg.corpus_path, "w", encoding="utf-8") as f:
        for row in corpus:
            f.write(json.dumps(row) + "\n")
    with open(cfg.queries_path, "w", encoding="utf-8") as f:
        for row in queries:
            f.write(json.dumps(row) + "\n")

    # Pre-seed the .npy cache in the exact format _embed_cached expects.
    cache = cfg.embeddings_dir
    cache.mkdir(parents=True)
    np.save(cache / "corpus.npy", doc_vecs)
    (cache / "corpus.ids.json").write_text(json.dumps(docids), encoding="utf-8")
    np.save(cache / "queries.npy", qvecs)
    (cache / "queries.ids.json").write_text(json.dumps([q["qid"] for q in queries]), encoding="utf-8")
    return cfg


def test_run_end_to_end(cfg, tmp_path):
    results = bench.run(cfg)
    assert len(results) == 1
    r = results[0]
    assert "error" not in r, r.get("error")

    assert r["num_queries"] == 4
    assert r["corpus_size"] == N_DOCS
    for metric in (f"recall@{TOP_K}", f"hit@{TOP_K}", f"mrr@{TOP_K}", f"ndcg@{TOP_K}"):
        assert r["quality"][metric] == pytest.approx(1.0), metric
    assert r["latency_ms"]["p50"] > 0
    assert r["self_check"]["ann_index_used"] is True

    md_path = report.save(cfg, results, out_dir=tmp_path / "results")
    md = md_path.read_text(encoding="utf-8")
    assert "chroma-e2e" in md
    assert "## Embedded backends" in md


def test_run_isolates_backend_failure(cfg):
    cfg.backends.insert(
        0, {"type": "qdrant", "label": "unreachable", "host": "127.0.0.1", "port": 1, "grpc_port": 1}
    )
    pytest.importorskip("qdrant_client")
    results = bench.run(cfg)
    assert len(results) == 2
    assert "error" in results[0]
    assert "error" not in results[1]
    assert results[1]["quality"][f"recall@{TOP_K}"] == pytest.approx(1.0)
