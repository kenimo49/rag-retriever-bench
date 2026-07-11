"""Live-backend contract tests: setup -> load -> build_index -> search.

Synthetic unit vectors with a numpy brute-force ground truth — no dataset
download, no OpenAI. Server backends skip when their service isn't reachable
(`docker compose up -d` starts all of them); embedded backends always run.

All tables/collections use the rrb_it_* namespace so a developer's local
bench data (rrb_docs) is never touched.
"""

from __future__ import annotations

import os
import socket
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from rag_retriever_bench.retrievers import create_retriever

pytestmark = pytest.mark.integration

DIM = 64
# 4096 is the smallest power of two where the Postgres planner picks the HNSW
# index over a seq scan at ef_search=100 (measured: 2048 -> seq, 4096 -> index;
# higher ef raises the estimated index-scan cost). Below that, pgvector's
# self_check honestly reports ann_index_used=False — the pgvector flavor of
# the "small data is never indexed" trap.
N_DOCS = 4096
N_QUERIES = 32
TOP_K = 10
# A returned doc counts as correct when its true cosine is within EPS of the
# 10th-best. Random vectors make ranks 2..10 near-ties, and quantizing
# backends (LanceDB SQ) legitimately reorder those without losing quality.
EPS = 0.02

# Where the docker compose services live; override when they run on another
# machine (e.g. RRB_IT_HOST=bench-host pytest -m integration).
HOST = os.environ.get("RRB_IT_HOST", "localhost")


def _reachable(port: int, host: str = HOST) -> bool:
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


@dataclass
class Case:
    module: str
    options: Callable[[Any], dict]
    port: int | None = None  # None = embedded, no service needed
    expect_ann: bool | None = True
    min_recall: float = 0.95
    extra_ports: list[int] = field(default_factory=list)


CASES = {
    "pgvector": Case(
        module="psycopg",
        port=5432,
        options=lambda tmp: {
            "type": "pgvector",
            "label": "pgvector-it",
            "dsn": f"postgresql://bench:bench@{HOST}:5432/bench",
            "table": "rrb_it_docs",
        },
    ),
    "clickhouse-hnsw": Case(
        module="clickhouse_connect",
        port=8123,
        # Small granules so the tiny corpus still spans multiple granules and
        # the EXPLAIN check exercises the skip index for real.
        options=lambda tmp: {
            "type": "clickhouse",
            "label": "rrb-it-ch-hnsw",
            "host": HOST,
            "index_granularity": 128,
        },
    ),
    "clickhouse-brute": Case(
        module="clickhouse_connect",
        port=8123,
        options=lambda tmp: {"type": "clickhouse", "label": "rrb-it-ch-brute", "host": HOST, "index": "none"},
        expect_ann=False,  # no index by design; self_check must say so
        min_recall=0.999,  # exact search: matches brute-force ground truth
    ),
    "qdrant": Case(
        module="qdrant_client",
        port=6333,
        extra_ports=[6334],
        options=lambda tmp: {
            "type": "qdrant",
            "label": "qdrant-it",
            "host": HOST,
            "collection": "rrb_it_docs",
        },
    ),
    "weaviate": Case(
        module="weaviate",
        port=8087,  # compose publishes on 8087/50052 (host 8080 shadowing, see docker-compose.yml)
        extra_ports=[50052],
        options=lambda tmp: {
            "type": "weaviate",
            "label": "weaviate-it",
            "host": HOST,
            "port": 8087,
            "grpc_port": 50052,
            "collection": "RrbItDocs",
        },
    ),
    "milvus": Case(
        module="pymilvus",
        port=19530,
        options=lambda tmp: {
            "type": "milvus",
            "label": "milvus-it",
            "uri": f"http://{HOST}:19530",
            "collection": "rrb_it_docs",
        },
    ),
    "chroma": Case(
        module="chromadb",
        options=lambda tmp: {
            "type": "chroma",
            "label": "chroma-it",
            "path": str(tmp / "chroma"),
            "collection": "rrb_it_docs",
        },
    ),
    "lancedb": Case(
        module="lancedb",
        options=lambda tmp: {
            "type": "lancedb",
            "label": "lancedb-it",
            "path": str(tmp / "lancedb"),
            "table": "rrb_it_docs",
        },
        min_recall=0.9,  # IVF_HNSW_SQ quantizes; EPS absorbs near-tie reordering only
    ),
}


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(7)
    docs = rng.standard_normal((N_DOCS, DIM)).astype(np.float32)
    docs /= np.linalg.norm(docs, axis=1, keepdims=True)
    qidx = rng.choice(N_DOCS, size=N_QUERIES, replace=False)
    queries = docs[qidx] + 0.05 * rng.standard_normal((N_QUERIES, DIM)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    sims = queries @ docs.T  # exact cosine (unit vectors)
    docids = [f"d{i:04d}" for i in range(N_DOCS)]
    texts = [f"passage {i}" for i in range(N_DOCS)]
    return docids, texts, docs, queries, sims


@pytest.mark.parametrize("name", CASES)
def test_backend_lifecycle(name, data, tmp_path):
    case = CASES[name]
    pytest.importorskip(case.module)
    for port in ([case.port] if case.port else []) + case.extra_ports:
        if not _reachable(port):
            pytest.skip(f"service on port {port} not reachable — run `docker compose up -d`")

    docids, texts, docs, queries, sims = data
    index_of = {d: i for i, d in enumerate(docids)}
    retriever = create_retriever(case.options(tmp_path))
    try:
        retriever.setup(dim=DIM)
        load_s = retriever.load(docids, texts, docs)
        index_s = retriever.build_index()
        assert load_s >= 0 and index_s >= 0

        recalls = []
        first_hits = 0
        for qvec, qsims in zip(queries, sims, strict=True):
            got = retriever.search(qvec, TOP_K)
            assert len(got) == TOP_K
            assert len(set(got)) == TOP_K, "duplicate docids in results"
            assert set(got) <= set(docids), "unknown docid returned"
            tenth_best = np.sort(qsims)[-TOP_K]
            good = sum(1 for d in got if qsims[index_of[d]] >= tenth_best - EPS)
            recalls.append(good / TOP_K)
            if got[0] == docids[int(np.argmax(qsims))]:
                first_hits += 1
        mean_recall = sum(recalls) / len(recalls)
        assert mean_recall >= case.min_recall, f"eps-recall vs brute force {mean_recall:.3f}"
        # The true nearest neighbour has a huge margin (query = doc + noise);
        # every backend, quantized or not, must put it first.
        assert first_hits >= 0.9 * N_QUERIES, f"top-1 exact only {first_hits}/{N_QUERIES}"

        check = retriever.self_check(queries[0])
        if case.expect_ann is not None:
            assert check.get("ann_index_used") is case.expect_ann, check

        desc = retriever.describe()
        assert desc["type"] == case.options(tmp_path)["type"]
        assert "index" in desc
    finally:
        retriever.close()
