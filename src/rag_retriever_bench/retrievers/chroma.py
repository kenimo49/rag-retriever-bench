from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from .base import BaseRetriever


class ChromaRetriever(BaseRetriever):
    """Chroma backend, embedded (in-process) mode — its most common deployment.

    NOTE for fair comparison: there is no network hop or server process, so
    latency numbers are not directly comparable to server-type backends.
    describe() marks mode=embedded; keep the two classes separate in writeups.
    """

    type_name = "chroma"

    def __init__(self, options: dict[str, Any]):
        super().__init__(options)
        import chromadb

        hnsw = options.get("hnsw", {})
        self.m = int(hnsw.get("m", 16))
        self.ef_construction = int(hnsw.get("ef_construction", 64))
        self.ef_search = int(hnsw.get("ef_search", 100))
        self.path = Path(options.get("path", "data/embedded/chroma"))
        self.path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.path))
        self.collection_name = options.get("collection", "rrb_docs")
        self.collection = None

    def setup(self, dim: int) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.create_collection(
            self.collection_name,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:M": self.m,
                "hnsw:construction_ef": self.ef_construction,
                "hnsw:search_ef": self.ef_search,
            },
        )

    def load(self, docids: list[str], texts: list[str], embeddings: np.ndarray) -> float:
        t0 = time.perf_counter()
        chunk = 5_000  # Chroma caps add() batches (~5461)
        emb_list = embeddings.tolist()
        for i in range(0, len(docids), chunk):
            self.collection.add(
                ids=list(docids[i : i + chunk]),
                embeddings=emb_list[i : i + chunk],
            )
        return time.perf_counter() - t0

    def build_index(self) -> float:
        # HNSW is built incrementally during add(); nothing to finalize.
        return 0.0

    def _ensure_collection(self):
        # Search-only reconnect: lets a downstream app (e.g. rag-db-advisor)
        # open a previously ingested store without re-running setup()/load().
        if self.collection is None:
            self.collection = self.client.get_collection(self.collection_name)
        return self.collection

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[str]:
        res = self._ensure_collection().query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=[],
        )
        return list(res["ids"][0])

    def self_check(self, query_embedding: np.ndarray) -> dict[str, Any]:
        # Chroma exposes no query plan; report the collection's configured
        # params and count so a misconfigured space/M is at least visible.
        meta = dict(self._ensure_collection().metadata or {})
        return {
            "ann_index_used": True,  # HNSW is Chroma's only vector index
            "method": "config-only (no plan introspection in Chroma)",
            "configured": {key: meta[key] for key in sorted(meta) if key.startswith("hnsw:")},
            "count": self._ensure_collection().count(),
        }

    def describe(self) -> dict[str, Any]:
        import chromadb

        return {
            **super().describe(),
            "server": f"Chroma {chromadb.__version__}",
            "mode": "embedded (in-process, no network hop)",
            "index": f"hnsw(M={self.m}, construction_ef={self.ef_construction}, search_ef={self.ef_search})",
            "distance": "cosine",
        }

    def close(self) -> None:
        # PersistentClient has no close(); setup() recreates the collection,
        # so reruns are clean without touching the (possibly shared) data dir.
        pass
