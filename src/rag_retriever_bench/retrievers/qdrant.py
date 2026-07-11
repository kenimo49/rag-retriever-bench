from __future__ import annotations

import time
from typing import Any

import numpy as np

from .base import BaseRetriever

COLLECTION = "rrb_docs"


class QdrantRetriever(BaseRetriever):
    """Qdrant backend (HNSW is the only index type, always on).

    Qdrant builds HNSW segments asynchronously after upsert; build_index()
    forces indexing (indexing_threshold=0 would index during upsert, so we
    keep the default and measure the wait until status turns green).
    """

    type_name = "qdrant"

    def __init__(self, options: dict[str, Any]):
        super().__init__(options)
        from qdrant_client import QdrantClient

        hnsw = options.get("hnsw", {})
        self.m = int(hnsw.get("m", 16))
        self.ef_construction = int(hnsw.get("ef_construction", 64))
        self.ef_search = int(hnsw.get("ef_search", 100))
        self.client = QdrantClient(
            host=options.get("host", "localhost"),
            port=int(options.get("port", 6333)),
            grpc_port=int(options.get("grpc_port", 6334)),
            # REST rejects JSON bodies >32MB; gRPC has no such ceiling and is
            # what production loaders use anyway.
            prefer_grpc=True,
            timeout=600,
        )
        self._docids: list[str] = []

    def setup(self, dim: int) -> None:
        from qdrant_client import models

        if self.client.collection_exists(COLLECTION):
            self.client.delete_collection(COLLECTION)
        self.client.create_collection(
            COLLECTION,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            hnsw_config=models.HnswConfigDiff(m=self.m, ef_construct=self.ef_construction),
        )

    def load(self, docids: list[str], texts: list[str], embeddings: np.ndarray) -> float:
        from qdrant_client import models

        self._docids = list(docids)
        t0 = time.perf_counter()
        chunk = 2_000
        emb_list = embeddings.tolist()
        for i in range(0, len(docids), chunk):
            self.client.upsert(
                COLLECTION,
                points=models.Batch(
                    ids=list(range(i, min(i + chunk, len(docids)))),
                    vectors=emb_list[i : i + chunk],
                    payloads=[{"docid": d} for d in docids[i : i + chunk]],
                ),
                wait=True,
            )
        return time.perf_counter() - t0

    def build_index(self) -> float:
        # Wait until the async optimizer finishes building HNSW segments.
        t0 = time.perf_counter()
        deadline = t0 + 1800
        while time.perf_counter() < deadline:
            info = self.client.get_collection(COLLECTION)
            if str(info.status) in ("CollectionStatus.GREEN", "green"):
                break
            time.sleep(0.5)
        return time.perf_counter() - t0

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[str]:
        from qdrant_client import models

        res = self.client.query_points(
            COLLECTION,
            query=query_embedding.tolist(),
            limit=top_k,
            search_params=models.SearchParams(hnsw_ef=self.ef_search),
            with_payload=["docid"],
        )
        return [p.payload["docid"] for p in res.points]

    def self_check(self, query_embedding: np.ndarray) -> dict[str, Any]:
        info = self.client.get_collection(COLLECTION)
        indexed = int(info.indexed_vectors_count or 0)
        total = int(info.points_count or 0)
        uses_index = indexed > 0
        if not uses_index:
            print(f"WARNING [{self.label}]: 0 indexed vectors — searches run unindexed")
        return {
            "ann_index_used": uses_index,
            "method": "server-reported (indexed_vectors_count)",
            "indexed_vectors": indexed,
            "total_points": total,
        }

    def describe(self) -> dict[str, Any]:
        try:
            from importlib.metadata import version

            client_ver = version("qdrant-client")
        except Exception:
            client_ver = "?"
        return {
            **super().describe(),
            "server": f"Qdrant (client {client_ver})",
            "index": f"hnsw(m={self.m}, ef_construction={self.ef_construction}, ef_search={self.ef_search})",
            "distance": "cosine",
        }

    def close(self) -> None:
        self.client.close()
