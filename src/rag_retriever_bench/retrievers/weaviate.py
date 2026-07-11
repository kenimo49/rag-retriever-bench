from __future__ import annotations

import time
from typing import Any

import numpy as np

from .base import BaseRetriever

# Weaviate collection names must start with an uppercase letter.
COLLECTION = "RrbDocs"


class WeaviateRetriever(BaseRetriever):
    """Weaviate backend (HNSW, vectorizer disabled — we bring our own vectors).

    Weaviate indexes vectors on insert, so load() includes HNSW construction
    and build_index() only waits for the async indexing queue to drain.
    """

    type_name = "weaviate"

    def __init__(self, options: dict[str, Any]):
        super().__init__(options)
        import weaviate

        hnsw = options.get("hnsw", {})
        self.m = int(hnsw.get("m", 16))
        self.ef_construction = int(hnsw.get("ef_construction", 64))
        self.ef_search = int(hnsw.get("ef_search", 100))
        self.client = weaviate.connect_to_local(
            host=options.get("host", "localhost"),
            port=int(options.get("port", 8080)),
            grpc_port=int(options.get("grpc_port", 50051)),
        )
        self.collection = None

    def setup(self, dim: int) -> None:
        from weaviate.classes.config import Configure, DataType, Property, VectorDistances

        if self.client.collections.exists(COLLECTION):
            self.client.collections.delete(COLLECTION)
        self.collection = self.client.collections.create(
            name=COLLECTION,
            properties=[Property(name="docid", data_type=DataType.TEXT)],
            vectorizer_config=Configure.Vectorizer.none(),
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE,
                max_connections=self.m,
                ef_construction=self.ef_construction,
                ef=self.ef_search,
            ),
        )

    def load(self, docids: list[str], texts: list[str], embeddings: np.ndarray) -> float:
        t0 = time.perf_counter()
        emb_list = embeddings.tolist()
        with self.collection.batch.dynamic() as batch:
            for docid, vec in zip(docids, emb_list, strict=True):
                batch.add_object(properties={"docid": docid}, vector=vec)
        failed = self.collection.batch.failed_objects
        if failed:
            raise RuntimeError(f"{len(failed)} objects failed to insert (first: {failed[0]})")
        return time.perf_counter() - t0

    def build_index(self) -> float:
        # Indexing happens on insert; wait for the vector-index queue to drain.
        t0 = time.perf_counter()
        deadline = t0 + 1800
        while time.perf_counter() < deadline:
            try:
                shards = self.collection.config.get_shards()
            except Exception as exc:
                # Don't swallow this into a fake 0s build time: without the
                # queue metric we can't prove indexing finished, so say so.
                print(
                    f"NOTE [{self.label}]: shard queue metric unavailable ({exc}); "
                    "assuming synchronous indexing (Weaviate default, ASYNC_INDEXING off)"
                )
                break
            if all(getattr(s, "vector_queue_size", 0) == 0 for s in shards):
                break
            time.sleep(0.5)
        return time.perf_counter() - t0

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[str]:
        res = self.collection.query.near_vector(
            near_vector=query_embedding.tolist(),
            limit=top_k,
            return_properties=["docid"],
        )
        return [o.properties["docid"] for o in res.objects]

    def self_check(self, query_embedding: np.ndarray) -> dict[str, Any]:
        cfg = self.collection.config.get()
        index_cfg = cfg.vector_index_config
        index_type = str(getattr(cfg, "vector_index_type", "?"))
        uses_index = "hnsw" in index_type.lower()
        total = self.collection.aggregate.over_all(total_count=True).total_count
        if not uses_index:
            print(f"WARNING [{self.label}]: vector index type is {index_type}, not hnsw")
        return {
            "ann_index_used": uses_index,
            "method": "config-reported (no per-query EXPLAIN in Weaviate)",
            "index_type": index_type,
            "ef": getattr(index_cfg, "ef", None),
            "total_objects": total,
        }

    def describe(self) -> dict[str, Any]:
        meta = self.client.get_meta()
        return {
            **super().describe(),
            "server": f"Weaviate {meta.get('version', '?')}",
            "index": (
                f"hnsw(max_connections={self.m}, ef_construction={self.ef_construction}, ef={self.ef_search})"
            ),
            "distance": "cosine",
        }

    def close(self) -> None:
        self.client.close()
