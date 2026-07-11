from __future__ import annotations

import time
from typing import Any

import numpy as np

from .base import BaseRetriever


class MilvusRetriever(BaseRetriever):
    """Milvus backend via MilvusClient.

    uri "http://host:19530" targets a standalone server (HNSW supported).
    A file path uri would use Milvus Lite, which only supports FLAT — the
    self_check reports the actual index type so that degradation is visible.
    """

    type_name = "milvus"

    def __init__(self, options: dict[str, Any]):
        super().__init__(options)
        from pymilvus import MilvusClient

        hnsw = options.get("hnsw", {})
        self.m = int(hnsw.get("m", 16))
        self.ef_construction = int(hnsw.get("ef_construction", 64))
        self.ef_search = int(hnsw.get("ef_search", 100))
        self.uri = options.get("uri", "http://localhost:19530")
        self.client = MilvusClient(uri=self.uri)
        self.collection_name = options.get("collection", "rrb_docs")
        self._docids: list[str] = []

    def setup(self, dim: int) -> None:
        from pymilvus import DataType

        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
        # Explicit schema: the quick-setup path (dimension=...) silently
        # replaces custom index_params with AUTOINDEX — self_check caught
        # exactly that on the first smoke run.
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field("docid", DataType.VARCHAR, max_length=128)
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": self.m, "efConstruction": self.ef_construction},
        )
        self.client.create_collection(self.collection_name, schema=schema, index_params=index_params)

    def load(self, docids: list[str], texts: list[str], embeddings: np.ndarray) -> float:
        self._docids = list(docids)
        t0 = time.perf_counter()
        chunk = 2_000
        emb_list = embeddings.tolist()
        for i in range(0, len(docids), chunk):
            rows = [
                {"id": j, "vector": emb_list[j], "docid": docids[j]}
                for j in range(i, min(i + chunk, len(docids)))
            ]
            self.client.insert(self.collection_name, data=rows)
        return time.perf_counter() - t0

    def build_index(self) -> float:
        t0 = time.perf_counter()
        self.client.flush(self.collection_name)
        # Wait until every row is covered by the HNSW index.
        deadline = t0 + 1800
        while time.perf_counter() < deadline:
            try:
                desc = self.client.describe_index(self.collection_name, index_name="vector")
                pending = int(desc.get("pending_index_rows", 0))
                total = int(desc.get("total_rows", 0))
                indexed = int(desc.get("indexed_rows", 0))
                if pending == 0 and (total == 0 or indexed >= total):
                    break
            except Exception:
                break
            time.sleep(0.5)
        # The collection is auto-loaded at creation; that snapshot misses the
        # newly sealed segments and refresh_load reports Loaded while still
        # pulling them in (measured recall@10 0.922 / 0.958 vs 0.979 on the
        # same index). A full release + load is the only reliably synchronous
        # path to an all-segments-visible state.
        self.client.release_collection(self.collection_name)
        self.client.load_collection(self.collection_name)
        while time.perf_counter() < deadline:
            state = str(self.client.get_load_state(self.collection_name).get("state", ""))
            if "Loaded" in state:
                break
            time.sleep(0.5)
        return time.perf_counter() - t0

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[str]:
        res = self.client.search(
            self.collection_name,
            data=[query_embedding.tolist()],
            limit=top_k,
            search_params={"metric_type": "COSINE", "params": {"ef": self.ef_search}},
            output_fields=["docid"],
        )
        return [hit["entity"]["docid"] for hit in res[0]]

    def self_check(self, query_embedding: np.ndarray) -> dict[str, Any]:
        desc = self.client.describe_index(self.collection_name, index_name="vector")
        index_type = str(desc.get("index_type", "?"))
        uses_index = index_type.upper() == "HNSW"
        if not uses_index:
            print(f"WARNING [{self.label}]: index type is {index_type}, not HNSW")
        return {
            "ann_index_used": uses_index,
            "method": "server-reported (describe_index)",
            "index_type": index_type,
            "indexed_rows": int(desc.get("indexed_rows", -1)),
        }

    def describe(self) -> dict[str, Any]:
        try:
            from pymilvus import utility  # noqa: F401  (server version via client)

            version = self.client.get_server_version()
        except Exception:
            version = "?"
        return {
            **super().describe(),
            "server": f"Milvus {version}",
            "uri": self.uri,
            "index": f"hnsw(M={self.m}, efConstruction={self.ef_construction}, ef={self.ef_search})",
            "distance": "cosine",
        }

    def close(self) -> None:
        self.client.close()
