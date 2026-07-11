from __future__ import annotations

import re
import time
from typing import Any

import numpy as np

from .base import BaseRetriever


class ClickHouseRetriever(BaseRetriever):
    """ClickHouse backend.

    index: "hnsw" uses the vector_similarity skipping index (experimental,
    ClickHouse >= 24.8); "none" runs brute-force cosineDistance, which is
    ClickHouse's classic full-scan strength and needs no experimental flags.
    """

    type_name = "clickhouse"

    def __init__(self, options: dict[str, Any]):
        super().__init__(options)
        import clickhouse_connect

        self.index_type = options.get("index", "hnsw")
        hnsw = options.get("hnsw", {})
        self.m = int(hnsw.get("m", 16))
        self.ef_construction = int(hnsw.get("ef_construction", 64))
        self.ef_search = int(hnsw.get("ef_search", 100))
        self.client = clickhouse_connect.get_client(
            host=options.get("host", "localhost"),
            port=int(options.get("port", 8123)),
            username=options.get("username", "bench"),
            password=options.get("password", "bench"),
            database=options.get("database", "bench"),
        )
        # Distinct table per backend config, so two ClickHouse variants in one
        # run don't clobber each other and post-run inspection stays possible.
        self.table = "rrb_" + re.sub(r"[^a-z0-9]+", "_", self.label.lower()).strip("_")
        self._search_settings: dict[str, Any] = {}
        if self.index_type == "hnsw":
            # Without the experimental flag at query time the index is
            # silently ignored, which turns "HNSW" into brute force.
            self._search_settings = {
                "allow_experimental_vector_similarity_index": 1,
                "hnsw_candidate_list_size_for_search": self.ef_search,
            }

    def setup(self, dim: int) -> None:
        self.client.command(f"DROP TABLE IF EXISTS {self.table}")
        index_clause = ""
        if self.index_type == "hnsw":
            self.client.command("SET allow_experimental_vector_similarity_index = 1")
            index_clause = (
                f", INDEX vec_idx embedding TYPE vector_similarity("
                f"'hnsw', 'cosineDistance', {dim}, 'bf16', {self.m}, {self.ef_construction}) GRANULARITY 100000000"
            )
        self.client.command(
            f"CREATE TABLE {self.table} (docid String, body String, embedding Array(Float32)"
            f"{index_clause}) ENGINE = MergeTree ORDER BY docid",
            settings={"allow_experimental_vector_similarity_index": 1},
        )

    def load(self, docids: list[str], texts: list[str], embeddings: np.ndarray) -> float:
        t0 = time.perf_counter()
        chunk = 10_000
        emb_list = embeddings.tolist()
        for i in range(0, len(docids), chunk):
            self.client.insert(
                self.table,
                list(zip(docids[i : i + chunk], texts[i : i + chunk], emb_list[i : i + chunk])),
                column_names=["docid", "body", "embedding"],
            )
        return time.perf_counter() - t0

    def build_index(self) -> float:
        # The vector index is built per data part; OPTIMIZE FINAL merges parts
        # and (re)builds the index, making timing comparable to CREATE INDEX.
        t0 = time.perf_counter()
        self.client.command(
            f"OPTIMIZE TABLE {self.table} FINAL",
            settings={"optimize_throw_if_noop": 0, "mutations_sync": 2},
        )
        return time.perf_counter() - t0

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[str]:
        result = self.client.query(
            f"SELECT docid FROM {self.table} "
            f"ORDER BY cosineDistance(embedding, {{q:Array(Float32)}}) LIMIT {top_k}",
            parameters={"q": query_embedding.tolist()},
            settings=self._search_settings,
        )
        return [row[0] for row in result.result_rows]

    def self_check(self, query_embedding: np.ndarray) -> dict[str, Any]:
        result = self.client.query(
            f"EXPLAIN indexes = 1 SELECT docid FROM {self.table} "
            f"ORDER BY cosineDistance(embedding, {{q:Array(Float32)}}) LIMIT 10",
            parameters={"q": query_embedding.tolist()},
            settings=self._search_settings,
        )
        plan = [row[0] for row in result.result_rows]
        uses_index = any("vec_idx" in line for line in plan)
        if self.index_type == "hnsw" and not uses_index:
            print(f"WARNING [{self.label}]: vector_similarity index NOT used in query plan")
        return {
            "ann_index_used": uses_index,
            "plan_excerpt": [l.strip() for l in plan if "Skip" in l or "vec_idx" in l or "Granules" in l],
        }

    def describe(self) -> dict[str, Any]:
        version = self.client.command("SELECT version()")
        index = (
            f"vector_similarity hnsw(m={self.m}, ef_construction={self.ef_construction}, "
            f"ef_search={self.ef_search})"
            if self.index_type == "hnsw"
            else "none (brute force)"
        )
        return {
            **super().describe(),
            "server": f"ClickHouse {version}",
            "index": index,
            "distance": "cosine",
        }

    def close(self) -> None:
        self.client.close()
