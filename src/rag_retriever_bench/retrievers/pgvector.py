from __future__ import annotations

import time
from typing import Any

import numpy as np

from .base import BaseRetriever

TABLE = "rrb_docs"


class PgvectorRetriever(BaseRetriever):
    type_name = "pgvector"

    def __init__(self, options: dict[str, Any]):
        super().__init__(options)
        import psycopg

        self.dsn = options.get("dsn", "postgresql://bench:bench@localhost:5432/bench")
        hnsw = options.get("hnsw", {})
        self.m = int(hnsw.get("m", 16))
        self.ef_construction = int(hnsw.get("ef_construction", 64))
        self.ef_search = int(hnsw.get("ef_search", 100))
        self.conn = psycopg.connect(self.dsn, autocommit=True)

    def setup(self, dim: int) -> None:
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f"DROP TABLE IF EXISTS {TABLE}")
            cur.execute(
                f"CREATE TABLE {TABLE} (docid text PRIMARY KEY, body text, embedding vector({dim}))"
            )

    def load(self, docids: list[str], texts: list[str], embeddings: np.ndarray) -> float:
        t0 = time.perf_counter()
        with self.conn.cursor() as cur:
            with cur.copy(f"COPY {TABLE} (docid, body, embedding) FROM STDIN") as copy:
                for docid, text, vec in zip(docids, texts, embeddings):
                    copy.write_row((docid, text, _vec_literal(vec)))
        return time.perf_counter() - t0

    def build_index(self) -> float:
        t0 = time.perf_counter()
        with self.conn.cursor() as cur:
            cur.execute("SET maintenance_work_mem = '2GB'")
            cur.execute("SET max_parallel_maintenance_workers = 4")
            cur.execute(
                f"CREATE INDEX ON {TABLE} USING hnsw (embedding vector_cosine_ops) "
                f"WITH (m = {self.m}, ef_construction = {self.ef_construction})"
            )
            cur.execute(f"ANALYZE {TABLE}")
        return time.perf_counter() - t0

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[str]:
        with self.conn.cursor() as cur:
            cur.execute(f"SET hnsw.ef_search = {self.ef_search}")
            cur.execute(
                f"SELECT docid FROM {TABLE} ORDER BY embedding <=> %s::vector LIMIT %s",
                (_vec_literal(query_embedding), top_k),
            )
            return [row[0] for row in cur.fetchall()]

    def self_check(self, query_embedding: np.ndarray) -> dict[str, Any]:
        with self.conn.cursor() as cur:
            cur.execute(f"SET hnsw.ef_search = {self.ef_search}")
            cur.execute(
                f"EXPLAIN SELECT docid FROM {TABLE} ORDER BY embedding <=> %s::vector LIMIT 10",
                (_vec_literal(query_embedding),),
            )
            plan = [row[0] for row in cur.fetchall()]
        # The HNSW index is auto-named (e.g. rrb_docs_embedding_idx); an index
        # scan on this table can only be that index, so match the scan itself.
        uses_index = any("Index Scan using" in line and TABLE in line for line in plan)
        if not uses_index:
            print(f"WARNING [{self.label}]: HNSW index NOT used in query plan")
        return {"ann_index_used": uses_index, "plan_excerpt": [l.strip()[:160] for l in plan[:2]]}

    def describe(self) -> dict[str, Any]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT version()")
            pg_version = cur.fetchone()[0].split(" on ")[0]
            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            row = cur.fetchone()
        return {
            **super().describe(),
            "server": pg_version,
            "pgvector": row[0] if row else "?",
            "index": f"hnsw(m={self.m}, ef_construction={self.ef_construction}, ef_search={self.ef_search})",
            "distance": "cosine",
        }

    def close(self) -> None:
        self.conn.close()


def _vec_literal(vec: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.7f}" for x in vec) + "]"
