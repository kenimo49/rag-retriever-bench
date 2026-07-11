from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from .base import BaseRetriever

TABLE = "rrb_docs"


class LanceDBRetriever(BaseRetriever):
    """LanceDB backend, embedded (serverless, file-backed) mode.

    LanceDB's ANN indexes are IVF-family; the closest HNSW analogue is
    IVF_HNSW_SQ (HNSW over IVF partitions with scalar quantization). This is
    NOT parameter-identical to the other backends' flat HNSW — describe()
    spells out the actual index so the difference stays visible in reports.
    """

    type_name = "lancedb"

    def __init__(self, options: dict[str, Any]):
        super().__init__(options)
        import lancedb

        hnsw = options.get("hnsw", {})
        self.m = int(hnsw.get("m", 16))
        self.ef_construction = int(hnsw.get("ef_construction", 64))
        self.ef_search = int(hnsw.get("ef_search", 100))
        self.index_type = options.get("index", "IVF_HNSW_SQ")
        self.num_partitions = int(options.get("num_partitions", 1))
        self.path = Path(options.get("path", "data/embedded/lancedb"))
        self.path.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.path))
        self.table = None
        self._dim = 0

    def setup(self, dim: int) -> None:
        self._dim = dim
        try:
            self.db.drop_table(TABLE)
        except Exception:
            pass

    def load(self, docids: list[str], texts: list[str], embeddings: np.ndarray) -> float:
        import pyarrow as pa

        t0 = time.perf_counter()
        arr = pa.table(
            {
                "docid": pa.array(docids, type=pa.string()),
                "vector": pa.FixedSizeListArray.from_arrays(
                    pa.array(embeddings.astype(np.float32).reshape(-1), type=pa.float32()),
                    self._dim,
                ),
            }
        )
        self.table = self.db.create_table(TABLE, data=arr)
        return time.perf_counter() - t0

    def build_index(self) -> float:
        t0 = time.perf_counter()
        self.table.create_index(
            metric="cosine",
            vector_column_name="vector",
            index_type=self.index_type,
            num_partitions=self.num_partitions,
            m=self.m,
            ef_construction=self.ef_construction,
        )
        return time.perf_counter() - t0

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[str]:
        rows = (
            self.table.search(query_embedding.astype(np.float32))
            .metric("cosine")
            .nprobes(self.num_partitions)
            .limit(top_k)
            .select(["docid"])
            .to_list()
        )
        return [row["docid"] for row in rows]

    def self_check(self, query_embedding: np.ndarray) -> dict[str, Any]:
        indices = list(self.table.list_indices())
        has_vec_index = any("vector" in str(getattr(ix, "columns", ix)) for ix in indices)
        if not has_vec_index:
            print(f"WARNING [{self.label}]: no vector index on table — searches run flat")
        return {
            "ann_index_used": has_vec_index,
            "method": "table-reported (list_indices)",
            "indices": [str(ix) for ix in indices],
        }

    def describe(self) -> dict[str, Any]:
        import lancedb

        return {
            **super().describe(),
            "server": f"LanceDB {lancedb.__version__}",
            "mode": "embedded (in-process, file-backed)",
            "index": (
                f"{self.index_type}(num_partitions={self.num_partitions}, m={self.m}, "
                f"ef_construction={self.ef_construction}) — IVF-family, not flat HNSW"
            ),
            "distance": "cosine",
        }
