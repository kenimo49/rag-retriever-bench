from __future__ import annotations

from typing import Any

from .base import BaseRetriever

_REGISTRY: dict[str, str] = {
    "pgvector": "rag_retriever_bench.retrievers.pgvector:PgvectorRetriever",
    "clickhouse": "rag_retriever_bench.retrievers.clickhouse:ClickHouseRetriever",
}


def create_retriever(options: dict[str, Any]) -> BaseRetriever:
    type_name = options["type"]
    if type_name not in _REGISTRY:
        raise SystemExit(f"unknown backend type: {type_name} (available: {sorted(_REGISTRY)})")
    module_name, class_name = _REGISTRY[type_name].split(":")
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)(options)
