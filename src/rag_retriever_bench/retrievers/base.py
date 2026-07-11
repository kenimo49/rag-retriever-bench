from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class IndexStats:
    load_seconds: float = 0.0
    index_seconds: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


class BaseRetriever(ABC):
    """One retrieval backend under test.

    Lifecycle: setup() -> load() -> build_index() -> search()*N -> close().
    setup() must leave the backend empty so runs are reproducible.
    """

    type_name: str = "base"

    def __init__(self, options: dict[str, Any]):
        self.options = options
        self.label: str = options.get("label", self.type_name)

    @abstractmethod
    def setup(self, dim: int) -> None: ...

    @abstractmethod
    def load(self, docids: list[str], texts: list[str], embeddings: np.ndarray) -> float:
        """Bulk-load documents. Returns wall seconds spent."""

    @abstractmethod
    def build_index(self) -> float:
        """Build/finalize the ANN index. Returns wall seconds spent."""

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int) -> list[str]:
        """Return docids ranked by similarity, best first."""

    def describe(self) -> dict[str, Any]:
        """Backend + index parameters for the report."""
        return {"type": self.type_name, "label": self.label}

    def close(self) -> None:
        pass
