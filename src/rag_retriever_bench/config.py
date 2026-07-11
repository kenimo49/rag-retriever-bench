from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DatasetConfig:
    name: str = "miracl-ja"
    corpus_size: int = 100_000
    split: str = "dev"
    seed: int = 42
    data_dir: Path = Path("data")


@dataclass
class EmbeddingConfig:
    provider: str = "openai"
    model: str = "text-embedding-3-small"
    dim: int = 1536
    batch_size: int = 128
    workers: int = 8


@dataclass
class BenchConfig:
    top_k: int = 10
    warmup_queries: int = 20


@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    bench: BenchConfig = field(default_factory=BenchConfig)
    backends: list[dict[str, Any]] = field(default_factory=list)

    @property
    def data_dir(self) -> Path:
        return Path(self.dataset.data_dir)

    @property
    def corpus_path(self) -> Path:
        return self.data_dir / f"{self.dataset.name}-{self.dataset.corpus_size}" / "corpus.jsonl"

    @property
    def queries_path(self) -> Path:
        return self.data_dir / f"{self.dataset.name}-{self.dataset.corpus_size}" / "queries.jsonl"

    @property
    def embeddings_dir(self) -> Path:
        return (
            self.data_dir
            / f"{self.dataset.name}-{self.dataset.corpus_size}"
            / "embeddings"
            / self.embeddings.model
        )


def load_config(path: str | Path, corpus_size: int | None = None) -> Config:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = Config(
        dataset=DatasetConfig(**raw.get("dataset", {})),
        embeddings=EmbeddingConfig(**raw.get("embeddings", {})),
        bench=BenchConfig(**raw.get("bench", {})),
        backends=raw.get("backends", []),
    )
    if corpus_size is not None:
        cfg.dataset.corpus_size = corpus_size

    # Environment overrides for connection targets, so the same config file
    # works on a laptop and on a remote bench host.
    for backend in cfg.backends:
        env_host = os.environ.get(f"RRB_{backend['type'].upper()}_HOST")
        if env_host:
            backend["host"] = env_host
    return cfg
