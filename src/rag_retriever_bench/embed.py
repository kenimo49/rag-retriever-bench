"""OpenAI embeddings with a local .npy cache.

Corpus embeddings are cached as {embeddings_dir}/corpus.npy plus an id list,
so re-running the bench never re-pays the API cost.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm

from .config import Config

# text-embedding-3-small pricing, USD per 1M tokens; only used for the estimate.
PRICE_PER_MTOK = 0.02
MAX_CHARS = 6000  # passages are short; hard guard against pathological rows


def embed_corpus(cfg: Config, corpus: list[dict]) -> tuple[list[str], np.ndarray]:
    return _embed_cached(cfg, "corpus", [r["docid"] for r in corpus],
                         [_passage_input(r) for r in corpus])


def embed_queries(cfg: Config, queries: list[dict]) -> tuple[list[str], np.ndarray]:
    return _embed_cached(cfg, "queries", [r["qid"] for r in queries],
                         [r["text"] for r in queries])


def _passage_input(row: dict) -> str:
    title = row.get("title") or ""
    text = row["text"]
    return (f"{title}\n{text}" if title else text)[:MAX_CHARS]


def _embed_cached(cfg: Config, name: str, ids: list[str], texts: list[str]) -> tuple[list[str], np.ndarray]:
    cache_dir = cfg.embeddings_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    npy_path = cache_dir / f"{name}.npy"
    ids_path = cache_dir / f"{name}.ids.json"

    if npy_path.exists() and ids_path.exists():
        cached_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        if cached_ids == ids:
            print(f"embeddings cache hit: {npy_path}")
            return ids, np.load(npy_path)
        print(f"embeddings cache stale (id mismatch), re-embedding {name}")

    est_tokens = sum(len(t) for t in texts) / 2  # rough: ~2 chars/token for Japanese
    print(
        f"embedding {len(texts)} {name} texts with {cfg.embeddings.model} "
        f"(~{est_tokens/1e6:.1f}M tokens, ~${est_tokens/1e6*PRICE_PER_MTOK:.2f})"
    )

    vecs = _embed_texts(cfg, texts)
    np.save(npy_path, vecs)
    ids_path.write_text(json.dumps(ids), encoding="utf-8")
    return ids, vecs


def _embed_texts(cfg: Config, texts: list[str]) -> np.ndarray:
    from openai import OpenAI

    client = OpenAI()
    bs = cfg.embeddings.batch_size
    batches = [texts[i : i + bs] for i in range(0, len(texts), bs)]

    def one(batch: list[str]) -> list[list[float]]:
        for attempt in range(5):
            try:
                resp = client.embeddings.create(model=cfg.embeddings.model, input=batch)
                return [d.embedding for d in resp.data]
            except Exception as e:  # noqa: BLE001 - retry any transient API error
                if attempt == 4:
                    raise
                wait = 2**attempt
                print(f"embed retry in {wait}s: {e}")
                time.sleep(wait)
        raise RuntimeError("unreachable")

    out: list[np.ndarray] = []
    with ThreadPoolExecutor(max_workers=cfg.embeddings.workers) as pool:
        for vecs in tqdm(pool.map(one, batches), total=len(batches), desc="embedding"):
            out.append(np.asarray(vecs, dtype=np.float32))
    return np.vstack(out)
