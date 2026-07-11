"""Deterministic retrieval metrics — no LLM judge anywhere.

All metrics use binary relevance from the dataset's qrels (positive docids).
"""

from __future__ import annotations

import math


def recall_at_k(retrieved: list[str], positives: set[str], k: int) -> float:
    hits = sum(1 for d in retrieved[:k] if d in positives)
    return hits / min(len(positives), k) if positives else 0.0


def hit_at_k(retrieved: list[str], positives: set[str], k: int) -> float:
    return 1.0 if any(d in positives for d in retrieved[:k]) else 0.0


def mrr_at_k(retrieved: list[str], positives: set[str], k: int) -> float:
    for rank, d in enumerate(retrieved[:k], start=1):
        if d in positives:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], positives: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(rank + 1) for rank, d in enumerate(retrieved[:k], start=1) if d in positives)
    ideal_hits = min(len(positives), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    idx = (len(sorted_values) - 1) * pct / 100
    lo, hi = math.floor(idx), math.ceil(idx)
    if lo == hi:
        return sorted_values[lo]
    frac = idx - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac
