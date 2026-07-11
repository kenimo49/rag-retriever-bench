from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Config


def save(cfg: Config, results: list[dict[str, Any]], out_dir: str | Path = "results") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = f"{cfg.dataset.name}-{cfg.dataset.corpus_size}-{stamp}"

    jsonl_path = out / f"{base}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    md_path = out / f"{base}.md"
    md_path.write_text(render_markdown(cfg, results), encoding="utf-8")
    print(f"\nreport: {md_path}\nraw:    {jsonl_path}")
    return md_path


def render_markdown(cfg: Config, results: list[dict[str, Any]]) -> str:
    k = cfg.bench.top_k
    lines = [
        f"# rag-retriever-bench — {cfg.dataset.name} ({cfg.dataset.corpus_size:,} passages)",
        "",
        f"- embeddings: {cfg.embeddings.model} (dim={cfg.embeddings.dim})",
        f"- queries: {results[0]['num_queries'] if results else 0}, top_k={k}",
        f"- generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "| backend | recall@{k} | ndcg@{k} | mrr@{k} | p50 (ms) | p95 (ms) | load (s) | index (s) |".replace(
            "{k}", str(k)
        ),
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        q, lat, b = r["quality"], r["latency_ms"], r["build"]
        lines.append(
            f"| {r['backend']['label']} "
            f"| {q[f'recall@{k}']:.3f} | {q[f'ndcg@{k}']:.3f} | {q[f'mrr@{k}']:.3f} "
            f"| {lat['p50']:.1f} | {lat['p95']:.1f} "
            f"| {b['load_seconds']:.1f} | {b['index_seconds']:.1f} |"
        )
    lines += ["", "## Backend details", ""]
    for r in results:
        lines.append(f"- **{r['backend']['label']}**: " + ", ".join(
            f"{key}={value}" for key, value in r["backend"].items() if key != "label"
        ))
    return "\n".join(lines) + "\n"
