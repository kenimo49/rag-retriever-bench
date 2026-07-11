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
    ok = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]
    server = [r for r in ok if not _is_embedded(r)]
    embedded = [r for r in ok if _is_embedded(r)]

    lines = _header(cfg, ok, k)
    lines += _table("## Server backends (network hop included)", server, k)
    if embedded:
        lines += _table(
            "## Embedded backends (in-process — latency NOT comparable to server backends)",
            embedded,
            k,
        )
    lines += _failed_section(failed)
    lines += _details_section(ok)
    return "\n".join(lines) + "\n"


def _is_embedded(r: dict[str, Any]) -> bool:
    return "embedded" in str(r["backend"].get("mode", ""))


def _header(cfg: Config, ok: list[dict[str, Any]], k: int) -> list[str]:
    num_queries = ok[0]["num_queries"] if ok else 0
    return [
        f"# rag-retriever-bench — {cfg.dataset.name} ({cfg.dataset.corpus_size:,} passages)",
        "",
        f"- embeddings: {cfg.embeddings.model} (dim={cfg.embeddings.dim})",
        f"- queries: {num_queries}, top_k={k}",
        f"- generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
    ]


def _failed_section(failed: list[dict[str, Any]]) -> list[str]:
    if not failed:
        return []
    lines = ["", "## Failed backends", ""]
    for r in failed:
        lines.append(f"- **{r['backend']['label']}**: {r['error']}")
    return lines


def _details_section(ok: list[dict[str, Any]]) -> list[str]:
    lines = ["", "## Backend details", ""]
    for r in ok:
        params = ", ".join(f"{key}={value}" for key, value in r["backend"].items() if key != "label")
        lines.append(f"- **{r['backend']['label']}**: {params} — {_self_check_note(r)}")
    return lines


def _self_check_note(r: dict[str, Any]) -> str:
    check = r.get("self_check") or {}
    if not check:
        return "self_check: n/a"
    return f"ann_index_used={check.get('ann_index_used')} via {check.get('method', 'EXPLAIN')}"


def _table(heading: str, rows: list[dict[str, Any]], k: int) -> list[str]:
    if not rows:
        return []
    lines = [
        "",
        heading,
        "",
        f"| backend | recall@{k} | ndcg@{k} | mrr@{k} | hit@{k} "
        f"| p50 (ms) | p95 (ms) | p99 (ms) | load (s) | index (s) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        q, lat, b = r["quality"], r["latency_ms"], r["build"]
        lines.append(
            f"| {r['backend']['label']} "
            f"| {q[f'recall@{k}']:.3f} | {q[f'ndcg@{k}']:.3f} | {q[f'mrr@{k}']:.3f} "
            f"| {q[f'hit@{k}']:.3f} "
            f"| {lat['p50']:.1f} | {lat['p95']:.1f} | {lat['p99']:.1f} "
            f"| {b['load_seconds']:.1f} | {b['index_seconds']:.1f} |"
        )
    return lines
