"""render_markdown / save: fixed input dict -> markdown, no DB required."""

import json

from rag_retriever_bench.config import Config
from rag_retriever_bench.report import render_markdown, save


def _result(label, mode="server", **overrides):
    r = {
        "backend": {"label": label, "type": label, "mode": mode},
        "num_queries": 860,
        "quality": {"recall@10": 0.947, "ndcg@10": 0.9, "mrr@10": 0.88, "hit@10": 0.99},
        "latency_ms": {"p50": 3.3, "p95": 5.0, "p99": 8.1},
        "build": {"load_seconds": 18.0, "index_seconds": 1.0},
        "self_check": {"ann_index_used": True, "method": "server stats"},
    }
    r.update(overrides)
    return r


def test_server_and_embedded_tables_are_split():
    md = render_markdown(
        Config(),
        [_result("qdrant"), _result("chroma", mode="embedded")],
    )
    assert "## Server backends" in md
    assert "## Embedded backends" in md
    server_section = md.split("## Embedded backends")[0]
    assert "qdrant" in server_section
    assert "chroma" not in server_section


def test_no_embedded_section_when_all_server():
    md = render_markdown(Config(), [_result("pgvector")])
    assert "## Embedded backends" not in md


def test_metric_formatting():
    md = render_markdown(Config(), [_result("qdrant")])
    assert "| 0.947 |" in md  # recall 3dp
    assert "| 3.3 |" in md  # latency 1dp


def test_failed_backend_listed_but_not_in_table():
    md = render_markdown(
        Config(),
        [_result("pgvector"), {"backend": {"label": "milvus"}, "error": "connection refused"}],
    )
    assert "## Failed backends" in md
    assert "**milvus**: connection refused" in md
    # a failed backend must not leak into the results table
    table = md.split("## Failed backends")[0]
    assert "milvus" not in table


def test_self_check_recorded_in_details():
    md = render_markdown(Config(), [_result("qdrant")])
    assert "ann_index_used=True via server stats" in md


def test_self_check_missing_is_explicit():
    md = render_markdown(Config(), [_result("pgvector", self_check=None)])
    assert "self_check: n/a" in md


def test_header_survives_all_backends_failing():
    md = render_markdown(Config(), [{"backend": {"label": "x"}, "error": "boom"}])
    assert "- queries: 0, top_k=10" in md
    assert "## Failed backends" in md


def test_save_writes_jsonl_and_markdown(tmp_path):
    results = [_result("qdrant")]
    md_path = save(Config(), results, out_dir=tmp_path)
    assert md_path.exists()
    jsonl_path = md_path.with_suffix(".jsonl")
    assert jsonl_path.exists()
    lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["backend"]["label"] == "qdrant"
