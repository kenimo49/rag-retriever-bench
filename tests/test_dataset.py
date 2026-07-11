"""Dataset preparation against local fixture files (hf_hub_download faked)."""

from __future__ import annotations

import gzip
import json

import pytest

from rag_retriever_bench import dataset
from rag_retriever_bench.config import Config, DatasetConfig

TOPICS = "q1\tクエリその1\nq2\tクエリその2\nq3\tqrelsに無いクエリ\n"
QRELS = "q1 Q0 d1 1\nq1 Q0 d2 0\nq1 Q0 d4 2\nq2 Q0 d3 1\n"
CORPUS_DOCS = [
    {"docid": d, "title": f"title {d}", "text": f"text {d}"}
    for d in ("d1", "d2", "d3", "d4", "f1", "f2", "f3", "f4")
]


@pytest.fixture()
def fake_hub(tmp_path, monkeypatch):
    topics = tmp_path / "topics.tsv"
    topics.write_text(TOPICS, encoding="utf-8")
    qrels = tmp_path / "qrels.tsv"
    qrels.write_text(QRELS, encoding="utf-8")
    shard = tmp_path / "docs-0.jsonl.gz"
    with gzip.open(shard, "wt", encoding="utf-8") as f:
        for doc in CORPUS_DOCS:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    calls = []

    def fake_download(repo, filename, repo_type):
        assert repo_type == "dataset"
        calls.append(f"{repo}/{filename}")
        if "topics" in filename:
            return str(topics)
        if "qrels" in filename:
            return str(qrels)
        return str(shard)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    monkeypatch.setattr(dataset, "CORPUS_SHARDS", 1)
    return calls


def _cfg(tmp_path, corpus_size):
    ds = DatasetConfig(name="miracl-ja", corpus_size=corpus_size, data_dir=tmp_path / "data")
    return Config(dataset=ds)


def test_queries_keep_only_positive_qrels(fake_hub):
    queries, positive_ids = dataset._load_queries_and_qrels("dev")
    by_qid = {q["qid"]: q for q in queries}
    assert set(by_qid) == {"q1", "q2"}  # q3 has no qrels -> dropped
    assert sorted(by_qid["q1"]["positives"]) == ["d1", "d4"]  # rel 0 (d2) excluded, rel 2 kept
    assert by_qid["q2"]["positives"] == ["d3"]
    assert positive_ids == {"d1", "d3", "d4"}


def test_prepare_includes_every_positive(fake_hub, tmp_path):
    cfg = _cfg(tmp_path, corpus_size=6)
    dataset.prepare(cfg)

    corpus = dataset.load_corpus(cfg)
    queries = dataset.load_queries(cfg)
    docids = {r["docid"] for r in corpus}
    assert len(corpus) == 6
    assert {"d1", "d3", "d4"} <= docids  # complete ground truth guaranteed
    assert len(queries) == 2
    for q in queries:
        assert set(q["positives"]) <= docids


def test_prepare_is_idempotent(fake_hub, tmp_path):
    cfg = _cfg(tmp_path, corpus_size=6)
    dataset.prepare(cfg)
    downloads = len(fake_hub)
    dataset.prepare(cfg)  # files exist -> no second download
    assert len(fake_hub) == downloads


def test_prepare_rejects_too_small_corpus(fake_hub, tmp_path):
    with pytest.raises(SystemExit, match="smaller than"):
        dataset.prepare(_cfg(tmp_path, corpus_size=2))  # 3 positives don't fit
