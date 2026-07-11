from rag_retriever_bench.config import load_config


def _write_config(tmp_path):
    p = tmp_path / "bench.yaml"
    p.write_text(
        "dataset:\n  corpus_size: 10000\n"
        "backends:\n  - type: pgvector\n    host: localhost\n",
        encoding="utf-8",
    )
    return p


def test_corpus_size_override(tmp_path):
    cfg = load_config(_write_config(tmp_path), corpus_size=100_000)
    assert cfg.dataset.corpus_size == 100_000


def test_env_host_override(tmp_path, monkeypatch):
    monkeypatch.setenv("RRB_PGVECTOR_HOST", "bench-host")
    cfg = load_config(_write_config(tmp_path))
    assert cfg.backends[0]["host"] == "bench-host"


def test_paths_derive_from_dataset(tmp_path):
    cfg = load_config(_write_config(tmp_path))
    assert cfg.corpus_path.as_posix() == "data/miracl-ja-10000/corpus.jsonl"
    assert cfg.embeddings_dir.as_posix().endswith("embeddings/text-embedding-3-small")
