"""Embedding batching/retry/cache logic with a fake OpenAI client."""

from __future__ import annotations

import json

import numpy as np
import pytest

from rag_retriever_bench import embed
from rag_retriever_bench.config import Config, DatasetConfig, EmbeddingConfig


class FakeOpenAI:
    def __init__(self, fail_times=0):
        self.fail_times = fail_times
        self.calls: list[list[str]] = []
        self.embeddings = self

    def create(self, model, input):  # noqa: A002 - mirrors the OpenAI SDK signature
        if self.fail_times > 0:
            self.fail_times -= 1
            raise ConnectionError("transient")
        self.calls.append(list(input))

        class Item:
            def __init__(self, text):
                self.embedding = [float(len(text)), 0.0]

        class Resp:
            data = [Item(t) for t in input]

        return Resp()


@pytest.fixture()
def fake_client(monkeypatch):
    client = FakeOpenAI()
    monkeypatch.setattr("openai.OpenAI", lambda: client)
    monkeypatch.setattr(embed.time, "sleep", lambda s: None)
    return client


def _cfg(tmp_path, batch_size=2):
    return Config(
        dataset=DatasetConfig(name="tiny", corpus_size=4, data_dir=tmp_path),
        embeddings=EmbeddingConfig(dim=2, batch_size=batch_size, workers=1),
    )


def test_batching_and_order(tmp_path, fake_client):
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]
    vecs = embed._embed_texts(_cfg(tmp_path), texts)
    assert vecs.shape == (5, 2)
    assert vecs.dtype == np.float32
    assert [len(batch) for batch in fake_client.calls] == [2, 2, 1]
    assert [v[0] for v in vecs] == [1.0, 2.0, 3.0, 4.0, 5.0]  # order preserved


def test_transient_error_is_retried(tmp_path, monkeypatch):
    client = FakeOpenAI(fail_times=2)
    monkeypatch.setattr("openai.OpenAI", lambda: client)
    monkeypatch.setattr(embed.time, "sleep", lambda s: None)
    vecs = embed._embed_texts(_cfg(tmp_path, batch_size=8), ["a", "bb"])
    assert vecs.shape == (2, 2)


def test_persistent_error_raises(tmp_path, monkeypatch):
    client = FakeOpenAI(fail_times=99)
    monkeypatch.setattr("openai.OpenAI", lambda: client)
    monkeypatch.setattr(embed.time, "sleep", lambda s: None)
    with pytest.raises(ConnectionError):
        embed._embed_texts(_cfg(tmp_path, batch_size=8), ["a"])


def test_cache_hit_skips_api(tmp_path, fake_client):
    cfg = _cfg(tmp_path)
    cache = cfg.embeddings_dir
    cache.mkdir(parents=True)
    cached = np.ones((2, 2), dtype=np.float32)
    np.save(cache / "corpus.npy", cached)
    (cache / "corpus.ids.json").write_text(json.dumps(["d1", "d2"]), encoding="utf-8")

    ids, vecs = embed._embed_cached(cfg, "corpus", ["d1", "d2"], ["a", "b"])
    assert ids == ["d1", "d2"]
    assert np.array_equal(vecs, cached)
    assert fake_client.calls == []  # no API call


def test_stale_cache_reembeds(tmp_path, fake_client):
    cfg = _cfg(tmp_path)
    cache = cfg.embeddings_dir
    cache.mkdir(parents=True)
    np.save(cache / "corpus.npy", np.ones((2, 2), dtype=np.float32))
    (cache / "corpus.ids.json").write_text(json.dumps(["old1", "old2"]), encoding="utf-8")

    ids, vecs = embed._embed_cached(cfg, "corpus", ["d1", "d2"], ["a", "bb"])
    assert fake_client.calls  # id mismatch -> re-embedded
    assert [v[0] for v in vecs] == [1.0, 2.0]
    # cache updated for next run
    assert json.loads((cache / "corpus.ids.json").read_text(encoding="utf-8")) == ["d1", "d2"]


def test_passage_input_prepends_title_and_caps_length():
    assert embed._passage_input({"title": "T", "text": "body"}) == "T\nbody"
    assert embed._passage_input({"title": "", "text": "body"}) == "body"
    long = embed._passage_input({"title": "", "text": "x" * 10_000})
    assert len(long) == embed.MAX_CHARS
