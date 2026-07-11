"""Dataset preparation: MIRACL-ja corpus sampling + dev queries with qrels.

MIRACL's HF dataset uses a loading script (unsupported by datasets>=3), but
the underlying files are plain TSV (topics, qrels) and JSONL.gz (corpus), so
we download them directly via huggingface_hub — no `datasets` dependency.

Output format (both JSONL):
  corpus.jsonl:  {"docid": str, "title": str, "text": str}
  queries.jsonl: {"qid": str, "text": str, "positives": [docid, ...]}

The sample always contains every positive passage referenced by the query
set, so recall@k is measured against a complete ground truth.
"""

from __future__ import annotations

import gzip
import json
import random
from pathlib import Path

from tqdm import tqdm

from .config import Config

QUERIES_REPO = "miracl/miracl"
CORPUS_REPO = "miracl/miracl-corpus"
LANG = "ja"
CORPUS_SHARDS = 14  # miracl-corpus-v1.0-ja/docs-{0..13}.jsonl.gz, ~6.95M passages


def prepare(cfg: Config) -> None:
    out_dir = cfg.corpus_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.corpus_path.exists() and cfg.queries_path.exists():
        print(f"dataset already prepared: {out_dir}")
        return

    queries, positive_ids = _load_queries_and_qrels(cfg.dataset.split)
    n_fill = cfg.dataset.corpus_size - len(positive_ids)
    if n_fill < 0:
        raise SystemExit(
            f"corpus_size={cfg.dataset.corpus_size} is smaller than the "
            f"{len(positive_ids)} positive passages required by the query set"
        )
    print(
        f"{len(queries)} queries, {len(positive_ids)} positive passages; "
        f"reservoir-sampling {n_fill} filler passages from {CORPUS_REPO} ({LANG})"
    )

    positives, fillers = _sample_corpus(positive_ids, n_fill, cfg.dataset.seed)

    missing = positive_ids - set(positives)
    if missing:
        raise SystemExit(f"{len(missing)} positive docids not found in corpus (e.g. {sorted(missing)[:3]})")

    corpus = list(positives.values()) + fillers
    random.Random(cfg.dataset.seed).shuffle(corpus)

    _write_jsonl(cfg.corpus_path, corpus)
    _write_jsonl(cfg.queries_path, queries)
    print(f"wrote {len(corpus)} passages -> {cfg.corpus_path}")
    print(f"wrote {len(queries)} queries  -> {cfg.queries_path}")


def _load_queries_and_qrels(split: str) -> tuple[list[dict], set[str]]:
    from huggingface_hub import hf_hub_download

    topics_path = hf_hub_download(
        QUERIES_REPO,
        f"miracl-v1.0-{LANG}/topics/topics.miracl-v1.0-{LANG}-{split}.tsv",
        repo_type="dataset",
    )
    qrels_path = hf_hub_download(
        QUERIES_REPO,
        f"miracl-v1.0-{LANG}/qrels/qrels.miracl-v1.0-{LANG}-{split}.tsv",
        repo_type="dataset",
    )

    positives_by_qid: dict[str, list[str]] = {}
    with open(qrels_path, encoding="utf-8") as f:
        for line in f:
            qid, _q0, docid, rel = line.split()
            if int(rel) >= 1:
                positives_by_qid.setdefault(qid, []).append(docid)

    queries: list[dict] = []
    with open(topics_path, encoding="utf-8") as f:
        for line in f:
            qid, text = line.rstrip("\n").split("\t", 1)
            if qid in positives_by_qid:
                queries.append({"qid": qid, "text": text, "positives": positives_by_qid[qid]})

    all_positive_ids = {d for docids in positives_by_qid.values() for d in docids}
    return queries, all_positive_ids


def _sample_corpus(
    positive_ids: set[str], n_fill: int, seed: int
) -> tuple[dict[str, dict], list[dict]]:
    """One pass over all corpus shards: collect positives, reservoir-sample fillers."""
    from huggingface_hub import hf_hub_download

    rng = random.Random(seed)
    positives: dict[str, dict] = {}
    reservoir: list[dict] = []
    seen = 0

    for shard in tqdm(range(CORPUS_SHARDS), desc="corpus shards"):
        path = hf_hub_download(
            CORPUS_REPO,
            f"miracl-corpus-v1.0-{LANG}/docs-{shard}.jsonl.gz",
            repo_type="dataset",
        )
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                doc = {"docid": row["docid"], "title": row.get("title", ""), "text": row["text"]}
                if doc["docid"] in positive_ids:
                    positives[doc["docid"]] = doc
                    continue
                if len(reservoir) < n_fill:
                    reservoir.append(doc)
                else:
                    j = rng.randrange(seen + 1)
                    if j < n_fill:
                        reservoir[j] = doc
                seen += 1

    return positives, reservoir


def load_corpus(cfg: Config) -> list[dict]:
    return _read_jsonl(cfg.corpus_path)


def load_queries(cfg: Config) -> list[dict]:
    return _read_jsonl(cfg.queries_path)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]
