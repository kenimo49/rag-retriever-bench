from __future__ import annotations

import argparse

from . import __version__


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rag-retriever-bench",
        description="Benchmark RAG retrieval backends on the same corpus, queries, and metrics.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare", help="download dataset, sample corpus, embed")
    p_run = sub.add_parser("run", help="load, index, search, and score every backend")
    for p in (p_prepare, p_run):
        p.add_argument("-c", "--config", required=True, help="path to a YAML config")
        p.add_argument("--corpus-size", type=int, default=None, help="override dataset.corpus_size")
    p_run.add_argument("--out", default="results", help="output directory (default: results)")
    p_run.add_argument("--skip-prepare", action="store_true", help="fail instead of preparing missing data")

    args = parser.parse_args()

    from .config import load_config

    cfg = load_config(args.config, corpus_size=args.corpus_size)

    if args.command == "prepare":
        _prepare(cfg)
    elif args.command == "run":
        if not cfg.corpus_path.exists():
            if args.skip_prepare:
                raise SystemExit(f"dataset not prepared: {cfg.corpus_path}")
            _prepare(cfg)
        from . import bench, report

        results = bench.run(cfg)
        report.save(cfg, results, out_dir=args.out)


def _prepare(cfg) -> None:
    from . import dataset, embed

    dataset.prepare(cfg)
    corpus = dataset.load_corpus(cfg)
    queries = dataset.load_queries(cfg)
    embed.embed_corpus(cfg, corpus)
    embed.embed_queries(cfg, queries)


if __name__ == "__main__":
    main()
