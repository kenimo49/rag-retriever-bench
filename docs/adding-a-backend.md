# Adding a backend

A backend is a single file under `src/rag_retriever_bench/retrievers/` that
implements `BaseRetriever` (see [`base.py`](../src/rag_retriever_bench/retrievers/base.py)).
Once the file is in place and registered in the config YAML, every existing
run mode — smoke, integration test, published report — picks it up.

The rest of this page walks through the interface, points at the simplest
in-tree implementation (Chroma) as a template, and lists the conventions the
harness relies on so your numbers stay comparable to the others.

## Interface

```python
class BaseRetriever(ABC):
    type_name: str = "base"

    def setup(self, dim: int) -> None: ...
    def load(self, docids, texts, embeddings) -> float: ...   # returns wall seconds
    def build_index(self) -> float: ...                        # returns wall seconds
    def search(self, query_embedding, top_k) -> list[str]: ... # returns docids, best first
    def describe(self) -> dict[str, Any]: ...                  # for the report header
    def self_check(self, query_embedding) -> dict[str, Any]: ...  # optional but strongly recommended
    def close(self) -> None: ...
```

Lifecycle: `setup → load → build_index → search × N → close`. `setup()` must
leave the backend empty so reruns are reproducible.

Timing convention:

- `load()` returns wall seconds for bulk-loading all documents. If your driver
  batches for you, still time the whole thing from the outside.
- `build_index()` returns wall seconds for finalizing/building the ANN index.
  If the index is built incrementally during `load()` (as with Chroma), return
  `0.0` — the harness will report it correctly.
- The harness times `search()` per query; you don't need to time it yourself.

## The simplest in-tree example

[`chroma.py`](../src/rag_retriever_bench/retrievers/chroma.py) is the shortest
implementation (~110 lines) and shows every hook the harness uses. Read it
alongside this doc; the notes below reference it directly.

## Step-by-step

1. **Add an optional-dependency extra** in `pyproject.toml`, and add your
   backend's client to the `all` extra too:

   ```toml
   [project.optional-dependencies]
   mybackend = ["mybackend-client>=1.0"]
   all = [..., "mybackend-client>=1.0"]
   ```

2. **Create `src/rag_retriever_bench/retrievers/mybackend.py`** and implement
   `BaseRetriever`. Import the client library *inside* `__init__` (not at
   module top level) so the extra remains truly optional — see
   `chroma.py:24` and `pgvector.py` for the pattern.

3. **Register it in the factory.** Add your `type_name` to the dispatch table
   in `src/rag_retriever_bench/retrievers/__init__.py`.

4. **Add a `docker compose` service** in `docker-compose.yml` if it's a
   server-type backend. Use the same port scheme as the existing services
   (localhost, no auth for the bench user). Embedded backends skip this.

5. **Add a config block** to `configs/miracl-ja.yaml`:

   ```yaml
   - type: mybackend
     label: MyBackend (HNSW)
     host: localhost
     port: 6333
     hnsw: { m: 16, ef_construction: 64, ef_search: 100 }
   ```

6. **Run the smoke test:**

   ```bash
   docker compose up -d mybackend
   rag-retriever-bench run -c configs/miracl-ja.yaml --corpus-size 10000
   ```

7. **Wire up integration tests.** Add a `pytest.mark.integration` test under
   `tests/integration/` that uses the `rrb_it_*` table/collection namespace.
   Verify results against the numpy brute-force ground truth helper.

## Fair-comparison conventions

The harness assumes every backend is trying to make the same trade-off. To
keep the report meaningful:

- **Align HNSW parameters** with the other backends: `m=16,
  ef_construction=64, ef_search=100`, cosine distance. If your backend
  can't hit those exactly, `describe()` should say so explicitly.
- **Report `mode`** if you're not a networked server. Embedded backends
  (Chroma, LanceDB) are grouped into a separate table because their
  in-process latency isn't comparable to a server's TCP round-trip.
- **Deduplicate returned docids** if your driver can return the same id
  twice (rare, but happens with certain post-filter paths). The scorer
  dedupes downstream too, but do it at the source when you can.
- **Fail loud on config drift.** If the caller asks for HNSW and your
  backend silently falls back to full scan, that has to surface — see the
  next section.

## `self_check` — catching silent full-scan

Three of the in-tree backends had a way to silently degrade to a full scan
with no error. The harness caught all three via `self_check`. This is the
single most important optional hook.

`self_check()` runs once (post-warmup, pre-timing) and its output lands in
the report next to the backend row. The convention is:

```python
def self_check(self, query_embedding) -> dict[str, Any]:
    return {
        "ann_index_used": <bool>,
        "method": "<how you know — e.g. 'EXPLAIN', 'server-reported', 'config-only'>",
        # any backend-specific evidence: EXPLAIN text, index stats, etc.
    }
```

Prefer, in order:

1. **`EXPLAIN` on the actual query** — pgvector, ClickHouse.
2. **Server-side stats** — Qdrant `indexed_vectors_count`, Milvus
   `describe_index`.
3. **Config-only** — Chroma, Weaviate (report what was configured; note
   that this is weaker evidence).

If your backend supports EXPLAIN or a plan endpoint, use it — the harness
prints a note in the report when it falls back to config-only.

## Sending a PR

Small backend PRs are welcome. Please include:

- A screenshot or paste of the 10k smoke run against your backend.
- The integration test.
- A one-line entry in the Backends table in `README.md`.

If your backend has multiple modes worth benching (indexing variants,
quantization on/off), submit them as multiple config entries with distinct
`label`s rather than as separate `type`s — that keeps the report format
uniform. `ClickHouse (HNSW, g=128)` in the sample report is an example.
