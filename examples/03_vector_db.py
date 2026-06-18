"""HNSW vector database: filtered ANN search + DuckDB over the Parquet snapshot.

This shows the Phase-4 store end to end:

1. Build an in-memory HNSW index of synthetic "document" embeddings, each with
   metadata (a category and a year).
2. Run a metadata-filtered approximate-nearest-neighbour search.
3. Persist the live set to a self-describing Parquet file.
4. Query that same Parquet directly with **DuckDB** — exact, filtered, SQL-native
   cosine ranking over the ``vector`` LIST column — and confirm the approximate
   index agrees with the exact analytical baseline.

Run from the repo root with the project's virtualenv active::

    python examples/03_vector_db.py

DuckDB is optional; the script skips step 4 with a hint if it is not installed.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

import neuralforge as nf
from neuralforge import Filter, VectorIndex

CATEGORIES = ("science", "history", "sports", "tech")
DIM = 96
N = 5_000
K = 5


def build_index() -> tuple[VectorIndex, np.ndarray, list[dict]]:
    """Builds an index of N random unit-ish embeddings with metadata."""
    rng = np.random.default_rng(7)
    vectors = rng.standard_normal((N, DIM)).astype(np.float32)
    records = []
    index = VectorIndex(dim=DIM, metric="cosine", m=16, ef_construction=200)
    for i in range(N):
        meta = {"category": CATEGORIES[i % len(CATEGORIES)], "year": 2000 + (i % 25)}
        index.add(i, vectors[i], meta)
        records.append(meta)
    return index, vectors, records


def duckdb_baseline(path: Path, query: np.ndarray, category: str) -> list[tuple[int, float]]:
    """Exact, filtered cosine top-k computed in SQL over the Parquet snapshot."""
    import duckdb

    q = query.astype(np.float64).tolist()
    sql = """
        SELECT id,
               list_cosine_similarity(vector, ?::FLOAT[]) AS sim
        FROM read_parquet(?)
        WHERE json_extract_string(metadata, '$.category') = ?
        ORDER BY sim DESC
        LIMIT ?
    """
    rows = duckdb.execute(sql, [q, str(path), category, K]).fetchall()
    return [(int(r[0]), float(r[1])) for r in rows]


def main() -> None:
    print(f"NeuralForge-X v{nf.__version__} - vector DB demo\n")

    index, vectors, _ = build_index()
    print(f"Indexed {len(index)} vectors  ({index.dim}-d, metric={index.metric!r})")

    # A query near document 1234 (category cycles, so 1234 -> 'sports').
    rng = np.random.default_rng(99)
    query = vectors[1234] + 0.05 * rng.standard_normal(DIM).astype(np.float32)
    category = CATEGORIES[1234 % len(CATEGORIES)]

    # --- 1) Filtered approximate search via HNSW ----------------------------
    flt = Filter.eq("category", category) & Filter.ge("year", 2000)
    hits = index.search(query, k=K, ef=128, filter=flt)
    print(f"\nFiltered ANN search (category={category!r}, k={K}):")
    for rank, h in enumerate(hits, start=1):
        meta = index.get_metadata(h.id)
        print(f"  #{rank}: id={h.id:<5} cos={h.score:.4f}  {meta}")

    # --- 2) Persist to Parquet ---------------------------------------------
    path = Path(tempfile.gettempdir()) / "neuralforge_docs.parquet"
    index.save(str(path))
    print(f"\nSaved snapshot -> {path}  ({path.stat().st_size / 1024:.0f} KiB)")

    reloaded = VectorIndex.load(str(path))
    print(f"Reloaded index: {reloaded!r}")

    # --- 3) Exact DuckDB baseline over the same file ------------------------
    try:
        baseline = duckdb_baseline(path, query, category)
    except ModuleNotFoundError:
        print("\n(DuckDB not installed — `pip install duckdb` to see the SQL baseline.)")
        return

    print("\nExact filtered top-k via DuckDB SQL over the Parquet:")
    for rank, (doc_id, sim) in enumerate(baseline, start=1):
        print(f"  #{rank}: id={doc_id:<5} cos={sim:.4f}")

    ann_ids = {h.id for h in hits}
    exact_ids = {doc_id for doc_id, _ in baseline}
    overlap = len(ann_ids & exact_ids)
    print(f"\nANN vs exact agreement: {overlap}/{K} of the top-{K} match.")


if __name__ == "__main__":
    main()
