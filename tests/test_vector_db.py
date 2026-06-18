"""Tests for the Python ``VectorIndex`` surface over the Rust HNSW store.

These exercise the binding and the typed wrapper end to end: recall against an
exact NumPy baseline, metadata filtering, soft-delete/update semantics, the
``Filter`` builder's JSON encoding, and a Parquet save/load round-trip.
"""

from __future__ import annotations

import numpy as np
import pytest

import neuralforge as nf
from neuralforge import Filter, VectorHit, VectorIndex

RNG = np.random.default_rng(20260618)


def _corpus(n: int, d: int) -> np.ndarray:
    return RNG.standard_normal((n, d)).astype(np.float32)


def _exact_top_k(corpus: np.ndarray, query: np.ndarray, k: int) -> set[int]:
    """Exact cosine top-k via the project's own batch kernel as ground truth."""
    sims = nf.batch_similarity(query[None, :], corpus, metric="cosine")[0]
    return set(np.argsort(-sims)[:k].tolist())


def test_search_recall_matches_exact_baseline() -> None:
    n, d, k = 2_000, 64, 10
    corpus = _corpus(n, d)
    index = VectorIndex(dim=d, metric="cosine")
    for i in range(n):
        index.add(i, corpus[i])
    assert len(index) == n
    assert index.dim == d
    assert index.metric == "cosine"

    total = 0
    for _ in range(20):
        q = RNG.standard_normal(d).astype(np.float32)
        exact = _exact_top_k(corpus, q, k)
        hits = index.search(q, k=k, ef=128)
        assert all(isinstance(h, VectorHit) for h in hits)
        # Scores must be sorted best-first (cosine: descending).
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)
        total += len(exact & {h.id for h in hits})
    recall = total / (20 * k)
    assert recall > 0.9, f"recall@{k} too low: {recall:.3f}"


def test_metadata_filter_constrains_results() -> None:
    index = VectorIndex(dim=8)
    for i in range(300):
        v = RNG.standard_normal(8).astype(np.float32)
        index.add(i, v, {"bucket": i % 4, "flag": bool(i % 2)})

    q = RNG.standard_normal(8).astype(np.float32)
    f = Filter.eq("bucket", 1) & Filter.eq("flag", True)
    hits = index.search(q, k=10, ef=128, filter=f)
    assert hits, "expected some matches"
    for h in hits:
        md = index.get_metadata(h.id)
        assert md is not None
        assert md["bucket"] == 1
        assert md["flag"] is True


def test_filter_builder_json_shapes() -> None:
    import json

    assert json.loads(Filter.eq("a", "x").to_json()) == {"Eq": ["a", "x"]}
    assert json.loads(Filter.ge("y", 2).to_json()) == {"Ge": ["y", 2.0]}
    assert json.loads(Filter.is_in("z", [1, 2]).to_json()) == {"In": ["z", [1, 2]]}
    combined = (Filter.eq("a", 1) | Filter.exists("b")).to_json()
    assert json.loads(combined) == {"Or": [{"Eq": ["a", 1]}, {"Exists": "b"}]}
    assert json.loads((~Filter.missing("c")).to_json()) == {"Not": {"Missing": "c"}}


def test_delete_and_update_semantics() -> None:
    index = VectorIndex(dim=3, metric="cosine")
    index.add(1, np.array([1, 0, 0], dtype=np.float32), {"tag": "a"})
    index.add(2, np.array([0, 1, 0], dtype=np.float32), {"tag": "b"})
    index.add(3, np.array([0, 0, 1], dtype=np.float32), {"tag": "c"})

    # Metadata-only update keeps the vector.
    index.update(2, metadata={"tag": "B"})
    assert index.get_metadata(2)["tag"] == "B"

    # Vector update carries metadata forward and rewires the graph.
    index.update(3, vector=np.array([0.99, 0.01, 0.0], dtype=np.float32))
    assert index.get_metadata(3)["tag"] == "c"
    hits = index.search(np.array([1, 0, 0], dtype=np.float32), k=1)
    assert hits[0].id in (1, 3)

    index.delete(1)
    assert 1 not in index
    assert len(index) == 2
    assert index.tombstones() >= 1
    index.compact()
    assert index.tombstones() == 0
    assert len(index) == 2


def test_save_load_round_trip(tmp_path) -> None:
    index = VectorIndex(dim=5, metric="l2")
    for i in range(120):
        index.add(i, RNG.standard_normal(5).astype(np.float32), {"i": i})
    index.delete(0)

    path = tmp_path / "snapshot.parquet"
    index.save(str(path))
    loaded = VectorIndex.load(str(path))

    assert len(loaded) == len(index)
    assert loaded.metric == "l2"
    assert loaded.dim == 5
    assert 0 not in loaded
    assert loaded.get_metadata(5)["i"] == 5


def test_invalid_inputs_raise() -> None:
    index = VectorIndex(dim=4)
    index.add(1, np.zeros(4, dtype=np.float32))

    with pytest.raises(nf.InvalidInputError):
        index.add(2, np.zeros((2, 2), dtype=np.float32))  # not 1-D
    with pytest.raises(KeyError):
        index.add(1, np.ones(4, dtype=np.float32))  # duplicate id
    with pytest.raises(KeyError):
        index.delete(999)  # unknown id
    with pytest.raises(nf.InvalidInputError):
        index.search(np.zeros(4, dtype=np.float32), k=0)  # bad k
    with pytest.raises(ValueError):
        index.add(3, np.array([np.nan, 0, 0, 0], dtype=np.float32))  # non-finite
