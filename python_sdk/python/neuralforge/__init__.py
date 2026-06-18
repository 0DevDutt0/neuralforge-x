"""NeuralForge-X — high-performance vector similarity & retrieval kernels.

The compute core is implemented in Rust (hand-vectorised AVX2 + FMA, parallelised
with rayon) and exposed through the private :mod:`neuralforge._native` extension
module. This package is the typed, validated, NumPy-friendly public surface.

Example
-------
>>> import numpy as np, neuralforge as nf
>>> a = np.random.rand(768).astype(np.float32)
>>> b = np.random.rand(768).astype(np.float32)
>>> round(nf.cosine_similarity(a, b), 4)  # doctest: +SKIP
0.7421
>>> corpus = np.random.rand(10_000, 768).astype(np.float32)
>>> result = nf.top_k_search(a, corpus, k=5, metric="cosine")  # doctest: +SKIP
>>> result.indices[:3]                                          # doctest: +SKIP
array([8123, 91, 4570])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

from . import _native
from ._exceptions import (
    DimensionMismatchError,
    InvalidInputError,
    InvalidMetricError,
    NeuralForgeError,
)
from ._vector_db import Filter, VectorHit, VectorIndex

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Literal

    from numpy.typing import ArrayLike, NDArray

    Metric = Literal["cosine", "dot", "l2"]

__all__ = [
    "DimensionMismatchError",
    "Filter",
    "InvalidInputError",
    "InvalidMetricError",
    "NeuralForgeError",
    "SearchResult",
    "VectorHit",
    "VectorIndex",
    "__version__",
    "batch_similarity",
    "cosine_similarity",
    "dot_product",
    "l2_distance",
    "top_k_search",
]

__version__: Final[str] = _native.__version__

_LOGGER: Final = logging.getLogger("neuralforge")

# Canonical names plus the aliases the Rust layer also accepts.
_VALID_METRICS: Final[frozenset[str]] = frozenset(
    {"cosine", "cos", "dot", "dot_product", "ip", "inner_product", "l2", "euclidean", "euclid"}
)


def _as_vector(x: ArrayLike, *, name: str) -> NDArray[np.float32]:
    """Coerces ``x`` to a contiguous 1-D float32 array (zero-copy when possible)."""
    arr = np.ascontiguousarray(x, dtype=np.float32)
    if arr.ndim != 1:
        raise InvalidInputError(
            f"{name} must be 1-D, got a {arr.ndim}-D array of shape {arr.shape}"
        )
    if arr.size == 0:
        raise InvalidInputError(f"{name} must be non-empty")
    return arr


def _as_matrix(x: ArrayLike, *, name: str) -> NDArray[np.float32]:
    """Coerces ``x`` to a contiguous 2-D ``(rows, dim)`` float32 array."""
    arr = np.ascontiguousarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise InvalidInputError(
            f"{name} must be 2-D (n, dim), got a {arr.ndim}-D array of shape {arr.shape}"
        )
    if arr.size == 0:
        raise InvalidInputError(f"{name} must be non-empty")
    return arr


def _validate_metric(metric: str) -> str:
    if not isinstance(metric, str) or metric.strip().lower() not in _VALID_METRICS:
        raise InvalidMetricError(
            f"unknown metric {metric!r}; expected one of 'cosine', 'dot', 'l2'"
        )
    return metric


def cosine_similarity(a: ArrayLike, b: ArrayLike) -> float:
    """Cosine similarity ``⟨a, b⟩ / (‖a‖·‖b‖)`` of two vectors, in ``[-1, 1]``.

    A zero-norm vector yields ``0.0`` (never ``NaN``).

    Raises:
        DimensionMismatchError: if ``a`` and ``b`` have different lengths.
        InvalidInputError: if either input is not a non-empty 1-D array.
    """
    va, vb = _as_vector(a, name="a"), _as_vector(b, name="b")
    if va.shape != vb.shape:
        raise DimensionMismatchError(f"a has {va.size} elements, b has {vb.size}")
    return float(_native.cosine_similarity(va, vb))


def dot_product(a: ArrayLike, b: ArrayLike) -> float:
    """Inner product ``⟨a, b⟩`` of two vectors.

    Raises:
        DimensionMismatchError: if ``a`` and ``b`` have different lengths.
        InvalidInputError: if either input is not a non-empty 1-D array.
    """
    va, vb = _as_vector(a, name="a"), _as_vector(b, name="b")
    if va.shape != vb.shape:
        raise DimensionMismatchError(f"a has {va.size} elements, b has {vb.size}")
    return float(_native.dot_product(va, vb))


def l2_distance(a: ArrayLike, b: ArrayLike) -> float:
    """Euclidean (L2) distance ``‖a - b‖₂`` of two vectors.

    Raises:
        DimensionMismatchError: if ``a`` and ``b`` have different lengths.
        InvalidInputError: if either input is not a non-empty 1-D array.
    """
    va, vb = _as_vector(a, name="a"), _as_vector(b, name="b")
    if va.shape != vb.shape:
        raise DimensionMismatchError(f"a has {va.size} elements, b has {vb.size}")
    return float(_native.l2_distance(va, vb))


def batch_similarity(
    queries: ArrayLike, corpus: ArrayLike, metric: Metric = "cosine"
) -> NDArray[np.float32]:
    """Computes the full ``(n_queries, n_corpus)`` similarity matrix.

    Args:
        queries: ``(q, dim)`` array of query vectors.
        corpus: ``(n, dim)`` array of corpus vectors.
        metric: ``"cosine"``, ``"dot"``, or ``"l2"``.

    Returns:
        A ``(q, n)`` float32 array; entry ``[i, j]`` compares query ``i`` to
        corpus vector ``j`` under ``metric``.

    Raises:
        DimensionMismatchError: if query and corpus dimensionalities differ.
        InvalidMetricError: if ``metric`` is unknown.
        InvalidInputError: if either input is not a non-empty 2-D array.
    """
    _validate_metric(metric)
    q = _as_matrix(queries, name="queries")
    c = _as_matrix(corpus, name="corpus")
    if q.shape[1] != c.shape[1]:
        raise DimensionMismatchError(
            f"queries have dim {q.shape[1]} but corpus has dim {c.shape[1]}"
        )
    return _native.batch_similarity(q, c, metric)


@dataclass(frozen=True)
class SearchResult:
    """Outcome of a :func:`top_k_search`, ordered best-match first.

    Attributes:
        indices: ``(k,)`` int64 array of corpus row indices.
        scores: ``(k,)`` float32 array of metric values (similarity, or distance
            for the L2 metric).
    """

    indices: NDArray[np.int64]
    scores: NDArray[np.float32]

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __iter__(self) -> Iterator[tuple[int, float]]:
        """Iterates over ``(index, score)`` pairs."""
        return iter(zip(self.indices.tolist(), self.scores.tolist()))

    def __repr__(self) -> str:
        pairs = ", ".join(f"({i}, {s:.4f})" for i, s in list(self)[:5])
        suffix = ", ..." if len(self) > 5 else ""
        return f"SearchResult(k={len(self)}, top=[{pairs}{suffix}])"


def top_k_search(
    query: ArrayLike, corpus: ArrayLike, k: int, metric: Metric = "cosine"
) -> SearchResult:
    """Finds the ``k`` corpus vectors most similar to ``query``.

    Args:
        query: a ``(dim,)`` query vector.
        corpus: a ``(n, dim)`` array of corpus vectors.
        k: number of neighbours to return; must satisfy ``1 <= k <= n``.
        metric: ``"cosine"``, ``"dot"``, or ``"l2"``.

    Returns:
        A :class:`SearchResult` ordered best-match first.

    Raises:
        DimensionMismatchError: if the query width differs from the corpus width.
        InvalidMetricError: if ``metric`` is unknown.
        InvalidInputError: if ``k`` is out of range or inputs have bad shapes.
    """
    _validate_metric(metric)
    q = _as_vector(query, name="query")
    c = _as_matrix(corpus, name="corpus")
    if q.shape[0] != c.shape[1]:
        raise DimensionMismatchError(f"query has dim {q.shape[0]} but corpus has dim {c.shape[1]}")
    if not isinstance(k, (int, np.integer)) or k <= 0:
        raise InvalidInputError(f"k must be a positive integer, got {k!r}")
    if k > c.shape[0]:
        raise InvalidInputError(f"k={k} exceeds the corpus size of {c.shape[0]}")
    indices, scores = _native.top_k_search(q, c, int(k), metric)
    return SearchResult(indices=indices, scores=scores)
