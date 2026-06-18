"""Typed, validated Python surface for the HNSW vector store.

The heavy lifting — the graph, the filtered beam search, Parquet persistence —
lives in Rust (:mod:`neuralforge._native`). This module is the ergonomic layer:
it coerces vectors to contiguous float32, serialises metadata and filter
predicates to the JSON the native side expects, and returns plain Python
objects. Metadata is any JSON-scalar mapping; filters are composed with the
:class:`Filter` builder and the ``&``, ``|``, ``~`` operators.

Example
-------
>>> import numpy as np
>>> from neuralforge import VectorIndex, Filter
>>> index = VectorIndex(dim=3, metric="cosine")
>>> index.add(1, np.array([1, 0, 0], dtype=np.float32), {"lang": "rust"})
>>> index.add(2, np.array([0.9, 0.1, 0.0], dtype=np.float32), {"lang": "python"})
>>> hits = index.search(np.array([1, 0, 0], dtype=np.float32), k=1,
...                      filter=Filter.eq("lang", "rust"))
>>> hits[0].id
1
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from . import _native
from ._exceptions import InvalidInputError

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from typing import Literal

    from numpy.typing import ArrayLike, NDArray

    Metric = Literal["cosine", "dot", "l2"]
    Scalar = str | int | float | bool | None
    MetaDict = Mapping[str, Scalar]

__all__ = ["Filter", "VectorHit", "VectorIndex"]

_LOGGER: Final = logging.getLogger("neuralforge")


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


def _dump_metadata(metadata: MetaDict | None) -> str | None:
    """Serialises a metadata mapping to a JSON object string, or ``None``."""
    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        raise InvalidInputError("metadata must be a mapping of str -> scalar")
    for key in metadata:
        if not isinstance(key, str):
            raise InvalidInputError(f"metadata keys must be strings, got {key!r}")
    try:
        return json.dumps(metadata)
    except (TypeError, ValueError) as exc:
        raise InvalidInputError(f"metadata is not JSON-serialisable: {exc}") from exc


class Filter:
    """A composable metadata predicate for filtered search.

    Build leaves with the static constructors and combine them with the
    bitwise operators: ``a & b`` (and), ``a | b`` (or), ``~a`` (not). The object
    is an opaque handle around the JSON the native layer evaluates; it mirrors
    the Rust ``Filter`` enum exactly.

    Example
    -------
    >>> f = Filter.eq("lang", "rust") & (Filter.ge("year", 2024) | Filter.exists("pinned"))
    """

    __slots__ = ("_node",)

    def __init__(self, node: Any) -> None:
        self._node = node

    # -- leaves ------------------------------------------------------------
    @staticmethod
    def eq(field: str, value: Scalar) -> Filter:
        """Field equals ``value``."""
        return Filter({"Eq": [field, value]})

    @staticmethod
    def ne(field: str, value: Scalar) -> Filter:
        """Field is absent or differs from ``value``."""
        return Filter({"Ne": [field, value]})

    @staticmethod
    def lt(field: str, bound: float) -> Filter:
        """Field is numeric and ``< bound``."""
        return Filter({"Lt": [field, float(bound)]})

    @staticmethod
    def le(field: str, bound: float) -> Filter:
        """Field is numeric and ``<= bound``."""
        return Filter({"Le": [field, float(bound)]})

    @staticmethod
    def gt(field: str, bound: float) -> Filter:
        """Field is numeric and ``> bound``."""
        return Filter({"Gt": [field, float(bound)]})

    @staticmethod
    def ge(field: str, bound: float) -> Filter:
        """Field is numeric and ``>= bound``."""
        return Filter({"Ge": [field, float(bound)]})

    @staticmethod
    def is_in(field: str, values: Sequence[Scalar]) -> Filter:
        """Field is one of ``values``."""
        return Filter({"In": [field, list(values)]})

    @staticmethod
    def exists(field: str) -> Filter:
        """Field is present with a non-null value."""
        return Filter({"Exists": field})

    @staticmethod
    def missing(field: str) -> Filter:
        """Field is absent or null."""
        return Filter({"Missing": field})

    # -- combinators -------------------------------------------------------
    def __and__(self, other: Filter) -> Filter:
        return Filter({"And": [self._node, other._node]})

    def __or__(self, other: Filter) -> Filter:
        return Filter({"Or": [self._node, other._node]})

    def __invert__(self) -> Filter:
        return Filter({"Not": self._node})

    def to_json(self) -> str:
        """The JSON encoding handed to the native layer."""
        return json.dumps(self._node)

    def __repr__(self) -> str:
        return f"Filter({self._node!r})"


@dataclass(frozen=True)
class VectorHit:
    """A single search result: the caller's id and its metric score."""

    id: int
    score: float

    def __iter__(self) -> Iterator[float]:
        """Iterates as ``(id, score)`` so a hit unpacks like a tuple."""
        return iter((self.id, self.score))


class VectorIndex:
    """An HNSW approximate-nearest-neighbour index with metadata filtering.

    Args:
        dim: vector dimensionality.
        metric: ``"cosine"`` (default), ``"dot"``, or ``"l2"``.
        m: target graph out-degree; higher means denser, higher recall, more memory.
        ef_construction: build-time beam width.
        ef_search: default query-time beam width (overridable per search).
    """

    __slots__ = ("_native",)

    def __init__(
        self,
        dim: int,
        metric: Metric = "cosine",
        *,
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 64,
    ) -> None:
        if not isinstance(dim, (int, np.integer)) or dim <= 0:
            raise InvalidInputError(f"dim must be a positive integer, got {dim!r}")
        self._native = _native.VectorIndex(
            int(dim), metric, int(m), int(ef_construction), int(ef_search)
        )

    @classmethod
    def _wrap(cls, native: Any) -> VectorIndex:
        obj = cls.__new__(cls)
        obj._native = native
        return obj

    @property
    def dim(self) -> int:
        """Vector dimensionality."""
        return self._native.dim

    @property
    def metric(self) -> str:
        """The ranking metric name."""
        return self._native.metric

    def add(self, id: int, vector: ArrayLike, metadata: MetaDict | None = None) -> None:
        """Inserts ``vector`` under a fresh ``id`` with optional ``metadata``.

        Raises:
            KeyError: if ``id`` is already present.
            InvalidInputError: on a bad vector shape, dimension, or non-finite value.
        """
        v = _as_vector(vector, name="vector")
        self._native.insert(int(id), v, _dump_metadata(metadata))

    def delete(self, id: int) -> None:
        """Soft-deletes a live ``id``.

        Raises:
            KeyError: if ``id`` is not present.
        """
        self._native.delete(int(id))

    def update(
        self,
        id: int,
        vector: ArrayLike | None = None,
        metadata: MetaDict | None = None,
    ) -> None:
        """Replaces a live ``id``'s vector and/or metadata.

        A new ``vector`` rewires the graph; metadata is carried forward unless
        also supplied. With neither argument this is a no-op.

        Raises:
            KeyError: if ``id`` is not present.
        """
        v = None if vector is None else _as_vector(vector, name="vector")
        self._native.update(int(id), v, _dump_metadata(metadata))

    def search(
        self,
        query: ArrayLike,
        k: int,
        *,
        ef: int = 0,
        filter: Filter | None = None,
    ) -> list[VectorHit]:
        """Returns up to ``k`` nearest live vectors, best match first.

        Args:
            query: a ``(dim,)`` query vector.
            k: number of neighbours requested.
            ef: search beam width; ``0`` uses the index default. Larger trades
                latency for recall.
            filter: an optional :class:`Filter`; a selective filter may yield
                fewer than ``k`` hits.

        Raises:
            InvalidInputError: if ``k`` is out of range or the query shape is wrong.
        """
        q = _as_vector(query, name="query")
        if not isinstance(k, (int, np.integer)) or k <= 0:
            raise InvalidInputError(f"k must be a positive integer, got {k!r}")
        filter_json = None if filter is None else filter.to_json()
        ids, scores = self._native.search(q, int(k), int(ef), filter_json)
        return [VectorHit(int(i), float(s)) for i, s in zip(ids.tolist(), scores.tolist())]

    def get_metadata(self, id: int) -> dict[str, Any] | None:
        """Returns the metadata dict for a live ``id``, or ``None``."""
        raw = self._native.metadata_json(int(id))
        return None if raw is None else json.loads(raw)

    def __contains__(self, id: int) -> bool:
        return self._native.contains(int(id))

    def __len__(self) -> int:
        return len(self._native)

    def tombstones(self) -> int:
        """Number of soft-deleted vectors still occupying graph memory."""
        return self._native.tombstones()

    def compact(self) -> None:
        """Rebuilds the graph to physically drop tombstones and reclaim memory."""
        self._native.compact()

    def save(self, path: str) -> None:
        """Writes a self-describing Parquet snapshot of the live vectors."""
        self._native.save(str(path))

    @classmethod
    def load(cls, path: str) -> VectorIndex:
        """Loads an index from a Parquet snapshot, rebuilding the graph."""
        return cls._wrap(_native.VectorIndex.load(str(path)))

    def __repr__(self) -> str:
        return (
            f"VectorIndex(dim={self.dim}, metric={self.metric!r}, "
            f"len={len(self)}, tombstones={self.tombstones()})"
        )
