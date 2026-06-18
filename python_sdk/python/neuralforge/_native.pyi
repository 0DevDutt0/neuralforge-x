"""Type stubs for the native Rust extension module ``neuralforge._native``.

These signatures describe the low-level boundary; the public, validated API in
``neuralforge/__init__.py`` wraps them. All array arguments must be
C-contiguous float32.
"""

import numpy as np
from numpy.typing import NDArray

__version__: str

def cosine_similarity(a: NDArray[np.float32], b: NDArray[np.float32]) -> float: ...
def dot_product(a: NDArray[np.float32], b: NDArray[np.float32]) -> float: ...
def l2_distance(a: NDArray[np.float32], b: NDArray[np.float32]) -> float: ...
def batch_similarity(
    queries: NDArray[np.float32],
    corpus: NDArray[np.float32],
    metric: str = ...,
) -> NDArray[np.float32]: ...
def top_k_search(
    query: NDArray[np.float32],
    corpus: NDArray[np.float32],
    k: int,
    metric: str = ...,
) -> tuple[NDArray[np.int64], NDArray[np.float32]]: ...

class VectorIndex:
    """Native HNSW index. Metadata and filters cross as JSON strings."""

    def __init__(
        self,
        dim: int,
        metric: str = ...,
        m: int = ...,
        ef_construction: int = ...,
        ef_search: int = ...,
    ) -> None: ...
    def insert(self, id: int, vector: NDArray[np.float32], metadata: str | None = ...) -> None: ...
    def delete(self, id: int) -> None: ...
    def update(
        self,
        id: int,
        vector: NDArray[np.float32] | None = ...,
        metadata: str | None = ...,
    ) -> None: ...
    def search(
        self,
        query: NDArray[np.float32],
        k: int,
        ef: int = ...,
        filter: str | None = ...,
    ) -> tuple[NDArray[np.uint64], NDArray[np.float32]]: ...
    def contains(self, id: int) -> bool: ...
    def metadata_json(self, id: int) -> str | None: ...
    def compact(self) -> None: ...
    def tombstones(self) -> int: ...
    def save(self, path: str) -> None: ...
    @staticmethod
    def load(path: str) -> VectorIndex: ...
    @property
    def dim(self) -> int: ...
    @property
    def metric(self) -> str: ...
    def __len__(self) -> int: ...
