"""Reference implementations for the benchmark workloads.

These are the *baselines* — straightforward pure-Python and NumPy versions of
each kernel. They double as correctness oracles: the harness checks that every
accelerated backend (Rust, CUDA, ...) agrees with the NumPy result on small
inputs before trusting its timings, and the unit tests pin Python ≡ NumPy.

The accelerated backends are not here — they are imported from their own
packages (`neuralforge`, `neuralforge_cuda`) in :mod:`benchmark_lab.workloads`,
which is the only module that depends on them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def python_pairwise_cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two sequences in pure Python (the scalar baseline)."""
    dot = norm_a = norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    denom = math.sqrt(norm_a) * math.sqrt(norm_b)
    return 0.0 if denom == 0.0 else dot / denom


def numpy_pairwise_cosine(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """Cosine similarity of two vectors via NumPy."""
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    denom = np.linalg.norm(a64) * np.linalg.norm(b64)
    return 0.0 if denom == 0.0 else float(a64 @ b64 / denom)


def numpy_batch_cosine(
    queries: NDArray[np.float32], corpus: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Full ``(q, n)`` cosine similarity matrix via normalised GEMM."""
    qn = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    cn = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
    return (qn @ cn.T).astype(np.float32)


def numpy_topk_dot(
    query: NDArray[np.float32], corpus: NDArray[np.float32], k: int
) -> NDArray[np.int64]:
    """Indices of the top-``k`` inner products, best-first (the exact baseline)."""
    scores = corpus @ query
    part = np.argpartition(-scores, k - 1)[:k]
    return part[np.argsort(-scores[part])].astype(np.int64)


def numpy_topk_cosine(
    query: NDArray[np.float32], corpus: NDArray[np.float32], k: int
) -> NDArray[np.int64]:
    """Indices of the top-``k`` by cosine similarity, best-first (exact baseline).

    Cosine — unlike raw dot product — is a true angular metric, so it is the
    right oracle for the HNSW backend's recall (and a fair common metric across
    every retrieval backend).

    Scores are a BLAS GEMV divided by row norms computed with ``einsum`` (which
    streams the sum-of-squares without materialising ``corpus**2``), rather than
    normalising a whole copy of the corpus — the efficient implementation a NumPy
    user would write, so the Rust comparison stays fair.
    """
    q_norm = float(np.linalg.norm(query))
    corpus_norms = np.sqrt(np.einsum("ij,ij->i", corpus, corpus))
    scores = (corpus @ query) / (corpus_norms * q_norm + 1e-30)
    part = np.argpartition(-scores, k - 1)[:k]
    return part[np.argsort(-scores[part])].astype(np.int64)


def python_topk_dot(query: Sequence[float], corpus: Sequence[Sequence[float]], k: int) -> list[int]:
    """Pure-Python exact top-``k`` by inner product (small-input parity oracle)."""
    scored = [(sum(c * q for c, q in zip(row, query)), i) for i, row in enumerate(corpus)]
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [i for _, i in scored[:k]]
