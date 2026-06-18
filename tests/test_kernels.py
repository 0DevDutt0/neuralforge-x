"""Numerical parity tests: the Rust kernels must agree with NumPy references."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

import neuralforge as nf

RNG = np.random.default_rng(20260617)


def _np_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    na, nb = np.linalg.norm(a64), np.linalg.norm(b64)
    if na == 0 or nb == 0:
        return 0.0
    return float(a64 @ b64 / (na * nb))


def test_pairwise_kernels_match_numpy() -> None:
    a = RNG.standard_normal(768).astype(np.float32)
    b = RNG.standard_normal(768).astype(np.float32)
    assert nf.cosine_similarity(a, b) == pytest.approx(_np_cosine(a, b), abs=1e-4)
    assert nf.dot_product(a, b) == pytest.approx(float(a @ b), rel=1e-4)
    assert nf.l2_distance(a, b) == pytest.approx(float(np.linalg.norm(a - b)), rel=1e-4)


def test_known_small_values() -> None:
    a = np.array([1, 2, 3], dtype=np.float32)
    b = np.array([4, 5, 6], dtype=np.float32)
    assert nf.dot_product(a, b) == pytest.approx(32.0)
    assert nf.l2_distance(np.zeros(2, np.float32), np.array([3, 4], np.float32)) == pytest.approx(
        5.0
    )


def test_accepts_python_lists_and_non_float32() -> None:
    # The wrapper coerces dtype/contiguity; integer lists are fine.
    assert nf.dot_product([1, 2, 3], [1, 1, 1]) == pytest.approx(6.0)


def test_batch_similarity_cosine_matches_numpy() -> None:
    q = RNG.standard_normal((8, 64)).astype(np.float32)
    c = RNG.standard_normal((50, 64)).astype(np.float32)
    got = nf.batch_similarity(q, c, "cosine")
    qn = q / np.linalg.norm(q, axis=1, keepdims=True)
    cn = c / np.linalg.norm(c, axis=1, keepdims=True)
    ref = qn @ cn.T
    assert got.shape == (8, 50)
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, ref, atol=1e-4)


def test_batch_similarity_l2_matches_numpy() -> None:
    q = RNG.standard_normal((4, 16)).astype(np.float32)
    c = RNG.standard_normal((20, 16)).astype(np.float32)
    got = nf.batch_similarity(q, c, "l2")
    ref = np.linalg.norm(q[:, None, :] - c[None, :, :], axis=2)
    np.testing.assert_allclose(got, ref, atol=1e-3)


@pytest.mark.parametrize("metric", ["cosine", "dot", "l2"])
def test_top_k_matches_bruteforce(metric: str) -> None:
    c = RNG.standard_normal((300, 32)).astype(np.float32)
    query = RNG.standard_normal(32).astype(np.float32)
    k = 7
    res = nf.top_k_search(query, c, k=k, metric=metric)
    assert len(res) == k

    if metric == "dot":
        scores = c @ query
        order = np.argsort(-scores)[:k]
    elif metric == "cosine":
        cn = c / np.linalg.norm(c, axis=1, keepdims=True)
        qn = query / np.linalg.norm(query)
        scores = cn @ qn
        order = np.argsort(-scores)[:k]
    else:  # l2
        scores = np.linalg.norm(c - query, axis=1)
        order = np.argsort(scores)[:k]

    assert list(res.indices) == list(order)
    np.testing.assert_allclose(res.scores, scores[order], atol=1e-3)


def test_search_result_is_iterable_and_sized() -> None:
    c = RNG.standard_normal((10, 4)).astype(np.float32)
    res = nf.top_k_search(c[0], c, k=3, metric="cosine")
    pairs = list(res)
    assert len(pairs) == 3
    assert pairs[0][0] == 0  # the query equals corpus row 0 → best match
    assert pairs[0][1] == pytest.approx(1.0, abs=1e-4)


@given(
    arrays(
        np.float32,
        st.integers(1, 256),
        elements=st.floats(-1e3, 1e3, width=32, allow_nan=False, allow_infinity=False),
    )
)
@settings(max_examples=75, deadline=None)
def test_cosine_self_similarity_is_one_or_zero(v: np.ndarray) -> None:
    # Self-similarity is 1.0, except for a zero-norm vector where it is 0.0 by
    # convention. Decide via the kernel's own dot to stay self-consistent in f32.
    got = nf.cosine_similarity(v, v)
    if nf.dot_product(v, v) == 0.0:
        assert got == 0.0
    else:
        assert got == pytest.approx(1.0, abs=1e-3)
