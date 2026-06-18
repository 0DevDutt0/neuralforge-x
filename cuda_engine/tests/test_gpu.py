"""Correctness tests for the GPU backends, checked against the CPU core.

Skipped automatically when no CUDA device is present (e.g. on CI). Every
available backend (cuda / torch / triton) is validated against ``neuralforge``.
"""

from __future__ import annotations

import numpy as np
import pytest

nf = pytest.importorskip("neuralforge")
gpu = pytest.importorskip("neuralforge_cuda")

pytestmark = pytest.mark.skipif(not gpu.cuda_available(), reason="no CUDA device available")

BACKENDS = gpu.available_backends()
RNG = np.random.default_rng(20260617)
CORPUS = RNG.standard_normal((1500, 192)).astype(np.float32)
QUERIES = RNG.standard_normal((8, 192)).astype(np.float32)
METRICS = ["cosine", "dot", "l2"]


def test_at_least_cuda_backend_present():
    assert "cuda" in BACKENDS, f"expected the CUDA backend to be available; got {BACKENDS}"


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("metric", METRICS)
def test_batch_matches_cpu(backend: str, metric: str) -> None:
    got = gpu.gpu_batch_similarity(QUERIES, CORPUS, metric, backend=backend)
    ref = nf.batch_similarity(QUERIES, CORPUS, metric)
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("metric", METRICS)
def test_topk_matches_cpu(backend: str, metric: str) -> None:
    k = 10
    idx, scores = gpu.gpu_topk_search(CORPUS[5], CORPUS, k, metric, backend=backend)
    ref = nf.top_k_search(CORPUS[5], CORPUS, k, metric)
    # The query is corpus row 5 -> it must be the best match.
    assert idx[0] == 5
    # Compare score multisets (index identity can differ on boundary ties across
    # float paths, but the selected scores must agree).
    np.testing.assert_allclose(np.sort(scores), np.sort(ref.scores), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("backend", BACKENDS)
def test_pairwise_matches_cpu(backend: str) -> None:
    a, b = QUERIES[0], CORPUS[0]
    assert gpu.gpu_cosine_similarity(a, b, backend=backend) == pytest.approx(
        nf.cosine_similarity(a, b), abs=1e-3
    )
    assert gpu.gpu_l2_distance(a, b, backend=backend) == pytest.approx(
        nf.l2_distance(a, b), rel=1e-3
    )


def test_device_info_reports_blackwell_or_newer() -> None:
    info = gpu.device_info()
    assert "name" in info and info["total_mem_mb"] > 0
    assert int(info["compute_capability"]) >= 70  # sane CUDA arch
