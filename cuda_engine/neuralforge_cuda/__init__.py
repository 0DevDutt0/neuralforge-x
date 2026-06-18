"""NeuralForge-X GPU acceleration engine — CUDA C++ and Triton kernels.

This is a **hard GPU** module: there is no CPU fallback here (that is the role of
the ``neuralforge`` core). Three interchangeable backends implement the same
operations, selectable via ``backend=``:

- ``"cuda"``   — hand-written CUDA C++ kernels compiled by CuPy's NVRTC (default).
- ``"triton"`` — Triton kernels (the kernel-optimization showcase).
- ``"torch"``  — a PyTorch reference baseline.

Example
-------
>>> import numpy as np, neuralforge_cuda as gpu
>>> corpus = np.random.rand(100_000, 768).astype(np.float32)
>>> q = corpus[7]
>>> idx, scores = gpu.gpu_topk_search(q, corpus, k=5, metric="cosine")  # doctest: +SKIP
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._capability import GpuError, cuda_available, device_info, require_cuda

if TYPE_CHECKING:
    from typing import Literal

    Backend = Literal["cuda", "triton", "torch"]

__all__ = [
    "GpuError",
    "__version__",
    "available_backends",
    "cuda_available",
    "device_info",
    "gpu_batch_similarity",
    "gpu_cosine_similarity",
    "gpu_dot_product",
    "gpu_l2_distance",
    "gpu_topk_search",
    "require_cuda",
]

__version__ = "0.1.0"


def _load(backend: str):
    if backend == "cuda":
        from . import _cuda as mod
    elif backend == "triton":
        from . import _triton as mod
    elif backend == "torch":
        from . import _torch as mod
    else:
        raise ValueError(f"unknown backend {backend!r}; expected 'cuda', 'triton', or 'torch'")
    return mod


def available_backends() -> list[str]:
    """Return the backends whose dependencies import successfully on this machine."""
    found = []
    for name in ("cuda", "triton", "torch"):
        try:
            _load(name)
            found.append(name)
        except Exception:
            pass
    return found


def gpu_cosine_similarity(a, b, *, backend: Backend = "cuda") -> float:
    """Cosine similarity of two vectors on the GPU."""
    return _load(backend).gpu_cosine_similarity(a, b)


def gpu_dot_product(a, b, *, backend: Backend = "cuda") -> float:
    """Inner product of two vectors on the GPU."""
    return _load(backend).gpu_dot_product(a, b)


def gpu_l2_distance(a, b, *, backend: Backend = "cuda") -> float:
    """Euclidean distance of two vectors on the GPU."""
    return _load(backend).gpu_l2_distance(a, b)


def gpu_batch_similarity(queries, corpus, metric: str = "cosine", *, backend: Backend = "cuda"):
    """Full ``(q, n)`` similarity matrix on the GPU (NumPy array out)."""
    return _load(backend).gpu_batch_similarity(queries, corpus, metric)


def gpu_topk_search(query, corpus, k: int, metric: str = "cosine", *, backend: Backend = "cuda"):
    """Top-``k`` ``(indices, scores)`` on the GPU."""
    return _load(backend).gpu_topk_search(query, corpus, k, metric)
