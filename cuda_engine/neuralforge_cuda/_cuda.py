"""Raw CUDA C++ backend (hand-written kernels compiled by CuPy's NVRTC).

The kernels in ``kernels/similarity.cu`` are compiled at first use for the
device's compute capability and launched on the GPU. On Blackwell (`sm_120`)
this requires NVRTC >= 12.8; see :func:`_ensure_modern_nvrtc`.
"""

from __future__ import annotations

import os
from importlib.resources import files

import numpy as np

from ._capability import GpuError, require_cuda

_MIN_NVRTC = (12, 8)
_THREADS = 256


def _ensure_modern_nvrtc() -> None:
    """Make CuPy use a Blackwell-capable NVRTC.

    CuPy prefers the toolkit pointed to by ``CUDA_PATH`` for NVRTC. On a machine
    whose *system* toolkit predates Blackwell (``sm_120`` needs CUDA >= 12.8),
    that NVRTC cannot emit a runnable kernel. CuPy bundles its own newer NVRTC,
    so we drop a stale ``CUDA_PATH`` for *this process only* to let it win. This
    mutates neither the user's environment nor the system toolkit.
    """
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path and not any(tag in cuda_path for tag in ("12.8", "12.9", "13.")):
        os.environ.pop("CUDA_PATH", None)


_ensure_modern_nvrtc()

# We deliberately cleared a stale CUDA_PATH above; CuPy then runs entirely on its
# bundled libraries, so its "CUDA path could not be detected" notice is expected.
import warnings  # noqa: E402

warnings.filterwarnings("ignore", message="CUDA path could not be detected.*", category=UserWarning)

import cupy as cp  # noqa: E402  (imported after the NVRTC fix above)

_KERNEL_SOURCE = (files(__package__) / "kernels" / "similarity.cu").read_text(encoding="utf-8")
_MODULE: cp.RawModule | None = None

_METRICS = {"cosine": 0, "dot": 1, "l2": 2}


def _metric_code(metric: str) -> int:
    key = metric.strip().lower()
    aliases = {"cos": "cosine", "dot_product": "dot", "ip": "dot", "euclidean": "l2"}
    key = aliases.get(key, key)
    if key not in _METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected 'cosine', 'dot', or 'l2'")
    return _METRICS[key]


def _module() -> cp.RawModule:
    global _MODULE
    if _MODULE is None:
        require_cuda()
        version = cp.cuda.nvrtc.getVersion()
        if version < _MIN_NVRTC:
            raise GpuError(
                f"NVRTC {version} cannot target Blackwell sm_120 (need >= 12.8). "
                "Install 'nvidia-cuda-nvrtc-cu12>=12.8' or clear a stale CUDA_PATH."
            )
        _MODULE = cp.RawModule(code=_KERNEL_SOURCE, options=("--std=c++14",))
    return _MODULE


def _grid_1d(n: int) -> tuple[int]:
    return ((n + _THREADS - 1) // _THREADS,)


def _as_f32_2d(x, name: str) -> cp.ndarray:
    arr = cp.ascontiguousarray(cp.asarray(x, dtype=cp.float32))
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D (n, dim), got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    return arr


def _as_f32_1d(x, name: str) -> cp.ndarray:
    arr = cp.ascontiguousarray(cp.asarray(x, dtype=cp.float32).ravel())
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    return arr


def gpu_batch_similarity(queries, corpus, metric: str = "cosine") -> np.ndarray:
    """Full ``(q, n)`` similarity matrix on the GPU. Returns a NumPy array."""
    require_cuda()
    code = _metric_code(metric)
    q_arr = _as_f32_2d(queries, "queries")
    c_arr = _as_f32_2d(corpus, "corpus")
    q, dim = q_arr.shape
    n, dim_c = c_arr.shape
    if dim != dim_c:
        raise ValueError(f"queries dim {dim} != corpus dim {dim_c}")

    mod = _module()
    q_norm = cp.empty(q, dtype=cp.float32)
    c_norm = cp.empty(n, dtype=cp.float32)
    if code == 0:  # cosine — precompute norms once
        row_norms = mod.get_function("row_norms")
        row_norms(_grid_1d(q), (_THREADS,), (q_arr, np.int32(q), np.int32(dim), q_norm))
        row_norms(_grid_1d(n), (_THREADS,), (c_arr, np.int32(n), np.int32(dim), c_norm))

    out = cp.empty((q, n), dtype=cp.float32)
    block = (16, 16)
    grid = ((n + 15) // 16, (q + 15) // 16)
    mod.get_function("batch_sim")(
        grid,
        block,
        (
            q_arr,
            c_arr,
            np.int32(q),
            np.int32(n),
            np.int32(dim),
            np.int32(code),
            q_norm,
            c_norm,
            out,
        ),
    )
    return cp.asnumpy(out)


def gpu_topk_search(query, corpus, k: int, metric: str = "cosine"):
    """Top-``k`` ``(np.int64[k], np.float32[k])``.

    The O(n·d) scoring runs on the GPU (custom kernel); the final O(n) selection
    runs on the host via NumPy ``argpartition``. Besides being cheap, this avoids
    CuPy's CUB-based sort kernels, which fail to compile against this NVRTC/CCCL
    header combination on Blackwell.
    """
    require_cuda()
    code = _metric_code(metric)
    q_host = np.ascontiguousarray(np.asarray(query, dtype=np.float32).ravel())
    if q_host.size == 0:
        raise ValueError("query must be non-empty")
    c_arr = _as_f32_2d(corpus, "corpus")
    n, dim = c_arr.shape
    if q_host.size != dim:
        raise ValueError(f"query dim {q_host.size} != corpus dim {dim}")
    if not 1 <= k <= n:
        raise ValueError(f"k={k} must satisfy 1 <= k <= {n}")

    q_arr = cp.asarray(q_host)
    mod = _module()
    c_norm = cp.empty(n, dtype=cp.float32)
    q_norm = np.float32(0.0)
    if code == 0:
        mod.get_function("row_norms")(
            _grid_1d(n), (_THREADS,), (c_arr, np.int32(n), np.int32(dim), c_norm)
        )
        q_norm = np.float32(float(np.sqrt(np.dot(q_host, q_host))))  # tiny, on host

    scores_dev = cp.empty(n, dtype=cp.float32)
    mod.get_function("score_query")(
        _grid_1d(n),
        (_THREADS,),
        (q_arr, c_arr, np.int32(n), np.int32(dim), np.int32(code), q_norm, c_norm, scores_dev),
    )
    scores = cp.asnumpy(scores_dev)  # D2H, O(n)

    rank = scores if code == 2 else -scores  # cosine/dot higher-better; L2 lower-better
    part = np.argpartition(rank, k - 1)[:k]
    order = part[np.argsort(rank[part])]
    return order.astype(np.int64), scores[order]


def _pair_reduce(a, b) -> tuple[float, float, float]:
    require_cuda()
    a_arr = _as_f32_1d(a, "a")
    b_arr = _as_f32_1d(b, "b")
    if a_arr.size != b_arr.size:
        raise ValueError(f"a has {a_arr.size} elements, b has {b_arr.size}")
    out = cp.zeros(3, dtype=cp.float32)
    shared = 3 * _THREADS * 4  # 3 float arrays of blockDim.x
    _module().get_function("pair_reduce")(
        (1,), (_THREADS,), (a_arr, b_arr, np.int32(a_arr.size), out), shared_mem=shared
    )
    dot, na2, nb2 = (float(v) for v in cp.asnumpy(out))
    return dot, na2, nb2


def gpu_cosine_similarity(a, b) -> float:
    """Cosine similarity of two vectors on the GPU (block reduction kernel)."""
    dot, na2, nb2 = _pair_reduce(a, b)
    den = (na2**0.5) * (nb2**0.5)
    return dot / den if den > 0.0 else 0.0


def gpu_dot_product(a, b) -> float:
    """Inner product of two vectors on the GPU."""
    return _pair_reduce(a, b)[0]


def gpu_l2_distance(a, b) -> float:
    """Euclidean distance of two vectors on the GPU."""
    dot, na2, nb2 = _pair_reduce(a, b)
    return max(na2 - 2.0 * dot + nb2, 0.0) ** 0.5
