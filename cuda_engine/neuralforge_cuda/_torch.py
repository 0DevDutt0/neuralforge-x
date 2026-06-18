"""PyTorch GPU backend — a reference baseline using framework tensor ops.

This is the "obvious" GPU implementation a practitioner would reach for; it
serves as the correctness oracle and performance baseline that the hand-written
CUDA and Triton kernels are measured against.
"""

from __future__ import annotations

import numpy as np

from ._capability import GpuError

_METRIC_ALIASES = {
    "cos": "cosine",
    "cosine": "cosine",
    "dot": "dot",
    "dot_product": "dot",
    "ip": "dot",
    "l2": "l2",
    "euclidean": "l2",
}


def _torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent
        raise GpuError(
            "PyTorch is not installed. Install the cu128 build: "
            "pip install torch --index-url https://download.pytorch.org/whl/cu128"
        ) from exc
    if not torch.cuda.is_available():
        raise GpuError("PyTorch reports no available CUDA device")
    # Compare all backends in true fp32 (disable TF32) so results match the CPU
    # core and the CUDA/Triton kernels rather than diverging by ~1e-2.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch


def _metric(name: str) -> str:
    key = _METRIC_ALIASES.get(name.strip().lower())
    if key is None:
        raise ValueError(f"unknown metric {name!r}; expected 'cosine', 'dot', or 'l2'")
    return key


def _t(torch, x, ndim):
    arr = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    if ndim == 1:
        arr = arr.ravel()
    elif arr.ndim != 2:
        raise ValueError(f"expected {ndim}-D input, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("input must be non-empty")
    return torch.from_numpy(arr).cuda()


def gpu_batch_similarity(queries, corpus, metric: str = "cosine") -> np.ndarray:
    torch = _torch()
    m = _metric(metric)
    q = _t(torch, queries, 2)
    c = _t(torch, corpus, 2)
    if q.shape[1] != c.shape[1]:
        raise ValueError(f"queries dim {q.shape[1]} != corpus dim {c.shape[1]}")
    with torch.no_grad():
        if m == "l2":
            # Direct mode (not the matmul expansion) keeps self-distance exactly 0.
            out = torch.cdist(q, c, p=2.0, compute_mode="donot_use_mm_for_euclid_dist")
        elif m == "dot":
            out = q @ c.T
        else:  # cosine; zero-norm rows -> 0 (the eps only guards the division)
            qn = q / q.norm(dim=1, keepdim=True).clamp_min(1e-12)
            cn = c / c.norm(dim=1, keepdim=True).clamp_min(1e-12)
            out = qn @ cn.T
    return out.cpu().numpy()


def gpu_topk_search(query, corpus, k: int, metric: str = "cosine"):
    torch = _torch()
    m = _metric(metric)
    q = _t(torch, query, 1)
    c = _t(torch, corpus, 2)
    n = c.shape[0]
    if q.shape[0] != c.shape[1]:
        raise ValueError(f"query dim {q.shape[0]} != corpus dim {c.shape[1]}")
    if not 1 <= k <= n:
        raise ValueError(f"k={k} must satisfy 1 <= k <= {n}")
    with torch.no_grad():
        if m == "l2":
            scores = torch.cdist(
                q.unsqueeze(0), c, p=2.0, compute_mode="donot_use_mm_for_euclid_dist"
            ).squeeze(0)
            vals, idx = torch.topk(scores, k, largest=False)
        else:
            if m == "cosine":
                qn = q / q.norm().clamp_min(1e-12)
                cn = c / c.norm(dim=1, keepdim=True).clamp_min(1e-12)
                scores = cn @ qn
            else:
                scores = c @ q
            vals, idx = torch.topk(scores, k, largest=True)
    return idx.cpu().numpy().astype(np.int64), vals.cpu().numpy().astype(np.float32)


def _as_row(x) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(x, dtype=np.float32).ravel())
    if arr.size == 0:
        raise ValueError("input must be non-empty")
    return arr[None, :]


# Pairwise ops route through the (validated) 2-D batch kernel: 1-D reductions are
# unstable on some Torch+Blackwell builds, and a single pair on the GPU is
# launch-bound regardless, so reuse the well-tested path.
def gpu_cosine_similarity(a, b) -> float:
    return float(gpu_batch_similarity(_as_row(a), _as_row(b), "cosine")[0, 0])


def gpu_dot_product(a, b) -> float:
    return float(gpu_batch_similarity(_as_row(a), _as_row(b), "dot")[0, 0])


def gpu_l2_distance(a, b) -> float:
    return float(gpu_batch_similarity(_as_row(a), _as_row(b), "l2")[0, 0])
