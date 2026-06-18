"""Triton backend — a tiled similarity kernel (the kernel-optimization showcase).

A single ``@triton.jit`` kernel scores a block of corpus rows against one query,
tiling the reduction over the feature dimension. It is reused for both the full
batch matrix (2-D launch grid over corpus tiles x queries) and top-k scoring
(one query; selection via ``torch.topk``). Pairwise ops delegate to the PyTorch
backend — the Triton win is in the batched hot path.
"""

from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl

from ._capability import GpuError

_METRICS = {
    "cosine": 0,
    "cos": 0,
    "dot": 1,
    "dot_product": 1,
    "ip": 1,
    "l2": 2,
    "euclidean": 2,
}


def _metric(name: str) -> int:
    code = _METRICS.get(name.strip().lower())
    if code is None:
        raise ValueError(f"unknown metric {name!r}; expected 'cosine', 'dot', or 'l2'")
    return code


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise GpuError("Triton backend requires a CUDA device, but PyTorch reports none")


@triton.jit
def _sim_kernel(
    Q,
    C,
    OUT,
    QN,
    q,
    n,
    d,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    METRIC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_q = tl.program_id(1)
    rows = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = rows < n

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    cnorm = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for d0 in range(0, d, BLOCK_D):
        cols = d0 + tl.arange(0, BLOCK_D)
        mask_d = cols < d
        qv = tl.load(Q + pid_q * d + cols, mask=mask_d, other=0.0)
        cv = tl.load(
            C + rows[:, None] * d + cols[None, :], mask=mask_n[:, None] & mask_d[None, :], other=0.0
        )
        if METRIC == 2:
            diff = cv - qv[None, :]
            acc += tl.sum(diff * diff, axis=1)
        else:
            acc += tl.sum(cv * qv[None, :], axis=1)
            if METRIC == 0:
                cnorm += tl.sum(cv * cv, axis=1)

    if METRIC == 2:
        out = tl.sqrt(acc)
    elif METRIC == 0:
        qn = tl.load(QN + pid_q)
        denom = qn * tl.sqrt(cnorm)
        out = tl.where(denom > 0.0, acc / denom, 0.0)
    else:
        out = acc
    tl.store(OUT + pid_q * n + rows, out, mask=mask_n)


_BLOCK_N = 64
_BLOCK_D = 128


def _dev(x, ndim: int) -> torch.Tensor:
    arr = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    if ndim == 1:
        arr = arr.ravel()
    elif arr.ndim != 2:
        raise ValueError(f"expected {ndim}-D input, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("input must be non-empty")
    return torch.from_numpy(arr).cuda()


def gpu_batch_similarity(queries, corpus, metric: str = "cosine") -> np.ndarray:
    _require_cuda()
    m = _metric(metric)
    qt = _dev(queries, 2)
    ct = _dev(corpus, 2)
    q, d = qt.shape
    n, d_c = ct.shape
    if d != d_c:
        raise ValueError(f"queries dim {d} != corpus dim {d_c}")
    out = torch.empty((q, n), device="cuda", dtype=torch.float32)
    qn = qt.norm(dim=1).contiguous() if m == 0 else torch.empty(q, device="cuda")
    grid = (triton.cdiv(n, _BLOCK_N), q)
    _sim_kernel[grid](qt, ct, out, qn, q, n, d, BLOCK_N=_BLOCK_N, BLOCK_D=_BLOCK_D, METRIC=m)
    return out.cpu().numpy()


def gpu_topk_search(query, corpus, k: int, metric: str = "cosine"):
    _require_cuda()
    m = _metric(metric)
    qt = _dev(query, 1)
    ct = _dev(corpus, 2)
    n, d = ct.shape
    if qt.numel() != d:
        raise ValueError(f"query dim {qt.numel()} != corpus dim {d}")
    if not 1 <= k <= n:
        raise ValueError(f"k={k} must satisfy 1 <= k <= {n}")
    scores = torch.empty(n, device="cuda", dtype=torch.float32)
    qn = qt.norm().reshape(1) if m == 0 else torch.empty(1, device="cuda")
    grid = (triton.cdiv(n, _BLOCK_N), 1)
    _sim_kernel[grid](
        qt.reshape(1, d), ct, scores, qn, 1, n, d, BLOCK_N=_BLOCK_N, BLOCK_D=_BLOCK_D, METRIC=m
    )
    vals, idx = torch.topk(scores, k, largest=(m != 2))
    return idx.cpu().numpy().astype(np.int64), vals.cpu().numpy().astype(np.float32)


# Pairwise ops are not the Triton hot path; delegate to the PyTorch backend.
def gpu_cosine_similarity(a, b) -> float:
    from . import _torch

    return _torch.gpu_cosine_similarity(a, b)


def gpu_dot_product(a, b) -> float:
    from . import _torch

    return _torch.gpu_dot_product(a, b)


def gpu_l2_distance(a, b) -> float:
    from . import _torch

    return _torch.gpu_l2_distance(a, b)
