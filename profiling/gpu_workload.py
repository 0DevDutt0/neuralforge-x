"""A steady GPU workload for Nsight profiling.

Mirrors the Rust `neuralforge_profile` target on the GPU side: it drives the
`neuralforge_cuda` kernels in a loop for a fixed duration so Nsight Systems /
Compute have a stable signal to trace. Profiling the real engine (not a
re-implementation) means the timeline reflects exactly the H2D/D2H copies and
kernels that run in production.

    python profiling/gpu_workload.py [--seconds N] [--corpus N] [--backend cuda|triton|torch]
"""

from __future__ import annotations

import argparse
import time

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--corpus", type=int, default=100_000)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--queries", type=int, default=64)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--backend", default="cuda", choices=("cuda", "triton", "torch"))
    args = parser.parse_args()

    import neuralforge_cuda as gpu

    if not gpu.cuda_available():
        raise SystemExit("CUDA not available; cannot run the GPU profiling workload.")
    if args.backend not in gpu.available_backends():
        raise SystemExit(f"backend {args.backend!r} unavailable; have {gpu.available_backends()}")

    rng = np.random.default_rng(7)
    corpus = rng.standard_normal((args.corpus, args.dim)).astype(np.float32)
    queries = rng.standard_normal((args.queries, args.dim)).astype(np.float32)
    query = queries[0]

    info = gpu.device_info()
    print(
        f"GPU workload on {info['name']} (cc {info['compute_capability']}), backend={args.backend}"
    )

    deadline = time.perf_counter() + args.seconds
    iters = 0
    while time.perf_counter() < deadline:
        gpu.gpu_batch_similarity(queries, corpus, "cosine", backend=args.backend)
        gpu.gpu_topk_search(query, corpus, args.k, "cosine", backend=args.backend)
        iters += 1
    print(f"completed {iters} iterations in ~{args.seconds:.0f}s")


if __name__ == "__main__":
    main()
