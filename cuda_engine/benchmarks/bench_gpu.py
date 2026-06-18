"""Four-way benchmark: NumPy vs Rust(CPU) vs CUDA vs Triton (+ PyTorch baseline).

End-to-end timings (host array in, host array out) so every backend is measured
on the same, fair basis. Writes ``benchmark_lab/results/gpu_bench.json``.

Usage (from repo root, venv active, neuralforge_cuda importable)::

    python cuda_engine/benchmarks/bench_gpu.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import neuralforge_cuda as gpu
import numpy as np

import neuralforge as nf

REPEATS = 5
DIM = 768
CORPUS_SIZES = [50_000, 200_000]
N_QUERIES = 64
TOP_K = 10


def best_ms(fn) -> float:
    fn()  # warm-up: compiles kernels, spins thread pools, primes caches
    best = float("inf")
    for _ in range(REPEATS):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best * 1e3


def numpy_batch_cosine(q: np.ndarray, c: np.ndarray) -> np.ndarray:
    qn = q / np.linalg.norm(q, axis=1, keepdims=True)
    cn = c / np.linalg.norm(c, axis=1, keepdims=True)
    return qn @ cn.T


def numpy_topk_dot(query: np.ndarray, c: np.ndarray, k: int) -> np.ndarray:
    scores = c @ query
    part = np.argpartition(-scores, k)[:k]
    return part[np.argsort(-scores[part])]


def main() -> None:
    rng = np.random.default_rng(7)
    backends = [b for b in ("cuda", "triton", "torch") if b in gpu.available_backends()]
    info = gpu.device_info()
    print(f"GPU: {info['name']} (cc {info['compute_capability']}, {info['total_mem_mb']} MB)")
    print(f"backends: {backends} | dim={DIM}, queries={N_QUERIES}, k={TOP_K}, best-of-{REPEATS}\n")

    rows = []
    for n in CORPUS_SIZES:
        corpus = rng.standard_normal((n, DIM)).astype(np.float32)
        queries = rng.standard_normal((N_QUERIES, DIM)).astype(np.float32)
        query = queries[0]
        rec: dict[str, float | int] = {"corpus": n}

        rec["batch_numpy_ms"] = best_ms(lambda q=queries, c=corpus: numpy_batch_cosine(q, c))
        rec["batch_rust_ms"] = best_ms(
            lambda q=queries, c=corpus: nf.batch_similarity(q, c, "cosine")
        )
        for b in backends:
            rec[f"batch_{b}_ms"] = best_ms(
                lambda q=queries, c=corpus, bk=b: gpu.gpu_batch_similarity(
                    q, c, "cosine", backend=bk
                )
            )

        rec["topk_numpy_ms"] = best_ms(lambda q=query, c=corpus: numpy_topk_dot(q, c, TOP_K))
        rec["topk_rust_ms"] = best_ms(lambda q=query, c=corpus: nf.top_k_search(q, c, TOP_K, "dot"))
        for b in backends:
            rec[f"topk_{b}_ms"] = best_ms(
                lambda q=query, c=corpus, bk=b: gpu.gpu_topk_search(q, c, TOP_K, "dot", backend=bk)
            )
        rows.append({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()})

    _print_table("Batch cosine similarity (ms, lower better)", "batch", rows, backends)
    _print_table("Top-k dot search (ms, lower better)", "topk", rows, backends)

    out_dir = Path(__file__).resolve().parents[2] / "benchmark_lab" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "device": info,
        "config": {"dim": DIM, "n_queries": N_QUERIES, "top_k": TOP_K, "repeats": REPEATS},
        "backends": backends,
        "results": rows,
    }
    out_path = out_dir / "gpu_bench.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


def _print_table(title: str, op: str, rows: list[dict], backends: list[str]) -> None:
    cols = ["numpy", "rust", *backends]
    print(title)
    header = "| corpus  | " + " | ".join(f"{c:>9}" for c in cols) + " | best vs rust |"
    print(header)
    print("|" + "---|" * (len(cols) + 2))
    for r in rows:
        cells = []
        gpu_times = []
        for c in cols:
            val = r.get(f"{op}_{c}_ms")
            cells.append(f"{val:>9}" if val is not None else f"{'-':>9}")
            if c in backends and val is not None:
                gpu_times.append(val)
        rust = r.get(f"{op}_rust_ms")
        speed = f"{rust / min(gpu_times):.1f}x" if gpu_times and rust else "-"
        print(f"| {r['corpus']:>7,} | " + " | ".join(cells) + f" | {speed:>12} |")
    print()


if __name__ == "__main__":
    main()
