"""Micro-benchmark: NeuralForge-X (Rust) vs a tuned NumPy baseline.

Times batched cosine similarity and top-k retrieval on random data, prints a
Markdown table, and writes machine-readable results to
``benchmark_lab/results/python_vs_rust.json``.

Usage::

    python examples/02_numpy_vs_rust_benchmark.py
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Callable

import numpy as np

import neuralforge as nf

REPEATS = 5
DIM = 768
CORPUS_SIZES = [10_000, 100_000]
N_QUERIES = 32
TOP_K = 10


def _best_of(fn: Callable[[], object], repeats: int = REPEATS) -> float:
    """Returns the best (minimum) wall-clock time in milliseconds over `repeats`."""
    fn()  # warm-up (allocations, thread pool spin-up)
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best * 1e3


def _numpy_batch_cosine(q: np.ndarray, c: np.ndarray) -> np.ndarray:
    qn = q / np.linalg.norm(q, axis=1, keepdims=True)
    cn = c / np.linalg.norm(c, axis=1, keepdims=True)
    return qn @ cn.T


def _numpy_top_k_dot(query: np.ndarray, c: np.ndarray, k: int) -> np.ndarray:
    scores = c @ query
    part = np.argpartition(-scores, k)[:k]
    return part[np.argsort(-scores[part])]


def main() -> None:
    rng = np.random.default_rng(7)
    rows = []

    for n in CORPUS_SIZES:
        corpus = rng.standard_normal((n, DIM)).astype(np.float32)
        queries = rng.standard_normal((N_QUERIES, DIM)).astype(np.float32)
        query = queries[0]

        # Bind loop vars as defaults so each lambda closes over this iteration.
        np_batch = _best_of(lambda q=queries, c=corpus: _numpy_batch_cosine(q, c))
        nf_batch = _best_of(lambda q=queries, c=corpus: nf.batch_similarity(q, c, "cosine"))

        np_topk = _best_of(lambda q=query, c=corpus: _numpy_top_k_dot(q, c, TOP_K))
        nf_topk = _best_of(lambda q=query, c=corpus: nf.top_k_search(q, c, TOP_K, "dot"))

        rows.append(
            {
                "corpus_size": n,
                "dim": DIM,
                "batch_numpy_ms": round(np_batch, 3),
                "batch_neuralforge_ms": round(nf_batch, 3),
                "batch_speedup": round(np_batch / nf_batch, 2),
                "topk_numpy_ms": round(np_topk, 3),
                "topk_neuralforge_ms": round(nf_topk, 3),
                "topk_speedup": round(np_topk / nf_topk, 2),
            }
        )

    # --- Pretty print -------------------------------------------------------
    print(f"NeuralForge-X v{nf.__version__}  |  {platform.processor() or platform.machine()}")
    print(f"dim={DIM}, queries={N_QUERIES}, k={TOP_K}, best-of-{REPEATS}\n")
    header = (
        "| corpus | batch NumPy (ms) | batch NF (ms) | speedup "
        "| top-k NumPy (ms) | top-k NF (ms) | speedup |"
    )
    print(header)
    print("|" + "---|" * 7)
    for r in rows:
        cells = [
            f"{r['corpus_size']:>7,}",
            f"{r['batch_numpy_ms']:>14}",
            f"{r['batch_neuralforge_ms']:>11}",
            f"{r['batch_speedup']:>5}x",
            f"{r['topk_numpy_ms']:>14}",
            f"{r['topk_neuralforge_ms']:>11}",
            f"{r['topk_speedup']:>5}x",
        ]
        print("| " + " | ".join(cells) + " |")

    # --- Persist ------------------------------------------------------------
    out_dir = Path(__file__).resolve().parents[1] / "benchmark_lab" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "python_vs_rust.json"
    payload = {
        "tool": "examples/02_numpy_vs_rust_benchmark.py",
        "version": nf.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "config": {"dim": DIM, "n_queries": N_QUERIES, "top_k": TOP_K, "repeats": REPEATS},
        "results": rows,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path.relative_to(out_dir.parents[1])}")


if __name__ == "__main__":
    main()
