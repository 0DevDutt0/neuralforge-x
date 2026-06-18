"""Quickstart: the five core kernels in under a minute.

Run from the repo root with the project's virtualenv active::

    python examples/01_quickstart.py
"""

from __future__ import annotations

import numpy as np

import neuralforge as nf


def main() -> None:
    print(f"NeuralForge-X v{nf.__version__}\n")

    rng = np.random.default_rng(0)
    dim = 768

    # --- Pairwise kernels ---------------------------------------------------
    a = rng.standard_normal(dim).astype(np.float32)
    b = rng.standard_normal(dim).astype(np.float32)
    print("Pairwise (768-dim random vectors):")
    print(f"  cosine_similarity = {nf.cosine_similarity(a, b): .6f}")
    print(f"  dot_product       = {nf.dot_product(a, b): .4f}")
    print(f"  l2_distance       = {nf.l2_distance(a, b): .4f}\n")

    # --- Batch similarity ---------------------------------------------------
    queries = rng.standard_normal((4, dim)).astype(np.float32)
    corpus = rng.standard_normal((10_000, dim)).astype(np.float32)
    sims = nf.batch_similarity(queries, corpus, metric="cosine")
    print(f"batch_similarity: {queries.shape[0]} queries x {corpus.shape[0]} corpus")
    print(f"  -> matrix shape {sims.shape}, dtype {sims.dtype}\n")

    # --- Top-k retrieval ----------------------------------------------------
    query = corpus[42] + 0.01 * rng.standard_normal(dim).astype(np.float32)
    result = nf.top_k_search(query, corpus, k=5, metric="cosine")
    print("top_k_search (k=5, cosine) for a vector near corpus row 42:")
    for rank, (idx, score) in enumerate(result, start=1):
        print(f"  #{rank}: corpus[{idx:>5}]  score={score:.4f}")


if __name__ == "__main__":
    main()
