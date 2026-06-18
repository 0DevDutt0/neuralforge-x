"""Benchmark workloads — the orchestration layer.

Each workload runs one kernel across every applicable backend and a sweep of
problem sizes, verifies the accelerated backends against the NumPy oracle, and
returns a JSON-serialisable record of :class:`~benchmark_lab.metrics.Measurement`
data (plus GPU samples and, for the ANN backend, recall).

Three workloads span the stack the project spec calls out — Python, NumPy, Rust,
CUDA:

* ``pairwise_cosine`` — one vector pair over a dimension sweep: **Python → NumPy
  → Rust**, the scalar-to-SIMD ladder.
* ``batch_cosine`` — a ``q × n`` similarity matrix: **NumPy → Rust → GPU**
  (CUDA / Triton / PyTorch), the throughput headline.
* ``topk_search`` — k-nearest retrieval: **NumPy → Rust → GPU**, plus the Phase-4
  **HNSW** index as an approximate backend reporting latency *and* recall.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

import neuralforge as nf

from . import backends
from .gpu import sample_gpu
from .metrics import measure


@dataclass(frozen=True)
class BenchConfig:
    """Problem sizes and sampling parameters for a run."""

    pairwise_dims: tuple[int, ...] = (128, 384, 768, 1536)
    batch_corpus: tuple[int, ...] = (50_000, 200_000)
    topk_corpus: tuple[int, ...] = (50_000, 200_000)
    dim: int = 768
    queries: int = 64
    top_k: int = 10
    repeats: int = 7
    seed: int = 7
    use_gpu: bool = True
    gpu_window_s: float = 0.4
    # The ANN backend is built in a Python insert loop, so we cap the corpus it
    # indexes; larger top-k sizes still benchmark the exact NumPy/Rust/GPU paths.
    hnsw_max_n: int = 100_000

    @staticmethod
    def quick() -> BenchConfig:
        """A tiny, fast configuration for tests and smoke runs."""
        return BenchConfig(
            pairwise_dims=(64, 256),
            batch_corpus=(2_000,),
            topk_corpus=(2_000,),
            dim=64,
            queries=8,
            top_k=5,
            repeats=3,
            use_gpu=False,
            gpu_window_s=0.1,
        )


# Names of the registered workloads, for the CLI and docs.
WORKLOADS: tuple[str, ...] = ("pairwise_cosine", "batch_cosine", "topk_search")


def _gpu_backends(use_gpu: bool) -> list[str]:
    """The available GPU backends, or ``[]`` when GPU work is off/unavailable."""
    if not use_gpu:
        return []
    try:
        import neuralforge_cuda as gpu

        if not gpu.cuda_available():
            return []
        return [b for b in ("cuda", "triton", "torch") if b in gpu.available_backends()]
    except Exception:
        return []


def _topk_indices(result: Any) -> np.ndarray:
    """Normalises the various top-k return shapes to a 1-D index array."""
    if hasattr(result, "indices"):
        return np.asarray(result.indices)
    if isinstance(result, tuple):
        return np.asarray(result[0])
    return np.asarray(result)


# --------------------------------------------------------------------------
# Workload 1 — pairwise cosine: Python → NumPy → Rust.
# --------------------------------------------------------------------------
def run_pairwise_cosine(cfg: BenchConfig) -> dict[str, Any]:
    rng = np.random.default_rng(cfg.seed)
    sizes: list[dict[str, Any]] = []

    for dim in cfg.pairwise_dims:
        a = rng.standard_normal(dim).astype(np.float32)
        b = rng.standard_normal(dim).astype(np.float32)
        a_list, b_list = a.tolist(), b.tolist()
        reference = backends.numpy_pairwise_cosine(a, b)

        runners: dict[str, Callable[[], Any]] = {
            "python": lambda al=a_list, bl=b_list: backends.python_pairwise_cosine(al, bl),
            "numpy": lambda av=a, bv=b: backends.numpy_pairwise_cosine(av, bv),
            "rust": lambda av=a, bv=b: nf.cosine_similarity(av, bv),
        }
        measurements = []
        for name, fn in runners.items():
            _verify_scalar(name, float(fn()), reference)
            measurements.append(measure(name, fn, work_items=dim, repeats=cfg.repeats).as_dict())
        sizes.append({"params": {"dim": dim}, "measurements": measurements})

    return {
        "workload": "pairwise_cosine",
        "unit": "elements",
        "baseline": "python",
        "lower_is_better": True,
        "sizes": sizes,
    }


# --------------------------------------------------------------------------
# Workload 2 — batch cosine similarity: NumPy → Rust → GPU.
# --------------------------------------------------------------------------
def run_batch_cosine(cfg: BenchConfig) -> dict[str, Any]:
    rng = np.random.default_rng(cfg.seed)
    gpu_backends = _gpu_backends(cfg.use_gpu)
    gpu_mod = _import_gpu() if gpu_backends else None
    sizes: list[dict[str, Any]] = []

    for n in cfg.batch_corpus:
        corpus = rng.standard_normal((n, cfg.dim)).astype(np.float32)
        queries = rng.standard_normal((cfg.queries, cfg.dim)).astype(np.float32)
        reference = backends.numpy_batch_cosine(queries, corpus)
        work = cfg.queries * n

        runners: dict[str, Callable[[], Any]] = {
            "numpy": lambda q=queries, c=corpus: backends.numpy_batch_cosine(q, c),
            "rust": lambda q=queries, c=corpus: nf.batch_similarity(q, c, "cosine"),
        }
        for b in gpu_backends:
            runners[b] = lambda q=queries, c=corpus, bk=b: gpu_mod.gpu_batch_similarity(
                q, c, "cosine", backend=bk
            )

        measurements = []
        for name, fn in runners.items():
            _verify_close(name, np.asarray(fn()), reference, atol=1e-3)
            m = measure(name, fn, work_items=work, repeats=cfg.repeats)
            record = m.as_dict()
            if name in gpu_backends:
                _attach_gpu(record, fn, cfg.gpu_window_s)
            measurements.append(record)
        sizes.append(
            {
                "params": {"corpus": n, "queries": cfg.queries, "dim": cfg.dim},
                "measurements": measurements,
            }
        )

    return {
        "workload": "batch_cosine",
        "unit": "similarities",
        "baseline": "numpy",
        "lower_is_better": True,
        "sizes": sizes,
    }


# --------------------------------------------------------------------------
# Workload 3 — top-k retrieval: NumPy → Rust → GPU, plus HNSW (ANN + recall).
# --------------------------------------------------------------------------
def run_topk_search(cfg: BenchConfig) -> dict[str, Any]:
    rng = np.random.default_rng(cfg.seed)
    gpu_backends = _gpu_backends(cfg.use_gpu)
    gpu_mod = _import_gpu() if gpu_backends else None
    k = cfg.top_k
    sizes: list[dict[str, Any]] = []

    for n in cfg.topk_corpus:
        # Clustered (Gaussian-mixture) data, not pure noise: in high dimensions
        # uniform-random vectors are near-orthogonal with no neighbour structure,
        # so *any* ANN method scores ~0 recall. Real embeddings are clustered, and
        # that is what makes nearest-neighbour search — and HNSW recall — meaningful.
        corpus, query = _clustered_corpus(n, cfg.dim, rng)
        # Cosine is a true metric, so HNSW navigates it well and recall is
        # meaningful; every retrieval backend ranks by the same cosine order.
        exact = set(backends.numpy_topk_cosine(query, corpus, k).tolist())

        runners: dict[str, Callable[[], Any]] = {
            "numpy": lambda q=query, c=corpus: backends.numpy_topk_cosine(q, c, k),
            "rust": lambda q=query, c=corpus: nf.top_k_search(q, c, k, "cosine"),
        }
        for b in gpu_backends:
            runners[b] = lambda q=query, c=corpus, bk=b: gpu_mod.gpu_topk_search(
                q, c, k, "cosine", backend=bk
            )

        measurements = []
        for name, fn in runners.items():
            got = set(_topk_indices(fn()).tolist())
            _verify_recall(name, got, exact, exact_required=True)
            m = measure(name, fn, work_items=n, repeats=cfg.repeats)
            record = m.as_dict()
            if name in gpu_backends:
                _attach_gpu(record, fn, cfg.gpu_window_s)
            measurements.append(record)

        # HNSW: approximate retrieval over a prebuilt index (cosine-normalised
        # inputs make dot-ranked exact top-k the right oracle for recall). Capped
        # to keep the Python-loop build time bounded on the largest corpora.
        if n <= cfg.hnsw_max_n:
            measurements.append(_measure_hnsw(corpus, query, k, exact, cfg))

        sizes.append(
            {"params": {"corpus": n, "dim": cfg.dim, "k": k}, "measurements": measurements}
        )

    return {
        "workload": "topk_search",
        "unit": "vectors_scanned",
        "baseline": "numpy",
        "lower_is_better": True,
        "sizes": sizes,
    }


def _clustered_corpus(n: int, dim: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """A Gaussian-mixture corpus with real neighbour structure, and a query.

    Points are drawn around a set of random cluster centres with modest noise, so
    cosine nearest-neighbours are well-defined (cluster mates). The query is a
    lightly perturbed corpus point, guaranteeing it has genuine near neighbours.
    """
    n_clusters = max(8, n // 500)
    centers = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    labels = rng.integers(0, n_clusters, size=n)
    noise = 0.35 * rng.standard_normal((n, dim)).astype(np.float32)
    corpus = (centers[labels] + noise).astype(np.float32)
    anchor = int(rng.integers(0, n))
    query = (corpus[anchor] + 0.15 * rng.standard_normal(dim).astype(np.float32)).astype(np.float32)
    return corpus, query


def _measure_hnsw(
    corpus: np.ndarray, query: np.ndarray, k: int, exact: set[int], cfg: BenchConfig
) -> dict[str, Any]:
    """Builds an HNSW index, times a query, and reports recall@k vs exact."""
    index = nf.VectorIndex(dim=corpus.shape[1], metric="cosine")
    for i, vec in enumerate(corpus):
        index.add(i, vec)

    def query_fn() -> Any:
        return index.search(query, k=k, ef=64)

    got = {h.id for h in query_fn()}
    recall = len(got & exact) / max(1, len(exact))
    record = measure("hnsw", query_fn, work_items=corpus.shape[0], repeats=cfg.repeats).as_dict()
    record["recall_at_k"] = round(recall, 4)
    record["approximate"] = True
    return record


# --------------------------------------------------------------------------
# Helpers.
# --------------------------------------------------------------------------
def _import_gpu() -> Any:
    import neuralforge_cuda as gpu

    return gpu


def _attach_gpu(record: dict[str, Any], fn: Callable[[], Any], window_s: float) -> None:
    """Adds NVML GPU utilisation/memory to a measurement record, if available."""
    sample = sample_gpu(fn, window_s=window_s)
    if sample is not None:
        record["gpu"] = sample.as_dict()


def _verify_scalar(name: str, got: float, reference: float, atol: float = 1e-4) -> None:
    if not np.isclose(got, reference, atol=atol, rtol=1e-3):
        raise ValueError(f"{name} disagrees with NumPy: {got} vs {reference}")


def _verify_close(name: str, got: np.ndarray, reference: np.ndarray, atol: float) -> None:
    if got.shape != reference.shape or not np.allclose(got, reference, atol=atol, rtol=1e-3):
        raise ValueError(f"{name} output disagrees with the NumPy baseline")


def _verify_recall(name: str, got: set[int], exact: set[int], *, exact_required: bool) -> None:
    if exact_required and got != exact:
        overlap = len(got & exact)
        raise ValueError(f"{name} is not exact: {overlap}/{len(exact)} of the top-k match")


def run_all(cfg: BenchConfig | None = None) -> dict[str, Any]:
    """Runs every workload and returns the full result document."""
    from .environment import collect_environment

    cfg = cfg or BenchConfig()
    workloads = [
        run_pairwise_cosine(cfg),
        run_batch_cosine(cfg),
        run_topk_search(cfg),
    ]
    return {
        "schema": "neuralforge-x/benchmark/1",
        "environment": collect_environment(),
        "config": _config_dict(cfg),
        "workloads": workloads,
    }


def _config_dict(cfg: BenchConfig) -> dict[str, Any]:
    return {
        "pairwise_dims": list(cfg.pairwise_dims),
        "batch_corpus": list(cfg.batch_corpus),
        "topk_corpus": list(cfg.topk_corpus),
        "dim": cfg.dim,
        "queries": cfg.queries,
        "top_k": cfg.top_k,
        "repeats": cfg.repeats,
        "use_gpu": cfg.use_gpu,
    }
