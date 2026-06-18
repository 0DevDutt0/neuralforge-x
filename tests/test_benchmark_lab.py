"""Tests for the cross-stack benchmark lab.

These pin the parts that must be correct regardless of timing noise: the
statistics helpers, backend parity (the accelerated paths must match the NumPy
oracle), chart/report rendering, and a full quick end-to-end run.
"""

from __future__ import annotations

import numpy as np
import pytest
from benchmark_lab import backends, collect_environment, measure, render_markdown
from benchmark_lab.charts import grouped_bar_svg, render_charts
from benchmark_lab.metrics import _percentile
from benchmark_lab.workloads import BenchConfig, run_all

import neuralforge as nf

RNG = np.random.default_rng(0)


# --- metrics --------------------------------------------------------------
def test_percentile_interpolates() -> None:
    assert _percentile([10.0], 90) == 10.0
    assert _percentile([0.0, 10.0], 50) == pytest.approx(5.0)
    assert _percentile([0.0, 1.0, 2.0, 3.0, 4.0], 90) == pytest.approx(3.6)


def test_measure_reports_consistent_throughput() -> None:
    m = measure("noop", lambda: sum(range(100)), work_items=100, repeats=5, warmup=1)
    assert m.backend == "noop"
    assert m.repeats == 5
    assert m.best_ms >= 0.0
    assert m.median_ms >= m.best_ms - 1e-9
    # throughput == work_items / best_seconds
    assert m.throughput_per_s == pytest.approx(100 / (m.best_ms / 1e3), rel=1e-6)
    assert m.peak_host_kib >= 0.0


# --- backend parity -------------------------------------------------------
def test_pairwise_python_numpy_rust_agree() -> None:
    a = RNG.standard_normal(256).astype(np.float32)
    b = RNG.standard_normal(256).astype(np.float32)
    ref = backends.numpy_pairwise_cosine(a, b)
    assert backends.python_pairwise_cosine(a.tolist(), b.tolist()) == pytest.approx(ref, abs=1e-4)
    assert nf.cosine_similarity(a, b) == pytest.approx(ref, abs=1e-4)


def test_batch_and_topk_baselines_match_rust() -> None:
    corpus = RNG.standard_normal((500, 32)).astype(np.float32)
    queries = RNG.standard_normal((4, 32)).astype(np.float32)
    np_batch = backends.numpy_batch_cosine(queries, corpus)
    rust_batch = np.asarray(nf.batch_similarity(queries, corpus, "cosine"))
    assert np.allclose(np_batch, rust_batch, atol=1e-3)

    q = queries[0]
    np_idx = set(backends.numpy_topk_dot(q, corpus, 10).tolist())
    py_idx = set(backends.python_topk_dot(q.tolist(), corpus.tolist(), 10))
    rust_idx = set(nf.top_k_search(q, corpus, 10, "dot").indices.tolist())
    assert np_idx == py_idx == rust_idx


# --- environment ----------------------------------------------------------
def test_environment_has_core_keys() -> None:
    env = collect_environment()
    for key in ("platform", "processor", "python", "cpu_count_logical"):
        assert key in env


# --- charts ---------------------------------------------------------------
def test_grouped_bar_svg_is_wellformed_and_scaled() -> None:
    svg = grouped_bar_svg(
        title="t",
        subtitle="s",
        group_labels=["a", "b"],
        series=["numpy", "rust"],
        value_at=lambda g, s: 2.0 if s == "rust" else 1.0,
        value_label=lambda v: f"{v:g}×",
        baseline=(1.0, "1× base"),
    )
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    assert svg.count("<rect") >= 4  # panel + bars
    assert "2×" in svg and "rust" in svg
    # The emerald rust accent must appear.
    assert "#34d399" in svg


def test_render_charts_from_quick_run() -> None:
    doc = run_all(BenchConfig.quick())
    charts = render_charts(doc)
    assert "bench_pairwise.svg" in charts
    assert "bench_batch.svg" in charts
    assert "bench_topk.svg" in charts
    for svg in charts.values():
        assert svg.startswith("<svg")
        assert "</svg>" in svg


# --- report + end-to-end --------------------------------------------------
def test_quick_run_all_structure_and_report() -> None:
    doc = run_all(BenchConfig.quick())
    names = [w["workload"] for w in doc["workloads"]]
    assert names == ["pairwise_cosine", "batch_cosine", "topk_search"]

    # The top-k workload must include the HNSW backend with a recall figure.
    topk = next(w for w in doc["workloads"] if w["workload"] == "topk_search")
    hnsw = [m for size in topk["sizes"] for m in size["measurements"] if m["backend"] == "hnsw"]
    assert hnsw and all("recall_at_k" in m for m in hnsw)

    md = render_markdown(doc)
    assert "## Environment" in md
    assert "pairwise_cosine" in md
    assert "| backend |" in md
