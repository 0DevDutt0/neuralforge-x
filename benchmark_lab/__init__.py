"""NeuralForge-X benchmark lab — a unified cross-stack benchmark harness.

The lab measures the *same* workloads across the project's compute backends —
pure **Python**, **NumPy**, the **Rust** core (via the `neuralforge` SDK), and
the **GPU** engine (`neuralforge_cuda`: CUDA / Triton / PyTorch) — capturing
latency, throughput, host/device memory, CPU%, and GPU%. Runs are written as
machine-readable JSON (``benchmark_lab/results/``); the chart and report
generators turn that JSON into committed SVG charts and Markdown summaries.

Design: measurement (:mod:`benchmark_lab.metrics`), environment capture
(:mod:`benchmark_lab.environment`), reference backends
(:mod:`benchmark_lab.backends`), and the GPU sampler (:mod:`benchmark_lab.gpu`)
are independent and unit-tested; :mod:`benchmark_lab.workloads` orchestrates
them and :mod:`benchmark_lab.charts` / :mod:`benchmark_lab.report` render the
outputs. The CLI lives in :mod:`benchmark_lab.__main__` (``python -m benchmark_lab``).
"""

from __future__ import annotations

from .environment import collect_environment
from .metrics import Measurement, measure
from .report import render_markdown
from .workloads import WORKLOADS, run_all

__all__ = [
    "WORKLOADS",
    "Measurement",
    "collect_environment",
    "measure",
    "render_markdown",
    "run_all",
]
