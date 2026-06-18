"""Tests for the profiling-lab analyzer (criterion → report/chart).

The capture half (flamegraph / Nsight) is environment-dependent and exercised by
its scripts; here we pin the pure analysis: throughput math, the Markdown report,
and the SVG chart, all from a synthetic estimates dict so no benchmark run or
profiler is needed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load_analyze():
    spec = importlib.util.spec_from_file_location("nfx_analyze", _ROOT / "profiling" / "analyze.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analyze = _load_analyze()

# Real-shaped medians (ns): scalar vs AVX2+FMA dot product.
_DP = {
    "scalar": {128: 31.3, 384: 124.5, 768: 270.0, 1536: 567.3},
    "simd_avx2": {128: 4.06, 384: 10.27, 768: 23.31, 1536: 45.77},
}


def test_throughput_math() -> None:
    # Gelem/s for a length-d pass in `ns` is d / ns.
    assert analyze._gelem_s(768, 23.31) == pytest.approx(768 / 23.31)
    assert analyze._gelem_s(128, 0.0) == 0.0


def test_report_has_speedups_and_log() -> None:
    md = analyze.render_report(_DP)
    assert "scalar vs AVX2 + FMA" in md
    # 31.3 / 4.06 = 7.71 -> "7.7×"
    assert "7.7×" in md
    assert "Optimization log" in md
    assert "Gelem/s" in md


def test_chart_is_wellformed() -> None:
    svg = analyze.render_chart(_DP)
    assert svg is not None
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    assert "#34d399" in svg  # the SIMD (emerald) series
    assert svg.count("<rect") >= 8  # panel + 2 series x 4 dims


def test_chart_none_without_data() -> None:
    assert analyze.render_chart({}) is None
