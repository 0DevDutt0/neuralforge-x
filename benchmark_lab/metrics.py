"""Measurement primitives: latency statistics, throughput, memory, and CPU.

A :class:`Measurement` is the unit of every benchmark record. :func:`measure`
runs a callable ``repeats`` times after a warm-up, and reports:

* **latency** — best / median / mean / p90 / stdev, in milliseconds. We headline
  the *best* time (least perturbed by OS scheduling) but keep the distribution.
* **throughput** — caller-supplied ``work_items`` divided by the best time.
* **host memory** — peak transient allocation during one call, via ``tracemalloc``
  (stdlib, always available), plus the process RSS delta when ``psutil`` is present.
* **CPU%** — process CPU utilisation across the timed loop (``psutil``, optional),
  which can exceed 100% for the rayon-parallel Rust paths.

Timing uses :func:`time.perf_counter`, the highest-resolution monotonic clock.
"""

from __future__ import annotations

import gc
import statistics
import time
import tracemalloc
from dataclasses import asdict, dataclass
from typing import Any, Callable

try:
    import psutil

    _PROC = psutil.Process()
except Exception:
    psutil = None  # type: ignore[assignment]
    _PROC = None


@dataclass(frozen=True)
class Measurement:
    """Timing and resource statistics for one (workload, backend) measurement."""

    backend: str
    repeats: int
    best_ms: float
    median_ms: float
    mean_ms: float
    p90_ms: float
    stdev_ms: float
    work_items: int
    throughput_per_s: float
    peak_host_kib: float
    rss_delta_kib: float
    cpu_percent: float

    def as_dict(self) -> dict[str, Any]:
        """A rounded, JSON-friendly view of the measurement."""
        out = asdict(self)
        for key, value in out.items():
            if isinstance(value, float):
                # Throughput spans many orders of magnitude; keep it as an int-ish
                # number, everything else to 4 significant decimals.
                out[key] = round(value, 1) if key == "throughput_per_s" else round(value, 4)
        return out


def _percentile(sorted_vals: list[float], pct: float) -> float:
    """Linear-interpolated percentile of an already-sorted, non-empty list."""
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = pct / 100.0 * (len(sorted_vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def measure(
    backend: str,
    fn: Callable[[], Any],
    *,
    work_items: int,
    repeats: int = 7,
    warmup: int = 2,
    capture_resources: bool = True,
) -> Measurement:
    """Benchmarks ``fn`` and returns a :class:`Measurement`.

    Args:
        backend: label for this implementation (``"numpy"``, ``"rust"``, ...).
        fn: the zero-argument callable to time; its result is discarded.
        work_items: units of work per call (e.g. ``q * n`` similarities), used to
            derive throughput.
        repeats: number of timed iterations (the latency sample size).
        warmup: untimed calls first, to prime caches / JITs / thread pools.
        capture_resources: also measure peak host memory and CPU% (a little extra
            overhead and a couple of untimed calls).
    """
    for _ in range(max(0, warmup)):
        fn()

    # --- Latency distribution ---------------------------------------------
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        samples: list[float] = []
        for _ in range(max(1, repeats)):
            start = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - start) * 1e3)
    finally:
        if gc_was_enabled:
            gc.enable()

    ordered = sorted(samples)
    best = ordered[0]
    throughput = work_items / (best / 1e3) if best > 0 else float("inf")

    peak_host_kib = 0.0
    rss_delta_kib = 0.0
    cpu_percent = 0.0
    if capture_resources:
        peak_host_kib = _measure_host_peak(fn)
        rss_delta_kib, cpu_percent = _measure_process(fn, repeats)

    return Measurement(
        backend=backend,
        repeats=len(samples),
        best_ms=best,
        median_ms=statistics.median(ordered),
        mean_ms=statistics.fmean(ordered),
        p90_ms=_percentile(ordered, 90.0),
        stdev_ms=statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
        work_items=work_items,
        throughput_per_s=throughput,
        peak_host_kib=peak_host_kib,
        rss_delta_kib=rss_delta_kib,
        cpu_percent=cpu_percent,
    )


def _measure_host_peak(fn: Callable[[], Any]) -> float:
    """Peak transient Python-heap allocation during a single call, in KiB."""
    gc.collect()
    tracemalloc.start()
    try:
        fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak / 1024.0


def _measure_process(fn: Callable[[], Any], repeats: int) -> tuple[float, float]:
    """Process RSS delta (KiB) and CPU% across a short run; ``(0, 0)`` sans psutil."""
    if _PROC is None:
        return 0.0, 0.0
    gc.collect()
    rss_before = _PROC.memory_info().rss
    _PROC.cpu_percent(None)  # reset the CPU% reference point
    for _ in range(max(1, repeats)):
        fn()
    cpu = _PROC.cpu_percent(None)
    rss_after = _PROC.memory_info().rss
    return max(0.0, (rss_after - rss_before) / 1024.0), cpu
