"""Host, CPU, GPU, and library environment capture.

Every benchmark run records the environment it was measured on, so a results
file is self-describing and numbers are never quoted without their context.
Optional probes (``psutil`` for core counts / RAM, ``neuralforge_cuda`` for the
GPU) degrade gracefully when those packages are absent.
"""

from __future__ import annotations

import os
import platform
from typing import Any


def _optional(probe: str) -> dict[str, Any]:
    """Runs one best-effort probe, returning ``{}`` if its package is missing."""
    try:
        if probe == "psutil":
            import psutil

            return {
                "cpu_count_physical": psutil.cpu_count(logical=False),
                "ram_total_mb": round(psutil.virtual_memory().total / 1e6),
            }
        if probe == "numpy":
            import numpy

            return {"numpy": numpy.__version__}
        if probe == "neuralforge":
            import neuralforge

            return {"neuralforge": neuralforge.__version__}
        if probe == "gpu":
            import neuralforge_cuda as gpu

            if gpu.cuda_available():
                return {"gpu": gpu.device_info(), "gpu_backends": gpu.available_backends()}
    except Exception:
        return {}
    return {}


def collect_environment() -> dict[str, Any]:
    """Returns a JSON-serialisable snapshot of the benchmarking environment."""
    env: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "cpu_count_logical": os.cpu_count(),
    }
    for probe in ("psutil", "numpy", "neuralforge", "gpu"):
        env.update(_optional(probe))
    return env
