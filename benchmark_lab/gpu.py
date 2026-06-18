"""GPU utilisation and memory sampling via NVML.

CPU metrics come from :mod:`benchmark_lab.metrics`; for GPU backends we also want
device utilisation and memory. A short kernel is too brief to sample reliably, so
:func:`sample_gpu` runs the workload in a tight loop for a fixed wall-clock window
while a background thread polls NVML, then reports the mean/peak utilisation and
peak memory observed. Everything is best-effort: without ``pynvml`` (the
``nvidia-ml-py`` package) or a CUDA device, :func:`sample_gpu` returns ``None`` and
the harness simply omits GPU metrics.
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

try:
    import pynvml

    _NVML_OK = True
except Exception:
    pynvml = None  # type: ignore[assignment]
    _NVML_OK = False


@dataclass(frozen=True)
class GpuSample:
    """GPU utilisation/memory observed while a workload ran."""

    util_mean_pct: float
    util_peak_pct: float
    mem_peak_mb: float
    samples: int

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for key, value in out.items():
            if isinstance(value, float):
                out[key] = round(value, 2)
        return out


def gpu_available() -> bool:
    """Whether NVML is importable and at least one device is present."""
    if not _NVML_OK:
        return False
    try:
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
        return count > 0
    except Exception:
        return False


def sample_gpu(
    fn: Callable[[], Any],
    *,
    window_s: float = 0.4,
    poll_s: float = 0.005,
    device_index: int = 0,
) -> GpuSample | None:
    """Runs ``fn`` repeatedly for ``window_s`` while polling NVML.

    Returns the observed utilisation/memory, or ``None`` if NVML is unavailable.
    """
    if not _NVML_OK:
        return None
    try:
        pynvml.nvmlInit()
    except Exception:
        return None

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        utils: list[float] = []
        mems: list[float] = []
        stop = threading.Event()

        def poller() -> None:
            while not stop.is_set():
                try:
                    rate = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    utils.append(float(rate.gpu))
                    mems.append(mem.used / 1e6)
                except Exception:
                    pass
                time.sleep(poll_s)

        thread = threading.Thread(target=poller, daemon=True)
        thread.start()
        deadline = time.perf_counter() + window_s
        while time.perf_counter() < deadline:
            fn()
        stop.set()
        thread.join(timeout=1.0)

        if not utils:
            return None
        return GpuSample(
            util_mean_pct=sum(utils) / len(utils),
            util_peak_pct=max(utils),
            mem_peak_mb=max(mems) if mems else 0.0,
            samples=len(utils),
        )
    except Exception:
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
