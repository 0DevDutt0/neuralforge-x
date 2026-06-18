"""CUDA capability detection and the GPU error type.

The GPU engine is a *hard* GPU module: there is no CPU fallback here (that is the
job of the ``neuralforge`` core). If no CUDA device is present, operations raise
:class:`GpuError` with an actionable message.
"""

from __future__ import annotations

from typing import Any


class GpuError(RuntimeError):
    """Raised when a GPU operation cannot run (no device, driver, or toolkit)."""


def cuda_available() -> bool:
    """True if CuPy can see at least one CUDA device."""
    try:
        import cupy as cp

        return bool(cp.cuda.runtime.getDeviceCount() > 0)
    except Exception:
        return False


def require_cuda() -> None:
    """Raise :class:`GpuError` unless a CUDA device is available."""
    try:
        import cupy  # noqa: F401
    except Exception as exc:  # pragma: no cover - import-time environment issue
        raise GpuError(
            "CuPy is not installed. Install the GPU extra: pip install -e ./cuda_engine[cuda]"
        ) from exc
    if not cuda_available():
        raise GpuError(
            "no CUDA device detected. The GPU engine requires an NVIDIA GPU with a "
            "recent driver (>= 570 for Blackwell sm_120)."
        )


def device_info() -> dict[str, Any]:
    """Return a summary of CUDA device 0 (name, compute capability, memory)."""
    require_cuda()
    import cupy as cp

    props = cp.cuda.runtime.getDeviceProperties(0)
    free, total = cp.cuda.runtime.memGetInfo()
    name = props["name"]
    if isinstance(name, bytes):
        name = name.decode("utf-8", "replace")
    return {
        "name": name,
        "compute_capability": cp.cuda.Device(0).compute_capability,
        "total_mem_mb": total // (1024 * 1024),
        "free_mem_mb": free // (1024 * 1024),
        "cupy_version": cp.__version__,
        "runtime_version": cp.cuda.runtime.runtimeGetVersion(),
        "driver_version": cp.cuda.runtime.driverGetVersion(),
    }
