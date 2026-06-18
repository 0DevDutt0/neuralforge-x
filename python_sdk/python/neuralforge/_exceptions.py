"""Exception hierarchy for NeuralForge-X.

The custom errors subclass :class:`ValueError` so that callers who already
guard with ``except ValueError`` keep working, while callers who want to be
specific can catch :class:`NeuralForgeError` or one of its subclasses. Errors
raised by the native Rust layer surface as plain :class:`ValueError`.
"""

from __future__ import annotations

__all__ = [
    "DimensionMismatchError",
    "InvalidInputError",
    "InvalidMetricError",
    "NeuralForgeError",
]


class NeuralForgeError(Exception):
    """Base class for every error raised by NeuralForge-X."""


class InvalidInputError(NeuralForgeError, ValueError):
    """An input array had the wrong shape, dtype, size, or value."""


class DimensionMismatchError(InvalidInputError):
    """Two operands had incompatible dimensionalities."""


class InvalidMetricError(NeuralForgeError, ValueError):
    """An unrecognised metric name was supplied."""
