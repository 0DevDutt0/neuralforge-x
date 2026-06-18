"""Validation and error-handling tests for the public API."""

from __future__ import annotations

import numpy as np
import pytest

import neuralforge as nf
from neuralforge import DimensionMismatchError, InvalidInputError, InvalidMetricError


def test_dimension_mismatch_pairwise() -> None:
    with pytest.raises(DimensionMismatchError):
        nf.cosine_similarity(np.ones(3, np.float32), np.ones(4, np.float32))


def test_dimension_mismatch_batch() -> None:
    with pytest.raises(DimensionMismatchError):
        nf.batch_similarity(np.ones((2, 3), np.float32), np.ones((2, 4), np.float32))


def test_unknown_metric_rejected() -> None:
    with pytest.raises(InvalidMetricError):
        nf.batch_similarity(
            np.ones((2, 3), np.float32), np.ones((5, 3), np.float32), metric="manhattan"
        )


def test_k_out_of_range() -> None:
    corpus = np.ones((5, 3), np.float32)
    query = np.ones(3, np.float32)
    with pytest.raises(InvalidInputError):
        nf.top_k_search(query, corpus, k=0)
    with pytest.raises(InvalidInputError):
        nf.top_k_search(query, corpus, k=6)


def test_wrong_ndim_rejected() -> None:
    with pytest.raises(InvalidInputError):
        nf.cosine_similarity(np.ones((2, 2), np.float32), np.ones(4, np.float32))
    with pytest.raises(InvalidInputError):
        nf.top_k_search(np.ones((2, 2), np.float32), np.ones((5, 4), np.float32), k=1)


def test_empty_input_rejected() -> None:
    with pytest.raises(InvalidInputError):
        nf.dot_product(np.array([], np.float32), np.array([], np.float32))


def test_custom_errors_are_value_errors() -> None:
    # Subclassing ValueError keeps `except ValueError` callers working.
    assert issubclass(DimensionMismatchError, ValueError)
    assert issubclass(InvalidMetricError, ValueError)
    with pytest.raises(ValueError):
        nf.cosine_similarity(np.ones(2, np.float32), np.ones(3, np.float32))
