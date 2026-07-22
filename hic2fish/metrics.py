"""Ensemble-level accuracy, diversity and numerical-validity metrics."""

from __future__ import annotations

from typing import Any

import numpy as np

from .constants import MATRIX_SIZE


LOWER_INDICES = np.tril_indices(MATRIX_SIZE, k=-1)


def lower_triangle_vectors(matrices: np.ndarray) -> np.ndarray:
    data = np.asarray(matrices, dtype=np.float64)
    if data.ndim == 2:
        data = data[None, ...]
    if data.ndim != 3 or data.shape[1:] != (MATRIX_SIZE, MATRIX_SIZE):
        raise ValueError(
            f"Expected (n,{MATRIX_SIZE},{MATRIX_SIZE}) or "
            f"({MATRIX_SIZE},{MATRIX_SIZE}); received {data.shape}."
        )
    return data[:, LOWER_INDICES[0], LOWER_INDICES[1]]


def pearson_correlation(first: np.ndarray, second: np.ndarray) -> float:
    x = np.asarray(first, dtype=np.float64).reshape(-1)
    y = np.asarray(second, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size < 2:
        return float("nan")
    x = x - np.mean(x)
    y = y - np.mean(y)
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator <= 1e-12:
        return float("nan")
    return float(np.dot(x, y) / denominator)


def mean_pairwise_pcc(matrices: np.ndarray) -> float:
    """Mean PCC over every unordered pair in a generated ensemble."""
    vectors = lower_triangle_vectors(matrices)
    vectors = vectors - np.mean(vectors, axis=1, keepdims=True)
    norms = np.linalg.norm(vectors, axis=1)
    valid = np.isfinite(norms) & (norms > 1e-12)
    vectors = vectors[valid]
    norms = norms[valid]
    sample_count = vectors.shape[0]
    if sample_count < 2:
        return float("nan")
    standardized = vectors / norms[:, None]
    summed = np.sum(standardized, axis=0, dtype=np.float64)
    return float(
        (float(np.dot(summed, summed)) - sample_count)
        / (sample_count * (sample_count - 1))
    )


def validate_distance_ensemble(matrices: np.ndarray) -> dict[str, Any]:
    finite = bool(np.all(np.isfinite(matrices)))
    negative_fraction = float(
        np.mean(lower_triangle_vectors(matrices) < -1e-8)
    )
    symmetry_error = float(
        np.max(np.abs(matrices - np.transpose(matrices, (0, 2, 1))))
    )
    diagonal_error = float(
        np.max(np.abs(np.diagonal(matrices, axis1=1, axis2=2)))
    )
    valid = bool(
        finite
        and negative_fraction == 0.0
        and symmetry_error <= 1e-6
        and diagonal_error <= 1e-6
    )
    return {
        "finite": finite,
        "negative_distance_fraction": negative_fraction,
        "symmetry_max_abs_error": symmetry_error,
        "diagonal_max_abs_value": diagonal_error,
        "numerically_valid": valid,
    }
