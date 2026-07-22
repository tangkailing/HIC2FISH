"""Input validation, checkpoint loading and normalization utilities."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from .constants import MATRIX_SIZE
from .model import EnhancedUNet


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


def load_normalization_scalers(
    path: Path,
) -> tuple[float, float, float, float]:
    """Load global training-partition scalers from an NPZ archive."""
    required_keys = (
        "hic_min_train",
        "hic_max_train",
        "dna_fish_min_train_um",
        "dna_fish_max_train_um",
    )
    with np.load(path) as scalers:
        missing = [key for key in required_keys if key not in scalers]
        if missing:
            raise KeyError(
                f"Scaler file {path} is missing keys: {', '.join(missing)}."
            )
        values = tuple(
            float(np.asarray(scalers[key]).reshape(-1)[0])
            for key in required_keys
        )

    hic_min, hic_max, fish_min, fish_max = values
    if not all(np.isfinite(value) for value in values):
        raise ValueError(f"Scaler file {path} contains non-finite values.")
    if hic_max <= hic_min:
        raise ValueError(f"Invalid Hi-C scaler range: {hic_min}, {hic_max}.")
    if fish_max <= fish_min:
        raise ValueError(
            f"Invalid DNA-FISH scaler range: {fish_min}, {fish_max}."
        )
    return hic_min, hic_max, fish_min, fish_max


def select_matrix(
    array: np.ndarray,
    index: int,
    label: str,
    matrix_size: int = MATRIX_SIZE,
) -> np.ndarray:
    """Select one matrix from common NPY layouts and return a 2D array."""
    data = np.asarray(array)
    if data.ndim == 2:
        if index != 0:
            raise IndexError(f"A single {label} matrix only supports index 0.")
        matrix = data
    elif data.ndim == 3:
        if data.shape == (matrix_size, matrix_size, 1):
            if index != 0:
                raise IndexError(f"A single {label} matrix only supports index 0.")
            matrix = data[..., 0]
        else:
            matrix = data[index]
    elif data.ndim == 4:
        selected = data[index]
        if selected.shape == (matrix_size, matrix_size, 1):
            matrix = selected[..., 0]
        elif selected.shape == (1, matrix_size, matrix_size):
            matrix = selected[0]
        else:
            raise ValueError(f"Unsupported selected {label} shape {selected.shape}.")
    else:
        raise ValueError(f"Unsupported {label} array shape {data.shape}.")

    matrix = np.asarray(matrix, dtype=np.float64)
    expected_shape = (matrix_size, matrix_size)
    if matrix.shape != expected_shape:
        raise ValueError(
            f"{label} matrix must be {expected_shape}; got {matrix.shape}."
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} matrix contains NaN or infinite values.")
    return matrix


def load_checkpoint(path: Path, device: torch.device) -> EnhancedUNet:
    """Load a plain or wrapped state dictionary into EnhancedUNet."""
    model = EnhancedUNet().to(device)
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError("Unsupported checkpoint format.")
    cleaned = {
        key.removeprefix("module."): value for key, value in state_dict.items()
    }
    model.load_state_dict(cleaned, strict=True)
    model.eval()
    return model


def normalize_hic(
    matrix: np.ndarray,
    minimum: float,
    maximum: float,
    device: torch.device,
) -> torch.Tensor:
    normalized = (matrix - minimum) / (maximum - minimum)
    return (
        torch.from_numpy(np.asarray(normalized, dtype=np.float32))
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )
