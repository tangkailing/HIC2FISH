"""Relative-geometry MDS reconstruction and interactive visualization."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.manifold import MDS

from .constants import MATRIX_SIZE
from .metrics import LOWER_INDICES


def normalize_distance_matrix_for_shape(
    matrix: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Normalize a distance matrix by its positive lower-triangle median."""
    distance = np.asarray(matrix, dtype=np.float64)
    expected_shape = (MATRIX_SIZE, MATRIX_SIZE)
    if distance.shape != expected_shape:
        raise ValueError(f"Expected distance shape {expected_shape}; got {distance.shape}.")
    distance = 0.5 * (distance + distance.T)
    np.fill_diagonal(distance, 0.0)
    if not np.all(np.isfinite(distance)) or np.min(distance) < -1e-8:
        raise ValueError("Centroid distance matrix is invalid for MDS.")
    distance[distance < 0.0] = 0.0
    positive = distance[LOWER_INDICES]
    positive = positive[positive > 0.0]
    if positive.size == 0:
        raise ValueError("Centroid distance matrix has no positive distances.")
    median_distance = float(np.median(positive))
    return distance / median_distance, median_distance


def classical_mds_initialization(distance_matrix: np.ndarray) -> np.ndarray:
    sample_count = distance_matrix.shape[0]
    centering = (
        np.eye(sample_count)
        - np.ones((sample_count, sample_count)) / sample_count
    )
    gram = -0.5 * centering @ (distance_matrix**2) @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order[:3]], 0.0)
    eigenvectors = eigenvectors[:, order[:3]]
    coordinates = eigenvectors * np.sqrt(eigenvalues)[None, :]
    if coordinates.shape[1] < 3:
        coordinates = np.pad(
            coordinates,
            ((0, 0), (0, 3 - coordinates.shape[1])),
        )
    return np.asarray(coordinates, dtype=np.float64)


def metric_mds_with_initialization(
    distance_matrix: np.ndarray,
    initialization: np.ndarray,
    random_state: int,
) -> tuple[np.ndarray, float]:
    kwargs: dict[str, Any] = {
        "n_components": 3,
        "metric": True,
        "dissimilarity": "precomputed",
        "random_state": random_state,
        "n_init": 1,
        "max_iter": 500,
        "eps": 1e-7,
    }
    if "normalized_stress" in inspect.signature(MDS).parameters:
        kwargs["normalized_stress"] = False
    model = MDS(**kwargs)
    coordinates = model.fit_transform(
        distance_matrix,
        init=initialization.copy(),
    )
    return np.asarray(coordinates, dtype=np.float64), float(model.stress_)


def normalize_coordinates_to_unit_radius(
    coordinates: np.ndarray,
) -> np.ndarray:
    centered = coordinates - np.mean(coordinates, axis=0, keepdims=True)
    radius = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
    if radius <= 1e-12:
        raise ValueError("MDS coordinates have zero radius of gyration.")
    return centered / radius


def orthogonal_align(
    coordinates: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    left, _, right_transpose = np.linalg.svd(coordinates.T @ reference)
    return coordinates @ (left @ right_transpose)


def common_scene(coordinate_sets: list[np.ndarray]) -> dict[str, Any]:
    all_coordinates = np.concatenate(coordinate_sets, axis=0)
    minimum = np.min(all_coordinates, axis=0)
    maximum = np.max(all_coordinates, axis=0)
    center = 0.5 * (minimum + maximum)
    half_range = max(0.5 * float(np.max(maximum - minimum)) * 1.08, 1e-6)

    def axis(dimension: int) -> dict[str, Any]:
        return {
            "title": "",
            "range": [
                float(center[dimension] - half_range),
                float(center[dimension] + half_range),
            ],
            "backgroundcolor": "white",
            "gridcolor": "#D1D5DB",
            "showbackground": True,
            "zeroline": False,
            "showticklabels": False,
        }

    return {
        "xaxis": axis(0),
        "yaxis": axis(1),
        "zaxis": axis(2),
        "aspectmode": "cube",
        "camera": {"eye": {"x": 1.35, "y": 1.35, "z": 1.15}},
    }


def add_structure(
    figure: go.Figure,
    coordinates: np.ndarray,
    column: int,
    color: str,
) -> None:
    loci = np.arange(1, MATRIX_SIZE + 1)
    figure.add_trace(
        go.Scatter3d(
            x=coordinates[:, 0],
            y=coordinates[:, 1],
            z=coordinates[:, 2],
            mode="lines+markers",
            line={"color": color, "width": 5},
            marker={"size": 4, "color": color},
            text=[f"Locus {index}" for index in loci],
            hovertemplate=(
                "%{text}<br>X: %{x:.4f}<br>Y: %{y:.4f}<br>"
                "Z: %{z:.4f}<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=column,
    )


def write_relative_centroid_html(
    fish_centroid_um: np.ndarray,
    generated_centroid_um: np.ndarray,
    centroid_pcc: float,
    generated_diversity: float,
    generated_count: int,
    output_dir: Path,
    random_state: int,
) -> dict[str, Any]:
    """Reconstruct and compare relative ensemble-centroid geometries."""
    fish_normalized, fish_median_um = normalize_distance_matrix_for_shape(
        fish_centroid_um
    )
    generated_normalized, generated_median_um = (
        normalize_distance_matrix_for_shape(generated_centroid_um)
    )

    common_distance = 0.5 * (fish_normalized + generated_normalized)
    initialization = classical_mds_initialization(common_distance)
    fish_coordinates_raw, fish_stress = metric_mds_with_initialization(
        fish_normalized,
        initialization,
        random_state,
    )
    generated_coordinates_raw, generated_stress = (
        metric_mds_with_initialization(
            generated_normalized,
            initialization,
            random_state,
        )
    )

    fish_coordinates = normalize_coordinates_to_unit_radius(
        fish_coordinates_raw
    )
    generated_coordinates = normalize_coordinates_to_unit_radius(
        generated_coordinates_raw
    )
    generated_coordinates = orthogonal_align(
        generated_coordinates,
        fish_coordinates,
    )

    np.save(
        output_dir / "fish_centroid_relative_distance.npy",
        fish_normalized,
    )
    np.save(
        output_dir / "generated_centroid_relative_distance.npy",
        generated_normalized,
    )
    np.save(
        output_dir / "fish_centroid_relative_coordinates.npy",
        fish_coordinates,
    )
    np.save(
        output_dir / "generated_centroid_relative_coordinates.npy",
        generated_coordinates,
    )

    figure = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(
            "Experimental DNA-FISH ensemble centroid",
            "HiC2FISH-generated ensemble centroid",
        ),
    )
    add_structure(figure, fish_coordinates, column=1, color="#2563EB")
    add_structure(figure, generated_coordinates, column=2, color="#DC2626")
    scene = common_scene([fish_coordinates, generated_coordinates])
    figure.update_layout(
        template="plotly_white",
        title=(
            "HiC2FISH relative centroid geometry | "
            f"Centroid-PCC={centroid_pcc:.4f} | "
            f"generated mean pairwise PCC={generated_diversity:.4f} | "
            f"n={generated_count}"
        ),
        width=1450,
        height=720,
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"color": "black", "size": 14},
        margin={"l": 10, "r": 10, "t": 85, "b": 10},
        scene=scene,
        scene2=scene,
    )
    html_path = output_dir / "centroid_3d_comparison.html"
    html = figure.to_html(
        full_html=True,
        include_plotlyjs=True,
        config={"responsive": True},
    )
    html = html.replace(
        "</head>",
        "<style>html,body{background:#fff!important;margin:0;}</style></head>",
    )
    html_path.write_text(html, encoding="utf-8")

    return {
        "visualization": "relative shape only",
        "coordinate_units": "dimensionless normalized units",
        "fish_centroid_median_distance_um": fish_median_um,
        "generated_centroid_median_distance_um": generated_median_um,
        "fish_mds_raw_stress": fish_stress,
        "generated_mds_raw_stress": generated_stress,
        "html_path": str(html_path),
    }
