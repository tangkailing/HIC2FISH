#!/usr/bin/env python3
"""Prepare matched Hi-C--DNA-FISH matrices for HiC2FISH.

The script intentionally contains no raw-data paths. Input datasets are
described in a user-supplied CSV manifest. By default, outputs are written to
``data/processed`` relative to this script:

    X_train.npy, Y_train.npy, X_val.npy, Y_val.npy,
    X_minmax.npy and Y_minmax.npy.

Expected DNA-FISH input
-----------------------
Each Excel workbook contains one row per assayed locus and columns describing
the trace identifier, 3D coordinates and genomic interval. Column names are
matched case-insensitively to common variants of:

    Trace_ID, X, Y, Z, Chrom, Chrom_Start, Chrom_End

Rows belonging to one Trace_ID constitute one single-cell chromatin trace.
The union of genomic intervals in the selected region defines the canonical
locus order. Missing loci are restored at their known genomic positions and
their coordinates are linearly interpolated along genomic midpoint. Distance
matrices are calculated only after coordinate interpolation.

Expected Hi-C input
-------------------
Hi-C is read from a multi-resolution Cooler (``.mcool``) file at the resolution
specified in the manifest. A contact matrix is fetched for the requested
region and bilinearly interpolated from Hi-C bin centers to the DNA-FISH probe
midpoints.

The saved X/Y matrices are not min--max transformed. Training-partition global
minima and maxima are saved separately and must be used for normalization.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "data" / "processed"
DEFAULT_TEMPLATE_PATH = REPOSITORY_ROOT / "data" / "preprocessing_manifest.csv"

MANIFEST_COLUMNS = [
    "condition_id",
    "fish_xlsx",
    "fish_sheet",
    "mcool_path",
    "chromosome",
    "region_start",
    "region_end",
    "resolution",
    "balance",
    "coordinate_scale_to_um",
    "split",
]


@dataclass(frozen=True)
class ConditionSpec:
    condition_id: str
    fish_xlsx: Path
    fish_sheet: str | int | None
    mcool_path: Path
    chromosome: str
    region_start: int
    region_end: int
    resolution: int
    balance: bool
    coordinate_scale_to_um: float
    split: str


@dataclass
class ConditionData:
    spec: ConditionSpec
    hic_matrix: np.ndarray
    fish_matrices: np.ndarray
    trace_ids: list[str]
    fish_summary: dict[str, Any]
    hic_summary: dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert raw .mcool Hi-C and Excel DNA-FISH coordinates into "
            "matched HiC2FISH X/Y arrays."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="CSV manifest describing the raw input conditions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory (default: repository/data/processed).",
    )
    parser.add_argument(
        "--target-loci",
        type=int,
        default=50,
        help="Required number of ordered DNA-FISH loci per condition.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.20,
        help="Validation fraction for manifest rows whose split is 'auto'.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=2026,
        help="Random seed used only for train/validation assignment.",
    )
    parser.add_argument(
        "--edge-missing-policy",
        choices=("exclude", "nearest"),
        default="exclude",
        help=(
            "How to handle a trace missing a terminal probe coordinate. "
            "'exclude' avoids extrapolation; 'nearest' uses the nearest "
            "observed terminal coordinate."
        ),
    )
    parser.add_argument(
        "--duplicate-locus-policy",
        choices=("error", "mean"),
        default="error",
        help="How to handle duplicate rows for one Trace_ID and genomic locus.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of existing output arrays.",
    )
    parser.add_argument(
        "--write-template",
        nargs="?",
        const=str(DEFAULT_TEMPLATE_PATH),
        metavar="PATH",
        help=(
            "Write an empty manifest template and exit. If PATH is omitted, "
            "write data/preprocessing_manifest.csv."
        ),
    )
    return parser


def write_manifest_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=MANIFEST_COLUMNS).to_csv(path, index=False)
    print(f"Wrote empty manifest template: {path}")


def _clean_column_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def resolve_fish_columns(frame: pd.DataFrame) -> dict[str, str]:
    aliases = {
        "trace": {"traceid", "trace"},
        "x": {"x", "xcoordinate", "coordx"},
        "y": {"y", "ycoordinate", "coordy"},
        "z": {"z", "zcoordinate", "coordz"},
        "chrom": {"chrom", "chromosome", "chr"},
        "start": {"chromstart", "chromosomestart", "start", "startbp"},
        "end": {"chromend", "chromosomeend", "end", "endbp"},
    }
    normalized = {_clean_column_name(column): str(column) for column in frame.columns}
    resolved: dict[str, str] = {}
    for canonical, candidates in aliases.items():
        matches = [normalized[name] for name in candidates if name in normalized]
        if len(matches) != 1:
            raise ValueError(
                f"Could not uniquely resolve DNA-FISH column '{canonical}'. "
                f"Available columns: {list(frame.columns)}"
            )
        resolved[canonical] = matches[0]
    return resolved


def parse_bool(value: Any, label: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"{label} must be true or false; received {value!r}.")


def parse_sheet(value: Any) -> str | int | None:
    if pd.isna(value) or str(value).strip() == "":
        return None
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    return text


def resolve_input_path(value: Any, manifest_dir: Path, label: str) -> Path:
    if pd.isna(value) or str(value).strip() == "":
        raise ValueError(f"Manifest field '{label}' cannot be empty.")
    path = Path(str(value).strip()).expanduser()
    if not path.is_absolute():
        path = manifest_dir / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def load_manifest(path: Path) -> list[ConditionSpec]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    frame = pd.read_csv(path)
    missing = [column for column in MANIFEST_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")
    if frame.empty:
        raise ValueError("The manifest contains no input rows.")

    specs: list[ConditionSpec] = []
    seen_ids: set[str] = set()
    for row_index, row in frame.iterrows():
        prefix = f"Manifest row {row_index + 2}"
        condition_id = str(row["condition_id"]).strip()
        if not condition_id or condition_id.lower() == "nan":
            raise ValueError(f"{prefix}: condition_id cannot be empty.")
        if condition_id in seen_ids:
            raise ValueError(f"{prefix}: duplicate condition_id {condition_id!r}.")
        seen_ids.add(condition_id)

        chromosome = str(row["chromosome"]).strip()
        if not chromosome or chromosome.lower() == "nan":
            raise ValueError(f"{prefix}: chromosome cannot be empty.")
        region_start = int(row["region_start"])
        region_end = int(row["region_end"])
        resolution = int(row["resolution"])
        coordinate_scale = float(row["coordinate_scale_to_um"])
        if region_start < 0 or region_end <= region_start:
            raise ValueError(f"{prefix}: invalid genomic region.")
        if resolution <= 0:
            raise ValueError(f"{prefix}: resolution must be positive.")
        if not math.isfinite(coordinate_scale) or coordinate_scale <= 0:
            raise ValueError(
                f"{prefix}: coordinate_scale_to_um must be positive. "
                "Use 1 for micrometres or 0.001 for nanometres."
            )
        split = str(row["split"]).strip().lower()
        if split in {"", "nan"}:
            split = "auto"
        if split not in {"auto", "train", "val"}:
            raise ValueError(f"{prefix}: split must be auto, train or val.")

        specs.append(
            ConditionSpec(
                condition_id=condition_id,
                fish_xlsx=resolve_input_path(
                    row["fish_xlsx"], path.parent, f"{prefix} fish_xlsx"
                ),
                fish_sheet=parse_sheet(row["fish_sheet"]),
                mcool_path=resolve_input_path(
                    row["mcool_path"], path.parent, f"{prefix} mcool_path"
                ),
                chromosome=chromosome,
                region_start=region_start,
                region_end=region_end,
                resolution=resolution,
                balance=parse_bool(row["balance"], f"{prefix} balance"),
                coordinate_scale_to_um=coordinate_scale,
                split=split,
            )
        )
    return specs


def chromosome_equal(series: pd.Series, chromosome: str) -> pd.Series:
    target = chromosome.lower().removeprefix("chr")
    values = series.astype(str).str.lower().str.replace(r"^chr", "", regex=True)
    return values == target


def interpolate_coordinate_vector(
    positions: np.ndarray,
    values: np.ndarray,
    edge_policy: str,
) -> np.ndarray | None:
    known = np.isfinite(values)
    if known.sum() < 2:
        return None
    missing = ~known
    if not missing.any():
        return values.astype(np.float64, copy=True)

    known_positions = positions[known]
    if edge_policy == "exclude":
        if positions[missing].min() < known_positions.min():
            return None
        if positions[missing].max() > known_positions.max():
            return None

    result = values.astype(np.float64, copy=True)
    result[missing] = np.interp(
        positions[missing],
        known_positions,
        values[known],
    )
    return result


def load_fish_distance_matrices(
    spec: ConditionSpec,
    target_loci: int,
    edge_policy: str,
    duplicate_policy: str,
) -> tuple[np.ndarray, list[str], np.ndarray, dict[str, Any]]:
    read_kwargs: dict[str, Any] = {}
    if spec.fish_sheet is not None:
        read_kwargs["sheet_name"] = spec.fish_sheet
    frame = pd.read_excel(spec.fish_xlsx, **read_kwargs)
    columns = resolve_fish_columns(frame)

    working = pd.DataFrame(
        {
            "trace": frame[columns["trace"]].astype(str).str.strip(),
            "x": pd.to_numeric(frame[columns["x"]], errors="coerce"),
            "y": pd.to_numeric(frame[columns["y"]], errors="coerce"),
            "z": pd.to_numeric(frame[columns["z"]], errors="coerce"),
            "chrom": frame[columns["chrom"]].astype(str).str.strip(),
            "start": pd.to_numeric(frame[columns["start"]], errors="coerce"),
            "end": pd.to_numeric(frame[columns["end"]], errors="coerce"),
        }
    )
    working = working.loc[
        working["trace"].ne("")
        & working["trace"].str.lower().ne("nan")
        & chromosome_equal(working["chrom"], spec.chromosome)
        & working["start"].notna()
        & working["end"].notna()
    ].copy()
    working["start"] = working["start"].astype(np.int64)
    working["end"] = working["end"].astype(np.int64)
    working["midpoint"] = 0.5 * (working["start"] + working["end"])
    working = working.loc[
        (working["midpoint"] >= spec.region_start)
        & (working["midpoint"] < spec.region_end)
    ].copy()
    if working.empty:
        raise ValueError(
            f"{spec.condition_id}: no DNA-FISH rows overlap "
            f"{spec.chromosome}:{spec.region_start}-{spec.region_end}."
        )

    probes = (
        working[["chrom", "start", "end", "midpoint"]]
        .drop_duplicates(subset=["start", "end"])
        .sort_values(["midpoint", "start", "end"], kind="stable")
        .reset_index(drop=True)
    )
    if len(probes) != target_loci:
        raise ValueError(
            f"{spec.condition_id}: the selected DNA-FISH region contains "
            f"{len(probes)} unique probe intervals, expected {target_loci}. "
            "Check the region boundaries and workbook."
        )

    probe_keys = pd.MultiIndex.from_frame(probes[["start", "end"]])
    positions = probes["midpoint"].to_numpy(dtype=np.float64)
    matrices: list[np.ndarray] = []
    retained_trace_ids: list[str] = []
    excluded: dict[str, int] = {
        "duplicate_locus_rows": 0,
        "insufficient_coordinates": 0,
        "terminal_missing_coordinates": 0,
        "nonfinite_distance_matrix": 0,
    }
    interpolated_locus_counts: list[int] = []

    for trace_id, trace in working.groupby("trace", sort=False):
        duplicate_mask = trace.duplicated(subset=["start", "end"], keep=False)
        if duplicate_mask.any():
            excluded["duplicate_locus_rows"] += int(duplicate_mask.sum())
            if duplicate_policy == "error":
                raise ValueError(
                    f"{spec.condition_id}, Trace_ID={trace_id}: duplicate rows "
                    "were found for a genomic probe. Use "
                    "--duplicate-locus-policy mean only if averaging replicate "
                    "coordinates is scientifically appropriate."
                )

        per_probe = trace.groupby(["start", "end"], sort=False)[
            ["x", "y", "z"]
        ].mean()
        per_probe = per_probe.reindex(probe_keys)
        raw_coordinates = per_probe[["x", "y", "z"]].to_numpy(dtype=np.float64)
        interpolated_count = int((~np.isfinite(raw_coordinates).all(axis=1)).sum())

        coordinate_columns: list[np.ndarray] = []
        failed = False
        terminal_failure = False
        for axis in range(3):
            values = raw_coordinates[:, axis]
            known = np.isfinite(values)
            if known.sum() < 2:
                failed = True
                break
            if edge_policy == "exclude" and (
                not known[0] or not known[-1]
            ):
                terminal_failure = True
                failed = True
                break
            interpolated = interpolate_coordinate_vector(
                positions, values, edge_policy
            )
            if interpolated is None:
                failed = True
                break
            coordinate_columns.append(interpolated)
        if failed:
            key = (
                "terminal_missing_coordinates"
                if terminal_failure
                else "insufficient_coordinates"
            )
            excluded[key] += 1
            continue

        coordinates_um = (
            np.column_stack(coordinate_columns) * spec.coordinate_scale_to_um
        )
        differences = coordinates_um[:, None, :] - coordinates_um[None, :, :]
        distance_matrix = np.sqrt(np.sum(differences * differences, axis=-1))
        distance_matrix = 0.5 * (distance_matrix + distance_matrix.T)
        np.fill_diagonal(distance_matrix, 0.0)
        if not np.isfinite(distance_matrix).all():
            excluded["nonfinite_distance_matrix"] += 1
            continue

        matrices.append(distance_matrix.astype(np.float32))
        retained_trace_ids.append(str(trace_id))
        interpolated_locus_counts.append(interpolated_count)

    if not matrices:
        raise ValueError(f"{spec.condition_id}: no valid DNA-FISH traces remained.")

    fish_array = np.stack(matrices, axis=0)
    summary = {
        "input_trace_count": int(working["trace"].nunique()),
        "retained_trace_count": len(retained_trace_ids),
        "excluded": excluded,
        "mean_imputed_loci_per_retained_trace": float(
            np.mean(interpolated_locus_counts)
        ),
        "maximum_imputed_loci_in_retained_trace": int(
            np.max(interpolated_locus_counts)
        ),
        "coordinate_scale_to_um": spec.coordinate_scale_to_um,
        "distance_min_um": float(fish_array.min()),
        "distance_max_um": float(fish_array.max()),
    }
    return fish_array, retained_trace_ids, positions, summary


def load_hic_matrix_at_probes(
    spec: ConditionSpec,
    probe_midpoints: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        import cooler
    except ImportError as exc:
        raise ImportError(
            "The 'cooler' package is required to read .mcool files. "
            "Install the preprocessing dependencies first."
        ) from exc
    try:
        from scipy.interpolate import RegularGridInterpolator
    except ImportError as exc:
        raise ImportError(
            "The 'scipy' package is required for Hi-C interpolation."
        ) from exc

    uri = f"{spec.mcool_path}::resolutions/{spec.resolution}"
    clr = cooler.Cooler(uri)
    if clr.binsize is not None and int(clr.binsize) != spec.resolution:
        raise ValueError(
            f"{spec.condition_id}: requested resolution {spec.resolution}, "
            f"but Cooler reports {clr.binsize}."
        )
    available_chromosomes = set(clr.chromnames)
    chromosome = spec.chromosome
    if chromosome not in available_chromosomes:
        alternate = (
            chromosome.removeprefix("chr")
            if chromosome.startswith("chr")
            else f"chr{chromosome}"
        )
        if alternate not in available_chromosomes:
            raise ValueError(
                f"{spec.condition_id}: chromosome {spec.chromosome!r} is not "
                "present in the Cooler file."
            )
        chromosome = alternate

    region = f"{chromosome}:{spec.region_start}-{spec.region_end}"
    try:
        matrix = np.asarray(
            clr.matrix(balance=spec.balance, sparse=False).fetch(region),
            dtype=np.float64,
        )
    except Exception as exc:
        if spec.balance:
            raise RuntimeError(
                f"{spec.condition_id}: balanced Hi-C extraction failed. "
                "Confirm that the selected resolution contains balancing "
                "weights, or set balance=false in the manifest."
            ) from exc
        raise

    bins = clr.bins().fetch(region)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{spec.condition_id}: fetched Hi-C matrix is not square.")
    if len(bins) != matrix.shape[0]:
        raise ValueError(
            f"{spec.condition_id}: Hi-C bin table and matrix dimensions disagree."
        )
    if matrix.shape[0] < 2:
        raise ValueError(f"{spec.condition_id}: fewer than two Hi-C bins fetched.")

    invalid_fraction = float((~np.isfinite(matrix)).mean())
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix = 0.5 * (matrix + matrix.T)
    if (matrix < -1e-10).any():
        raise ValueError(
            f"{spec.condition_id}: negative Hi-C contacts were found before "
            "model min--max normalization."
        )
    matrix = np.maximum(matrix, 0.0)
    if not np.any(matrix > 0):
        raise ValueError(f"{spec.condition_id}: fetched Hi-C matrix is all zero.")

    bin_centers = 0.5 * (
        bins["start"].to_numpy(dtype=np.float64)
        + bins["end"].to_numpy(dtype=np.float64)
    )
    target = np.clip(
        probe_midpoints.astype(np.float64),
        bin_centers[0],
        bin_centers[-1],
    )
    interpolator = RegularGridInterpolator(
        (bin_centers, bin_centers),
        matrix,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    row_positions, column_positions = np.meshgrid(
        target, target, indexing="ij"
    )
    points = np.column_stack(
        [row_positions.reshape(-1), column_positions.reshape(-1)]
    )
    aligned = interpolator(points).reshape(len(target), len(target))
    aligned = 0.5 * (aligned + aligned.T)
    aligned = np.maximum(aligned, 0.0)
    if not np.isfinite(aligned).all():
        raise ValueError(
            f"{spec.condition_id}: non-finite values remained after Hi-C interpolation."
        )

    summary = {
        "resolution_bp": spec.resolution,
        "balance": spec.balance,
        "fetched_bin_count": int(matrix.shape[0]),
        "invalid_fetched_contact_fraction_replaced_with_zero": invalid_fraction,
        "aligned_contact_min": float(aligned.min()),
        "aligned_contact_max": float(aligned.max()),
    }
    return aligned.astype(np.float32), summary


def prepare_condition(
    spec: ConditionSpec,
    target_loci: int,
    edge_policy: str,
    duplicate_policy: str,
) -> ConditionData:
    fish_matrices, trace_ids, probe_midpoints, fish_summary = (
        load_fish_distance_matrices(
            spec,
            target_loci=target_loci,
            edge_policy=edge_policy,
            duplicate_policy=duplicate_policy,
        )
    )
    hic_matrix, hic_summary = load_hic_matrix_at_probes(spec, probe_midpoints)
    if hic_matrix.shape != (target_loci, target_loci):
        raise AssertionError("Internal error: aligned Hi-C matrix has wrong shape.")
    return ConditionData(
        spec=spec,
        hic_matrix=hic_matrix,
        fish_matrices=fish_matrices,
        trace_ids=trace_ids,
        fish_summary=fish_summary,
        hic_summary=hic_summary,
    )


def split_indices(
    condition: ConditionData,
    val_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(condition.trace_ids)
    indices = np.arange(count, dtype=np.int64)
    if condition.spec.split == "train":
        return indices, np.empty(0, dtype=np.int64)
    if condition.spec.split == "val":
        return np.empty(0, dtype=np.int64), indices
    if count < 2:
        raise ValueError(
            f"{condition.spec.condition_id}: split='auto' requires at least "
            "two retained traces."
        )
    shuffled = rng.permutation(indices)
    validation_count = int(round(count * val_fraction))
    validation_count = min(max(validation_count, 1), count - 1)
    val_indices = np.sort(shuffled[:validation_count])
    train_indices = np.sort(shuffled[validation_count:])
    return train_indices, val_indices


def paired_arrays(
    condition: ConditionData,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(indices) == 0:
        size = condition.hic_matrix.shape[0]
        empty = np.empty((0, size, size, 1), dtype=np.float32)
        return empty.copy(), empty.copy()
    y = condition.fish_matrices[indices, :, :, None].astype(np.float32)
    x = np.repeat(
        condition.hic_matrix[None, :, :, None],
        repeats=len(indices),
        axis=0,
    ).astype(np.float32)
    return x, y


def validate_output_pair(x: np.ndarray, y: np.ndarray, label: str) -> None:
    if x.shape != y.shape or x.ndim != 4 or x.shape[-1] != 1:
        raise ValueError(
            f"{label}: expected matched arrays shaped (n, N, N, 1); "
            f"received X{x.shape} and Y{y.shape}."
        )
    if len(x) == 0:
        raise ValueError(f"{label}: no samples were assigned.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f"{label}: non-finite values were found.")
    if (y < 0).any():
        raise ValueError(f"{label}: negative DNA-FISH distances were found.")
    symmetry_error = float(np.max(np.abs(y - y.swapaxes(1, 2))))
    diagonal_error = float(
        np.max(np.abs(np.diagonal(y[..., 0], axis1=1, axis2=2)))
    )
    if symmetry_error > 1e-5 or diagonal_error > 1e-5:
        raise ValueError(
            f"{label}: invalid distance matrices; symmetry error="
            f"{symmetry_error}, diagonal error={diagonal_error}."
        )


def save_outputs(
    output_dir: Path,
    arrays: dict[str, np.ndarray],
    summary: dict[str, Any],
    overwrite: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        name: output_dir / f"{name}.npy" for name in arrays
    }
    summary_path = output_dir / "preprocessing_summary.json"
    existing = [path for path in [*output_paths.values(), summary_path] if path.exists()]
    if existing and not overwrite:
        formatted = "\n  - ".join(str(path) for path in existing)
        raise FileExistsError(
            "Output files already exist. Re-run with --overwrite to replace them:\n"
            f"  - {formatted}"
        )
    for name, array in arrays.items():
        np.save(output_paths[name], array)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def run(args: argparse.Namespace) -> None:
    if args.manifest is None:
        raise ValueError("--manifest is required unless --write-template is used.")
    if args.target_loci < 2:
        raise ValueError("--target-loci must be at least 2.")
    if not 0 < args.val_fraction < 1:
        raise ValueError("--val-fraction must be between 0 and 1.")

    specs = load_manifest(args.manifest)
    conditions: list[ConditionData] = []
    for index, spec in enumerate(specs, start=1):
        print(f"[{index}/{len(specs)}] Processing {spec.condition_id}")
        conditions.append(
            prepare_condition(
                spec,
                target_loci=args.target_loci,
                edge_policy=args.edge_missing_policy,
                duplicate_policy=args.duplicate_locus_policy,
            )
        )

    rng = np.random.default_rng(args.split_seed)
    x_train_parts: list[np.ndarray] = []
    y_train_parts: list[np.ndarray] = []
    x_val_parts: list[np.ndarray] = []
    y_val_parts: list[np.ndarray] = []
    condition_summaries: list[dict[str, Any]] = []

    for condition in conditions:
        train_indices, val_indices = split_indices(
            condition, args.val_fraction, rng
        )
        x_train, y_train = paired_arrays(condition, train_indices)
        x_val, y_val = paired_arrays(condition, val_indices)
        if len(train_indices):
            x_train_parts.append(x_train)
            y_train_parts.append(y_train)
        if len(val_indices):
            x_val_parts.append(x_val)
            y_val_parts.append(y_val)
        condition_summaries.append(
            {
                "condition_id": condition.spec.condition_id,
                "chromosome": condition.spec.chromosome,
                "region_start": condition.spec.region_start,
                "region_end": condition.spec.region_end,
                "split_setting": condition.spec.split,
                "training_pairs": int(len(train_indices)),
                "validation_pairs": int(len(val_indices)),
                "fish": condition.fish_summary,
                "hic": condition.hic_summary,
            }
        )

    if not x_train_parts or not x_val_parts:
        raise ValueError(
            "Both training and validation outputs must contain samples. Check "
            "the manifest split column and --val-fraction."
        )
    x_train = np.concatenate(x_train_parts, axis=0).astype(np.float32)
    y_train = np.concatenate(y_train_parts, axis=0).astype(np.float32)
    x_val = np.concatenate(x_val_parts, axis=0).astype(np.float32)
    y_val = np.concatenate(y_val_parts, axis=0).astype(np.float32)
    validate_output_pair(x_train, y_train, "training")
    validate_output_pair(x_val, y_val, "validation")

    x_minmax = np.asarray([x_train.min(), x_train.max()], dtype=np.float32)
    y_minmax = np.asarray([y_train.min(), y_train.max()], dtype=np.float32)
    if not x_minmax[1] > x_minmax[0]:
        raise ValueError("Training Hi-C data have a zero-valued scaler range.")
    if not y_minmax[1] > y_minmax[0]:
        raise ValueError("Training DNA-FISH data have a zero-valued scaler range.")

    arrays = {
        "X_train": x_train,
        "Y_train": y_train,
        "X_val": x_val,
        "Y_val": y_val,
        "X_minmax": x_minmax,
        "Y_minmax": y_minmax,
    }
    summary = {
        "target_loci": args.target_loci,
        "matrix_layout": "NHWC",
        "matrix_shape": [args.target_loci, args.target_loci, 1],
        "training_pairs": int(len(x_train)),
        "validation_pairs": int(len(x_val)),
        "split_seed": args.split_seed,
        "validation_fraction_for_auto_rows": args.val_fraction,
        "edge_missing_policy": args.edge_missing_policy,
        "duplicate_locus_policy": args.duplicate_locus_policy,
        "normalization": {
            "applied_to_saved_X_Y": False,
            "scalers_fitted_on_training_partition_only": True,
            "X_min_train": float(x_minmax[0]),
            "X_max_train": float(x_minmax[1]),
            "Y_min_train_um": float(y_minmax[0]),
            "Y_max_train_um": float(y_minmax[1]),
        },
        "conditions": condition_summaries,
    }
    save_outputs(
        args.output_dir.expanduser().resolve(),
        arrays,
        summary,
        overwrite=args.overwrite,
    )

    print("\nPreprocessing completed successfully.")
    print(f"Output directory: {args.output_dir.expanduser().resolve()}")
    print(f"X_train: {x_train.shape}")
    print(f"Y_train: {y_train.shape}")
    print(f"X_val:   {x_val.shape}")
    print(f"Y_val:   {y_val.shape}")
    print(f"Hi-C training range: {x_minmax.tolist()}")
    print(f"DNA-FISH training range (um): {y_minmax.tolist()}")


def main() -> None:
    args = build_parser().parse_args()
    if args.write_template is not None:
        write_manifest_template(Path(args.write_template).expanduser().resolve())
        return
    run(args)


if __name__ == "__main__":
    main()
