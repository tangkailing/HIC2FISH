

from __future__ import annotations

import argparse
import json
import webbrowser
from pathlib import Path

import numpy as np

from .diffusion import generate_ensemble
from .io import (
    load_checkpoint,
    load_normalization_scalers,
    normalize_hic,
    resolve_device,
    select_matrix,
    set_global_seed,
)
from .metrics import (
    lower_triangle_vectors,
    mean_pairwise_pcc,
    pearson_correlation,
    validate_distance_ensemble,
)
from .visualization import write_relative_centroid_html


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DATA_DIR = REPOSITORY_ROOT / "data" / "example_data"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Minimal HiC2FISH ensemble-centroid."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=REPOSITORY_ROOT / "pretrained" / "hic2fish.pt",
    )
    parser.add_argument(
        "--hic-path",
        type=Path,
        default=EXAMPLE_DATA_DIR / "example_hic.npy",
    )
    parser.add_argument(
        "--reference-fish-centroid-path",
        type=Path,
        default=EXAMPLE_DATA_DIR / "example_dna_fish_centroid_um.npy",
    )
    parser.add_argument(
        "--scaler-path",
        type=Path,
        default=EXAMPLE_DATA_DIR / "normalization_scalers.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "output",
    )
    parser.add_argument("--hic-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--ddim-steps", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument("--generation-batch-size", type=int, default=2)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the generated interactive HTML in the default browser.",
    )
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    required = (
        args.model_path,
        args.hic_path,
        args.reference_fish_centroid_path,
        args.scaler_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required files:\n  - " + "\n  - ".join(missing)
        )
    if args.num_samples < 2:
        raise ValueError("--num-samples must be at least 2.")
    if args.ddim_steps < 1 or args.generation_batch_size < 1:
        raise ValueError("DDIM steps and generation batch size must be positive.")


def run(args: argparse.Namespace) -> dict[str, object]:
    """Execute generation, validation, metrics and visualization."""
    validate_arguments(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(args.base_seed)
    device = resolve_device(args.device)

    hic_min, hic_max, fish_min_um, fish_max_um = (
        load_normalization_scalers(args.scaler_path)
    )
    hic_raw = select_matrix(
        np.load(args.hic_path),
        args.hic_index,
        "Hi-C",
    )
    fish_centroid_um = select_matrix(
        np.load(args.reference_fish_centroid_path),
        0,
        "reference DNA-FISH centroid",
    )
    fish_centroid_um = 0.5 * (fish_centroid_um + fish_centroid_um.T)
    np.fill_diagonal(fish_centroid_um, 0.0)

    condition = normalize_hic(hic_raw, hic_min, hic_max, device)
    model = load_checkpoint(args.model_path, device)
    seeds = [args.base_seed + index for index in range(args.num_samples)]

    print(f"Using device: {device}")
    print(
        f"Generating {args.num_samples} cells "
        f"with seeds {seeds[0]}-{seeds[-1]}"
    )
    generated_um = generate_ensemble(
        model=model,
        condition=condition,
        seeds=seeds,
        steps=args.ddim_steps,
        batch_size=args.generation_batch_size,
        minimum_um=fish_min_um,
        maximum_um=fish_max_um,
    )
    validity = validate_distance_ensemble(generated_um)
    if not validity["numerically_valid"]:
        raise RuntimeError(f"Generated ensemble failed validation: {validity}")

    generated_centroid_um = np.mean(generated_um, axis=0)
    centroid_pcc = pearson_correlation(
        lower_triangle_vectors(generated_centroid_um)[0],
        lower_triangle_vectors(fish_centroid_um)[0],
    )
    generated_diversity = mean_pairwise_pcc(generated_um)

    np.save(args.output_dir / "input_hic.npy", hic_raw)
    np.save(
        args.output_dir / "reference_dna_fish_centroid_um.npy",
        fish_centroid_um,
    )
    np.save(
        args.output_dir / "generated_single_cell_distances_um.npy",
        generated_um,
    )
    np.save(
        args.output_dir / "generated_ensemble_centroid_um.npy",
        generated_centroid_um,
    )

    visualization = write_relative_centroid_html(
        fish_centroid_um=fish_centroid_um,
        generated_centroid_um=generated_centroid_um,
        centroid_pcc=centroid_pcc,
        generated_diversity=generated_diversity,
        generated_count=args.num_samples,
        output_dir=args.output_dir,
        random_state=args.base_seed,
    )
    summary: dict[str, object] = {
        "checkpoint": str(args.model_path),
        "num_generated_cells": args.num_samples,
        "ddim_steps": args.ddim_steps,
        "eta": 0.0,
        "base_seed": args.base_seed,
        "seeds": seeds,
        "centroid_pcc": centroid_pcc,
        "generated_mean_pairwise_pcc": generated_diversity,
        "representative_cell_selection": False,
        "experimental_nearest_neighbor_search": False,
        "complete_fish_library_exported": False,
        "normalization_scalers": {
            "hic_min_train": hic_min,
            "hic_max_train": hic_max,
            "dna_fish_min_train_um": fish_min_um,
            "dna_fish_max_train_um": fish_max_um,
        },
        "validity": validity,
        "visualization": visualization,
    }
    with (args.output_dir / "summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print("\nHiC2FISH centroid  completed.")
    print(f"Centroid-PCC: {centroid_pcc:.4f}")
    print(f"Generated mean pairwise PCC: {generated_diversity:.4f}")
    print(f"Output directory: {args.output_dir}")
    print(f"Interactive HTML: {visualization['html_path']}")

    if args.show:
        webbrowser.open(Path(visualization["html_path"]).resolve().as_uri())
    return summary


def main() -> None:
    run(build_parser().parse_args())
