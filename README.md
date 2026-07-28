<div align="center">
  <img src="docs/figures/HiC2FISH_logo_readme.png"
       width="210"
       alt="HiC2FISH logo">

  <p>
    <strong>A conditional diffusion framework for generating single-cell DNA-FISH distance matrices from bulk Hi-C contact maps</strong>
  </p>
</div>

## Contents

- [Overview](#overview)
- [Quick start](#quick-start)
- [Installation](#installation)
- [Running the supplied example](#running-the-supplied-example)
  - [Lightweight CPU check](#lightweight-cpu-check)
  - [Full generation example](#full-generation-example)
  - [Generation options](#generation-options)
- [Outputs](#outputs)
- [Using custom Hi-C inputs](#using-custom-hi-c-inputs)
  - [Input requirements](#input-requirements)
  - [Supported NumPy shapes](#supported-numpy-shapes)
  - [Running on custom data](#running-on-custom-data)
- [Preparing Hi-C and DNA-FISH data](#preparing-hi-c-and-dna-fish-data)
  - [Input data formats](#input-data-formats)
  - [Preprocessing manifest](#preprocessing-manifest)
  - [Preprocessing workflow](#preprocessing-workflow)
  - [Generated processed files](#generated-processed-files)
- [Evaluation and visualization](#evaluation-and-visualization)
- [Repository structure](#repository-structure)
- [Reproducibility](#reproducibility)
- [Pretrained model](#pretrained-model)
- [Data availability](#data-availability)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

## Overview

HiC2FISH is a conditional diffusion framework that generates ensembles of
single-cell, DNA-FISH-calibrated chromatin distance matrices from a
population-level Hi-C contact map. During inference, a prepared Hi-C matrix
conditions the denoising of independently sampled Gaussian noise. The released
workflow produces symmetric 50 × 50 distance matrices in micrometres, with one
matrix representing one generated cell.

The repository supports three distinct workflows:

| Goal | Required user data | Result |
| --- | --- | --- |
| Generate single-cell structures | A prepared 50 × 50 Hi-C matrix | Generated single-cell distance matrices and their ensemble centroid |
| Evaluate generated structures | The matching Hi-C matrix plus an experimental DNA-FISH ensemble centroid | Generation outputs, Centroid-PCC and comparative 3D visualization |
| Rebuild paired training data | Raw Hi-C `.mcool` and DNA-FISH coordinate workbook | Matched Hi-C–DNA-FISH training and validation arrays |

## Quick start

The quickest route to a complete and visually interpretable result is to run
the supplied K562 example. HiC2FISH generates 100 single-cell distance matrices
using only the supplied Hi-C condition, and then compares the generated
ensemble centroid with the supplied experimental DNA-FISH centroid.

Clone the repository with Git LFS and create an isolated Python environment:

```bash
git lfs install
git clone https://github.com/tangkailing/HIC2FISH.git
cd HIC2FISH
git lfs pull
python -m venv .venv
```

Activate the environment using the command for your operating system:

```bat
:: Windows Command Prompt
.venv\Scripts\activate.bat
```

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS or Linux
source .venv/bin/activate
```

Install the dependencies and run the complete supplied example:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run.py --reference-fish-centroid-path data/example_data/example_dna_fish_centroid_um.npy --output-dir output/full_evaluation --show
```

The default settings generate 100 cells using 100 deterministic DDIM steps. All
results are written to `output/full_evaluation/`, including the generated
single-cell distance matrices, their ensemble centroid, and `summary.json`
with the sampling settings and evaluation metrics. See [Outputs](#outputs) for
a complete description of every generated file. Because this command uses
`--show`, the interactive three-dimensional comparison is also generated and
opened automatically.

The supplied DNA-FISH centroid is used only for evaluation and visualization.
It is not an input to the diffusion model and is not required when generating
structures from a new Hi-C condition. The complete example may take longer on
a CPU-only computer; use the [lightweight CPU check](#lightweight-cpu-check)
below to verify the installation first.

## Installation

HiC2FISH requires Python 3.10 or later. The workflow was also tested in a clean
Python 3.12 environment on Windows. CPU execution is supported; a compatible
CUDA-enabled PyTorch installation can be used for faster generation.

The required packages are listed in `requirements.txt` and include:

- PyTorch;
- NumPy and SciPy;
- pandas and openpyxl;
- cooler;
- scikit-learn;
- Plotly.

Git LFS must be installed before cloning because the checkpoint and NumPy
arrays are stored as large-file objects. If these files appear as 1–2 kB text
files, they are LFS pointers rather than the actual data. Run:

```bash
git lfs pull
```

The following supplied files should then be present:

```text
pretrained/hic2fish.pt
data/example_data/example_hic.npy
data/example_data/normalization_scalers.npz
data/example_data/example_dna_fish_centroid_um.npy
```

The first three files are required by the supplied generation workflow. The
DNA-FISH centroid is an optional reference used only for evaluation.

## Running the supplied example

The supplied input represents K562 chromosome 21,
`chr21:28,000,000–30,000,000`, using 50 ordered loci.

### Lightweight CPU check

Use the reduced command below to verify checkpoint loading, input handling,
CPU execution and output writing:

```bash
python run.py \
  --device cpu \
  --num-samples 2 \
  --ddim-steps 1 \
  --generation-batch-size 1 \
  --output-dir output/cpu_smoke_test
```

On Windows Command Prompt:

```bat
python run.py ^
  --device cpu ^
  --num-samples 2 ^
  --ddim-steps 1 ^
  --generation-batch-size 1 ^
  --output-dir output\cpu_smoke_test
```

This is an actual reduced inference run, but it is intended only as an
installation check. Two cells and one DDIM step are not sufficient to
reproduce the reported example metrics.

### Full generation example

Generate 100 cells from the supplied Hi-C input without using an experimental
DNA-FISH reference:

```bash
python run.py --output-dir output/full_generation
```

The defaults use 100 deterministic DDIM steps, `eta = 0`, and random seeds
2026–2125.

To additionally evaluate the generated ensemble against the supplied
experimental DNA-FISH centroid, run:

```bash
python run.py \
  --reference-fish-centroid-path data/example_data/example_dna_fish_centroid_um.npy \
  --output-dir output/full_evaluation
```

On Windows Command Prompt:

```bat
python run.py ^
  --reference-fish-centroid-path data\example_data\example_dna_fish_centroid_um.npy ^
  --output-dir output\full_evaluation
```

For the supplied reference configuration, 100 generated matrices produced a
Centroid-PCC of approximately 0.8612 and a generated mean pairwise PCC of
approximately 0.0804. Small numerical differences can occur between software
versions and hardware platforms. These values are reference values rather
than pass/fail thresholds.

### Generation options

Run `python run.py --help` to display all options.

| Option | Purpose | Default |
| --- | --- | --- |
| `--hic-path` | Prepared Hi-C NumPy input | `data/example_data/example_hic.npy` |
| `--hic-index` | Matrix selected from a stacked NumPy input | `0` |
| `--model-path` | Pretrained checkpoint | `pretrained/hic2fish.pt` |
| `--scaler-path` | Training-derived Hi-C and DNA-FISH scalers | `data/example_data/normalization_scalers.npz` |
| `--reference-fish-centroid-path` | Optional experimental DNA-FISH centroid | Not used |
| `--num-samples` | Number of generated cells | `100` |
| `--ddim-steps` | Number of deterministic sampling steps | `100` |
| `--base-seed` | First random seed | `2026` |
| `--generation-batch-size` | Number of cells generated in each batch | `2` |
| `--device` | `auto`, `cpu` or `cuda` | `auto` |
| `--output-dir` | Directory receiving generated files | `output/` |
| `--show` | Open the comparative HTML after reference-based evaluation | Off |

## Outputs

The following files are always written to the selected output directory:

| File | Description |
| --- | --- |
| `input_hic.npy` | Hi-C matrix used as the generation condition |
| `generated_single_cell_distances_um.npy` | Generated matrices with shape `(n_cells, 50, 50)`, in micrometres |
| `generated_ensemble_centroid_um.npy` | Mean of the generated single-cell matrices |
| `summary.json` | Sampling settings, seeds, diversity, evaluation status and validity checks |

When `--reference-fish-centroid-path` is supplied, the following files are also
written:

| File | Description |
| --- | --- |
| `reference_dna_fish_centroid_um.npy` | Experimental ensemble centroid used for evaluation |
| `fish_centroid_relative_distance.npy` | Scale-normalized experimental centroid used for visualization |
| `generated_centroid_relative_distance.npy` | Scale-normalized generated centroid used for visualization |
| `fish_centroid_relative_coordinates.npy` | Experimental centroid MDS coordinates |
| `generated_centroid_relative_coordinates.npy` | Aligned generated centroid MDS coordinates |
| `centroid_3d_comparison.html` | Interactive comparison of experimental and generated relative geometries |

The repository's `expected_output/` directory contains a reference summary. It
is not an output destination and is not read by the model.

## Using custom Hi-C inputs

### Input requirements

`run.py` accepts a prepared NumPy matrix, not a raw `.hic`, `.cool` or `.mcool`
file. A custom input must:

1. describe 50 ordered genomic loci;
2. resolve to one 50 × 50 matrix after selection;
3. contain finite numeric contact values;
4. use the same locus order in any optional DNA-FISH reference;
5. remain on the processed contact-value scale expected by the supplied
   training scaler.

Do not manually min–max normalize a custom matrix before running `run.py`.
Input normalization is applied automatically using the values in
`normalization_scalers.npz`.

The released checkpoint is fixed to 50 × 50. A matrix with another size cannot
be passed to this checkpoint by changing a command-line option; another input
size requires compatible preprocessing and model retraining.

### Supported NumPy shapes

`run.py` accepts the following layouts and uses `--hic-index` to select one
matrix:

| Input shape | Interpretation |
| --- | --- |
| `(50, 50)` | One matrix; only `--hic-index 0` is valid |
| `(50, 50, 1)` | One channels-last matrix |
| `(n, 50, 50)` | Stack of matrices |
| `(n, 50, 50, 1)` | Channels-last stack |
| `(n, 1, 50, 50)` | Channels-first stack |

All selected matrices are checked for the required 50 × 50 shape and for NaN
or infinite values before inference.

### Running on custom data

For a single prepared matrix:

```bash
python run.py \
  --hic-path path/to/hic_50x50.npy \
  --output-dir output/custom_hic
```

For a stacked array, select one matrix:

```bash
python run.py \
  --hic-path path/to/hic_stack.npy \
  --hic-index 3 \
  --output-dir output/custom_hic_index_3
```

If the matching experimental DNA-FISH centroid is available, add:

```bash
--reference-fish-centroid-path path/to/dna_fish_centroid_um.npy
```

The current repository does not provide a one-command conversion from an
arbitrary raw Hi-C file to a model-ready matrix when no locus or probe
definition is available. Users starting from raw Hi-C must first define the 50
ordered target loci and construct the corresponding matrix on a scale
compatible with the supplied preprocessing workflow.

## Preparing Hi-C and DNA-FISH data

This section is for rebuilding paired training/validation arrays, retraining,
or preparing an experimental reference. It is not required to generate cells
from a model-ready Hi-C NumPy matrix with the released checkpoint.

### Input data formats

| Modality | Expected format | Required information |
| --- | --- | --- |
| Hi-C | Multi-resolution Cooler `.mcool` | Chromosome, genomic start/end, resolution and balancing choice |
| DNA-FISH | Excel workbook readable by pandas/openpyxl | `Trace_ID`, X/Y/Z coordinates, chromosome, probe start and probe end |

Rows belonging to one `Trace_ID` represent one experimental cell. Coordinate
units are converted to micrometres using `coordinate_scale_to_um` in the
manifest.

### Preprocessing manifest

Copy `data/preprocessing_manifest.csv` and add one row for each biological
condition and genomic window.

| Field | Description |
| --- | --- |
| `condition_id` | Unique label for one cell type and genomic window |
| `fish_xlsx` | Path to the raw DNA-FISH coordinate workbook |
| `fish_sheet` | Optional sheet name or zero-based sheet index |
| `mcool_path` | Path to the multi-resolution Cooler file |
| `chromosome` | Chromosome, such as `chr21` or `21` |
| `region_start` | Zero-based genomic start in base pairs |
| `region_end` | Genomic end in base pairs |
| `resolution` | Cooler resolution in base pairs |
| `balance` | `true` or `false` for balanced contact extraction |
| `coordinate_scale_to_um` | Use `1` for micrometres or `0.001` for nanometres |
| `split` | `train`, `val` or `auto` |

Relative input paths are resolved from the directory containing the manifest.
Duplicate `condition_id` values are rejected.

### Preprocessing workflow

Run from the repository root:

```bash
python preprocess_hic2fish_data.py \
  --manifest data/preprocessing_manifest.csv
```

The workflow:

1. selects the requested cell type and genomic region;
2. orders DNA-FISH probes by genomic interval;
3. restores missing loci at their known genomic positions;
4. interpolates eligible missing coordinates;
5. converts each retained trace into a 50 × 50 Euclidean distance matrix;
6. extracts the Hi-C region from the requested Cooler resolution;
7. aligns Hi-C contacts to the ordered DNA-FISH probe midpoints;
8. constructs matched Hi-C–DNA-FISH pairs;
9. assigns training and validation cells;
10. calculates Hi-C and DNA-FISH min–max scalers from the training partition
    only.

Existing processed files are preserved unless `--overwrite` is supplied.

### Generated processed files

By default, files are written to `data/processed/`.

| File | Description |
| --- | --- |
| `X_train.npy` | Training Hi-C conditioning matrices |
| `Y_train.npy` | Training single-cell DNA-FISH distance matrices in micrometres |
| `X_val.npy` | Validation Hi-C conditioning matrices |
| `Y_val.npy` | Validation single-cell DNA-FISH distance matrices in micrometres |
| `X_minmax.npy` | Hi-C minimum and maximum from `X_train.npy` |
| `Y_minmax.npy` | DNA-FISH minimum and maximum from `Y_train.npy` |
| `preprocessing_summary.json` | Settings, retained-cell counts, exclusions and scaler values |

The saved paired arrays use channels-last layout:

```text
X_train, Y_train: (n_train, 50, 50, 1)
X_val,   Y_val:   (n_val, 50, 50, 1)
```

## Evaluation and visualization

Reference-based evaluation is optional. The experimental DNA-FISH reference
must represent the same cell type, genomic interval, 50 loci and locus order as
the Hi-C condition.

Run reference-based evaluation with:

```bash
python run.py \
  --hic-path path/to/matching_hic_50x50.npy \
  --reference-fish-centroid-path data/evaluation/dna_fish_centroid_um.npy \
  --output-dir output/evaluation
```

The workflow reports:

- **Centroid-PCC:** Pearson correlation between the strict lower triangles of
  the generated and experimental ensemble-centroid distance matrices;
- **Generated mean pairwise PCC:** average lower-triangle correlation among
  independently generated cells under the same Hi-C condition. Values near 1
  indicate highly similar generated cells; lower values indicate greater
  ensemble diversity.

Generated mean pairwise PCC is available without experimental DNA-FISH.
Centroid-PCC is calculated only when a reference centroid is supplied.

The comparative HTML uses metric multidimensional scaling to display the
relative geometry of the two ensemble centroids. Each distance matrix is
normalized by its own median lower-triangular distance, and each coordinate
configuration is normalized to unit radius of gyration. 

<p align="center">
  <img src="docs/figures/hic2fish_example_output.png"
       width="900"
       alt="HiC2FISH example output">
</p>

## Repository structure

```text
HiC2FISH/
├── data/
│   ├── example_data/              # Supplied inference input and optional reference
│   ├── processed/                 # Matched training and validation arrays
│   └── preprocessing_manifest.csv # Raw-data preprocessing configuration
├── docs/figures/                   # README figures
├── expected_output/               # Reference summary for verification
├── hic2fish/                       # Reusable Python package
├── output/                         # Generated files
├── pretrained/                     # Pretrained checkpoint
├── preprocess_hic2fish_data.py     # Paired-data preprocessing entry point
├── run.py                          # Generation and optional evaluation entry point
├── requirements.txt
└── CODE_STRUCTURE.md
```

Detailed file-format descriptions are provided in the README files inside the
main subdirectories.

## Reproducibility

- The supplied checkpoint and example arrays are tracked with Git LFS.
- Hi-C and DNA-FISH values use separate global scalers estimated from the
  training partition.
- `run.py` applies the supplied scalers automatically.
- Deterministic DDIM sampling uses `eta = 0`.
- Independent cells are initialized with consecutive random seeds beginning
  at `--base-seed`.
- The final strict lower triangle is mirrored to produce a symmetric matrix,
  and diagonal entries are set to zero.
- Generated matrices are checked for finiteness, non-negativity, symmetry and
  zero diagonals.
- `summary.json` records checkpoint, sampling settings, seeds, metrics,
  evaluation status and validity checks.
- Small floating-point differences can occur across PyTorch versions,
  operating systems, CPUs and GPUs.

The lightweight CPU command verifies installation and execution. The full
100-cell, 100-step command should be used when comparing against the supplied
reference summary.

## Pretrained model

`pretrained/hic2fish.pt` is a PyTorch state dictionary for the
`EnhancedUNet` architecture defined in `hic2fish/model.py`. During diffusion,
the network receives:

- a noisy distance matrix with shape `(batch, 1, 50, 50)`;
- a normalized Hi-C condition with shape `(batch, 1, 50, 50)`;
- a continuous diffusion time with shape `(batch, 1)`.

The noisy distance matrix is generated from Gaussian noise during inference;
it is not an experimental DNA-FISH input.

The checkpoint must be used with the associated training-derived scalers in
`data/example_data/normalization_scalers.npz`, which contains:

```text
hic_min_train
hic_max_train
dna_fish_min_train_um
dna_fish_max_train_um
```

A custom checkpoint can be supplied with `--model-path`. It must use the same
architecture, layer names and tensor dimensions as `EnhancedUNet`. Models
trained for other matrix sizes require corresponding code, checkpoint and
scaler changes.

## Data availability

The supplied example and processed arrays were derived from paired K562
datasets from the 4D Nucleome Data Portal:

- bulk in situ Hi-C:
  [`4DNFI18UHVRO`](https://data.4dnucleome.org/files-processed/4DNFI18UHVRO/);
- multiplexed DNA-FISH:
  [`4DNEST5FUQKC`](https://data.4dnucleome.org/experiment-set-replicates/4DNEST5FUQKC/).

The example is restricted to K562 chromosome 21,
`chr21:28,000,000–30,000,000`. Processed arrays, compact example inputs and the
pretrained checkpoint are distributed through this repository using Git LFS.
The original DNA-FISH workbooks and full `.mcool` files are not redistributed
in the repository and should be obtained from the source portal when
rebuilding the processed data.

## Citation


## License


## Contact

