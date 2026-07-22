# Processed data

This directory contains matched Hi-C–DNA-FISH arrays produced by `preprocess_hic2fish_data.py`.

## Files

| File | Description |
| --- | --- |
| `X_train.npy` | Training Hi-C conditioning matrices |
| `Y_train.npy` | Training single-cell DNA-FISH distance matrices in micrometres |
| `X_val.npy` | Validation Hi-C conditioning matrices |
| `Y_val.npy` | Validation single-cell DNA-FISH distance matrices in micrometres |
| `X_minmax.npy` | Global Hi-C minimum and maximum estimated from `X_train.npy` |
| `Y_minmax.npy` | Global DNA-FISH minimum and maximum estimated from `Y_train.npy` |
| `preprocessing_summary.json` | Optional preprocessing settings, retained-cell counts, scaler values and quality-control summary |

## Array layout

The paired arrays use channels-last layout:

```text
X_train: (n_train, 50, 50, 1)
Y_train: (n_train, 50, 50, 1)
X_val:   (n_val, 50, 50, 1)
Y_val:   (n_val, 50, 50, 1)
```

For a given cell type and genomic window, the same population Hi-C condition can be paired with multiple experimental DNA-FISH single-cell matrices. There is no assumed one-to-one identity between a generated cell and a particular experimental cell.

## Normalization

The saved `X` and `Y` arrays retain their processed numeric scales. Min–max transformation is applied when the arrays are loaded:

```text
X_scaled = (X - X_min_train) / (X_max_train - X_min_train)
Y_scaled = (Y - Y_min_train) / (Y_max_train - Y_min_train)
```

The scaler parameters are estimated from the training partition only and are applied unchanged to the validation partition.

## Regenerating the files

After filling `data/preprocessing_manifest.csv`, run from the repository root:

```bash
python preprocess_hic2fish_data.py --manifest data/preprocessing_manifest.csv
```

Use `--overwrite` only when the existing processed arrays should be replaced.
