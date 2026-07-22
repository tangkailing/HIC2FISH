# Expected output

This directory contains a reference summary generated with the supplied checkpoint, example input and default command-line settings.

## File

### `demo_summary.json`

The summary records:

- checkpoint and sampling settings;
- number of generated cells and random seeds;
- Centroid-PCC;
- generated mean pairwise PCC;
- normalization parameters;
- matrix-validity checks;
- MDS visualization metadata.

With 100 generated cells, 100 DDIM steps, `eta = 0` and base seed 2026, the supplied configuration gives approximately:

```text
Centroid-PCC:                  0.8612
Generated mean pairwise PCC:  0.0804
```

All generated matrices are expected to be finite, non-negative and symmetric, with zero-valued diagonals. Small floating-point differences can occur between PyTorch versions, CUDA versions, GPUs and CPU execution.

The reference summary is provided for verification only and is not read by the model during generation.
