# HiC2FISH Python package

This directory contains the reusable implementation of the HiC2FISH generation, evaluation, visualization and preprocessing workflow. The root-level scripts provide command-line entry points and call functions from this package.

## Modules

| Module | Responsibility |
| --- | --- |
| `model.py` | Enhanced U-Net architecture, time embedding and self-attention |
| `diffusion.py` | Cosine signal-retention schedule and deterministic DDIM sampling |
| `io.py` | Checkpoint loading, scaler loading, matrix selection and device handling |
| `metrics.py` | Lower-triangle vectorization, Pearson correlation, diversity and validity checks |
| `visualization.py` | Relative-distance normalization, metric MDS and interactive 3D output |
| `demo.py` | End-to-end generation and evaluation orchestration |
| `preprocessing.py` | DNA-FISH coordinate processing, `.mcool` extraction and dataset construction |
| `constants.py` | Shared 50-locus matrix dimensions |
| `__init__.py` | Package metadata |

## Model architecture

`EnhancedUNet` receives two single-channel 50 × 50 matrices: the current noisy DNA-FISH state and the corresponding Hi-C condition. The encoder uses 64, 128 and 256 channels. A learned 64-dimensional time embedding is added to the first encoder representation. The decoder includes skip connections and an eight-head self-attention layer operating on the 128-channel spatial representation. A final 1 × 1 convolution produces a single-channel noise estimate.

## Programmatic use

The model can be imported with:

```python
from hic2fish.model import EnhancedUNet
```

Generation and evaluation functions can be imported independently:

```python
from hic2fish.diffusion import generate_ensemble
from hic2fish.metrics import mean_pairwise_pcc
```

For a complete reproducible execution, use the root entry point:

```bash
python run.py
```
