# Pretrained model

This directory contains the pretrained HiC2FISH checkpoint used by `run.py`.

## File

### `hic2fish_demo.pt`

PyTorch state dictionary for the `EnhancedUNet` architecture defined in `hic2fish/model.py`. The checkpoint expects:

- a normalized noisy DNA-FISH matrix with shape `(batch, 1, 50, 50)`;
- a normalized Hi-C condition with shape `(batch, 1, 50, 50)`;
- a continuous diffusion time with shape `(batch, 1)`.

The network returns a single-channel 50 × 50 prediction of the Gaussian noise associated with the current diffusion state.

The checkpoint is loaded automatically by:

```bash
python run.py
```

A custom checkpoint can be supplied with:

```bash
python run.py --model-path /path/to/checkpoint.pt
```

The checkpoint must use the same layer names and tensor dimensions as `EnhancedUNet`. The loader accepts a plain state dictionary or dictionaries containing `state_dict` or `model_state_dict`.
