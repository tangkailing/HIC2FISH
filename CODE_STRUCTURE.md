# Code organization

The two root scripts are command-line entry points only. Reusable code lives
inside the `hic2fish` package.

| Module | Responsibility |
| --- | --- |
| `hic2fish/model.py` | Enhanced U-Net architecture used by the checkpoint |
| `hic2fish/diffusion.py` | Cosine schedule and deterministic DDIM sampling |
| `hic2fish/io.py` | Checkpoint loading, scaler loading and input validation |
| `hic2fish/metrics.py` | Centroid-PCC support, diversity and validity checks |
| `hic2fish/visualization.py` | MDS reconstruction and interactive 3D output |
| `hic2fish/demo.py` | End-to-end demo orchestration |
| `hic2fish/preprocessing.py` | Raw Excel/.mcool preprocessing pipeline |

Run the public demo from the repository root:

```bash
python run_demo.py
```

Prepare data after filling `data/preprocessing_manifest.csv`:

```bash
python preprocess_hic2fish_data.py --manifest data/preprocessing_manifest.csv
```
