# Data

This directory contains the example inputs, preprocessing manifest and processed matrices used by HiC2FISH.

## Contents

- [`example_data/`](example_data/): compact inputs used by `run.py`.
- [`processed/`](processed/): matched training and validation arrays and their training-partition scalers.
- `preprocessing_manifest.csv`: configuration table used by `preprocess_hic2fish_data.py` to locate raw DNA-FISH Excel workbooks, `.mcool` files and genomic regions.

Raw DNA-FISH workbooks and `.mcool` files are not required for running the supplied inference workflow. They are needed only when regenerating the processed arrays.
