# Modelling

Data curation, training, evaluation and explainability for fetal ultrasound plane
classification. Results are in [`../docs/RESULTS.md`](../docs/RESULTS.md) and the full walkthrough is in [`../docs/REPRODUCING.md`](../docs/REPRODUCING.md).

## Questions this pipeline answers

1. Which lightweight architecture gives the best accuracy for its size and speed?
2. How much does a model trained on Spanish images degrade on images from five African countries?
3. Does selective CLAHE (contrast enhancement for low-contrast images only) reduce that gap?
4. Does Leave-One-Country-Out fine-tuning with weight averaging recover the performance, and
   does it beat simpler alternatives?

## Pipeline

```
Zenodo archives (mirrored on Hugging Face)
        |  scripts/prepare_datasets.py   (src/data/hf_sync.py, src/data/curate.py)
        v
data/fetal_db, data/african     train / val / test, one folder per class
        |  notebook 3, scripts/train.py
        v
Spain-trained checkpoints (8 architectures, 3- and 4-class heads)
        |  notebook 4                    zero-shot evaluation on Malawi, with / without CLAHE
        |  notebook 5, scripts/run_loco_cv.py
        v
4 LOCO fold checkpoints per model (one non-Malawi country as validation per fold)
        |  notebook 6                    weight averaging ("model soup"), Grad-CAM, FPS
        v
averaged checkpoint  ->  notebook 7  validation study  ->  deployment/
```

Malawi is never used for training or model selection in the LOCO experiments.

## Layout

```
modelling/
|-- configs/
|   |-- base_config.yaml      data paths, preprocessing, augmentation, training, logging
|   |-- experiments.yaml      named experiments for scripts/train.py
|   `-- models.yaml           architecture reference (timm names, parameter counts)
|-- notebooks/                1 to 7, run in order
|-- raw_data/
|   `-- African_planes_database.csv   reference copy, used to verify downloads
|-- scripts/
|   |-- prepare_datasets.py   download raw archives and curate splits
|   |-- download_data.sh     shell wrapper for prepare_datasets.py
|   |-- fetch_checkpoints.py  download pretrained checkpoints
|   |-- download_weights.sh   shell wrapper for fetch_checkpoints.py
|   |-- train.py              train one experiment from experiments.yaml
|   |-- evaluate.py           evaluate a checkpoint on Spain and / or Malawi test sets
|   |-- run_loco_cv.py        LOCO fine-tuning from a pretrained checkpoint
|   `-- publish_to_huggingface.py   upload data mirror and checkpoints (maintainer only)
|-- src/
|   |-- data/                 datasets, curation, preprocessing (CLAHE, border crop), augmentation, LOCO
|   |-- models/model_zoo.py   timm model factory, freezing helpers, FPS benchmark
|   |-- training/             Trainer (AMP, early stopping, cosine LR, W&B / MLflow), focal loss
|   |-- evaluation/           metrics, cross-domain evaluation, feature extraction
|   |-- explainability/       Grad-CAM
|   `-- utils/                logging, seeding
`-- tests/
```

## Setup

```bash
pip install -r requirements.txt
python scripts/prepare_datasets.py
python scripts/fetch_checkpoints.py
pytest -q
```

## Notebooks

| # | Notebook | Purpose |
|---|---|---|
| 1 | `1-dataset-curation` | Download raw data, inspect African images, build splits |
| 2 | `2-eda-preprocessing-experiments` | Class balance, image quality by domain, CLAHE tuning |
| 3 | `3-model-experiments` | Benchmark 8 architectures: accuracy, F1, AUC, FPS, size |
| 4 | `4-domain-shift-analysis` | Zero-shot Spain to Malawi, CLAHE effect, per-class drop, t-SNE |
| 5 | `5-loco-experiments` | LOCO fine-tuning for EfficientNet-B0, ConvNeXt-Tiny, EfficientFormerV2-S0 |
| 6 | `6-fetal-explainability` | Model soup, Grad-CAM, speed benchmark |
| 7 | `7-validation-study` | Corrected Malawi result, bootstrap CIs, symmetric hold-out, baselines, CLAHE sweep, Grad-CAM metrics, CPU cost |

## Key design choices

| Choice | Value | Reason |
|---|---|---|
| Backbone | EfficientNet-B0 (timm, ImageNet weights) | 0.936 macro-F1 on Spain test at 4.0M parameters |
| Input | 224 x 224, ImageNet normalisation | Matches pretrained weights |
| Preprocessing | Crop scanner borders; CLAHE (clip 2.0, 8 x 8) only if grayscale std < 35 | Removes UI overlays; enhances only low-contrast images |
| Augmentation | Flip, rotate +/- 15 deg, brightness / contrast, Gaussian noise, small coarse dropout | No large shears or zooms that distort anatomy |
| Optimiser | AdamW, lr 1e-4 (5e-5 for fine-tuning), weight decay 0.01, cosine schedule | |
| Fine-tuning | Last ~4 blocks plus head unfrozen, 20 epochs, early stopping on val loss | Full fine-tuning overfits with about 225 images per fold |
| Class balance | Inverse-frequency weighted sampler | |
| Checkpoint selection | Best validation PR-AUC | |

Class order: 3-class checkpoints use `["brain", "femur", "abdomen"]`; 4-class checkpoints use
`["abdomen", "brain", "femur", "other"]` (alphabetical, from the folder layout).

## Experiment tracking

`--tracker wandb` (default in `train.py`) needs `WANDB_API_KEY`; use `--tracker none` to
disable, or `mlflow` / `both` for MLflow.

## Data sources

- Burgos-Artizzu X. P. et al. *Evaluation of deep convolutional neural networks for automatic
  classification of common maternal fetal ultrasound planes.* Scientific Reports 10, 2020.
  [Zenodo 3904280](https://zenodo.org/records/3904280). CC BY 4.0.
- Sendra-Balcells C. et al. *Generalisability of fetal ultrasound deep learning models to
  low-resource imaging settings in five African countries.* Scientific Reports 13, 2023.
  [Zenodo 7540448](https://zenodo.org/records/7540448). CC BY 4.0.
