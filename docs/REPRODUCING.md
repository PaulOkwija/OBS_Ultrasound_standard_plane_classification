# Reproducing the Results

This guide walks through the modelling pipeline from an empty machine to the numbers in
[`RESULTS.md`](RESULTS.md), then shows how to serve a trained model.

## Requirements

| Item | Needed for |
|---|---|
| Python 3.10+ | everything |
| NVIDIA GPU (a Kaggle or Colab T4 is enough) | notebooks 3, 5, 6, 7 (training). CPU is fine for notebooks 1, 2, 4 and for inference |
| About 10 GB free disk | raw archives (about 2 GB), curated data, checkpoints (about 1.5 GB) |
| Total GPU time | roughly 2.5 hours on a T4 for all notebooks |

## 1. Set up the environment

```bash
git clone https://github.com/PaulOkwija/OBS_Ultrasound_standard_plane_classification.git
cd OBS_Ultrasound_standard_plane_classification/modelling
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                   # optional: makes `import src` work from anywhere
```

Every notebook starts with a setup cell that moves into `modelling/` (and clones the repository
first when run on Kaggle or Colab), so you can launch Jupyter from any folder.

## 2. Prepare the datasets

```bash
python scripts/prepare_datasets.py
```

This downloads both raw archives, verifies the African CSV against the committed copy in
`raw_data/`, and curates them into:

```
data/
|-- fetal_db/{train,val,test}/{abdomen,brain,femur,other}/*.png
`-- african/{train,val,test}/{abdomen,brain,femur,other}/*.png
```

Useful flags: `--skip-download` reuses `data/raw/`; `--force` re-curates over existing output.

Expected image counts (printed at the end):

| | train | val | test |
|---|---|---|---|
| fetal_db | 5,121 | 2,008 | 5,271 |
| african | 240 | 110 | 100 (Malawi) |

## 3. Download the pretrained checkpoints

```bash
python scripts/fetch_checkpoints.py           # into modelling/checkpoints/
```

Main files:

| File | Contents |
|---|---|
| `efficientnetb0_base_3c.pt` | EfficientNet-B0 trained on Spain, 3-class head. Starting point for LOCO |
| `efficientnetb0_base_4c.pt`, `best_efficientnet_b0_fetaldb.pt` | EfficientNet-B0 trained on Spain, 4 classes. Notebook 4 loads it as `checkpoints/efficientnet_b0.pt` |
| `effb0_best_f{0..3}.pt` | EfficientNet-B0 LOCO fold checkpoints |
| `eff_former_best_f{0..3}.pt` | EfficientFormerV2-S0 LOCO fold checkpoints |
| `averaged/*_loco_averaged.pt` | Model soups produced by notebook 6 |

Checkpoints are dicts with `model_state_dict`, `optimizer_state_dict`, `metrics`, `config` and,
for checkpoints written by the current trainer, `class_names`. Load them with:

```python
import torch
from src.models import build_model

ckpt = torch.load("checkpoints/averaged/efficientnet_b0_loco_averaged.pt", map_location="cpu")
model = build_model("efficientnet_b0", num_classes=3, pretrained=False)
model.load_state_dict(ckpt.get("model_state_dict", ckpt))
model.eval()
# 3-class output order: ["brain", "femur", "abdomen"]
```

## 4. Run the notebooks in order

| # | Notebook | Needs GPU | Produces |
|---|---|---|---|
| 1 | `1-dataset-curation` | no | `data/` (same as step 2) |
| 2 | `2-eda-preprocessing-experiments` | no | class balance, image-quality and CLAHE analysis |
| 3 | `3-model-experiments` | yes | 8-architecture benchmark (RESULTS Section 2) |
| 4 | `4-domain-shift-analysis` | no | zero-shot Spain to Malawi, CLAHE effect, t-SNE (Section 3) |
| 5 | `5-loco-experiments` | yes | LOCO fold checkpoints and scores (Section 4) |
| 6 | `6-fetal-explainability` | yes | model soups, Grad-CAM figures, FPS |
| 7 | `7-validation-study` | yes | corrected Malawi result, CIs, baselines, CLAHE sweep (Section 5) |

Notebook 7 writes its tables and figures to `results/validation/`.

## 5. Command-line equivalents

```bash
# Train one experiment from configs/experiments.yaml
python scripts/train.py --experiment baseline_fetal_3c --tracker none

# Evaluate a checkpoint (run directories get a random 4-digit suffix)
python scripts/evaluate.py --checkpoint checkpoints/<run_dir>/best_pr_auc.pt \
    --test_data african_test --fps --output results/eval.json

# LOCO fine-tuning from a Spain-trained 3-class checkpoint
python scripts/run_loco_cv.py --model efficientnet_b0 \
    --checkpoint checkpoints/efficientnetb0_base_3c.pt \
    --african_csv raw_data/African_planes_database.csv --tracker none
```

`--tracker wandb` logs to Weights & Biases if `WANDB_API_KEY` is set.

## 6. Run the tests

```bash
cd modelling && pytest -q        # 18 tests: curation, datasets, models, preprocessing, LOCO folds
cd ../deployment && pytest -q    # 37 tests: preprocessing, inference engine, API
```

## 7. Serve a trained model

```bash
cd deployment
pip install -r requirements.txt
python scripts/register_model.py \
    --name efficientnet_b0_soup --version v1 \
    --weights ../modelling/checkpoints/averaged/efficientnet_b0_loco_averaged.pt \
    --architecture efficientnet_b0 --num-classes 3 \
    --classes brain femur abdomen --format state_dict
python run.py
```

Then classify an image:

```bash
python client.py predict --image samples/sample_frame.png --model efficientnet_b0_soup
```

## Hosting data and checkpoints

Data and checkpoints are hosted on Hugging Face:

- `Okwija/afro-fetalnet-data` (dataset): an unmodified mirror of the two Zenodo archives
- `Okwija/afro-fetalnet-checkpoints` (model): the contents of `modelling/checkpoints/`

`scripts/publish_to_huggingface.py` creates or updates both (requires a Hugging Face write
token via `huggingface-cli login` or `HF_TOKEN`):

```bash
cd modelling
python scripts/publish_to_huggingface.py --target data
python scripts/publish_to_huggingface.py --target checkpoints
python scripts/publish_to_huggingface.py --target both --dry-run
```

To use another account, pass `--hf-repo-id` to `prepare_datasets.py` and `fetch_checkpoints.py`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `RepositoryNotFoundError` from Hugging Face | Check the repository id and your network access |
| African CSV hash mismatch | The mirror differs from `raw_data/African_planes_database.csv`. Investigate before continuing; use `--skip-verify` only if the change is intended |
| CUDA out of memory | Lower `training.batch_size` in `configs/base_config.yaml` |
| `size mismatch for classifier.weight` | The checkpoint's class count differs from the config. Use the `*_3c` checkpoints with 3 classes and `*_4c` with 4 |
| Force CPU | `export CUDA_VISIBLE_DEVICES=""` (Windows: `set CUDA_VISIBLE_DEVICES=`) |
