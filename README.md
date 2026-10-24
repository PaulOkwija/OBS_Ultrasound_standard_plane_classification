# Afro-FetalNet

Fetal ultrasound standard-plane classification (brain, abdomen, femur) for low-resource
African clinical settings. The project trains lightweight CNNs on a large Spanish dataset,
measures how badly they degrade on images from five African countries, adapts them with
Leave-One-Country-Out (LOCO) fine-tuning and weight averaging ("model soup"), and serves the
result through a small FastAPI server with a live-video view.

The repository has two independent parts:

| Part | What it is |
|---|---|
| [`modelling/`](modelling/README.md) | Data curation, training, evaluation, Grad-CAM, and 7 notebooks that record every experiment |
| [`deployment/`](deployment/README.md) | Offline inference server: REST API, batch inference, model registry, web UI, UDP video streaming |

## Results at a glance

Each number below comes from a saved notebook output. Full tables are in
[`docs/RESULTS.md`](docs/RESULTS.md).

| Measurement | Value | Source |
|---|---|---|
| EfficientNet-B0 size / speed (T4 GPU, batch 1) | 4.0M params, 130 FPS, 7.7 ms | notebook 3 |
| Spain test set, 4 classes (n=5,271) | 95.7% accuracy, 0.936 macro-F1 | notebook 3 |
| Zero-shot on Malawi, 4 classes (n=100) | 62.0% accuracy, 0.568 macro-F1 | notebook 4 |
| Zero-shot on Malawi with selective CLAHE | 70.0% accuracy, 0.700 macro-F1 | notebook 4 |
| LOCO fine-tune + model soup, Malawi, 3 classes (n=75, 26 patients) | 0.987 macro-F1, patient-level 95% CI [0.959, 1.000] | notebook 7 |
| Pooled fine-tune baseline (no LOCO, no soup), same test set | 1.000 macro-F1 (single run) | notebook 7 |

Two findings matter when reading these numbers:

- **The simplest baseline is at least as good as the full method.** On this 75-image test set,
  pooled fine-tuning on all four training countries matched or beat LOCO + model soup, and
  prediction-level ensembling gave identical predictions to the soup.
- **Malawi is not an outlier.** When each of the five countries is held out in turn, held-out
  F1 falls in a tight 0.92 to 0.96 band, with Malawi at the top. The earlier impression that the
  unseen country beat every validation fold came from an asymmetric protocol.

## Quick start

### Run the inference server (no GPU or dataset needed)

```bash
cd deployment
pip install -r requirements.txt
python run.py
```

Open http://localhost:8000 for the image UI, http://localhost:8000/video for the video UI, and
http://localhost:8000/docs for the API. A TorchScript EfficientNet-B0 is included in
`deployment/model_registry/efficientnet_b0/v1/`.

### Reproduce the modelling pipeline

```bash
cd modelling
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/prepare_datasets.py                       # download + curate both datasets
python scripts/fetch_checkpoints.py                      # pretrained checkpoints
jupyter lab notebooks/
```

Run the notebooks in numeric order. Full instructions, expected outputs and compute
requirements are in [`docs/REPRODUCING.md`](docs/REPRODUCING.md).

## Repository layout

```
.
|-- README.md
|-- docs/
|   |-- RESULTS.md            results, with the notebook each number comes from
|   `-- REPRODUCING.md        step-by-step reproduction guide
|-- modelling/
|   |-- configs/              YAML configs (base, experiment matrix, model zoo)
|   |-- notebooks/            1 to 7, run in order
|   |-- raw_data/             African dataset CSV (used to verify downloads)
|   |-- scripts/              CLI entry points: prepare, train, evaluate, LOCO, publish
|   |-- src/                  data, models, training, evaluation, explainability, utils
|   `-- tests/
`-- deployment/
    |-- app/                  FastAPI application and static web UI
    |-- model_registry/       versioned models: <name>/<version>/{model.pt, config.json}
    |-- samples/              sample ultrasound frame for quick tests
    |-- scripts/              register_model.py, batch_infer.py
    |-- tests/
    |-- client.py             CLI client (predict, batch, models, stream)
    `-- sender.py             UDP video sender that simulates an ultrasound machine
```

## Data

Both datasets are public and licensed CC BY 4.0. Cite the original authors when you use them.

| Dataset | Images | Classes used | Source |
|---|---|---|---|
| FETAL_PLANES_DB (Spain) | 12,400 from 1,792 patients | brain, femur, abdomen, other | Burgos-Artizzu et al., *Scientific Reports* 2020, [Zenodo 3904280](https://zenodo.org/records/3904280) |
| African fetal planes (Algeria, Egypt, Ghana, Malawi, Uganda) | 450 | brain, femur, abdomen (thorax mapped to other) | Sendra-Balcells et al., *Scientific Reports* 2023, [Zenodo 7540448](https://zenodo.org/records/7540448) |

Malawi is held out entirely as the test country in every LOCO experiment.

## Intended use

This is a research prototype, not a medical device. It has been evaluated on 75 test images
from a single country, with no external validation and no clinician review of its
explanations. Predictions are meant to support, never replace, a trained sonographer.

## Citation

```bibtex
@misc{okwija_afrofetalnet,
  author = {Okwija, Paul},
  title  = {Afro-FetalNet: Fetal Ultrasound Plane Classification for Low-Resource African Settings},
  year   = {2026},
  url    = {https://github.com/PaulOkwija/OBS_Ultrasound_standard_plane_classification}
}
```
