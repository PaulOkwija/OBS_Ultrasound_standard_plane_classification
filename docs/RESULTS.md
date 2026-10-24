# Results

Every number in this file is copied from a saved notebook output. The source notebook is
given next to each table. All notebooks were run on Kaggle (T4 GPU).

## 1. Datasets after curation

Source: notebooks 1 and 7 (Section 1).

| Split | FETAL_PLANES_DB (4 classes) | African (4 classes) |
|---|---|---|
| train | 5,121 | 240 |
| val | 2,008 | 110 |
| test | 5,271 (CSV `Train == 0`) | 100 (Malawi only) |

African dataset per country: Algeria, Egypt and Malawi have 100 images each (25 of them thorax,
mapped to `other`); Ghana and Uganda have 75 each and no thorax images. Restricted to the three
standard planes, each country has 75 images, so the LOCO pool is 4 x 75 = 300 images and the
Malawi test set is 75 images from 26 patients.

## 2. Architecture comparison on FETAL_PLANES_DB

Source: notebook 3. 4 classes, trained on Spain train, evaluated on Spain test (n=5,271).
Speed measured on a Kaggle T4, batch size 1, 224 x 224 input.

| Model | Params (M) | FPS | Latency (ms) | Test accuracy | Test macro-F1 |
|---|---|---|---|---|---|
| MobileNetV2 | 2.2 | 192.8 | 5.19 | 0.892 | 0.854 |
| MobileNetV3-Small | 1.5 | 192.0 | 5.21 | 0.917 | 0.882 |
| **EfficientNet-B0** | **4.0** | **130.3** | **7.67** | **0.957** | **0.936** |
| EfficientNet-B2 | 7.7 | 89.2 | 11.21 | 0.952 | 0.925 |
| ResNet50 | 23.5 | 153.6 | 6.51 | 0.934 | 0.905 |
| DenseNet169 | 12.5 | 47.6 | 21.00 | 0.957 | 0.937 |
| Xception | 20.8 | 184.9 | 5.41 | 0.964 | 0.945 |
| ConvNeXt-Tiny | 27.8 | 130.8 | 7.65 | 0.922 | 0.885 |

EfficientNet-B0 was selected for its accuracy at 4.0M parameters. Xception scored slightly
higher but is five times larger.

## 3. Zero-shot domain shift (Spain to Africa)

Source: notebook 4. EfficientNet-B0 trained on Spain only, evaluated without any African
training data. 4 classes. African test = Malawi, n=100.

| Evaluation set | Preprocessing | Accuracy | ROC-AUC | Macro-F1 |
|---|---|---|---|---|
| Spain test | none | 0.955 | 0.995 | n/a |
| Malawi | none | 0.620 | 0.919 | 0.568 |
| Spain test | selective CLAHE | 0.966 | 0.996 | n/a |
| Malawi | selective CLAHE | 0.700 | 0.966 | 0.700 |

Per-class F1 without CLAHE (Spain / Malawi): abdomen 0.962 / 0.370, brain 0.992 / 1.000,
femur 0.953 / 0.323, other 0.917 / 0.578. Brain transfers perfectly; abdomen and femur
collapse.

"Selective CLAHE" applies CLAHE (clip limit 2.0, 8 x 8 tiles) only to images whose grayscale
standard deviation is below 35.

## 4. LOCO fine-tuning (3 classes)

Source: notebook 5. Starting from Spain-trained 3-class checkpoints, each model is fine-tuned in
4 folds. In each fold one non-Malawi country is the validation set and the other three are
training data. Each fold's best checkpoint is then scored on Malawi (n=75) and the four scores
are averaged.

| Model | CV macro-F1 (mean +/- std) | Malawi macro-F1 (mean of 4 fold models) | Malawi accuracy |
|---|---|---|---|
| EfficientNet-B0 | 0.970 +/- 0.015 | 0.970 | 0.970 |
| ConvNeXt-Tiny | 1.000 +/- 0.000 | 0.979 | 0.980 |
| EfficientFormerV2-S0 | 0.990 +/- 0.011 | 0.973 | 0.973 |

## 5. Validation study

Source: notebook 7.

### 5.1 Corrected headline result

EfficientNet-B0, 4 LOCO folds averaged in weight space (model soup), evaluated on the
75-image, 26-patient Malawi test set with the 3-class loader:

| Metric | Value |
|---|---|
| Macro-F1 | 0.9867 |
| Accuracy | 0.9867 (74 of 75 correct) |
| Image-level bootstrap 95% CI (F1) | [0.955, 1.000] |
| Patient-clustered bootstrap 95% CI (F1) | [0.959, 1.000] |
| Multi-seed (seeds 42, 1, 7) | 0.9867 for every seed |

The identical F1 across seeds is expected rather than suspicious: per-fold training curves
differ between seeds, but with 75 test images macro-F1 can only take a few values near the top,
and a near-perfect model keeps landing on 74 of 75.

### 5.2 Symmetric five-way country hold-out

Each country held out in turn as a true test set, single pooled fine-tune on the other four:

| Held-out country | Macro-F1 | Accuracy |
|---|---|---|
| Algeria | 0.9456 | 0.9467 |
| Egypt | 0.9217 | 0.9200 |
| Ghana | 0.9330 | 0.9333 |
| Uganda | 0.9463 | 0.9467 |
| Malawi | 0.9599 | 0.9600 |

Mean of the four non-Malawi countries: 0.9367. Malawi is at the top of a tight band, not an
outlier.

### 5.3 Baselines on the Malawi test set

| Method | Macro-F1 |
|---|---|
| Pooled fine-tune on all 4 countries (no LOCO, no soup), 1 run | 1.0000 |
| LOCO + weight-space model soup | 0.9867 |
| LOCO + prediction-level ensemble of the 4 fold models | 0.9867 |

Patient-clustered paired bootstrap (2,000 resamples):

- Soup minus pooled: mean -0.0128, 95% CI [-0.041, 0.000].
- Ensemble minus soup: 0.0000, CI [0.000, 0.000]. The two produce identical predictions.

Because the pooled model made zero errors, every resample of it is also error-free, so the soup
can tie it but never beat it. The confidence interval is the informative part.

### 5.4 CLAHE

- `contrast_threshold=35` is a hand-picked default, not fit on any data.
- 36.7% of training-country images fall below the threshold, but only 1.0% of Malawi images do,
  so selective CLAHE almost never fires on the 3-class Malawi test set.
- A 4 x 4 sweep over clip limit {1.5, 2.0, 2.5, 3.0} and threshold {15, 25, 35, 45}, applied at
  inference to the soup model, gives F1 = 0.9867 in 15 of 16 cells and 1.0000 only at
  threshold 45.

CLAHE helps the zero-shot Spain model on the 4-class Malawi set (Section 3), but has almost no
effect once the model has been fine-tuned on African data.

### 5.5 Grad-CAM attention sharpness

Malawi test set, EfficientNet-B0 soup, target layer `conv_head`.
Concentration = share of CAM mass in the top 25% of pixels. Entropy = Shannon entropy in nats.

| Class | Concentration | Entropy |
|---|---|---|
| Abdomen | 0.751 +/- 0.077 | 3.080 +/- 0.218 |
| Brain | 0.663 +/- 0.076 | 3.207 +/- 0.158 |
| Femur | 0.780 +/- 0.089 | 2.974 +/- 0.239 |
| Overall | 0.731 +/- 0.094 | 3.087 +/- 0.227 |

These measure how focused the attention is, not whether it is on the correct anatomy. That
would need clinician-annotated landmarks.

### 5.6 Compute cost (EfficientNet-B0, batch 1)

| Device | FPS | Latency (ms) | Peak memory (MB) |
|---|---|---|---|
| Kaggle T4 GPU | 125.2 | 7.98 | 203.5 |
| Kaggle VM CPU (proxy only) | 30.7 | 32.60 | 3,059.1 |

The CPU row is a cloud VM, not a phone or edge device. Energy use was not measured.

## Limitations

- The Malawi test set has 75 images from 26 patients in one country. Confidence intervals reach
  1.0, and a one-image difference between methods is within noise.
- The pooled fine-tuning baseline is a single run.
- There is no external validation set and no clinician review of Grad-CAM maps.
- CPU latency was measured on a cloud VM, not on mobile or edge hardware. Energy use was not measured.
