# 🗺️ Image Geolocation Challenge: Complete Project Documentation & Structure Reference

This document provides a comprehensive, single-source-of-truth reference for the entire codebase, directory structure, data formats, model architectures, loss formulations, training scripts, evaluation pipelines, and optimization plans. 

---

## 1. 📌 Challenge Rules & Constraints

- **Task:** Predict decimal GPS coordinates `(pred_lat, pred_lng)` from street-level photos taken across **12 European countries**.
- **12 Countries:** Belarus, Finland, France, Germany, Iceland, Italy, Norway, Poland, Spain, Sweden, Turkey, United Kingdom.
- **Evaluation Metric:** **Median Haversine Distance (km)** across the 2,400 hidden test images. Lower is better.
  - *Tie-breaker:* Mean Haversine distance (km).
  - *Secondary Metrics:* $< 200\text{ km}$ rate (city tier accuracy) and $< 750\text{ km}$ rate (country tier accuracy).
- **Hard Rule 1 (Parameter Budget):** Submitted model must contain **$\le 5,000,000$ trainable parameters**.
- **Hard Rule 2 (Data Policy):** Models must be trained **strictly from scratch** (random initialization) using **only** the provided training dataset (~11,758 images). No external datasets, pre-trained ImageNet weights, or synthetic/generated images are permitted.
- **Python Environment:** Anaconda Python `/home/utn/uzis83et/anaconda3/bin/python` (`torch 2.13.0+cu130`, `timm 1.0.28`, `torchvision`, `scikit-learn`, `pandas`, `numpy`, `tqdm`).

---

## 2. 📂 Project Directory Structure

```
/home/utn/uzis83et/DL project/Final Project/
├── geo_dataset/
│   ├── train/                  # ~11,758 labeled training images (.jpg)
│   ├── train_labels.csv        # Supervision labels: filename, country, iso, lat, lng
│   └── holdout_public/         # 2,400 unlabelled test images for prediction submission
├── src/
│   ├── dataset.py              # Fast 512-cell spatial clustering dataset loader & precomputed indexing
│   ├── model.py                # RegNetYGeolocationModel architecture (RegNet-Y 400MF + GeM Pooling, ~4.50M params)
│   ├── train.py                # Multi-task training loop, Spatially-Constrained Local Neighborhood Softmax, AdamW, Cosine LR
│   ├── evaluate.py             # Inference pipeline with 10-View Multi-Crop TTA
│   ├── test_geolocation.py     # Unit test suite verifying shapes, gradients & param budget
│   └── predict.py              # Execution entrypoint wrapping evaluate.py
├── report/
│   └── report.tex              # 1-2 page LaTeX technical writeup
├── best_model.pth              # Saved model weights checkpoint (~18MB)
├── cluster_centroids.npy       # Saved 512 3D spherical Voronoi cell centroids (lat, lng)
├── predictions.csv             # Submitted predictions (filename, pred_lat, pred_lng)
├── requirements.txt            # Dependency specification
├── CHALLENGE.md                # Official challenge guidelines & evaluation specifications
├── README.md                   # Environment setup and run instructions
└── PROJECT_STRUCTURE.md        # [THIS FILE] Complete detailed structural reference
```

---

## 3. 📊 Dataset & Spatial Cell Discretization Pipeline

### 3.1 `train_labels.csv` Format
- `filename`: Image basename (e.g. `0006762b474e4b4d9be9a61ed0247fd9.jpg`).
- `country`: Country name string matching one of the 12 countries.
- `iso`: 2-letter country code.
- `lat`: Target latitude in decimal degrees.
- `lng`: Target longitude in decimal degrees.

### 3.2 3D Spherical Coordinate Representation
To eliminate latitudinal metric distortion across Europe, coordinates $(\text{lat}, \text{lng})$ in degrees are mapped to 3D unit sphere Cartesian vectors $(x, y, z)$:
$$x = \cos(\text{lat}_{\text{rad}}) \cdot \cos(\text{lng}_{\text{rad}})$$
$$y = \cos(\text{lat}_{\text{rad}}) \cdot \sin(\text{lng}_{\text{rad}})$$
$$z = \sin(\text{lat}_{\text{rad}})$$

### 3.3 576-Cell Voronoi Spatial Grid (`fine_centroids.npy`)
- **Clustering:** Balanced country-stratified K-Means clustering (48 fine cells per country, 576 fine cells total) fitted on 3D spherical Cartesian coordinates.
- **Target Cell:** Each photo is assigned to its nearest spatial cell index $c_k \in \{0, \dots, 575\}$.
- **Oracle Median Distance:** 23.95 km (theoretical lower bound).

---

## 4. 🏗️ Model Architecture (`src/model.py`)

### 4.1 Key Architecture Components

1. **RegNet-Y 400MF Backbone (`regnety_004`):**
   - RegNet-Y architecture with Squeeze-and-Excitation blocks and BatchNorm layers designed for fast, stable training from scratch.
   - Backbone parameter count: **3,903,144 parameters**.

2. **Generalized Mean (GeM) Pooling (`GeMPooling`):**
   $$f_{\text{GeM}} = \left( \frac{1}{H \times W} \sum_{i=1}^H \sum_{j=1}^W x_{i,j}^p \right)^{\frac{1}{p}}$$
   - Learnable exponent parameter $p$ (initialized at $3.0$). Emphasizes salient localized visual features (architectural details, road markings, landscape motifs) over uniform global averaging.

3. **Multi-Task Heads:**
   - **Shared Projection:** `Linear(440 -> 384) -> BatchNorm1d -> Hardswish -> Dropout(0.2)`.
   - **Country Head:** `Linear(384 -> 12)` (12 countries).
   - **Coarse Head:** `Linear(384 -> 48)` (4 coarse regions per country).
   - **Fine-Cell Head:** `Linear(384 -> 576)` (48 fine Voronoi cells per country).
   - **Metric Retrieval Head:** `Linear(384 -> 128)` with L2 normalization.
   - **Offset Head:** `Linear(384 -> 64) -> Hardswish -> Linear(64 -> 2) -> Tanh` ($\pm 50\text{ km}$ local displacement).
   - **Cartesian 3D Head:** `Linear(384 -> 64) -> Hardswish -> Linear(64 -> 3) -> L2 Norm`.

- **Total Parameter Count:** **4,417,002 parameters** ($\le 5,000,000$ constraint satisfied, 582,998 budget remaining).

---

## 5. 🎯 Loss Formulation & Training Curriculum (`src/train.py`)

### 5.1 Training Budget & Schedule (65 Epochs Total)
- **Full Curriculum (Phases B + C):** 65 epochs combined:
  - Phase B (15 epochs): Country and coarse-region hierarchical supervision with boosted focal cross-entropy.
  - Phase C (50 epochs): Joint multi-task localization with cosine LR decay, cross-border soft neighborhood decoding, and EMA weight averaging.
- **Optimizer:** `AdamW(backbone_lr=6e-4, head_lr=1.2e-3, weight_decay=1e-4)`.
- **Scheduler:** Cosine annealing with min LR ratio of 0.05.

---

## 6. 🔮 Inference & Evaluation Pipeline (`src/evaluate.py`, `src/predict.py`)

- **Spherical Decoding Strategies:**
  - `cell_only`: Soft spherical expectation over top-$k$ Voronoi cell centroids + continuous offset, allowing top-2 predicted countries within local spatial radius.
  - `retrieval_only`: k-NN metric retrieval from training embeddings database across top-2 predicted countries.
  - `blended` (Default): Cosine-weighted spherical interpolation between cell prediction and retrieval neighborhood.
- **Output Submission File (`predictions.csv`):**
  - Header: `filename,pred_lat,pred_lng`
  - Exactly 2,400 rows corresponding to `geo_dataset/holdout_public/`.
  - Automatic 13-point submission audit verifying strict compliance on both `experiments/` and project root.

---

## 7. 🧪 Unit Tests (`src/test_geolocation.py`)

Run unit tests via:
```bash
python run.py --test
# or
python src/test_geolocation.py
```
- Complete 23-test verification suite covering coordinate math, numerical stability, parameter budget, data transforms, and model heads.
