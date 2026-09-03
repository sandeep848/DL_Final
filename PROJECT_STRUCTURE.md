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

### 3.3 512-Cell Voronoi Spatial Grid (`cluster_centroids.npy`)
- **Clustering:** K-Means clustering ($K=512$) fitted on 3D spherical Cartesian coordinates of the training set.
- **Target Cell:** Each photo is assigned to its nearest spatial cell index $c_k \in \{0, \dots, 511\}$ based on 3D Euclidean distance (chord distance).
- **Oracle Median Distance:** 25.97 km (theoretical lower bound).

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
   - **Cell Head:** `Linear(in_features -> 384) -> BatchNorm1d -> Hardswish -> Dropout(0.2) -> Linear(384 -> 512)`.
   - **Offset Head:** `Linear(in_features -> 192) -> BatchNorm1d -> Hardswish -> Dropout(0.2) -> Linear(192 -> 2) -> Tanh`. Continuous local coordinate refinement ($\pm 2.0^\circ$).
   - **Cartesian 3D Head:** `Linear(in_features -> 128) -> BatchNorm1d -> Hardswish -> Linear(128 -> 3) -> L2 Norm`. Unit 3D sphere vector prediction.
   - **Country Head:** `Linear(in_features -> 192) -> BatchNorm1d -> Hardswish -> Dropout(0.2) -> Linear(192 -> 12)`. Auxiliary country classification supervision.

- **Total Parameter Count:** **4,500,282 parameters** ($\le 5,000,000$ constraint satisfied).

---

## 5. 🎯 Loss Formulation & Spatially-Constrained Decoding (`src/train.py`)

### 5.1 Spatially-Constrained Local Neighborhood Softmax Decoding
To eliminate cross-continent multimodal spatial averaging failure, logits are masked beyond $R_{\text{local}} = 150\text{ km}$ of the top-1 predicted cell $c^*$:
$$P(c_k \mid x) = \frac{\exp(z_k / \tau) \cdot \mathbb{I}[d(c_k, c^*) \le R_{\text{local}}]}{\sum_{j} \exp(z_j / \tau) \cdot \mathbb{I}[d(c_j, c^*) \le R_{\text{local}}]}$$

Continuous Tanh offset is scaled by $\cos(\text{lat})$ for longitude:
$$\hat{\text{lat}} = \text{lat}_{\text{soft}} + \Delta_{\text{lat}} \cdot 2.0^\circ, \quad \hat{\text{lng}} = \text{lng}_{\text{soft}} + \Delta_{\text{lng}} \cdot \frac{2.0^\circ}{\max(\cos(\text{lat}_{\text{soft}}), 0.2)}$$

### 5.2 Multi-Task Training Loss
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cell}} + 0.3 \cdot \mathcal{L}_{\text{country}} + 1.5 \cdot \mathcal{L}_{\text{focal\_log\_hav}} + 0.5 \cdot \mathcal{L}_{\text{xyz}}$$
- $\mathcal{L}_{\text{cell}}$: Cross-Entropy Loss with label smoothing ($0.05$).
- $\mathcal{L}_{\text{country}}$: Cross-Entropy Loss with label smoothing ($0.05$).
- $\mathcal{L}_{\text{focal\_log\_hav}}$: Focal Log-Haversine distance loss in km: $\left(1 - \exp\left(-\frac{d_{\text{hav}}}{150}\right)\right) \cdot \log\left(1 + \frac{d_{\text{hav}}}{10.0}\right)$.
- $\mathcal{L}_{\text{xyz}}$: MSE Loss between predicted unit 3D vector $\hat{\mathbf{v}}$ and true vector $\mathbf{v}$.

### 5.3 Optimization & Schedule
- **Optimizer:** `AdamW(lr=1e-3, weight_decay=1e-4)`.
- **Epochs:** 120 total.
- **Scheduler:** 5-epoch linear warmup followed by Cosine Annealing decay down to $10^{-5}$.

---

## 6. 🔮 Inference & Evaluation Pipeline (`src/evaluate.py`)

- **10-View Multi-Crop Test-Time Augmentation (TTA):**
  - Resizes test image to $416 \times 416$.
  - Extracts 5 spatial crops (`FiveCrop(384)`).
  - Generates 5 horizontal flips $\rightarrow$ total 10 views per image.
  - Averages cell logits and continuous offsets across all 10 views before applying Spatially-Constrained Local Neighborhood Softmax.
- **Output Submission File (`predictions.csv`):**
  - Header: `filename,pred_lat,pred_lng`
  - Exactly 2,400 rows corresponding to `geo_dataset/holdout_public/`.

---

## 7. 🧪 Unit Tests (`src/test_geolocation.py`)

Run unit tests via:
```bash
/home/utn/uzis83et/anaconda3/bin/python src/test_geolocation.py
```
- `test_coordinate_conversion_roundtrip()`: Validates $(\text{lat}, \text{lng}) \leftrightarrow (x, y, z)$ numerical roundtrip precision.
- `test_haversine_nan_safety()`: Ensures gradient computation near zero distance does not yield `NaN`.
- `test_model_parameter_constraint_and_shapes()`: Verifies output tensor shapes and checks parameter limit $\le 5,000,000$.
- `test_spatially_constrained_soft_expectation()`: Verifies stability of Spatially-Constrained Local Neighborhood Softmax.
