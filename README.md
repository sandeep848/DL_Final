# 🌍 Image Geolocation Challenge: RegNet-Y European Photo Localization

This repository contains the single-model neural network pipeline for the Image Geolocation Challenge. The model predicts decimal latitude and longitude coordinates (`pred_lat, pred_lng`) from street-level photos across **12 European countries**.

The entire pipeline strictly satisfies the challenge rules:
1. **Parameter Budget:** Submitted model contains **$\le 5,000,000$ parameters** (~4,196,778 total parameters, verified dynamically).
2. **Data Policy:** The model is trained strictly from scratch with `pretrained=False` using only the provided training images and labels. No external datasets, ImageNet weights, synthetic images, or downloaded geographic data are used.
3. **Reproducibility:** Fully deterministic seeding, persistent split manifests, and SHA256-verified geographic hierarchy metadata.
4. **Preserved Artifacts:** Existing submission files (`best_model.pth`, `predictions.csv`, `report/report.tex`, `CHALLENGE.md`) remain preserved; all new checkpoints, centroids, metadata, retrieval databases, and predictions are isolated in experiment directories (e.g. `experiments/exp_regnety_baseline/`).

---

## 🏗️ Model Architecture: RegNet-Y 400MF

The solution implements **only one neural-network architecture**:
- **Backbone:** `regnety_004` (RegNet-Y 400MF) initialized strictly with `pretrained=False`.
- **Pooling:** Generalized Mean (GeM) pooling with learnable exponent $p$ (initialized at 3.0).
- **Shared Feature Projection:** Compact bottleneck projection (`Linear(440 -> 256) + BatchNorm1d + Hardswish + Dropout(0.2)`).
- **Multi-Task Heads:**
  - **12-Country Head:** `Linear(256 -> 12)`
  - **Coarse-Region Head:** `Linear(256 -> 48)` (4 coarse regions per country)
  - **Fine-Cell Head:** `Linear(256 -> 384)` (32 fine Voronoi cells per country)
  - **Metric Retrieval Head:** `Linear(256 -> 128)` with L2-normalization
  - **East/North Offset Head:** `Linear(256 -> 64) -> Hardswish -> Linear(64 -> 2) -> Tanh` ($\pm 50\text{ km}$ local displacement)
  - **3D Cartesian Head:** `Linear(256 -> 64) -> Hardswish -> Linear(64 -> 3) -> L2 Norm`

### Parameter Breakdown
```
  Backbone (RegNet-Y 400MF):    3,903,144 params
  GeM Pooling:                          1 param
  Shared Feature Projection:      113,408 params
  Country Classification Head:      3,084 params
  Coarse-Region Head:              12,336 params
  Fine-Cell Head:                  98,688 params
  Metric Retrieval Head:           32,896 params
  East/North Offset Head:          16,578 params
  3D Cartesian Head:               16,643 params
  -------------------------------------------------------------
  TOTAL PARAMETERS:             4,196,778 / 5,000,000 max (PASS)
  BUDGET REMAINING:               803,222 params
```

---

## 📁 Repository Structure

```
Final Project/
├── geo_dataset/
│   ├── train/                         # ~11,758 labelled training images (.jpg)
│   ├── train_labels.csv               # Supervision labels (filename, country, iso, lat, lng)
│   └── holdout_public/                # 2,400 unlabelled test images
├── run.py                             # [PRIMARY] Single master runner for the entire pipeline
├── src/
│   ├── config.py                      # Central configuration dataclass and SHA256 config hash
│   ├── dataset.py                     # Geodesic math, 384-cell hierarchy, spatial splits & oracle analysis
│   ├── model.py                       # Single RegNet-Y 400MF architecture + parameter enforcement (4.19M)
│   ├── train.py                       # Fast 20-30 epoch training curriculum, SupCon loss, multi-task loop, EMA
│   ├── evaluate.py                    # Multi-strategy spherical decoding, decoder calibration & detailed metrics
│   ├── predict.py                     # Prediction entrypoint (2,400 rows) & 13-point submission audit
│   └── test_geolocation.py            # Complete 23-test verification suite
├── experiments/
│   ├── splits/                        # Persistent train/val CSV manifests & diagnostics
│   └── exp_regnety_baseline/          # Experiment outputs (checkpoints, centroids, predictions)
├── best_model.pth                     # Preserved legacy submission model
├── predictions.csv                    # Preserved legacy submission predictions
├── requirements.txt                   # Pinned dependency specification
├── CHALLENGE.md                       # Official challenge specification
└── README.md                          # [THIS FILE] Project documentation
```

---

## ⚡ Quick Start: Master Commands

### 1. Official 5-Fold Stratified Cross-Validation (Course Leaderboard Scoring)
Scores with 5-fold CV on `train_labels.csv`, stratified by country. Pools metrics over all 11,758 out-of-fold predictions and generates 5-fold ensemble predictions:
```bash
python run.py --cv --epochs 20
```

### 2. Fast Single-Fold Training (~12 mins)
Train on Fold 0 to quickly verify country-stratified validation performance (~80 km median error):
```bash
python run.py --train --fold 0 --epochs 20    # Train Fold 0 (~12 mins)
python run.py --eval --fold 0                 # Evaluate Fold 0 validation score
python run.py --oof                          # Print pooled leaderboard scorecard across completed folds
```

### 3. Direct End-to-End Single-Model Pipeline (~15 mins)
Run the complete pipeline (Tests $\to$ Splits $\to$ Oracle $\to$ Direct 20-Epoch Training $\to$ Decoder Tuning $\to$ Predictions $\to$ Submission Audit):
```bash
python run.py --all
```

Or execute any single stage modularly:
```bash
python run.py --test        # 1. Run 23 unit tests & parameter checks
python run.py --split       # 2. Generate persistent split manifests
python run.py --oracle      # 3. Compute oracle bounds & feasibility diagnostics
python run.py --train       # 4. Train model (Direct 20-Epoch Localization, ~15 mins)
python run.py --eval        # 5. Evaluate best model on validation split & tune decoder
python run.py --predict     # 6. Generate holdout predictions (2,400 rows)
python run.py --validate    # 7. Run 13-point read-only submission audit
```

---

## ⚙️ Environment Setup

Install the pinned dependencies:
```bash
pip install -r requirements.txt
```

Verify the environment imports:
```bash
python -c "import torch, timm, torchvision, sklearn, scipy, PIL; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
```

---

## 🧪 Verification & Unit Tests

Run the complete 23-test unit test suite verifying coordinate conversions, offset round-trips, wraparounds, parameter limits, and gradient flows:
```bash
python src/test_geolocation.py
```

Verify that the model strictly satisfies the 5,000,000 parameter budget:
```bash
python -c "from model import get_model; get_model()"
```

---

## 📊 Step-by-Step Workflow

### 1. Split Generation
Generate persistent, reproducible validation manifests. Spatial validation (primary test of generalization) clusters nearby coordinates to eliminate geographic group leakage:
```bash
# Generate spatially grouped validation split (group radius 35 km)
python src/dataset.py --generate-splits --split spatial

# Alternatively, generate country-stratified random split
python src/dataset.py --generate-splits --split random
```

### 2. Oracle & Feasibility Analysis
Analyze dataset density, nearest-training distances, cell populations, and assess whether the under-40 km stretch target is supported by the data:
```bash
python src/oracle_analysis.py --split spatial
```

### 3. Training Curriculum (3 Phases)

#### Phase A: Representation Learning (50–60 Epochs)
Trains the RegNet-Y backbone and 128-d metric embedding head from scratch using two augmented views per image and distance-aware Supervised Contrastive Loss ($\text{weight} = \exp(-\text{dist} / 50)$):
```bash
python src/train.py --phase A --epochs 60
```

#### Phase B: Country and Coarse-Region Heads (20–25 Epochs)
Freezes the backbone and trains the 12-country and 48-coarse-region classification heads:
```bash
python src/train.py --phase B --epochs 25
```

#### Phase C: Joint Localization (30–40 Epochs)
Jointly optimizes all heads with soft-start temperature annealing and gradual introduction of the differentiable Haversine loss:
```bash
python src/train.py --phase C --epochs 40
```

*To execute all three phases sequentially:*
```bash
python src/train.py --phase all
```

*To resume training from the last checkpoint:*
```bash
python src/train.py --phase C --resume experiments/exp_regnety_baseline/last_model.pth
```

### 4. Build Training Retrieval Database
Extracts 128-d L2-normalized embeddings for training images only to enable test-time retrieval:
```bash
python -c "
from config import get_default_config
from dataset import load_geographic_hierarchy
from model import load_saved_model
from train import build_retrieval_database
import pandas as pd, torch

cfg = get_default_config()
train_df = pd.read_csv(cfg.get_path(cfg.spatial_train_manifest))
f_c, c_c, f2c, f2r, _ = load_geographic_hierarchy(cfg.exp_dir)
model, _ = load_saved_model(cfg.get_exp_path(cfg.checkpoint_best_name), device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
build_retrieval_database(model, train_df, cfg.get_path(cfg.train_img_dir), f_c, f2c, f2r, cfg.get_exp_path(cfg.retrieval_db_name), device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
"
```

### 5. Decoder Tuning
Calibrate the retrieval neighbor count ($k$), country filtering, and unit-sphere blending weight ($\alpha$) on the validation split:
```bash
python src/evaluate.py --split spatial --tune-decoder
```

### 6. Validation Evaluation
Evaluate performance across random or spatial splits, comparing cell-only, retrieval-only, and blended decoding:
```bash
# Evaluate blended decoding on spatial validation split
python src/evaluate.py --split spatial --mode blended

# Evaluate cell-only decoding on random validation split
python src/evaluate.py --split random --mode cell_only
```

### 7. Holdout Prediction Generation
Generate the final prediction file (`experiments/exp_regnety_baseline/predictions.csv`) for the 2,400 test images without overwriting root `predictions.csv`:
```bash
python src/predict.py --mode blended --tta direct
```

### 8. Strict Submission Audit
Run the read-only validator to verify model offline construction, parameter counts ($\le 5\text{M}$), exact headers, 2,400 rows, coordinate bounds, and holdout file integrity:
```bash
python src/submission_validator.py --exp-dir experiments/exp_regnety_baseline
```

---

## 📌 Known Limitations & Practical Considerations

1. **Strictly Training from Scratch:** Without ImageNet pretraining, early visual representations require sufficient Phase A contrastive epochs to converge.
2. **Geographic Clustering Bounds:** Discretizing Europe into 384 country-pure cells yields a theoretical cell oracle floor of ~18–25 km. Point-level local tangent plane offsets ($\pm 50\text{ km}$) and retrieval refinement are required to push errors below this floor.
3. **Asymmetric Cues:** Horizontal image flipping is disabled by default because road signage, traffic driving sides (e.g. UK vs Continental Europe), and architectural motifs lose geographic fidelity when flipped.
