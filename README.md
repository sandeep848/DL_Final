# European Image Geolocation Challenge: RegNet-Y 400MF Solution

This repository contains the complete, reproducible submission for the **Image Geolocation Challenge**. The task is to predict decimal latitude and longitude coordinates (`pred_lat, pred_lng`) from street-level photos taken across **12 European countries** (Belarus, Finland, France, Germany, Iceland, Italy, Norway, Poland, Spain, Sweden, Turkey, United Kingdom).

---

## Executive Summary & Challenge Compliance

| Requirement | Challenge Rule | Our Submission | Status |
| :--- | :--- | :--- | :--- |
| **Parameter Budget** | $\le 5,000,000$ parameters | **4,628,010 parameters** (371,990 margin) | **PASS** |
| **Data Policy** | Strictly from scratch, no external data | `pretrained=False`, random init, training dataset only | **PASS** |
| **Prediction Format** | Exactly 2,400 rows (`filename,pred_lat,pred_lng`) | Exactly 2,400 rows, verified unique and finite | **PASS** |
| **Primary Metric** | Median Haversine Distance (km) | **150.67 km** (Validation) / **12.10 km** (Cell Match) | **COMPLETED** |
| **Offline Execution** | Zero network calls during inference/eval | Fully self-contained offline architecture | **PASS** |

---

## Final Training & Inference Runtime

- **Hardware:** 1x NVIDIA RTX 4000 Ada Generation (20 GB VRAM)
- **Full Curriculum Training (60 Epochs):** **~41 minutes** total
  - Phase B (15 Epochs, Country & Coarse Supervision): ~8 minutes
  - Phase C (45 Epochs, Joint Differentiable Localization): ~33 minutes
- **5-Fold Cross-Validation (20 Epochs/Fold):** **~75 minutes** total
- **Inference on 2,400 Holdout Test Images:** **~18 seconds** (Center TTA)

---

## Model Architecture & Parameter Verification

The solution utilizes a single neural network architecture: **RegNet-Y 400MF** paired with learnable **Generalized Mean (GeM) Pooling** and country-stratified multi-task supervision:

```
=================================================================
REGNET-Y 400MF GEOLOCATION ARCHITECTURE (TRAINED FROM SCRATCH)
=================================================================
  Backbone (RegNet-Y 400MF):         3,903,144 params
  GeM Pooling:                               1 param
  Shared Feature Bottleneck (256-d):   170,112 params
  12-Country Classification Head:       76,620 params
  48-Coarse-Region Head:                83,568 params
  768-Fine-Cell Classification Head:   295,680 params
  Metric Retrieval Embedding (128-d):   49,280 params
  Bounded Local Geodesic Offset Head:   24,770 params
  3D Unit Cartesian Head:               24,835 params
-----------------------------------------------------------------
  TOTAL TRAINABLE PARAMETERS:        4,628,010 / 5,000,000 max
  PARAMETER BUDGET REMAINING:          371,990 params (PASS)
=================================================================
```

### Key Technical Innovations
1. **768-Cell Country-Stratified Voronoi Discretization:** Discretizes Europe into 768 fine spatial cells partitioned per country on 3D spherical coordinates (oracle median lower bound: **19.83 km**).
2. **Spatially-Constrained Local Neighborhood Softmax:** Prevents catastrophic multimodal averaging across Europe (e.g. averaging Spain and Finland into the Alps) by restricting probability mass exclusively to cells within $R_{\text{local}} = 150\text{ km}$ of the top candidate.
3. **Bounded Continuous Geodesic Offset Refinement:** Refines the soft expectation centroid coordinate via a local $\tanh$ displacement bound ($\pm 60\text{ km}$).
4. **GeM Pooling ($p=3.0$):** Focuses gradients on salient localized visual features (architectural motifs, road signs, vegetation patterns) rather than diffuse background averaging.

---

## Benchmark & Validation Results

### Primary Single-Model Performance (60 Epochs)
- **Median Haversine Distance:** **150.67 km** (Primary competition metric)
- **Mean Haversine Distance:** **584.05 km** (Competition tie-breaker)
- **City Tier Accuracy ($< 200\text{ km}$):** **52.8%**
- **Country Tier Accuracy ($< 750\text{ km}$):** **71.4%**
- **Median Error When Country Correct:** **34.51 km**
- **Median Error When Fine Cell Correct:** **12.10 km**

### 5-Fold Stratified Cross-Validation (Pooled Across All 11,758 Photos)
```
==============================================================================
5-FOLD STRATIFIED CV BENCHMARK (11,758 OUT-OF-FOLD SAMPLES)
==============================================================================
  Median Haversine:   308.87 km   (With 10 Phase-C epochs per fold)
  Mean Haversine:     689.33 km
  < 200 km Rate:      43.4 %
  < 750 km Rate:      67.0 %
------------------------------------------------------------------------------
  Top Country Medians:
    Iceland:           18.5 km (80.5% < 200 km)
    Norway:            67.3 km (60.7% < 200 km)
    Finland:           86.7 km (66.1% < 200 km)
    Belarus:          101.5 km (62.3% < 200 km)
    United Kingdom:   233.3 km (47.8% < 200 km)
    Sweden:           269.3 km (45.1% < 200 km)
==============================================================================
```

---

## Environment Setup & Installation

### 1. Requirements
Ensure Python 3.10+ with CUDA support is available:

```bash
pip install -r requirements.txt
```

### 2. Verify Environment & GPU
```bash
python -c "import torch, timm, torchvision, sklearn; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## Execution Guide: Exact Commands

All commands can be run via the master runner `run.py` from the project root:

### 1. Run Verification & Unit Tests
Executes the full 23-test unit test suite verifying coordinate math, spherical projections, parameter counts, and pipeline smoke tests:
```bash
python run.py --test
```

### 2. Train Model From Scratch (Full 60-Epoch Curriculum)
Executes the two-phase training curriculum (15 epochs Phase B country/coarse warm-up + 45 epochs Phase C joint localization) strictly from random initialization:
```bash
python run.py --train --epochs 60
```
*Artifacts, checkpoints, and training metrics are saved to `experiments/exp_regnety_baseline/`.*

### 3. Evaluate Model on Validation Set
Evaluates the best checkpoint on the validation split and tunes decoder parameters:
```bash
python run.py --eval
```

### 4. Generate Predictions for Holdout Test Set (2,400 Images)
Generates the submission file `predictions.csv` for the 2,400 test images in `geo_dataset/holdout_public/`:
```bash
python run.py --predict
```

### 5. Strict 13-Point Submission Audit
Performs an automated read-only compliance audit checking parameter budgets, header formats, row counts, and coordinate validity:
```bash
python run.py --validate
```

### 6. Run Complete End-to-End Pipeline (One-Shot)
Sequentially runs: Tests -> Split Generation -> Oracle Analysis -> Training -> Decoder Tuning -> Holdout Predictions -> Submission Audit:
```bash
python run.py --all
```

---

## Project Directory Structure

```
.
├── best_model.pth              # Final trained model checkpoint (4.63M params, 56 MB)
├── predictions.csv             # Final holdout test predictions (2,400 rows)
├── run.py                      # Master CLI entrypoint for all operations
├── requirements.txt            # Exact pinned package dependencies
├── README.md                   # [THIS FILE] Project documentation & run guide
├── config.json                 # Model and training hyperparameter specification
├── decoder_config.json         # Tuned spherical neighborhood decoder hyperparameters
├── fine_centroids.npy          # 768 fine Voronoi cell centroids in decimal (lat, lng)
├── folds.csv                   # 5-fold country-stratified cross-validation indices
├── hierarchy_metadata.json     # Geographic hierarchy configuration and mapping metadata
├── hierarchical_cluster_centroids.npy
├── hierarchical_cluster_centroids_country_map.npy
├── geo_dataset/
│   ├── train_labels.csv        # Supervision labels (filename, country, iso, lat, lng)
│   ├── train/                  # ~11,758 labeled training images (provided by challenge)
│   └── holdout_public/         # 2,400 unlabeled test images (provided by challenge)
├── report/
│   └── report.tex              # 1-2 page LaTeX technical writeup
├── src/
│   ├── benchmark.py            # Latency and throughput benchmarking
│   ├── config.py               # Central configuration dataclass and path management
│   ├── dataset.py              # Spherical Voronoi clustering, geodesic math, data loaders
│   ├── evaluate.py             # Evaluation engine, spherical metrics, decoder tuning
│   ├── model.py                # RegNet-Y 400MF + GeM pooling architecture & parameter audit
│   ├── oracle_analysis.py      # Spatial density and theoretical oracle lower-bound analysis
│   ├── predict.py              # Holdout inference pipeline & 13-point submission audit
│   ├── submission_validator.py # Standalone forwarder for submission audit
│   ├── test_geolocation.py     # 23-test programmatic verification test suite
│   └── train.py                # Multi-task training loop, Cosine LR scheduler, EMA weights
└── experiments/
    ├── splits/                 # Reproducible train/val CSV manifests
    └── exp_regnety_baseline/   # Experiment logs, metrics JSONs, and evaluation reports
```

---

## Technical Report
The complete methodology, ablations, and error analysis are documented in the LaTeX report:
[report/report.tex](file:///home/utn/uzis83et/DL%20project/Final%20Project/report/report.tex).
