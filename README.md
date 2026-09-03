# Image Geolocation Challenge: Ultra-Fine European Photo Localization

This repository contains our solution for the Image Geolocation Challenge. The model predicts latitude and longitude coordinates across 12 European countries using images trained strictly from scratch without external datasets or pre-trained weights.

## Constraints Satisfied
- **Parameter Limit:** The submitted model architecture uses **RegNet-Y 400MF with GeM (Generalized Mean) Pooling** and multi-task heads, totaling **4,500,282 parameters** (strictly under the 5,000,000 parameter constraint).
- **Data Policy:** Trained completely from scratch (random initialization) using only the provided `~11,758` training images. No pre-trained ImageNet weights or synthetic data were used.

## Key Strategy & Innovations
1. **RegNet-Y 400MF Backbone (~3.9M Params):** Standard Conv+BatchNorm+ReLU architecture with Squeeze-and-Excitation modules designed for high representation learning and stable convergence when training from scratch.
2. **GeM (Generalized Mean) Pooling:** Replaces standard average pooling to retain salient localized visual features (street signs, architectural motifs, road markings, landscape elements).
3. **512-Cell Voronoi Discretization:** Europe is discretized into $K=512$ spatial Voronoi cells via 3D K-Means clustering on the unit sphere (~25 km average cell radius, oracle median distance 25.97 km).
4. **Spatially-Constrained Local Neighborhood Softmax:** Masks out candidate cells beyond a local radius ($R_{\text{local}} = 150\text{ km}$) of the top-1 cell, eliminating cross-continent spatial probability averaging.
5. **Tanh Local Offset Refinement:** Continuous regression head predicts local coordinate offsets bounded within $\pm 2.0^\circ$ ($\sim 200\text{ km}$ local neighborhood) around the predicted cell.
6. **Smooth Log-Haversine Geodesic Loss:** Directly optimizes log-Haversine distance to align loss gradients with median evaluation metric.
7. **10-View Multi-Crop Test-Time Augmentation (TTA):** Evaluates 10 views per test image (Five-Crop spatial views + 5 horizontal flips at $384 \times 384$) during test inference.

## Requirements
```bash
pip install -r requirements.txt
```

## Directory Structure
```
Final Project/
├── geo_dataset/
│   ├── train/                # Training images (~11,758 images)
│   ├── train_labels.csv      # Training labels
│   └── holdout_public/       # Unlabelled test images (2,400 images)
├── src/
│   ├── dataset.py            # 512-cell spatial clustering dataset loader
│   ├── model.py              # RegNetYGeolocationModel + GeM Pooling (~4.50M params)
│   ├── train.py              # Multi-task training script with Log-Haversine Loss
│   ├── evaluate.py           # Evaluation script with 10-View Multi-Crop TTA
│   ├── test_geolocation.py   # Unit verification tests
│   └── predict.py            # Prediction entrypoint
├── report/
│   ├── report.tex            # 1-2 page LaTeX writeup
├── best_model.pth            # Trained checkpoint weights
├── cluster_centroids.npy     # 512 3D Voronoi centroids
├── predictions.csv           # Holdout test set predictions (2,400 rows)
├── requirements.txt
└── README.md
```

## Training the Model
To train the model from scratch for 120 epochs:
```bash
cd src
python train.py
```

## Generating Predictions
To generate `predictions.csv` for the holdout test set with 10-View Multi-Crop TTA:
```bash
cd src
python evaluate.py
```

## Running Unit Tests
To verify parameter constraints, tensor shapes, gradient flow, and spatial functions:
```bash
python src/test_geolocation.py
```
