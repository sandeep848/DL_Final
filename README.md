# European Image Geolocation with RegNet-Y 400MF

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Geolocation-EE4C2C.svg)](https://pytorch.org/)
[![Parameters](https://img.shields.io/badge/Parameters-4.63M-success.svg)](#model)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A complete image-geolocation pipeline that predicts latitude and longitude from street-level photographs across 12 European countries. The solution uses one RegNet-Y 400MF model trained from scratch and remains below the challenge limit of five million parameters.

## Results

| Metric | Result |
|---|---:|
| Validation median Haversine error | **150.67 km** |
| Validation mean Haversine error | **584.05 km** |
| Predictions within 200 km | **52.8%** |
| Predictions within 750 km | **71.4%** |
| Median when country is correct | **34.51 km** |
| Median when fine cell is correct | **12.10 km** |
| Trainable parameters | **4,628,010** |

The 12.10 km value is conditional on a correct fine-cell match; it is not the overall validation score.

## Model

~~~mermaid
flowchart TD
    A["Street-level image"] --> B["RegNet-Y 400MF"]
    B --> C["GeM pooling"]
    C --> D["Shared embedding"]
    D --> E["Country and region heads"]
    D --> F["Fine-cell head"]
    D --> G["Local offset head"]
    E --> H["Constrained decoder"]
    F --> H
    G --> H
    H --> I["Latitude and longitude"]
~~~

The network combines country and hierarchical cell classification with a bounded continuous offset. A spatially constrained local softmax prevents probability mass from averaging geographically distant modes.

## Challenge compliance

| Constraint | Implementation |
|---|---|
| Maximum 5M parameters | 4,628,010 parameters |
| No external training data | Random initialization with `pretrained=False` |
| One model | Single RegNet-Y 400MF pipeline |
| Offline inference | No network calls during evaluation |
| Submission format | Validated 2,400-row latitude/longitude CSV |

## Installation

~~~bash
git clone https://github.com/sandeep848/european-image-geolocation.git
cd european-image-geolocation
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~

On Windows, activate with `.venv\Scripts\activate`.

## Usage

~~~bash
python run.py --test
python run.py --train --epochs 60
python run.py --eval
python run.py --predict
python run.py --validate
~~~

Run the complete pipeline with:

~~~bash
python run.py --all
~~~

## Reproducibility

- Country-stratified folds are stored in `folds.csv`.
- Model, decoder and geographic hierarchy settings are versioned.
- The test suite checks spherical geometry, parameter counts and output validity.
- `run.py --validate` audits headers, row counts, coordinate ranges and finite values.
- Inference operates offline using repository artifacts.

## Runtime

Measured on one NVIDIA RTX 4000 Ada GPU:

| Stage | Runtime |
|---|---:|
| Full 60-epoch curriculum | Approximately 41 minutes |
| Five-fold validation, 20 epochs per fold | Approximately 75 minutes |
| Inference on 2,400 images | Approximately 18 seconds |

## Repository structure

~~~text
run.py                  # Unified command-line entry point
src/                    # Data, model, training, evaluation and prediction
experiments/            # Splits, metrics and experiment outputs
report/report.tex       # Technical report
config.json             # Model and training configuration
decoder_config.json     # Decoder configuration
folds.csv               # Reproducible validation folds
predictions.csv         # Holdout predictions
~~~

## Limitations

- The overall validation error remains substantially higher than the conditional fine-cell error.
- Country confusion dominates long-distance failures.
- The checkpoint is stored in the repository for offline challenge reproduction and may later be moved to Git LFS or a release asset.
- Results apply to the supplied challenge distribution and should not be assumed to generalize globally.

## Technical report

See [`report/report.tex`](report/report.tex) for methodology, experiments and error analysis.

## License

Distributed under the [MIT License](LICENSE).
