"""
Consolidated into src/predict.py.
Run: python src/predict.py --validate
"""
import sys
import argparse
from predict import audit_submission

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strict Read-Only Submission Validator (Forwarder to predict.py)")
    parser.add_argument("--exp-dir", type=str, default="experiments/exp_regnety_baseline", help="Experiment directory")
    parser.add_argument("--prediction-file", type=str, default=None, help="Path to predictions CSV")
    parser.add_argument("--holdout-dir", type=str, default=None, help="Path to holdout_public/")
    args = parser.parse_args()

    success = audit_submission(
        exp_dir=args.exp_dir,
        prediction_file=args.prediction_file,
        holdout_dir=args.holdout_dir
    )
    sys.exit(0 if success else 1)
