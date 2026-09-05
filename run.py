#!/usr/bin/env python3
"""
Single Master Entrypoint for the Image Geolocation Challenge Pipeline.

Usage:
  # 1. Run entire end-to-end pipeline (Tests -> Splits -> Train A/B/C -> Eval -> Predict -> Audit):
  python run.py --all

  # 2. Or run individual stages:
  python run.py --test        # Run 22 unit tests and parameter limit verification
  python run.py --split       # Generate persistent spatial/random split manifests
  python run.py --oracle      # Compute oracle bounds and feasibility diagnostics
  python run.py --train       # Execute 3-phase training curriculum (Phases A, B, C)
  python run.py --eval        # Evaluate best checkpoint on validation split
  python run.py --predict     # Generate holdout predictions (2,400 rows)
  python run.py --validate    # Run strict 13-point read-only submission audit
"""

import os
import sys
import argparse
import subprocess
from typing import List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, "src")
PYTHON_EXE = sys.executable

def print_banner(stage_name: str, step_num: int, total_steps: int):
    print("\n" + "=" * 80)
    print(f"[{step_num}/{total_steps}] STAGE: {stage_name.upper()}")
    print("=" * 80, flush=True)

def run_cmd(cmd: List[str], desc: str) -> None:
    print(f"\n>> Running: {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, cwd=SCRIPT_DIR)
    if res.returncode != 0:
        print(f"\n[ERROR] Stage failed during: {desc} (Exit code: {res.returncode})", file=sys.stderr)
        sys.exit(res.returncode)

def stage_test():
    print_banner("Unit Tests & Parameter Verification", 1, 1)
    run_cmd([PYTHON_EXE, os.path.join(SRC_DIR, "test_geolocation.py")], "Unit Tests")

def stage_split(split_type: str = "spatial"):
    print_banner(f"Generate Split Manifests ({split_type})", 1, 1)
    run_cmd([
        PYTHON_EXE, os.path.join(SRC_DIR, "dataset.py"),
        "--generate-splits", "--split", split_type
    ], "Split Generation")

def stage_oracle(split_type: str = "spatial"):
    print_banner(f"Oracle & Feasibility Diagnostics ({split_type})", 1, 1)
    run_cmd([
        PYTHON_EXE, os.path.join(SRC_DIR, "dataset.py"),
        "--oracle", "--split", split_type
    ], "Oracle Analysis")

def stage_train(phase: str = "C", split_type: str = "spatial", resume: str = None, epochs: int = 20, fold: int = None):
    banner_name = f"Model Training (Phase: {phase}, Epochs: {epochs}{', Fold: ' + str(fold) if fold is not None else ''})"
    print_banner(banner_name, 1, 1)
    cmd = [
        PYTHON_EXE, os.path.join(SRC_DIR, "train.py"),
        "--phase", phase,
        "--split", "cv" if fold is not None else split_type,
        "--epochs", str(epochs)
    ]
    if fold is not None:
        cmd.extend(["--fold", str(fold)])
    if resume:
        cmd.extend(["--resume", resume])
    run_cmd(cmd, f"Training Phase {phase}")

def stage_eval(split_type: str = "spatial", mode: str = "blended", tune: bool = False, fold: int = None):
    banner_name = f"Validation Evaluation ({'Fold ' + str(fold) if fold is not None else split_type}, {mode})"
    print_banner(banner_name, 1, 1)
    cmd = [
        PYTHON_EXE, os.path.join(SRC_DIR, "evaluate.py"),
        "--split", "cv" if fold is not None else split_type,
        "--mode", mode
    ]
    if fold is not None:
        cmd.extend(["--fold", str(fold)])
    if tune:
        cmd.append("--tune-decoder")
    run_cmd(cmd, "Evaluation")

def stage_oof():
    print_banner("Pooled Out-of-Fold Leaderboard Benchmark Evaluation", 1, 1)
    run_cmd([PYTHON_EXE, os.path.join(SRC_DIR, "evaluate.py"), "--oof"], "OOF Evaluation")

def stage_predict(mode: str = "blended", tta: str = "direct", ensemble: bool = False):
    desc = "5-Fold Ensemble Prediction" if ensemble else f"Holdout Prediction (TTA: {tta})"
    print_banner(desc, 1, 1)
    cmd = [
        PYTHON_EXE, os.path.join(SRC_DIR, "predict.py"),
        "--mode", mode,
        "--tta", tta
    ]
    if ensemble:
        cmd.append("--ensemble")
    run_cmd(cmd, "Holdout Predictions")

def stage_validate(exp_dir: str = "experiments/exp_regnety_baseline"):
    print_banner("Strict 13-Point Submission Audit", 1, 1)
    run_cmd([
        PYTHON_EXE, os.path.join(SRC_DIR, "predict.py"),
        "--validate", "--exp-dir", exp_dir
    ], "Submission Audit")

def stage_cv(phase: str = "C", epochs: int = 20, mode: str = "blended", exp_dir: str = "experiments/exp_regnety_baseline"):
    TOTAL_STAGES = 6
    print("\n" + "#" * 80)
    print("STARTING 5-FOLD STRATIFIED CROSS-VALIDATION PIPELINE")
    print("Score with 5-fold cross-validation on train_labels.csv, stratified by country.")
    print("Report the metrics pooled over all out-of-fold predictions.")
    print("#" * 80)

    # 1. Unit Tests
    print_banner("Unit Tests & Parameter Budget Audit", 1, TOTAL_STAGES)
    run_cmd([PYTHON_EXE, os.path.join(SRC_DIR, "test_geolocation.py")], "Unit Tests")

    # 2. Split Generation
    print_banner("Generate 5-Fold Stratified Split Manifests", 2, TOTAL_STAGES)
    run_cmd([
        PYTHON_EXE, os.path.join(SRC_DIR, "dataset.py"),
        "--generate-splits", "--split", "cv"
    ], "5-Fold Split Generation")

    # 3. Train all 5 folds
    for f in range(5):
        print_banner(f"Train Fold {f}/5 ({epochs} Epochs, Phase {phase})", 3, TOTAL_STAGES)
        run_cmd([
            PYTHON_EXE, os.path.join(SRC_DIR, "train.py"),
            "--phase", phase,
            "--split", "cv",
            "--fold", str(f),
            "--epochs", str(epochs)
        ], f"Training Fold {f}")

    # 4. Out-of-Fold Evaluation & Official Leaderboard Score
    print_banner("Evaluate Pooled Out-of-Fold Predictions & Official Score", 4, TOTAL_STAGES)
    run_cmd([
        PYTHON_EXE, os.path.join(SRC_DIR, "evaluate.py"),
        "--oof"
    ], "Pooled OOF Evaluation")

    # 5. Holdout 5-Fold Ensemble Predictions
    print_banner("Generate 5-Fold Ensemble Holdout Predictions (2,400 Rows)", 5, TOTAL_STAGES)
    run_cmd([
        PYTHON_EXE, os.path.join(SRC_DIR, "predict.py"),
        "--ensemble",
        "--mode", mode
    ], "Ensemble Predictions")

    # 6. Final Submission Audit
    print_banner("Final Read-Only Submission Audit", 6, TOTAL_STAGES)
    run_cmd([
        PYTHON_EXE, os.path.join(SRC_DIR, "predict.py"),
        "--validate", "--exp-dir", exp_dir
    ], "Submission Audit")

    print("\n" + "#" * 80)
    print("🎉 5-FOLD STRATIFIED CROSS-VALIDATION PIPELINE COMPLETED!")
    print(f"Pooled OOF Metrics: {os.path.join(exp_dir, 'oof_metrics.json')}")
    print(f"Final Ensemble Predictions: {os.path.join(exp_dir, 'predictions.csv')}")
    print("#" * 80 + "\n")

def run_all_pipeline(split_type: str = "random", exp_dir: str = "experiments/exp_regnety_baseline", phase: str = "BC", epochs: int = 65):
    TOTAL_STAGES = 7
    print("\n" + "#" * 80)
    print("STARTING COMPLETE END-TO-END REGNET-Y GEOLOCATION PIPELINE")
    print("#" * 80)

    # 1. Tests & Parameter limit
    print_banner("Unit Tests & Parameter Budget Audit", 1, TOTAL_STAGES)
    run_cmd([PYTHON_EXE, os.path.join(SRC_DIR, "test_geolocation.py")], "Unit Tests")

    # 2. Manifests & Splits
    print_banner("Generate Persistent Split Manifests", 2, TOTAL_STAGES)
    run_cmd([
        PYTHON_EXE, os.path.join(SRC_DIR, "dataset.py"),
        "--generate-splits", "--split", split_type
    ], "Split Generation")

    # 3. Oracle Analysis
    print_banner("Dataset Density & Oracle Analysis", 3, TOTAL_STAGES)
    run_cmd([
        PYTHON_EXE, os.path.join(SRC_DIR, "dataset.py"),
        "--oracle", "--split", split_type
    ], "Oracle Analysis")

    # 4. Training (65-Epoch Curriculum: 15 Ep Country + 50 Ep Joint Localization)
    banner_desc = f"2-Phase Curriculum Training (15 Ep Country Warm-up + {max(1, epochs-15)} Ep Joint Localization)" if phase in ("BC", "all") else f"Model Training ({phase})"
    print_banner(banner_desc, 4, TOTAL_STAGES)
    run_cmd([
        PYTHON_EXE, os.path.join(SRC_DIR, "train.py"),
        "--phase", phase,
        "--epochs", str(epochs),
        "--split", split_type
    ], "Model Training")

    # 5. Decoder Tuning & Evaluation
    print_banner("Decoder Tuning & Validation Evaluation", 5, TOTAL_STAGES)
    run_cmd([
        PYTHON_EXE, os.path.join(SRC_DIR, "evaluate.py"),
        "--split", split_type,
        "--mode", "blended",
        "--tune-decoder"
    ], "Evaluation & Decoder Tuning")

    # 6. Holdout Predictions with Center-Crop TTA
    print_banner("Generate Holdout Test Predictions (2,400 Rows)", 6, TOTAL_STAGES)
    run_cmd([
        PYTHON_EXE, os.path.join(SRC_DIR, "predict.py"),
        "--mode", "blended",
        "--tta", "center"
    ], "Holdout Predictions")

    # Synchronize submission artifacts to project root
    import shutil
    print("\n" + "=" * 80)
    print("SYNCHRONIZING SUBMISSION ARTIFACTS TO PROJECT ROOT")
    print("=" * 80)
    for fname in ["predictions.csv", "best_model.pth", "config.json", "hierarchy_metadata.json", "fine_centroids.npy"]:
        src_f = os.path.join(exp_dir, fname)
        dst_f = os.path.join(SCRIPT_DIR, fname)
        if os.path.exists(src_f):
            shutil.copy2(src_f, dst_f)
            print(f"✓ Synchronized {fname} to project root: {dst_f}")
    print("=" * 80)

    # 7. Final Submission Validation (Dual Audit: Experiments Dir & Project Root)
    print_banner("Final Read-Only Submission Audit (Experiments & Root)", 7, TOTAL_STAGES)
    run_cmd([
        PYTHON_EXE, os.path.join(SRC_DIR, "predict.py"),
        "--validate", "--exp-dir", exp_dir
    ], "Experiment Submission Audit")
    run_cmd([
        PYTHON_EXE, os.path.join(SRC_DIR, "predict.py"),
        "--validate", "--exp-dir", SCRIPT_DIR
    ], "Root Submission Audit")

    print("\n" + "#" * 80)
    print("🎉 FULL PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Predictions saved to: {os.path.join(exp_dir, 'predictions.csv')}")
    print(f"Root submission file: {os.path.join(SCRIPT_DIR, 'predictions.csv')}")
    print("Ready for packaging and submission.")
    print("#" * 80 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Single Master Entrypoint for the Image Geolocation Challenge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Execute complete end-to-end pipeline in sequence")
    group.add_argument("--cv", action="store_true", help="Execute complete 5-fold cross-validation & pooled leaderboard evaluation")
    group.add_argument("--oof", action="store_true", help="Evaluate pooled out-of-fold predictions across existing CV folds")
    group.add_argument("--test", action="store_true", help="Run 22 unit tests & parameter checks")
    group.add_argument("--split", action="store_true", help="Generate persistent train/val split manifests")
    group.add_argument("--oracle", action="store_true", help="Run dataset density & oracle analysis")
    group.add_argument("--train", action="store_true", help="Train the model (Phases A, B, C)")
    group.add_argument("--eval", action="store_true", help="Evaluate checkpoint on validation split")
    group.add_argument("--predict", action="store_true", help="Generate predictions for 2,400 holdout images")
    group.add_argument("--validate", action="store_true", help="Run strict 13-point submission audit")

    # Modifiers
    parser.add_argument("--split-type", type=str, default="random", choices=["spatial", "random", "cv"], help="Validation split type (default: random for country-stratified benchmark)")
    parser.add_argument("--fold", type=int, default=None, help="Specific CV fold index (0..4) for training or evaluation")
    parser.add_argument("--ensemble", action="store_true", help="Ensemble predictions across all CV fold models")
    parser.add_argument("--phase", type=str, default="BC", choices=["A", "B", "C", "BC", "all"], help="Training phase (default: BC for 15 ep Country warm-up + 50 ep Fine localization)")
    parser.add_argument("--epochs", type=int, default=65, help="Number of training epochs (default: 65)")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume training from")
    parser.add_argument("--mode", type=str, default="blended", choices=["cell_only", "retrieval_only", "blended"], help="Decoding strategy (default: blended)")
    parser.add_argument("--tta", type=str, default="direct", choices=["direct", "center", "5crop", "6view", "multiscale"], help="TTA mode (default: direct)")
    parser.add_argument("--tune", action="store_true", help="Tune decoder during evaluation")
    parser.add_argument("--exp-dir", type=str, default="experiments/exp_regnety_baseline", help="Experiment directory")

    args = parser.parse_args()

    if args.all:
        run_all_pipeline(split_type=args.split_type, exp_dir=args.exp_dir, phase=args.phase, epochs=args.epochs)
    elif args.cv:
        stage_cv(phase=args.phase, epochs=args.epochs, mode=args.mode, exp_dir=args.exp_dir)
    elif args.oof:
        stage_oof()
    elif args.test:
        stage_test()
    elif args.split:
        stage_split(split_type=args.split_type)
    elif args.oracle:
        stage_oracle(split_type=args.split_type)
    elif args.train:
        stage_train(phase=args.phase, split_type=args.split_type, resume=args.resume, epochs=args.epochs, fold=args.fold)
    elif args.eval:
        stage_eval(split_type=args.split_type, mode=args.mode, tune=args.tune, fold=args.fold)
    elif args.predict:
        stage_predict(mode=args.mode, tta=args.tta, ensemble=args.ensemble)
    elif args.validate:
        stage_validate(exp_dir=args.exp_dir)

if __name__ == "__main__":
    main()
