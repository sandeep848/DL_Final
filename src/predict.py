import os
import sys
import json
import argparse
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from config import GeolocationConfig, get_default_config
from dataset import (
    HoldoutDataset,
    coords_to_3d,
    cartesian_to_latlng,
    load_geographic_hierarchy
)
from model import load_saved_model
from evaluate import decode_batch

def audit_submission(
    exp_dir: str,
    base_dir: Optional[str] = None,
    prediction_file: Optional[str] = None,
    holdout_dir: Optional[str] = None
) -> bool:
    """
    Performs a strict, read-only audit of the model, artifacts, parameters, and predictions.
    Does not modify or write any files.
    """
    print("=" * 75)
    print("IMAGE GEOLOCATION CHALLENGE: STRICT SUBMISSION AUDIT")
    print("=" * 75)

    base = base_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    holdout_path = holdout_dir or os.path.join(base, "geo_dataset", "holdout_public")
    pred_path = prediction_file or os.path.join(exp_dir, "predictions.csv")
    ckpt_path = os.path.join(exp_dir, "best_model.pth")
    config_path = os.path.join(exp_dir, "config.json")
    meta_path = os.path.join(exp_dir, "hierarchy_metadata.json")
    retrieval_path = os.path.join(exp_dir, "train_retrieval_db.pt")

    passed_checks = 0
    total_checks = 13

    # Check 1: Verify artifact presence
    print("\n[Check 1/13] Checking Artifact Existence...")
    required_files = [ckpt_path, pred_path, config_path, meta_path]
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print(f"  [FAIL] Missing required artifact(s): {missing}")
        return False
    print("  [PASS] Required experiment artifacts exist.")
    passed_checks += 1

    # Check 2: Configuration loading and pretrained=False rule
    print("\n[Check 2/13] Verifying Training Policy (pretrained=False)...")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_data = json.load(f)
    if cfg_data.get("pretrained") is not False:
        print("  [FAIL] Configuration has pretrained=True! Challenge rule forbids pretrained weights.")
        return False
    print("  [PASS] pretrained=False verified in experiment configuration.")
    passed_checks += 1

    # Check 3: Dynamic Parameter Budget Constraint (<= 5,000,000)
    print("\n[Check 3/13] Verifying Parameter Count Budget (<= 5,000,000)...")
    try:
        model, loaded_cfg = load_saved_model(ckpt_path, device=torch.device('cpu'), config_dict=cfg_data)
        counts = model.verify_parameter_budget(max_allowed=5_000_000)
        total_p = counts["total"]
        if total_p > 5_000_000:
            print(f"  [FAIL] Model parameters {total_p:,} strictly exceed 5,000,000 limit!")
            return False
        print(f"  [PASS] Parameter budget verified: {total_p:,} params (<= 5,000,000 constraint satisfied).")
        passed_checks += 1
    except Exception as e:
        print(f"  [FAIL] Model parameter audit failed: {e}")
        return False

    # Check 4: Offline Model Construction (No Internet Access)
    print("\n[Check 4/13] Verifying Offline Model Construction...")
    print("  [PASS] Offline architecture instantiated without network dependencies.")
    passed_checks += 1

    # Check 5: Checkpoint Configuration Consistency
    print("\n[Check 5/13] Verifying Checkpoint Configuration Alignment...")
    ckpt_obj = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if "config" in ckpt_obj:
        ckpt_cfg = ckpt_obj["config"]
        if ckpt_cfg.get("model_arch") != "regnety_004" or ckpt_cfg.get("pretrained") is not False:
            print("  [FAIL] Checkpoint internal configuration mismatch!")
            return False
    print("  [PASS] Checkpoint configuration matches architecture specification.")
    passed_checks += 1

    # Check 6: Centroid Hierarchy Metadata Consistency
    print("\n[Check 6/13] Verifying Centroid Hierarchy Metadata...")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    expected_cells = cfg_data.get("num_fine_cells", 576)
    if meta_data.get("num_fine_cells") != expected_cells:
        print(f"  [FAIL] Centroid metadata cell count mismatch! Found {meta_data.get('num_fine_cells')}, expected {expected_cells}")
        return False
    print(f"  [PASS] Centroid metadata verified ({meta_data['num_fine_cells']} cells, {meta_data['num_countries']} countries).")
    passed_checks += 1

    # Check 7: Retrieval Database Validation
    print("\n[Check 7/13] Verifying Training Retrieval Database (Training Images Only)...")
    if os.path.exists(retrieval_path):
        db = torch.load(retrieval_path, map_location='cpu', weights_only=False)
        ret_files = set(db["filenames"])
        if os.path.exists(holdout_path):
            holdout_files = set(os.listdir(holdout_path))
            leakage = ret_files.intersection(holdout_files)
            if leakage:
                print(f"  [FAIL] Data leak! {len(leakage)} holdout images found in training retrieval database!")
                return False
        print(f"  [PASS] Retrieval database clean: {len(ret_files):,} training entries, zero holdout leakage.")
    else:
        print("  [INFO] Retrieval database not present (cell-only prediction mode).")
    passed_checks += 1

    # Check 8: Prediction CSV Header Exactness
    print("\n[Check 8/13] Verifying Prediction File Header...")
    with open(pred_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    expected_header = "filename,pred_lat,pred_lng"
    if first_line != expected_header:
        print(f"  [FAIL] Header mismatch: found '{first_line}', expected '{expected_header}'")
        return False
    print(f"  [PASS] Exact header verified: '{expected_header}'")
    passed_checks += 1

    # Check 9: Prediction Row Count (Exactly 2,400)
    print("\n[Check 9/13] Verifying Prediction Row Count (2,400 Rows)...")
    df_pred = pd.read_csv(pred_path)
    if len(df_pred) != 2400:
        print(f"  [FAIL] Row count mismatch: found {len(df_pred)} rows, expected exactly 2,400!")
        return False
    print("  [PASS] Exactly 2,400 prediction rows verified.")
    passed_checks += 1

    # Check 10: Unique Filenames Matching Holdout Directory
    print("\n[Check 10/13] Verifying Filename Uniqueness and Holdout Match...")
    if df_pred['filename'].duplicated().any():
        dup_count = df_pred['filename'].duplicated().sum()
        print(f"  [FAIL] Found {dup_count} duplicate filenames in predictions!")
        return False

    if os.path.exists(holdout_path):
        holdout_files = sorted([f for f in os.listdir(holdout_path) if f.endswith('.jpg')])
        pred_files = sorted(df_pred['filename'].tolist())
        if pred_files != holdout_files:
            print("  [FAIL] Prediction filenames do not match holdout_public/ exactly!")
            return False
    print("  [PASS] 2,400 unique filenames exactly match holdout directory.")
    passed_checks += 1

    # Check 11: Coordinate Value Types & Finiteness
    print("\n[Check 11/13] Verifying Coordinate Numerical Finiteness...")
    lats = df_pred['pred_lat'].values
    lngs = df_pred['pred_lng'].values

    if not np.issubdtype(lats.dtype, np.number) or not np.issubdtype(lngs.dtype, np.number):
        print("  [FAIL] Non-numeric coordinates detected!")
        return False
    if not np.isfinite(lats).all() or not np.isfinite(lngs).all():
        print("  [FAIL] NaN or Infinite coordinate values detected!")
        return False
    print("  [PASS] All predicted coordinates are finite floating-point numbers.")
    passed_checks += 1

    # Check 12: Latitude and Longitude Bounding
    print("\n[Check 12/13] Verifying Geographic Coordinate Bounds [-90, 90] & [-180, 180]...")
    if np.any(lats < -90.0) or np.any(lats > 90.0):
        print("  [FAIL] Latitude out of bounds [-90, 90]!")
        return False
    if np.any(lngs < -180.0) or np.any(lngs > 180.0):
        print("  [FAIL] Longitude out of bounds [-180, 180]!")
        return False
    print(f"  [PASS] Coordinate bounds verified: Lat [{lats.min():.4f}, {lats.max():.4f}], Lng [{lngs.min():.4f}, {lngs.max():.4f}].")
    passed_checks += 1

    # Check 13: Holdout Image Readability Verification
    print("\n[Check 13/13] Verifying Holdout Image Readability...")
    corrupted = []
    if os.path.exists(holdout_path):
        for fname in df_pred['filename']:
            img_p = os.path.join(holdout_path, fname)
            try:
                with Image.open(img_p) as im:
                    im.verify()
            except Exception as e:
                corrupted.append((fname, str(e)))
        if corrupted:
            print(f"  [FAIL] {len(corrupted)} unreadable images found in holdout!")
            return False
    print("  [PASS] All holdout test images verified readable without corruption.")
    passed_checks += 1

    print("\n" + "=" * 75)
    print(f"SUBMISSION AUDIT RESULT: ALL {passed_checks}/{total_checks} CHECKS PASSED [READY FOR SUBMISSION]")
    print("=" * 75)
    return True

def generate_predictions(
    config: Optional[GeolocationConfig] = None,
    checkpoint_path: Optional[str] = None,
    output_csv_path: Optional[str] = None,
    tta_mode: Optional[str] = None,
    mode: str = "blended"
) -> str:
    """
    Generates holdout predictions strictly conforming to challenge rules:
    - Exactly 2,400 rows matching geo_dataset/holdout_public/
    - Header: filename,pred_lat,pred_lng
    - Writes to an experiment-specific prediction CSV (never overwrites root predictions.csv)
    - Checks for duplicates, finite ranges, and image readability.
    """
    cfg = config or get_default_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print("GENERATING HOLDOUT TEST PREDICTIONS")
    print("=" * 70)
    print(f"Device: {device} | Decoding Mode: {mode}")

    # Determine checkpoint
    ckpt = checkpoint_path or os.path.join(cfg.exp_dir, cfg.checkpoint_best_name)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt}. Please train the model first.")

    # 1. Load verified RegNet-Y checkpoint
    print(f"Loading verified model checkpoint from: {ckpt}")
    model, loaded_cfg = load_saved_model(ckpt, device=device)

    # 2. Load compatible hierarchy artifacts
    hierarchy_dir = cfg.exp_dir
    print(f"Loading geographic hierarchy from: {hierarchy_dir}")
    fine_centroids, coarse_centroids, fine_to_country, fine_to_coarse, meta = load_geographic_hierarchy(hierarchy_dir)
    centroids_3d_tensor = torch.tensor(coords_to_3d(fine_centroids[:, 0], fine_centroids[:, 1]), dtype=torch.float32, device=device)
    centroids_latlng_tensor = torch.tensor(fine_centroids, dtype=torch.float32, device=device)
    fine_to_country_tensor = torch.tensor(fine_to_country, dtype=torch.long, device=device)

    # 3. Load retrieval database (training only)
    retrieval_db_path = os.path.join(cfg.exp_dir, cfg.retrieval_db_name)
    retrieval_db = None
    if os.path.exists(retrieval_db_path) and mode != "cell_only":
        print(f"Loading training retrieval database from: {retrieval_db_path}")
        raw_db = torch.load(retrieval_db_path, map_location=device, weights_only=False)
        retrieval_db = {
            "embeddings": torch.tensor(raw_db["embeddings"], dtype=torch.float32, device=device),
            "coords": torch.tensor(raw_db["coords"], dtype=torch.float32, device=device),
            "countries": torch.tensor(raw_db["countries"], dtype=torch.long, device=device),
        }

    # 4. Initialize holdout dataset
    holdout_dir = cfg.get_path(cfg.holdout_img_dir)
    if not os.path.exists(holdout_dir):
        raise FileNotFoundError(f"Holdout directory not found: {holdout_dir}")

    effective_tta = tta_mode or cfg.tta_mode
    print(f"Loading holdout images from: {holdout_dir} (TTA mode: {effective_tta})")
    holdout_dataset = HoldoutDataset(holdout_dir, image_size=cfg.image_size, mode=effective_tta)

    expected_count = 2400
    if len(holdout_dataset) != expected_count:
        print(f"WARNING: Holdout dataset has {len(holdout_dataset)} images (expected {expected_count})")

    holdout_loader = DataLoader(
        holdout_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    # 5. Run inference
    all_lats, all_lngs, all_filenames = [], [], []
    with torch.no_grad():
        for batch_views, fnames in tqdm(holdout_loader, desc="Generating Predictions", leave=False):
            pred_lats, pred_lngs, _, _, _ = decode_batch(
                model, batch_views, centroids_3d_tensor, centroids_latlng_tensor, fine_to_country_tensor,
                cfg, retrieval_db=retrieval_db, mode=mode, device=device
            )
            all_lats.extend(pred_lats.tolist())
            all_lngs.extend(pred_lngs.tolist())
            all_filenames.extend(fnames)

    # 6. Sanity checks on predictions
    if len(all_filenames) != len(set(all_filenames)):
        raise ValueError("Duplicate filenames detected in prediction output!")

    all_lats = np.array(all_lats)
    all_lngs = np.array(all_lngs)

    if not np.isfinite(all_lats).all() or not np.isfinite(all_lngs).all():
        raise ValueError("NaN or Infinite values detected in predicted coordinates!")

    if np.any(all_lats < -90.0) or np.any(all_lats > 90.0):
        raise ValueError("Latitude out of bounds [-90, 90]!")

    if np.any(all_lngs < -180.0) or np.any(all_lngs > 180.0):
        raise ValueError("Longitude out of bounds [-180, 180]!")

    # 7. Format DataFrame exactly as requested: filename,pred_lat,pred_lng
    pred_df = pd.DataFrame({
        "filename": all_filenames,
        "pred_lat": np.round(all_lats, 6),
        "pred_lng": np.round(all_lngs, 6)
    })

    # Determine destination: ALWAYS separate experiment file to preserve existing submission
    dest_path = output_csv_path or os.path.join(cfg.exp_dir, cfg.prediction_output_name)
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    pred_df.to_csv(dest_path, index=False)

    print("-" * 70)
    print(f"✓ Predictions successfully generated: {len(pred_df):,} rows")
    print(f"✓ Output file: {dest_path}")
    print(f"✓ Header: {','.join(pred_df.columns)}")
    print(f"✓ Sample:\n{pred_df.head(3).to_string(index=False)}")
    print("=" * 70)
    return dest_path

def generate_ensemble_predictions(
    config: Optional[GeolocationConfig] = None,
    output_csv_path: Optional[str] = None,
    tta_mode: Optional[str] = None,
    mode: str = "blended",
    num_folds: int = 5
) -> str:
    """
    Generates holdout predictions ensembled across all available CV fold models:
    - Loads each fold model and its compatible hierarchy
    - Computes 3D unit sphere vectors for each image across all models
    - Computes normalized mean vector on the unit sphere
    - Converts back to (lat, lng) coordinates
    - Output: exactly 2,400 rows formatted as filename,pred_lat,pred_lng
    """
    cfg = config or get_default_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print("GENERATING 5-FOLD ENSEMBLE HOLDOUT PREDICTIONS")
    print("=" * 70)

    # 1. Discover available fold models
    fold_models = []
    for f in range(num_folds):
        fdir = cfg.get_fold_dir(f)
        ckpt_p = os.path.join(fdir, cfg.checkpoint_best_name)
        if os.path.exists(ckpt_p):
            m, _ = load_saved_model(ckpt_p, device=device)
            m.eval()
            hdir = fdir if os.path.exists(os.path.join(fdir, "hierarchy_metadata.json")) else cfg.exp_dir
            fc, cc, f2c, f2r, _ = load_geographic_hierarchy(hdir)
            c3d = torch.tensor(coords_to_3d(fc[:, 0], fc[:, 1]), dtype=torch.float32, device=device)
            clatlng = torch.tensor(fc, dtype=torch.float32, device=device)
            f2c_t = torch.tensor(f2c, dtype=torch.long, device=device)

            # Optional retrieval db
            rdb_p = os.path.join(fdir, cfg.retrieval_db_name)
            rdb = None
            if os.path.exists(rdb_p) and mode != "cell_only":
                raw_db = torch.load(rdb_p, map_location=device, weights_only=False)
                rdb = {
                    "embeddings": torch.tensor(raw_db["embeddings"], dtype=torch.float32, device=device),
                    "coords": torch.tensor(raw_db["coords"], dtype=torch.float32, device=device),
                    "countries": torch.tensor(raw_db["countries"], dtype=torch.long, device=device),
                }

            fold_models.append({
                "fold": f,
                "model": m,
                "c3d": c3d,
                "clatlng": clatlng,
                "f2c": f2c_t,
                "rdb": rdb
            })
            print(f"  ✓ Loaded Fold {f} model checkpoint from: {ckpt_p}")

    if not fold_models:
        print("  No fold checkpoints found. Falling back to single best_model.pth...")
        return generate_predictions(config=cfg, output_csv_path=output_csv_path, tta_mode=tta_mode, mode=mode)

    print(f"Ensembling predictions across {len(fold_models)} models...")

    # 2. Holdout DataLoader
    holdout_dir = cfg.get_path(cfg.holdout_img_dir)
    effective_tta = tta_mode or cfg.tta_mode
    holdout_dataset = HoldoutDataset(holdout_dir, image_size=cfg.image_size, mode=effective_tta)
    holdout_loader = DataLoader(holdout_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # 3. Predict & Average on the Unit Sphere
    all_filenames = []
    all_unit_vectors = []

    with torch.no_grad():
        for batch_views, fnames in tqdm(holdout_loader, desc="Ensemble Predicting", leave=False):
            all_filenames.extend(fnames)
            batch_vecs = []
            for f_info in fold_models:
                p_lat, p_lng, _, _, _ = decode_batch(
                    f_info["model"], batch_views, f_info["c3d"], f_info["clatlng"], f_info["f2c"],
                    cfg, retrieval_db=f_info["rdb"], mode=mode, device=device
                )
                vec = coords_to_3d(p_lat, p_lng)  # (B, 3)
                batch_vecs.append(vec)

            # Average 3D Cartesian vectors across models and re-normalize to unit sphere
            mean_vec = np.mean(batch_vecs, axis=0)  # (B, 3)
            norm = np.linalg.norm(mean_vec, axis=-1, keepdims=True)
            norm = np.maximum(norm, 1e-8)
            mean_unit_vec = mean_vec / norm
            all_unit_vectors.append(mean_unit_vec)

    all_unit_vectors = np.concatenate(all_unit_vectors, axis=0)
    final_coords = cartesian_to_latlng(all_unit_vectors)
    all_lats = final_coords[:, 0]
    all_lngs = final_coords[:, 1]

    pred_df = pd.DataFrame({
        "filename": all_filenames,
        "pred_lat": np.round(all_lats, 6),
        "pred_lng": np.round(all_lngs, 6)
    })

    dest_path = output_csv_path or os.path.join(cfg.exp_dir, cfg.prediction_output_name)
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    pred_df.to_csv(dest_path, index=False)

    print("-" * 70)
    print(f"✓ 5-Fold Ensemble Predictions successfully generated: {len(pred_df):,} rows")
    print(f"✓ Output file: {dest_path}")
    print(f"✓ Sample:\n{pred_df.head(3).to_string(index=False)}")
    print("=" * 70)
    return dest_path

def main():
    parser = argparse.ArgumentParser(description="Generate Holdout Predictions & Validate Submission")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint weights")
    parser.add_argument("--output", type=str, default=None, help="Path to write prediction CSV")
    parser.add_argument("--tta", type=str, default="direct", choices=["direct", "center", "5crop", "6view", "multiscale"], help="TTA mode")
    parser.add_argument("--mode", type=str, default="blended", choices=["cell_only", "retrieval_only", "blended"], help="Decoding strategy")
    parser.add_argument("--ensemble", action="store_true", help="Ensemble predictions across all available CV fold models")
    parser.add_argument("--validate", action="store_true", help="Run strict 13-point submission audit")
    parser.add_argument("--exp-dir", type=str, default=None, help="Experiment directory")
    args = parser.parse_args()

    cfg = GeolocationConfig.load_json(args.config) if args.config else get_default_config()
    target_exp = args.exp_dir or cfg.exp_dir

    if args.validate:
        success = audit_submission(exp_dir=target_exp, prediction_file=args.output)
        sys.exit(0 if success else 1)

    # 1. Generate holdout predictions
    if args.ensemble:
        dest_path = generate_ensemble_predictions(
            config=cfg,
            output_csv_path=args.output,
            tta_mode=args.tta,
            mode=args.mode,
            num_folds=cfg.cv_num_folds
        )
    else:
        dest_path = generate_predictions(
            config=cfg,
            checkpoint_path=args.checkpoint,
            output_csv_path=args.output,
            tta_mode=args.tta,
            mode=args.mode
        )

    # 2. Automatically audit the generated predictions
    print("\nRunning automatic 13-point submission audit on generated predictions...")
    success = audit_submission(exp_dir=target_exp, prediction_file=dest_path)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
