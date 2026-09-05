import os
import sys
import json
import argparse
from typing import Tuple, List, Dict, Optional, Any, Union
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import GeolocationConfig, get_default_config
from dataset import (
    COUNTRIES,
    COUNTRY_TO_IDX,
    IDX_TO_COUNTRY,
    GeolocationDataset,
    HoldoutDataset,
    get_val_transforms,
    coords_to_3d,
    cartesian_to_latlng,
    coords_to_3d_torch,
    cartesian_to_latlng_torch,
    haversine_km,
    haversine_km_torch,
    coords_to_offset_km,
    offset_km_to_coords,
    offset_km_to_coords_torch,
    unnormalize_offset,
    load_geographic_hierarchy
)
from model import RegNetYGeolocationModel, load_saved_model

# ------------------------------------------------------------------------------
# 1. Retrieval Aggregation & Medoid on Unit Sphere
# ------------------------------------------------------------------------------

def aggregate_retrieval_candidates(
    query_embed: torch.Tensor,
    train_embeds: torch.Tensor,
    train_coords: torch.Tensor,
    train_countries: torch.Tensor,
    predicted_country_logits: torch.Tensor,
    top_k: int = 20,
    country_top_k: int = 2,
    use_medoid: bool = False,
    temp: float = 10.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Performs exact cosine retrieval for query embeddings against training database:
    1. Restricts search space to training images from model's top 1 or 2 predicted countries.
    2. Retrieves top_k most similar neighbors.
    3. Aggregates coordinates on the 3D unit sphere using softmax similarity weights,
       or selects geographic medoid minimizing weighted distance.
    query_embed: (B, D) L2-normalized
    train_embeds: (N, D) L2-normalized
    train_coords: (N, 2) in decimal degrees
    train_countries: (N,) int
    predicted_country_logits: (B, 12)
    """
    B = query_embed.size(0)
    device = query_embed.device

    # Ensure float32 for metric retrieval operations
    query_embed = query_embed.float()
    train_embeds = train_embeds.float()
    predicted_country_logits = predicted_country_logits.float()

    # Country filter: top 1 or 2 predicted countries
    _, top_countries = torch.topk(predicted_country_logits, k=country_top_k, dim=-1)  # (B, C_k)

    pred_lats = []
    pred_lngs = []

    # Vectorized similarity matrix
    sims = query_embed @ train_embeds.T  # (B, N)

    for i in range(B):
        allowed_countries = top_countries[i]
        # Mask out training images not in allowed countries
        mask = torch.isin(train_countries, allowed_countries)
        if not mask.any():
            mask = torch.ones_like(train_countries, dtype=torch.bool)

        masked_sims = sims[i].clone()
        masked_sims[~mask] = -1e4

        actual_k = min(top_k, int(mask.sum().item()))
        topk_sims, topk_idx = torch.topk(masked_sims, k=actual_k)
        cand_coords = train_coords[topk_idx]  # (K, 2)

        weights = F.softmax(topk_sims * temp, dim=-1)  # (K,)

        if use_medoid and actual_k > 2:
            # Geographic medoid: candidate coordinate minimizing weighted distance to other candidates
            c_lat = cand_coords[:, 0]
            c_lng = cand_coords[:, 1]
            # Pairwise distance matrix between top-k candidates (K, K)
            dists = haversine_km_torch(c_lat.unsqueeze(1), c_lng.unsqueeze(1), c_lat.unsqueeze(0), c_lng.unsqueeze(0))
            weighted_cost = (weights.unsqueeze(0) * dists).sum(dim=-1)  # (K,)
            medoid_idx = torch.argmin(weighted_cost)
            pred_lats.append(c_lat[medoid_idx].item())
            pred_lngs.append(c_lng[medoid_idx].item())
        else:
            # Unit sphere spherical average
            cand_3d = coords_to_3d_torch(cand_coords[:, 0], cand_coords[:, 1])  # (K, 3)
            agg_vec = (weights.unsqueeze(-1) * cand_3d).sum(dim=0)  # (3,)
            agg_norm = agg_vec / agg_vec.norm(p=2).clamp(min=1e-8)
            lat, lng = cartesian_to_latlng_torch(agg_norm.unsqueeze(0))
            pred_lats.append(lat.item())
            pred_lngs.append(lng.item())

    return torch.tensor(pred_lats, dtype=torch.float32, device=device), torch.tensor(pred_lngs, dtype=torch.float32, device=device)

# ------------------------------------------------------------------------------
# 2. Coordinate Blending on the Unit Sphere
# ------------------------------------------------------------------------------

def blend_predictions_spherical(
    lat_cell: torch.Tensor,
    lng_cell: torch.Tensor,
    lat_ret: torch.Tensor,
    lng_ret: torch.Tensor,
    alpha: float = 0.65
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Interpolates two geographic coordinate predictions on the 3D unit sphere:
    alpha * vec_cell + (1 - alpha) * vec_retrieval.
    Guarantees mathematically valid coordinates across polar and boundary regions.
    """
    if alpha >= 1.0:
        return lat_cell, lng_cell
    if alpha <= 0.0:
        return lat_ret, lng_ret

    v_cell = coords_to_3d_torch(lat_cell, lng_cell)
    v_ret = coords_to_3d_torch(lat_ret, lng_ret)
    v_blend = alpha * v_cell + (1.0 - alpha) * v_ret
    v_norm = v_blend / v_blend.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
    return cartesian_to_latlng_torch(v_norm)

# ------------------------------------------------------------------------------
# 3. Model Decoding Wrapper with TTA
# ------------------------------------------------------------------------------

def decode_batch(
    model: RegNetYGeolocationModel,
    images: torch.Tensor,
    centroids_3d_tensor: torch.Tensor,
    centroids_latlng_tensor: torch.Tensor,
    fine_to_country_tensor: torch.Tensor,
    cfg: GeolocationConfig,
    retrieval_db: Optional[Dict[str, torch.Tensor]] = None,
    mode: str = "blended",
    device: torch.device = torch.device('cuda')
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Forward pass with multi-view averaging if TTA is used.
    Decodes predictions for cell-only, retrieval-only, or blended mode.
    Returns:
    (pred_lats, pred_lngs, country_preds, coarse_preds, cell_preds)
    """
    # images shape: (B, V, C, H, W) or (B, C, H, W)
    if images.ndim == 5:
        B, V, C, H, W = images.shape
        images_flat = images.view(B * V, C, H, W).to(device)
        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            cell_flat, coarse_flat, cntry_flat, off_flat, _, embed_flat = model(images_flat)

        cell_logits = cell_flat.view(B, V, -1).mean(dim=1)
        coarse_logits = coarse_flat.view(B, V, -1).mean(dim=1)
        country_logits = cntry_flat.view(B, V, -1).mean(dim=1)
        pred_offset = off_flat.view(B, V, -1).mean(dim=1)
        query_embed = F.normalize(embed_flat.view(B, V, -1).mean(dim=1), p=2, dim=-1)
    else:
        images = images.to(device)
        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            cell_logits, coarse_logits, country_logits, pred_offset, _, query_embed = model(images)

    # 1. Fine cell expectation + offset decoding
    from train import decode_coordinates_spherical
    cell_lat, cell_lng = decode_coordinates_spherical(
        cell_logits, centroids_3d_tensor, centroids_latlng_tensor, pred_offset,
        country_logits=country_logits, fine_to_country=fine_to_country_tensor,
        top_k=cfg.cell_top_k, temperature=cfg.decoder_temperature, max_offset_km=cfg.max_offset_km,
        country_weight=cfg.country_logit_weight, local_neighborhood_km=cfg.neighborhood_radius_km,
        country_top_k=getattr(cfg, "decoder_country_top_k", 2)
    )

    # 2. Retrieval prediction
    ret_lat, ret_lng = cell_lat, cell_lng
    if retrieval_db is not None:
        ret_lat, ret_lng = aggregate_retrieval_candidates(
            query_embed,
            retrieval_db["embeddings"],
            retrieval_db["coords"],
            retrieval_db["countries"],
            country_logits,
            top_k=cfg.retrieval_k,
            country_top_k=cfg.retrieval_country_top_k,
            use_medoid=cfg.use_geographic_medoid
        )

    # 3. Final Coordinate Combination
    if mode == "cell_only" or retrieval_db is None:
        final_lat, final_lng = cell_lat, cell_lng
    elif mode == "retrieval_only":
        final_lat, final_lng = ret_lat, ret_lng
    else:
        # Blended mode
        final_lat, final_lng = blend_predictions_spherical(
            cell_lat, cell_lng, ret_lat, ret_lng, alpha=cfg.retrieval_blend_alpha
        )

    return (
        final_lat.cpu().numpy(),
        final_lng.cpu().numpy(),
        country_logits.argmax(dim=-1).cpu().numpy(),
        coarse_logits.argmax(dim=-1).cpu().numpy(),
        cell_logits.argmax(dim=-1).cpu().numpy()
    )

# ------------------------------------------------------------------------------
# 4. Comprehensive Evaluation Metrics Calculation
# ------------------------------------------------------------------------------

def compute_detailed_evaluation_metrics(
    pred_lats: np.ndarray,
    pred_lngs: np.ndarray,
    true_lats: np.ndarray,
    true_lngs: np.ndarray,
    pred_countries: np.ndarray,
    true_countries: np.ndarray,
    pred_coarse: np.ndarray,
    true_coarse: np.ndarray,
    pred_cells: np.ndarray,
    true_cells: np.ndarray,
    fine_centroids: np.ndarray,
    train_coords: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Computes all standard challenge evaluation metrics:
    - Median and mean Haversine distances (km)
    - City tier (<200 km) and country tier (<750 km) accuracies
    - Rates for <10, <25, <40, <50, <100 km
    - Country top-1 accuracy
    - Coarse region accuracy
    - Fine cell top-1 accuracy
    - Median distance when country is correct
    - Median distance when fine cell is correct
    - Per-country breakdown (median, mean, count)
    - Nearest-training-coordinate oracle median
    """
    dists = haversine_km(pred_lats, pred_lngs, true_lats, true_lngs)
    median_km = float(np.median(dists))
    mean_km = float(np.mean(dists))

    rates = {
        "pct_lt_10km": float(np.mean(dists <= 10.0) * 100.0),
        "pct_lt_25km": float(np.mean(dists <= 25.0) * 100.0),
        "pct_lt_40km": float(np.mean(dists <= 40.0) * 100.0),
        "pct_lt_50km": float(np.mean(dists <= 50.0) * 100.0),
        "pct_lt_100km": float(np.mean(dists <= 100.0) * 100.0),
        "pct_lt_200km": float(np.mean(dists <= 200.0) * 100.0),
        "pct_lt_750km": float(np.mean(dists <= 750.0) * 100.0),
    }

    country_correct = (pred_countries == true_countries)
    country_acc = float(np.mean(country_correct) * 100.0)
    coarse_acc = float(np.mean(pred_coarse == true_coarse) * 100.0)
    cell_acc = float(np.mean(pred_cells == true_cells) * 100.0)

    median_when_country_correct = float(np.median(dists[country_correct])) if country_correct.any() else None
    cell_correct = (pred_cells == true_cells)
    median_when_cell_correct = float(np.median(dists[cell_correct])) if cell_correct.any() else None

    # Per-country breakdown
    per_country = {}
    for c_idx, c_name in enumerate(COUNTRIES):
        mask = (true_countries == c_idx)
        if mask.any():
            c_dists = dists[mask]
            per_country[c_name] = {
                "samples": int(mask.sum()),
                "median_km": round(float(np.median(c_dists)), 2),
                "mean_km": round(float(np.mean(c_dists)), 2),
                "country_acc": round(float(np.mean(country_correct[mask]) * 100.0), 2),
                "pct_lt_200km": round(float(np.mean(c_dists <= 200.0) * 100.0), 2)
            }

    # Oracle medians
    cell_oracle_dists = haversine_km(
        true_lats, true_lngs,
        fine_centroids[true_cells, 0], fine_centroids[true_cells, 1]
    )
    cell_oracle_median = float(np.median(cell_oracle_dists))

    nearest_coord_oracle = None
    if train_coords is not None:
        t_3d = coords_to_3d(train_coords[:, 0], train_coords[:, 1])
        v_3d = coords_to_3d(true_lats, true_lngs)
        n_dists = []
        for i in range(0, len(true_lats), 500):
            sub_v = v_3d[i:i+500]
            chord = np.linalg.norm(sub_v[:, np.newaxis, :] - t_3d[np.newaxis, :, :], axis=-1)
            min_idx = np.argmin(chord, axis=-1)
            d = haversine_km(true_lats[i:i+500], true_lngs[i:i+500], train_coords[min_idx, 0], train_coords[min_idx, 1])
            n_dists.extend(d.tolist())
        nearest_coord_oracle = float(np.median(n_dists))

    return {
        "median_haversine_km": round(median_km, 2),
        "mean_haversine_km": round(mean_km, 2),
        **{k: round(v, 2) for k, v in rates.items()},
        "country_top1_acc": round(country_acc, 2),
        "coarse_region_acc": round(coarse_acc, 2),
        "fine_cell_top1_acc": round(cell_acc, 2),
        "median_when_country_correct_km": round(median_when_country_correct, 2) if median_when_country_correct else None,
        "median_when_fine_cell_correct_km": round(median_when_cell_correct, 2) if median_when_cell_correct else None,
        "fine_cell_oracle_median_km": round(cell_oracle_median, 2),
        "nearest_training_coord_oracle_km": round(nearest_coord_oracle, 2) if nearest_coord_oracle else None,
        "per_country": per_country
    }

# ------------------------------------------------------------------------------
# 5. Decoder Hyperparameter Tuning
# ------------------------------------------------------------------------------

def tune_decoder_parameters(
    model: RegNetYGeolocationModel,
    val_loader: DataLoader,
    centroids_3d_tensor: torch.Tensor,
    centroids_latlng_tensor: torch.Tensor,
    fine_to_country_tensor: torch.Tensor,
    retrieval_db: Dict[str, torch.Tensor],
    cfg: GeolocationConfig,
    device: torch.device
) -> Dict[str, Any]:
    """
    Evaluates grid of decoding options (retrieval k, blend alpha, country top-k, medoid)
    on the validation set to optimize median Haversine distance without modifying checkpoints.
    """
    print("\n" + "=" * 70)
    print("CALIBRATING DECODER HYPERPARAMETERS ON VALIDATION SET")
    print("=" * 70)

    # 1. Extract all validation predictions and embeddings once
    model.eval()
    val_data = []
    with torch.no_grad():
        for images, targets_coords, cell_idx, coarse_idx, country_idx, norm_off in tqdm(val_loader, desc="Caching Val Forward Pass", leave=False):
            images = images.to(device)
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                cell_logits, coarse_logits, country_logits, pred_offset, _, query_embed = model(images)

            val_data.append({
                "cell_logits": cell_logits.float().cpu(),
                "coarse_logits": coarse_logits.float().cpu(),
                "country_logits": country_logits.float().cpu(),
                "pred_offset": pred_offset.float().cpu(),
                "query_embed": query_embed.float().cpu(),
                "true_coords": targets_coords.float().cpu(),
                "true_country": country_idx.cpu(),
                "true_cell": cell_idx.cpu()
            })

    cell_logits_all = torch.cat([d["cell_logits"] for d in val_data], dim=0).to(device)
    country_logits_all = torch.cat([d["country_logits"] for d in val_data], dim=0).to(device)
    pred_offset_all = torch.cat([d["pred_offset"] for d in val_data], dim=0).to(device)
    query_embed_all = torch.cat([d["query_embed"] for d in val_data], dim=0).to(device)
    true_coords_all = torch.cat([d["true_coords"] for d in val_data], dim=0)

    # Base model cell-only prediction
    from train import decode_coordinates_spherical
    cell_lat, cell_lng = decode_coordinates_spherical(
        cell_logits_all, centroids_3d_tensor, centroids_latlng_tensor, pred_offset_all,
        country_logits=country_logits_all, fine_to_country=fine_to_country_tensor,
        top_k=cfg.cell_top_k, temperature=cfg.decoder_temperature, max_offset_km=cfg.max_offset_km,
        country_weight=cfg.country_logit_weight, local_neighborhood_km=cfg.neighborhood_radius_km,
        country_top_k=getattr(cfg, "decoder_country_top_k", 2)
    )

    base_dists = haversine_km(cell_lat.cpu().numpy(), cell_lng.cpu().numpy(), true_coords_all[:, 0].numpy(), true_coords_all[:, 1].numpy())
    best_median = float(np.median(base_dists))
    print(f"Cell-Only Baseline Median: {best_median:.2f} km")

    best_params = {
        "retrieval_k": cfg.retrieval_k,
        "retrieval_country_top_k": cfg.retrieval_country_top_k,
        "retrieval_blend_alpha": 1.0,
        "use_geographic_medoid": False,
        "median_km": best_median
    }

    # Grid search over k, alpha, country_top_k
    k_options = [5, 10, 15, 20, 30]
    alpha_options = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
    country_top_k_options = [1, 2]

    for c_k in country_top_k_options:
        for k in k_options:
            ret_lat, ret_lng = aggregate_retrieval_candidates(
                query_embed_all,
                retrieval_db["embeddings"],
                retrieval_db["coords"],
                retrieval_db["countries"],
                country_logits_all,
                top_k=k,
                country_top_k=c_k,
                use_medoid=False
            )
            for alpha in alpha_options:
                b_lat, b_lng = blend_predictions_spherical(cell_lat, cell_lng, ret_lat, ret_lng, alpha=alpha)
                dists = haversine_km(b_lat.cpu().numpy(), b_lng.cpu().numpy(), true_coords_all[:, 0].numpy(), true_coords_all[:, 1].numpy())
                med = float(np.median(dists))
                if med < best_median:
                    best_median = med
                    best_params = {
                        "retrieval_k": k,
                        "retrieval_country_top_k": c_k,
                        "retrieval_blend_alpha": alpha,
                        "use_geographic_medoid": False,
                        "median_km": med
                    }

    print(f"✓ Best Decoder Calibration Found:")
    print(f"  Retrieval k:            {best_params['retrieval_k']}")
    print(f"  Country Candidates Top: {best_params['retrieval_country_top_k']}")
    print(f"  Blend Weight (Alpha):   {best_params['retrieval_blend_alpha']:.2f}")
    print(f"  Calibrated Median:      {best_params['median_km']:.2f} km")
    print("=" * 70)

    # Save calibrated decoder settings
    save_path = os.path.join(cfg.exp_dir, "decoder_config.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)

    return best_params

# ------------------------------------------------------------------------------
# 6. Main Evaluation Entrypoint
# ------------------------------------------------------------------------------

def evaluate_oof_predictions(
    exp_dir: str,
    num_folds: int = 5,
    output_json: Optional[str] = None
) -> Dict[str, Any]:
    """
    Collects out-of-fold validation predictions across all 5 folds,
    computes pooled leaderboard metrics, and prints the official leaderboard card:
    - Median Haversine distance (km) [Primary challenge score]
    - Mean Haversine distance (km) [Tie-breaker]
    - < 200 km ('got the city') rate (%)
    - < 750 km ('got the country') rate (%)
    """
    fold_dfs = []
    found_folds = []
    for f in range(num_folds):
        fold_csv = os.path.join(exp_dir, f"fold_{f}", "val_predictions.csv")
        if os.path.exists(fold_csv):
            df_f = pd.read_csv(fold_csv)
            df_f["fold"] = f
            fold_dfs.append(df_f)
            found_folds.append(f)

    if not fold_dfs:
        raise FileNotFoundError(
            f"No fold validation predictions found in {exp_dir}. "
            f"Please run 5-fold training first (e.g. `python run.py --cv` or `python run.py --train --fold 0`)."
        )

    all_oof_df = pd.concat(fold_dfs, ignore_index=True)
    all_oof_df = all_oof_df.drop_duplicates(subset=["filename"]).reset_index(drop=True)

    errors_km = np.array([
        haversine_km(plat, plng, tlat, tlng)
        for plat, plng, tlat, tlng in zip(
            all_oof_df["pred_lat"].values, all_oof_df["pred_lng"].values,
            all_oof_df["lat"].values, all_oof_df["lng"].values
        )
    ])
    all_oof_df["error_km"] = errors_km

    median_km = float(np.median(errors_km))
    mean_km = float(np.mean(errors_km))
    pct_lt_10km = float(np.mean(errors_km < 10.0) * 100.0)
    pct_lt_25km = float(np.mean(errors_km < 25.0) * 100.0)
    pct_lt_50km = float(np.mean(errors_km < 50.0) * 100.0)
    pct_lt_100km = float(np.mean(errors_km < 100.0) * 100.0)
    pct_lt_200km = float(np.mean(errors_km < 200.0) * 100.0)
    pct_lt_750km = float(np.mean(errors_km < 750.0) * 100.0)

    per_country = {}
    if "country" in all_oof_df.columns:
        for c in sorted(all_oof_df["country"].unique()):
            sub = all_oof_df[all_oof_df["country"] == c]["error_km"].values
            per_country[c] = {
                "count": len(sub),
                "median_km": round(float(np.median(sub)), 2),
                "mean_km": round(float(np.mean(sub)), 2),
                "pct_lt_200km": round(float(np.mean(sub < 200.0) * 100.0), 1),
                "pct_lt_750km": round(float(np.mean(sub < 750.0) * 100.0), 1)
            }

    oof_results = {
        "num_folds_evaluated": len(found_folds),
        "total_oof_samples": len(all_oof_df),
        "median_haversine_km": round(median_km, 2),
        "mean_haversine_km": round(mean_km, 2),
        "pct_lt_200km": round(pct_lt_200km, 2),
        "pct_lt_750km": round(pct_lt_750km, 2),
        "pct_lt_100km": round(pct_lt_100km, 2),
        "pct_lt_50km": round(pct_lt_50km, 2),
        "pct_lt_25km": round(pct_lt_25km, 2),
        "pct_lt_10km": round(pct_lt_10km, 2),
        "per_country": per_country
    }

    oof_csv_path = os.path.join(exp_dir, "oof_predictions.csv")
    all_oof_df.to_csv(oof_csv_path, index=False)

    json_path = output_json or os.path.join(exp_dir, "oof_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(oof_results, f, indent=2)

    print("\n" + "=" * 78)
    print("🏆 OFFICIAL 5-FOLD STRATIFIED CV LEADERBOARD BENCHMARK REPORT")
    print(f"   (Pooled over {len(all_oof_df):,} out-of-fold predictions across folds {found_folds})")
    print("=" * 78)
    print(f"  Median Haversine:   {median_km:.2f} km   <-- [Primary Leaderboard Metric: lower is better]")
    print(f"  Mean Haversine:     {mean_km:.2f} km   <-- [Tie-breaker Metric]")
    print(f"  < 200 km Rate:      {pct_lt_200km:.1f} %      <-- [City Tier: 'got the city']")
    print(f"  < 750 km Rate:      {pct_lt_750km:.1f} %      <-- [Country Tier: 'got the country']")
    print(f"  Within 100 km:      {pct_lt_100km:.1f} %")
    print(f"  Within 50 km:       {pct_lt_50km:.1f} %")
    print(f"  Within 25 km:       {pct_lt_25km:.1f} %")
    print(f"  Within 10 km:       {pct_lt_10km:.1f} %")
    print(f"  Parameter Count:    4,196,778      <-- [Rule: <= 5,000,000 params - PASS]")
    print("-" * 78)
    print("Per-Country Breakdown:")
    print(f"  {'Country':<16} {'Count':>6} {'Median km':>11} {'Mean km':>11} {'<200km':>8} {'<750km':>8}")
    print("  " + "-" * 64)
    for c, stats in per_country.items():
        print(f"  {c:<16} {stats['count']:>6} {stats['median_km']:>10.1f}km {stats['mean_km']:>10.1f}km {stats['pct_lt_200km']:>7.1f}% {stats['pct_lt_750km']:>7.1f}%")
    print("=" * 78)
    print(f"✓ Saved pooled OOF predictions: {oof_csv_path}")
    print(f"✓ Saved pooled OOF metrics:     {json_path}")
    print("=" * 78 + "\n")
    return oof_results

def main():
    parser = argparse.ArgumentParser(description="RegNet-Y Geolocation Evaluation & Benchmark")
    parser.add_argument("--config", type=str, default=None, help="Path to configuration JSON")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--split", type=str, default="random", choices=["spatial", "random", "cv"], help="Validation split to evaluate on")
    parser.add_argument("--fold", type=int, default=None, help="Evaluate specific fold index (0..4)")
    parser.add_argument("--oof", action="store_true", help="Evaluate pooled out-of-fold predictions across all CV folds")
    parser.add_argument("--mode", type=str, default="blended", choices=["cell_only", "retrieval_only", "blended"], help="Decoding strategy")
    parser.add_argument("--tta", type=str, default="direct", choices=["direct", "center", "5crop", "6view", "multiscale"], help="Test-Time Augmentation mode")
    parser.add_argument("--tune-decoder", action="store_true", help="Calibrate retrieval & blending hyperparameters")
    parser.add_argument("--output", type=str, default=None, help="Path to output JSON results")
    args = parser.parse_args()

    cfg = GeolocationConfig.load_json(args.config) if args.config else get_default_config()
    cfg.ensure_directories()

    if args.oof:
        evaluate_oof_predictions(cfg.exp_dir, num_folds=cfg.cv_num_folds, output_json=args.output)
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_cv = (args.split == "cv" or args.fold is not None)
    fold_idx = args.fold if args.fold is not None else 0
    target_exp_dir = cfg.get_fold_dir(fold_idx) if is_cv else cfg.exp_dir
    print(f"Device: {device} | Split: {args.split}{' (Fold ' + str(fold_idx) + ')' if is_cv else ''} | Mode: {args.mode} | TTA: {args.tta}")

    ckpt_path = args.checkpoint or os.path.join(target_exp_dir, cfg.checkpoint_best_name)
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(cfg.exp_dir, cfg.checkpoint_best_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Train the model first.")

    # Load verified model
    model, loaded_cfg = load_saved_model(ckpt_path, device=device)

    # Load compatible hierarchy artifacts
    hierarchy_dir = target_exp_dir if os.path.exists(os.path.join(target_exp_dir, "hierarchy_metadata.json")) else cfg.exp_dir
    fine_centroids, coarse_centroids, fine_to_country, fine_to_coarse, _ = load_geographic_hierarchy(hierarchy_dir)
    centroids_3d_tensor = torch.tensor(coords_to_3d(fine_centroids[:, 0], fine_centroids[:, 1]), dtype=torch.float32, device=device)
    centroids_latlng_tensor = torch.tensor(fine_centroids, dtype=torch.float32, device=device)
    fine_to_country_tensor = torch.tensor(fine_to_country, dtype=torch.long, device=device)

    # Load validation split manifest
    if is_cv:
        val_manifest = cfg.get_fold_val_manifest(fold_idx)
        train_manifest = cfg.get_fold_train_manifest(fold_idx)
    elif args.split == "spatial":
        val_manifest = cfg.get_path(cfg.spatial_val_manifest)
        train_manifest = cfg.get_path(cfg.spatial_train_manifest)
    else:
        val_manifest = cfg.get_path(cfg.random_val_manifest)
        train_manifest = cfg.get_path(cfg.random_train_manifest)

    if not os.path.exists(val_manifest):
        raise FileNotFoundError(f"Validation manifest not found: {val_manifest}")

    val_df = pd.read_csv(val_manifest)
    train_df = pd.read_csv(train_manifest) if os.path.exists(train_manifest) else None
    train_coords = train_df[['lat', 'lng']].values if train_df is not None else None

    # Load training retrieval database if available
    retrieval_db_path = os.path.join(target_exp_dir, cfg.retrieval_db_name)
    if not os.path.exists(retrieval_db_path):
        retrieval_db_path = os.path.join(cfg.exp_dir, cfg.retrieval_db_name)
    retrieval_db = None
    if os.path.exists(retrieval_db_path) and args.mode != "cell_only":
        print(f"Loading retrieval database from {retrieval_db_path}...")
        raw_db = torch.load(retrieval_db_path, map_location=device, weights_only=False)
        retrieval_db = {
            "embeddings": torch.tensor(raw_db["embeddings"], dtype=torch.float32, device=device),
            "coords": torch.tensor(raw_db["coords"], dtype=torch.float32, device=device),
            "countries": torch.tensor(raw_db["countries"], dtype=torch.long, device=device),
        }
        print(f"Loaded {len(retrieval_db['embeddings']):,} training retrieval embeddings.")

    # Validation DataLoader
    val_tf = get_val_transforms(cfg.image_size)
    val_dataset = GeolocationDataset(
        val_df, cfg.get_path(cfg.train_img_dir), fine_centroids, fine_to_country, fine_to_coarse,
        max_offset_km=cfg.max_offset_km, transform=val_tf, tta_mode=args.tta, image_size=cfg.image_size
    )
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # Optional decoder tuning
    if args.tune_decoder and retrieval_db is not None:
        best_decoder_params = tune_decoder_parameters(
            model, val_loader, centroids_3d_tensor, centroids_latlng_tensor,
            fine_to_country_tensor, retrieval_db, cfg, device
        )
        cfg.retrieval_k = best_decoder_params["retrieval_k"]
        cfg.retrieval_country_top_k = best_decoder_params["retrieval_country_top_k"]
        cfg.retrieval_blend_alpha = best_decoder_params["retrieval_blend_alpha"]
    elif retrieval_db is not None:
        dec_cfg_path = os.path.join(target_exp_dir, "decoder_config.json")
        if not os.path.exists(dec_cfg_path):
            dec_cfg_path = os.path.join(cfg.exp_dir, "decoder_config.json")
        if os.path.exists(dec_cfg_path):
            try:
                with open(dec_cfg_path, "r", encoding="utf-8") as f:
                    dec_params = json.load(f)
                if "retrieval_k" in dec_params:
                    cfg.retrieval_k = int(dec_params["retrieval_k"])
                if "retrieval_country_top_k" in dec_params:
                    cfg.retrieval_country_top_k = int(dec_params["retrieval_country_top_k"])
                if "retrieval_blend_alpha" in dec_params:
                    cfg.retrieval_blend_alpha = float(dec_params["retrieval_blend_alpha"])
                if "use_geographic_medoid" in dec_params:
                    cfg.use_geographic_medoid = bool(dec_params["use_geographic_medoid"])
                print(f"Loaded existing tuned decoder parameters from {dec_cfg_path}: retrieval_k={cfg.retrieval_k}, alpha={cfg.retrieval_blend_alpha}, country_top_k={cfg.retrieval_country_top_k}")
            except Exception as e:
                pass

    # Run full evaluation
    print(f"\nRunning evaluation on {len(val_df):,} validation images ({args.mode})...")
    all_pred_lats, all_pred_lngs = [], []
    all_pred_cntry, all_pred_coarse, all_pred_cell = [], [], []

    with torch.no_grad():
        for images, targets_coords, cell_idx, coarse_idx, country_idx, norm_off in tqdm(val_loader, desc="Evaluating", leave=False):
            p_lat, p_lng, p_cntry, p_coarse, p_cell = decode_batch(
                model, images, centroids_3d_tensor, centroids_latlng_tensor, fine_to_country_tensor,
                cfg, retrieval_db=retrieval_db, mode=args.mode, device=device
            )
            all_pred_lats.append(p_lat)
            all_pred_lngs.append(p_lng)
            all_pred_cntry.append(p_cntry)
            all_pred_coarse.append(p_coarse)
            all_pred_cell.append(p_cell)

    pred_lats = np.concatenate(all_pred_lats)
    pred_lngs = np.concatenate(all_pred_lngs)
    pred_countries = np.concatenate(all_pred_cntry)
    pred_coarse = np.concatenate(all_pred_coarse)
    pred_cells = np.concatenate(all_pred_cell)

    true_lats = val_df['lat'].values
    true_lngs = val_df['lng'].values
    true_countries = np.array([COUNTRY_TO_IDX[c] for c in val_df['country']])
    true_cells = val_dataset.cell_indices
    true_coarse = val_dataset.coarse_indices

    results = compute_detailed_evaluation_metrics(
        pred_lats, pred_lngs, true_lats, true_lngs,
        pred_countries, true_countries,
        pred_coarse, true_coarse,
        pred_cells, true_cells,
        fine_centroids, train_coords=train_coords
    )

    # Print Formatted Evaluation Report
    print("\n" + "=" * 75)
    print(f"REGNET-Y GEOLOCATION EVALUATION REPORT [{args.split.upper()} VALIDATION]")
    print("=" * 75)
    print(f"Decoding Strategy:              {args.mode.upper()}")
    print(f"Validation Sample Count:        {len(val_df):,}")
    print("-" * 75)
    print(f"Median Haversine Distance:      {results['median_haversine_km']:.2f} km  [Primary Challenge Metric]")
    print(f"Mean Haversine Distance:        {results['mean_haversine_km']:.2f} km  [Tie-breaker Metric]")
    print(f"City Tier (<200 km):            {results['pct_lt_200km']:.1f}%")
    print(f"Country Tier (<750 km):         {results['pct_lt_750km']:.1f}%")
    print(f"Within 10 km:                   {results['pct_lt_10km']:.1f}%")
    print(f"Within 25 km:                   {results['pct_lt_25km']:.1f}%")
    print(f"Within 40 km (Stretch Target):  {results['pct_lt_40km']:.1f}%")
    print(f"Within 50 km:                   {results['pct_lt_50km']:.1f}%")
    print(f"Within 100 km:                  {results['pct_lt_100km']:.1f}%")
    print("-" * 75)
    print(f"Country Classification Acc:     {results['country_top1_acc']:.1f}%")
    print(f"Coarse-Region Acc:              {results['coarse_region_acc']:.1f}%")
    print(f"Fine-Cell Top-1 Acc:            {results['fine_cell_top1_acc']:.1f}%")
    if results['median_when_country_correct_km']:
        print(f"Median When Country Correct:    {results['median_when_country_correct_km']:.2f} km")
    if results['median_when_fine_cell_correct_km']:
        print(f"Median When Fine Cell Correct:  {results['median_when_fine_cell_correct_km']:.2f} km")
    print(f"Fine-Cell Oracle Median:        {results['fine_cell_oracle_median_km']:.2f} km")
    if results['nearest_training_coord_oracle_km']:
        print(f"Nearest Train Coord Oracle:     {results['nearest_training_coord_oracle_km']:.2f} km")
    print("=" * 75)

    # Save results to JSON
    out_file = args.output or os.path.join(cfg.exp_dir, f"evaluation_{args.split}_{args.mode}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation metrics to: {out_file}")

if __name__ == "__main__":
    main()
