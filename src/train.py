import os
import sys
import time
import json
import math
import random
import argparse
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from config import GeolocationConfig, get_default_config
from dataset import (
    COUNTRIES,
    COUNTRY_TO_IDX,
    GeolocationDataset,
    TwoViewGeolocationDataset,
    get_train_transforms,
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
    create_random_split,
    create_spatial_split,
    create_stratified_cv_splits,
    compute_split_diagnostics,
    build_geographic_hierarchy,
    load_geographic_hierarchy,
    EARTH_RADIUS_KM
)
from model import (
    RegNetYGeolocationModel,
    get_model,
    set_phase_trainable,
    load_saved_model
)

def create_grad_scaler(device_type: str = 'cuda', enabled: bool = True):
    """Creates a backward-compatible GradScaler across PyTorch versions."""
    if hasattr(torch.amp, 'GradScaler'):
        try:
            return torch.amp.GradScaler(device_type, enabled=enabled)
        except (TypeError, ValueError):
            return torch.amp.GradScaler(enabled=enabled)
    elif hasattr(torch.cuda.amp, 'GradScaler'):
        return torch.cuda.amp.GradScaler(enabled=enabled)
    else:
        return None

# ------------------------------------------------------------------------------
# 1. Deterministic Seeding & Reproducibility
# ------------------------------------------------------------------------------

def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """Sets deterministic seeds across Python, NumPy, PyTorch, CUDA, and cuDNN."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def seed_worker(worker_id: int) -> None:
    """Worker init function to guarantee reproducible DataLoader multiprocessing."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ------------------------------------------------------------------------------
# 2. Exponential Moving Average (EMA)
# ------------------------------------------------------------------------------

class EMA:
    """Maintains Exponential Moving Average of model parameters."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self.backup = {}
        self.step = 0

    def update(self, model: nn.Module) -> None:
        self.step += 1
        # Warmup decay
        decay = min(self.decay, (1.0 + self.step) / (10.0 + self.step))
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(decay).add_(v.detach(), alpha=1.0 - decay)
            else:
                self.shadow[k].copy_(v)

    def apply_to(self, model: nn.Module) -> None:
        self.backup = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)

    def restore(self, model: nn.Module) -> None:
        if self.backup:
            model.load_state_dict(self.backup)
            self.backup = {}

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.shadow

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.clone().detach() for k, v in state_dict.items()}

# ------------------------------------------------------------------------------
# 3. Coordinate Decoding on Unit Sphere
# ------------------------------------------------------------------------------

def decode_coordinates_spherical(
    cell_logits: torch.Tensor,
    centroids_3d_tensor: torch.Tensor,
    centroids_latlng_tensor: torch.Tensor,
    pred_offset: torch.Tensor,
    country_logits: Optional[torch.Tensor] = None,
    fine_to_country: Optional[torch.Tensor] = None,
    top_k: int = 4,
    temperature: float = 0.1,
    max_offset_km: float = 50.0,
    country_weight: float = 3.0,
    local_neighborhood_km: float = 150.0,
    country_top_k: int = 2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decodes continuous GPS coordinates via spatially-constrained local neighborhood
    expectation on the 3D unit sphere plus local tangent-plane displacement in km.

    Prevents multimodal averaging collapse:
    - Incorporates country log-probabilities as hierarchical prior.
    - Identifies dominant cell c* = argmax(logits).
    - Restricts candidate cells to local spatial neighborhood (<= local_neighborhood_km)
      and top predicted countries (country_top_k, default 2 for smooth continental border transitions),
      ensuring expectation cannot jump across continents or distant borders while allowing seamless border crossings.
    - Performs soft spherical vector sum over top-k local candidates.
    - Adds learned tangent-plane displacement (north_km, east_km).
    """
    logits = cell_logits.clone().float()
    if country_logits is not None and fine_to_country is not None:
        country_log_probs = F.log_softmax(country_logits.float(), dim=-1)
        country_prior = country_log_probs[:, fine_to_country]
        logits = logits + country_weight * country_prior

    batch_size = logits.size(0)
    top1_cell = torch.argmax(logits, dim=-1)  # (B,)
    top1_3d = centroids_3d_tensor[top1_cell]  # (B, 3)

    # Angular distance on unit sphere between top-1 cell and all centroids: (B, N)
    cos_sim = torch.sum(top1_3d.unsqueeze(1) * centroids_3d_tensor.unsqueeze(0), dim=-1).clamp(-1.0, 1.0)
    dist_to_top1_km = EARTH_RADIUS_KM * torch.acos(cos_sim)

    # Local neighborhood mask: within local_neighborhood_km
    local_mask = dist_to_top1_km <= local_neighborhood_km
    if fine_to_country is not None:
        if country_logits is not None and country_top_k > 1:
            # Allow candidate cells belonging to top-K predicted countries
            k_countries = min(country_top_k, country_logits.size(-1))
            _, top_c = torch.topk(country_logits.float(), k=k_countries, dim=-1)  # (B, k_countries)
            # fine_to_country: (N,), top_c: (B, k_countries) -> per-row comparison: (B, N)
            allowed_country_mask = (fine_to_country[None, :, None] == top_c[:, None, :]).any(dim=-1)
            local_mask = local_mask & allowed_country_mask
        else:
            top1_country = fine_to_country[top1_cell]
            same_country = (fine_to_country.unsqueeze(0) == top1_country.unsqueeze(1))
            local_mask = local_mask & same_country

    # Guarantee top-1 cell is always preserved in the mask
    batch_idx = torch.arange(batch_size, device=logits.device)
    local_mask[batch_idx, top1_cell] = True

    # Mask out distant cells with large negative logit
    masked_logits = logits.masked_fill(~local_mask, -1e4)
    scaled_logits = masked_logits / max(temperature, 1e-4)

    effective_k = min(top_k, logits.size(-1)) if (top_k is not None and top_k > 0) else 4
    topk_logits, topk_idx = torch.topk(scaled_logits, k=effective_k, dim=-1)
    topk_probs = F.softmax(topk_logits, dim=-1)  # (B, K)

    topk_vecs = centroids_3d_tensor[topk_idx]     # (B, K, 3)
    weighted_vec = torch.sum(topk_probs.unsqueeze(-1) * topk_vecs, dim=1)  # (B, 3)

    # Convert unit sphere vector to (lat, lng) degrees
    soft_lat, soft_lng = cartesian_to_latlng_torch(weighted_vec)

    # Unnormalize offset [-1, 1] to displacement in km
    norm_north = pred_offset[:, 0].float()
    norm_east = pred_offset[:, 1].float()
    north_km, east_km = unnormalize_offset(norm_north, norm_east, max_offset_km=max_offset_km)

    # Apply local displacement
    final_lat, final_lng = offset_km_to_coords_torch(soft_lat, soft_lng, north_km, east_km)
    return final_lat, final_lng

# ------------------------------------------------------------------------------
# 4. Geographic Supervised Contrastive Loss (Phase A)
# ------------------------------------------------------------------------------

class GeographicContrastiveLoss(nn.Module):
    """
    Distance-aware Supervised Contrastive Loss (SupCon) for Phase A:
    - Augmented views of the same image: strong positive (weight = 1.0)
    - Different training images within 10-75 km: distance-weighted positive (weight = exp(-dist / 50.0))
    - Images > 75 km: negatives (hard negatives if same country)
    - Numerically stable with log-sum-exp formulation.
    """
    def __init__(self, temperature: float = 0.1, dist_scale: float = 50.0):
        super(GeographicContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.dist_scale = dist_scale

    def forward(
        self,
        embed1: torch.Tensor,
        embed2: torch.Tensor,
        coords: torch.Tensor,
        country_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        embed1, embed2: (B, D) L2-normalized metric embeddings
        coords: (B, 2) in decimal degrees (lat, lng)
        country_indices: (B,)
        """
        # Ensure float32 precision for metric embeddings and coordinates
        embed1 = embed1.float()
        embed2 = embed2.float()
        coords = coords.float()

        B = embed1.size(0)
        # Concatenate views: 2*B representations
        embeddings = torch.cat([embed1, embed2], dim=0)  # (2B, D)
        all_coords = torch.cat([coords, coords], dim=0)  # (2B, 2)
        all_countries = torch.cat([country_indices, country_indices], dim=0)  # (2B,)

        # Compute pairwise distance matrix on CPU or GPU
        lat = all_coords[:, 0]
        lng = all_coords[:, 1]
        lat_diff = lat.unsqueeze(1) - lat.unsqueeze(0)
        lng_diff = lng.unsqueeze(1) - lng.unsqueeze(0)
        dists = haversine_km_torch(lat.unsqueeze(1), lng.unsqueeze(1), lat.unsqueeze(0), lng.unsqueeze(0))

        # Weighting function: exp(-dist / dist_scale) for d <= 75km
        weights = torch.exp(-dists / self.dist_scale)
        # Distance cutoff: zero out distant pairs
        weights = torch.where(dists <= 75.0, weights, torch.zeros_like(weights))
        
        # Augmented pair of same sample gets full weight 1.0
        sample_indices = torch.cat([torch.arange(B), torch.arange(B)], dim=0).to(embed1.device)
        same_sample_mask = sample_indices.unsqueeze(1) == sample_indices.unsqueeze(0)
        weights = torch.where(same_sample_mask, torch.ones_like(weights), weights)

        # Self-contrast mask: zero out exact diagonal
        self_mask = torch.eye(2 * B, dtype=torch.bool, device=embed1.device)
        weights.masked_fill_(self_mask, 0.0)

        # Pairwise cosine similarity matrix scaled by temperature
        sim_matrix = (embeddings @ embeddings.T) / self.temperature

        # For numerical stability, subtract row max (excluding self)
        logits_mask = ~self_mask
        # Use -1e4 which fits in both float16 and float32 without overflow
        row_max, _ = torch.max(sim_matrix.masked_fill(~logits_mask, -1e4), dim=1, keepdim=True)
        logits = sim_matrix - row_max.detach()

        # Denominator: sum exp over all j != i
        exp_logits = torch.exp(logits) * logits_mask.float()
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp(min=1e-8))

        # Numerator: weighted sum over positive pairs
        pos_weight_sum = weights.sum(dim=1)
        valid_rows = pos_weight_sum > 0

        loss_per_row = -(weights * log_prob).sum(dim=1) / pos_weight_sum.clamp(min=1e-8)
        loss = loss_per_row[valid_rows].mean() if valid_rows.any() else torch.tensor(0.0, device=embed1.device)
        return loss

# ------------------------------------------------------------------------------
# 5. Training Epoch Handlers for Phases A, B, and C
# ------------------------------------------------------------------------------

def train_phase_a_epoch(
    model: RegNetYGeolocationModel,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion_contrastive: GeographicContrastiveLoss,
    scaler: Any,
    device: torch.device,
    grad_accum_steps: int = 1,
    grad_clip_norm: float = 1.0
) -> float:
    """Phase A: Representation learning with 2 augmented views and SupCon loss."""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc="Phase A [Representation]", leave=False)
    for step, (v1, v2, coords, countries, _) in enumerate(pbar):
        v1, v2 = v1.to(device), v2.to(device)
        coords = coords.to(device)
        countries = countries.to(device)

        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            e1 = model.extract_features(v1)
            e2 = model.extract_features(v2)
            loss = criterion_contrastive(e1, e2, coords, countries)
            loss_scaled = loss / grad_accum_steps

        scaler.scale(loss_scaled).backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * v1.size(0)
        pbar.set_postfix({'metric_loss': f"{loss.item():.4f}"})

    return total_loss / len(loader.dataset)

def compute_focal_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 1.5,
    label_smoothing: float = 0.05
) -> torch.Tensor:
    """
    Focal cross-entropy with label smoothing.
    Down-weights easy examples to focus model capacity on hard European border cases.
    """
    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = torch.exp(log_probs)
    target_probs = probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    focal_weight = (1.0 - target_probs).clamp(min=0.0, max=1.0) ** gamma
    ce_loss = F.cross_entropy(logits.float(), targets, label_smoothing=label_smoothing, reduction='none')
    return (focal_weight * ce_loss).mean()

def train_phase_b_epoch(
    model: RegNetYGeolocationModel,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: Any,
    device: torch.device,
    grad_accum_steps: int = 1,
    grad_clip_norm: float = 1.0
) -> Tuple[float, float, float]:
    """Phase B: Country and coarse-region learning with trainable backbone."""
    model.train()
    total_loss = 0.0
    correct_country, correct_coarse, total_samples = 0, 0, 0
    optimizer.zero_grad()

    criterion_coarse = nn.CrossEntropyLoss(label_smoothing=0.05)

    pbar = tqdm(loader, desc="Phase B [Country/Coarse]", leave=False)
    for step, (images, _, _, coarse_idx, country_idx, _) in enumerate(pbar):
        images = images.to(device)
        coarse_idx = coarse_idx.to(device)
        country_idx = country_idx.to(device)

        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            _, coarse_logits, country_logits, _, _, _ = model(images)
            loss_cntry = compute_focal_cross_entropy(country_logits, country_idx, gamma=1.5, label_smoothing=0.05)
            loss_coarse = criterion_coarse(coarse_logits.float(), coarse_idx)
            loss = 4.0 * loss_cntry + 1.0 * loss_coarse
            loss_scaled = loss / grad_accum_steps

        scaler.scale(loss_scaled).backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * images.size(0)
        pred_cntry = country_logits.argmax(dim=-1)
        pred_coarse = coarse_logits.argmax(dim=-1)
        correct_country += (pred_cntry == country_idx).sum().item()
        correct_coarse += (pred_coarse == coarse_idx).sum().item()
        total_samples += images.size(0)

        pbar.set_postfix({'loss': f"{loss.item():.3f}", 'cntry_acc': f"{100.0 * correct_country / total_samples:.1f}%"})

    return (
        total_loss / total_samples,
        100.0 * correct_country / total_samples,
        100.0 * correct_coarse / total_samples
    )

def train_phase_c_epoch(
    model: RegNetYGeolocationModel,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: Any,
    ema: EMA,
    centroids_3d_tensor: torch.Tensor,
    centroids_latlng_tensor: torch.Tensor,
    fine_to_country_tensor: torch.Tensor,
    epoch: int,
    cfg: GeolocationConfig,
    device: torch.device
) -> Dict[str, float]:
    """Phase C: Multi-task joint localization."""
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()

    criterion_ce = nn.CrossEntropyLoss(label_smoothing=0.05)
    
    # Temperature schedule (soft start annealing)
    t_progress = min(1.0, epoch / max(1, cfg.phase_c_epochs))
    current_temp = cfg.initial_train_temp - t_progress * (cfg.initial_train_temp - cfg.final_train_temp)
    
    # Gradual introduction of Haversine loss
    w_hav_current = cfg.loss_weight_haversine * min(1.0, (epoch + 1) / max(1, cfg.haversine_warmup_epochs))

    pbar = tqdm(loader, desc=f"Phase C Epoch {epoch+1}", leave=False)
    for step, (images, targets_coords, cell_idx, coarse_idx, country_idx, norm_offset) in enumerate(pbar):
        images = images.to(device)
        targets_coords = targets_coords.to(device)
        cell_idx = cell_idx.to(device)
        coarse_idx = coarse_idx.to(device)
        country_idx = country_idx.to(device)
        norm_offset = norm_offset.to(device)

        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            cell_logits, coarse_logits, country_logits, pred_offset, pred_xyz, metric_embed = model(images)

            # 1. Classification losses
            loss_fine = criterion_ce(cell_logits.float(), cell_idx)
            loss_coarse = criterion_ce(coarse_logits.float(), coarse_idx)
            loss_country = compute_focal_cross_entropy(country_logits, country_idx, gamma=1.5, label_smoothing=0.05)

            # 2. Local tangent-plane offset loss (Smooth L1 on normalized displacement)
            loss_offset = F.smooth_l1_loss(pred_offset.float(), norm_offset.float())

            # 3. 3D Cartesian loss
            loss_xyz = torch.tensor(0.0, device=device)
            if pred_xyz is not None:
                true_xyz = coords_to_3d_torch(targets_coords[:, 0], targets_coords[:, 1])
                loss_xyz = F.mse_loss(pred_xyz.float(), true_xyz.float())

            # 4. Metric distance consistency loss (smooth geographic embedding space)
            batch_dists_km = haversine_km_torch(
                targets_coords[:, 0:1], targets_coords[:, 1:2],
                targets_coords[:, 0:1].T, targets_coords[:, 1:2].T
            )
            sim_mat = metric_embed @ metric_embed.T
            target_sim = torch.exp(-batch_dists_km / cfg.metric_dist_scale)
            loss_metric = F.mse_loss(sim_mat, target_sim)

            # 5. Decoded coordinates & Differentiable Haversine Loss
            decoded_lat, decoded_lng = decode_coordinates_spherical(
                cell_logits, centroids_3d_tensor, centroids_latlng_tensor, pred_offset,
                country_logits=country_logits, fine_to_country=fine_to_country_tensor,
                top_k=cfg.cell_top_k, temperature=current_temp, max_offset_km=cfg.max_offset_km,
                country_weight=cfg.country_logit_weight, local_neighborhood_km=cfg.neighborhood_radius_km,
                country_top_k=cfg.decoder_country_top_k
            )
            dists_km = haversine_km_torch(decoded_lat, decoded_lng, targets_coords[:, 0], targets_coords[:, 1])
            loss_hav = (torch.log(1.0 + dists_km / 10.0)).mean()

            # Multi-task total loss
            loss = (
                cfg.loss_weight_fine * loss_fine +
                cfg.loss_weight_coarse * loss_coarse +
                cfg.loss_weight_country * loss_country +
                cfg.loss_weight_metric * loss_metric +
                cfg.loss_weight_offset * loss_offset +
                cfg.loss_weight_cartesian * loss_xyz +
                w_hav_current * loss_hav
            )
            loss_scaled = loss / cfg.grad_accum_steps

        scaler.scale(loss_scaled).backward()

        if (step + 1) % cfg.grad_accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            ema.update(model)

        running_loss += loss.item() * images.size(0)
        pbar.set_postfix({
            'loss': f"{loss.item():.3f}",
            'ctry': f"{loss_country.item():.2f}",
            'fine': f"{loss_fine.item():.2f}",
            'hav_med': f"{dists_km.median().item():.1f}km"
        })

    return {
        "train_loss": running_loss / len(loader.dataset),
        "temp": current_temp,
        "w_hav": w_hav_current
    }

# ------------------------------------------------------------------------------
# 6. Validation Evaluation Function
# ------------------------------------------------------------------------------

def validate_epoch(
    model: RegNetYGeolocationModel,
    loader: DataLoader,
    centroids_3d_tensor: torch.Tensor,
    centroids_latlng_tensor: torch.Tensor,
    fine_to_country_tensor: torch.Tensor,
    cfg: GeolocationConfig,
    device: torch.device
) -> Dict[str, float]:
    """Evaluates validation performance, computing median and accuracy rates."""
    model.eval()
    all_dists = []
    correct_country = 0
    correct_cell = 0
    total = 0

    with torch.no_grad():
        for images, targets_coords, cell_idx, _, country_idx, norm_offset in tqdm(loader, desc="Validating", leave=False):
            images = images.to(device)
            targets_coords = targets_coords.to(device)
            cell_idx = cell_idx.to(device)
            country_idx = country_idx.to(device)

            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                cell_logits, _, country_logits, pred_offset, _, _ = model(images)
                decoded_lat, decoded_lng = decode_coordinates_spherical(
                    cell_logits, centroids_3d_tensor, centroids_latlng_tensor, pred_offset,
                    country_logits=country_logits, fine_to_country=fine_to_country_tensor,
                    top_k=cfg.cell_top_k, temperature=cfg.decoder_temperature, max_offset_km=cfg.max_offset_km,
                    country_weight=cfg.country_logit_weight, local_neighborhood_km=cfg.neighborhood_radius_km,
                    country_top_k=cfg.decoder_country_top_k
                )
                dists_km = haversine_km_torch(decoded_lat, decoded_lng, targets_coords[:, 0], targets_coords[:, 1])

            all_dists.append(dists_km.cpu().numpy())
            pred_country = country_logits.argmax(dim=-1)
            pred_cell = cell_logits.argmax(dim=-1)
            correct_country += (pred_country == country_idx).sum().item()
            correct_cell += (pred_cell == cell_idx).sum().item()
            total += images.size(0)

    all_dists = np.concatenate(all_dists)
    return {
        "median_km": float(np.median(all_dists)),
        "mean_km": float(np.mean(all_dists)),
        "pct_lt_10km": float(np.mean(all_dists <= 10.0) * 100.0),
        "pct_lt_25km": float(np.mean(all_dists <= 25.0) * 100.0),
        "pct_lt_40km": float(np.mean(all_dists <= 40.0) * 100.0),
        "pct_lt_50km": float(np.mean(all_dists <= 50.0) * 100.0),
        "pct_lt_100km": float(np.mean(all_dists <= 100.0) * 100.0),
        "pct_lt_200km": float(np.mean(all_dists <= 200.0) * 100.0),
        "country_acc": float(100.0 * correct_country / total),
        "cell_acc": float(100.0 * correct_cell / total)
    }

# ------------------------------------------------------------------------------
# 7. Retrieval Database Construction & Calibration
# ------------------------------------------------------------------------------

def build_retrieval_database(
    model: RegNetYGeolocationModel,
    train_df: pd.DataFrame,
    img_dir: str,
    fine_centroids: np.ndarray,
    fine_to_country: np.ndarray,
    fine_to_coarse: np.ndarray,
    output_path: str,
    device: torch.device,
    image_size: int = 512,
    batch_size: int = 32
) -> Dict[str, Any]:
    """
    Extracts L2-normalized 128-d embeddings for all training images.
    Saves database with metadata for exact test-time cosine retrieval refinement.
    Never includes validation or test images.
    """
    print(f"\nBuilding retrieval database on {len(train_df):,} training images...")
    model.eval()
    val_tf = get_val_transforms(image_size)
    dataset = GeolocationDataset(
        train_df, img_dir, fine_centroids, fine_to_country, fine_to_coarse, transform=val_tf
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    filenames = train_df['filename'].tolist()
    coords = train_df[['lat', 'lng']].values.astype(np.float32)
    countries = [COUNTRY_TO_IDX[c] for c in train_df['country']]
    fine_cells = dataset.cell_indices.tolist()
    coarse_regions = dataset.coarse_indices.tolist()

    all_embeddings = []
    with torch.no_grad():
        for images, _, _, _, _, _ in tqdm(loader, desc="Extracting Retrieval Embeddings", leave=False):
            images = images.to(device)
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                embed = model.extract_features(images)
            all_embeddings.append(embed.cpu().numpy())

    all_embeddings = np.vstack(all_embeddings).astype(np.float32)
    db = {
        "filenames": filenames,
        "coords": coords,
        "countries": np.array(countries, dtype=np.int64),
        "fine_cells": np.array(fine_cells, dtype=np.int64),
        "coarse_regions": np.array(coarse_regions, dtype=np.int64),
        "embeddings": all_embeddings
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(db, output_path)
    print(f"✓ Retrieval database saved to {output_path} ({len(filenames):,} samples, dim={all_embeddings.shape[1]}).")
    return db

# ------------------------------------------------------------------------------
# 8. Main Training Orchestration
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Image Geolocation Challenge RegNet-Y Training")
    parser.add_argument("--config", type=str, default=None, help="Path to configuration JSON")
    parser.add_argument("--phase", type=str, default="BC", choices=["A", "B", "C", "BC", "all"], help="Training phase to execute (BC: 10 ep Country warm-up + 30 ep Fine)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--split", type=str, default="random", choices=["spatial", "random", "cv"], help="Validation split type")
    parser.add_argument("--fold", type=int, default=None, help="Fold index (0..4) for 5-fold CV")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count for chosen phase")
    args = parser.parse_args()

    cfg = GeolocationConfig.load_json(args.config) if args.config else get_default_config()
    cfg.ensure_directories()

    is_cv = (args.split == "cv" or args.fold is not None)
    fold_idx = args.fold if args.fold is not None else 0
    target_exp_dir = cfg.get_fold_dir(fold_idx) if is_cv else cfg.exp_dir
    os.makedirs(target_exp_dir, exist_ok=True)
    cfg.save_json(os.path.join(target_exp_dir, "config.json"))

    seed_everything(cfg.seed, cfg.deterministic)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Experiment Directory: {target_exp_dir}")
    if is_cv:
        print(f"5-Fold Cross-Validation Active: Training Fold {fold_idx} / {cfg.cv_num_folds}")

    # 1. Dataset Split Manifests
    if is_cv:
        train_manifest_path = cfg.get_fold_train_manifest(fold_idx)
        val_manifest_path = cfg.get_fold_val_manifest(fold_idx)
    elif args.split == "spatial":
        train_manifest_path = cfg.get_path(cfg.spatial_train_manifest)
        val_manifest_path = cfg.get_path(cfg.spatial_val_manifest)
    else:
        train_manifest_path = cfg.get_path(cfg.random_train_manifest)
        val_manifest_path = cfg.get_path(cfg.random_val_manifest)

    if not os.path.exists(train_manifest_path) or not os.path.exists(val_manifest_path):
        raw_df = pd.read_csv(cfg.get_path(cfg.train_csv))
        if is_cv:
            print(f"Generating 5-fold country-stratified CV manifests (seed={cfg.seed})...")
            cv_folds = create_stratified_cv_splits(raw_df, n_splits=cfg.cv_num_folds, seed=cfg.seed)
            os.makedirs(cfg.splits_dir, exist_ok=True)
            for f_i, (t_f, v_f) in enumerate(cv_folds):
                t_f.to_csv(cfg.get_fold_train_manifest(f_i), index=False)
                v_f.to_csv(cfg.get_fold_val_manifest(f_i), index=False)
                print(f"✓ Saved Fold {f_i} manifests: Train={len(t_f):,}, Val={len(v_f):,}")
        elif args.split == "spatial":
            print(f"Generating {args.split} split manifests...")
            train_df, val_df = create_spatial_split(raw_df, group_radius_km=cfg.spatial_group_radius_km, val_ratio=cfg.val_ratio, seed=cfg.seed)
            os.makedirs(os.path.dirname(train_manifest_path), exist_ok=True)
            train_df.to_csv(train_manifest_path, index=False)
            val_df.to_csv(val_manifest_path, index=False)
            print(f"Saved manifests: Train={len(train_df):,}, Val={len(val_df):,}")
            diag = compute_split_diagnostics(train_df, val_df, cfg.get_path(cfg.train_img_dir))
            diag_path = os.path.join(cfg.splits_dir, f"{args.split}_split_diagnostics.json")
            with open(diag_path, "w", encoding="utf-8") as f:
                json.dump(diag, f, indent=2)
        else:
            print(f"Generating {args.split} split manifests...")
            train_df, val_df = create_random_split(raw_df, val_ratio=cfg.val_ratio, seed=cfg.seed)
            os.makedirs(os.path.dirname(train_manifest_path), exist_ok=True)
            train_df.to_csv(train_manifest_path, index=False)
            val_df.to_csv(val_manifest_path, index=False)
            print(f"Saved manifests: Train={len(train_df):,}, Val={len(val_df):,}")

    train_df = pd.read_csv(train_manifest_path)
    val_df = pd.read_csv(val_manifest_path)

    # 2. Geographic Hierarchy (Fitted strictly on this fold's training data - zero leak)
    hierarchy_dir = target_exp_dir
    hierarchy_meta_path = os.path.join(hierarchy_dir, "hierarchy_metadata.json")
    loaded = False
    if os.path.exists(hierarchy_meta_path):
        try:
            fine_centroids, coarse_centroids, fine_to_country, fine_to_coarse, _ = load_geographic_hierarchy(
                hierarchy_dir, train_manifest_path, expected_num_fine_cells=cfg.num_fine_cells
            )
            loaded = True
        except ValueError as e:
            print(f"Existing hierarchy metadata does not match current configuration ({e}).\nRebuilding hierarchy...")
    if not loaded:
        hierarchy_dict = build_geographic_hierarchy(
            train_df,
            num_fine_cells=cfg.num_fine_cells,
            num_coarse_regions=cfg.num_coarse_regions,
            seed=cfg.seed,
            output_dir=hierarchy_dir,
            config_hash=cfg.compute_config_hash(),
            manifest_path=train_manifest_path
        )
        fine_centroids = hierarchy_dict["fine_centroids"]
        coarse_centroids = hierarchy_dict["coarse_centroids"]
        fine_to_country = hierarchy_dict["fine_to_country"]
        fine_to_coarse = hierarchy_dict["fine_to_coarse"]

    centroids_3d_tensor = torch.tensor(coords_to_3d(fine_centroids[:, 0], fine_centroids[:, 1]), dtype=torch.float32, device=device)
    centroids_latlng_tensor = torch.tensor(fine_centroids, dtype=torch.float32, device=device)
    fine_to_country_tensor = torch.tensor(fine_to_country, dtype=torch.long, device=device)

    # 3. Model Construction (Strictly pretrained=False, <= 5M params)
    model = get_model(
        num_fine_cells=cfg.num_fine_cells,
        num_coarse_regions=cfg.num_coarse_regions,
        num_countries=cfg.num_countries,
        embedding_dim=cfg.embedding_dim,
        max_offset_km=cfg.max_offset_km,
        include_cartesian_head=cfg.include_cartesian_head
    ).to(device)

    ema = EMA(model, decay=cfg.ema_decay)
    scaler = create_grad_scaler('cuda', enabled=device.type == 'cuda' and cfg.use_amp)

    # Resume checkpoint if specified
    start_epoch = 0
    best_val_median = float('inf')
    checkpoint = None
    if args.resume:
        print(f"Resuming training from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "ema_state_dict" in checkpoint:
            ema.load_state_dict(checkpoint["ema_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_median = checkpoint.get("best_val_median", float('inf'))
        print(f"Resumed at epoch {start_epoch} with best validation median: {best_val_median:.2f} km")

    # Dataloaders
    train_tf = get_train_transforms(cfg.image_size, allow_hflip=cfg.horizontal_flip)
    val_tf = get_val_transforms(cfg.image_size)

    # Generator for reproducible DataLoader shuffling
    g = torch.Generator()
    g.manual_seed(cfg.seed)

    val_dataset = GeolocationDataset(
        val_df, cfg.get_path(cfg.train_img_dir), fine_centroids, fine_to_country, fine_to_coarse,
        max_offset_km=cfg.max_offset_km, transform=val_tf
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory, worker_init_fn=seed_worker, generator=g
    )

    if args.phase in ("all", "BC"):
        phases_to_run = ["B", "C"]
        num_epochs_a = 0
        num_epochs_b = cfg.phase_b_epochs
        num_epochs_c = max(10, (args.epochs or (cfg.phase_b_epochs + cfg.phase_c_epochs)) - num_epochs_b)
    elif args.phase == "B":
        phases_to_run = ["B"]
        num_epochs_a = 0
        num_epochs_b = args.epochs or cfg.phase_b_epochs
        num_epochs_c = 0
    elif args.phase == "C":
        phases_to_run = ["C"]
        num_epochs_a = 0
        num_epochs_b = 0
        num_epochs_c = args.epochs or cfg.phase_c_epochs
    elif args.phase == "A":
        phases_to_run = ["A"]
        num_epochs_a = args.epochs or cfg.phase_a_epochs
        num_epochs_b, num_epochs_c = 0, 0
    else:
        phases_to_run = ["C"]
        num_epochs_a = 0
        num_epochs_b = 0
        num_epochs_c = args.epochs or cfg.phase_c_epochs

    total_planned_epochs = sum([
        num_epochs_a if "A" in phases_to_run else 0,
        num_epochs_b if "B" in phases_to_run else 0,
        num_epochs_c if "C" in phases_to_run else 0
    ])
    print(f"Scheduled Training Epochs: Total={total_planned_epochs} (Phase A: {num_epochs_a if 'A' in phases_to_run else 0}, Phase B: {num_epochs_b if 'B' in phases_to_run else 0}, Phase C: {num_epochs_c if 'C' in phases_to_run else 0})")

    # --- Phase A: Metric Representation Learning ---
    if "A" in phases_to_run:
        print("\n" + "=" * 70 + "\nSTARTING PHASE A: METRIC REPRESENTATION LEARNING\n" + "=" * 70)
        set_phase_trainable(model, "A")
        train_ds_a = TwoViewGeolocationDataset(
            train_df, cfg.get_path(cfg.train_img_dir), fine_centroids, fine_to_country, transform=train_tf
        )
        loader_a = DataLoader(
            train_ds_a, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory, worker_init_fn=seed_worker, generator=g
        )
        crit_contrastive = GeographicContrastiveLoss(temperature=cfg.metric_temperature, dist_scale=cfg.metric_dist_scale)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(trainable_params, lr=cfg.phase_a_lr, weight_decay=cfg.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs_a, eta_min=cfg.phase_a_lr * cfg.min_lr_ratio)

        for ep in range(num_epochs_a):
            loss = train_phase_a_epoch(model, loader_a, optimizer, crit_contrastive, scaler, device, cfg.grad_accum_steps, cfg.grad_clip_norm)
            scheduler.step()
            print(f"[Phase A] Epoch {ep+1:02d}/{num_epochs_a:02d} | Contrastive Loss: {loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        ckpt_a_path = os.path.join(target_exp_dir, "checkpoint_phase_a.pth")
        torch.save({
            "phase": "A",
            "model_state_dict": model.state_dict(),
            "config": cfg.to_dict()
        }, ckpt_a_path)
        print(f"✓ Saved Phase A representation checkpoint: {ckpt_a_path}")

    # --- Phase B: Country and Coarse-Region Heads ---
    if "B" in phases_to_run:
        print("\n" + "=" * 70 + "\nSTARTING PHASE B: COUNTRY & COARSE REGION SUPERVISION\n" + "=" * 70)
        if "A" not in phases_to_run and not args.resume:
            ckpt_a_path = os.path.join(target_exp_dir, "checkpoint_phase_a.pth")
            if os.path.exists(ckpt_a_path):
                print(f"Loading weights from Phase A checkpoint: {ckpt_a_path}")
                ckpt_a = torch.load(ckpt_a_path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt_a["model_state_dict"])

        set_phase_trainable(model, "B")
        train_ds = GeolocationDataset(
            train_df, cfg.get_path(cfg.train_img_dir), fine_centroids, fine_to_country, fine_to_coarse,
            max_offset_km=cfg.max_offset_km, transform=train_tf
        )
        loader_b = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory, worker_init_fn=seed_worker, generator=g
        )
        backbone_params = list(model.backbone.parameters()) + list(model.gem_pool.parameters())
        head_params = [p for n, p in model.named_parameters() if 'backbone' not in n and 'gem_pool' not in n and p.requires_grad]
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': cfg.phase_b_backbone_lr},
            {'params': head_params, 'lr': cfg.phase_b_lr}
        ], weight_decay=cfg.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs_b, eta_min=cfg.phase_b_backbone_lr * cfg.min_lr_ratio)

        for ep in range(num_epochs_b):
            loss, cntry_acc, coarse_acc = train_phase_b_epoch(model, loader_b, optimizer, scaler, device, cfg.grad_accum_steps, cfg.grad_clip_norm)
            scheduler.step()
            print(f"[Phase B] Epoch {ep+1:02d}/{num_epochs_b:02d} | Loss: {loss:.4f} | Country Acc: {cntry_acc:.2f}% | Coarse Acc: {coarse_acc:.2f}%")

        ckpt_b_path = os.path.join(target_exp_dir, "checkpoint_phase_b.pth")
        torch.save({
            "phase": "B",
            "model_state_dict": model.state_dict(),
            "config": cfg.to_dict()
        }, ckpt_b_path)
        print(f"✓ Saved Phase B country/coarse checkpoint: {ckpt_b_path}")

    # --- Phase C: Joint Localization ---
    if "C" in phases_to_run:
        print("\n" + "=" * 70 + "\nSTARTING PHASE C: FULL MULTI-TASK JOINT LOCALIZATION\n" + "=" * 70)
        if "B" not in phases_to_run and not args.resume:
            ckpt_b_path = os.path.join(target_exp_dir, "checkpoint_phase_b.pth")
            if os.path.exists(ckpt_b_path):
                print(f"Loading weights from Phase B checkpoint: {ckpt_b_path}")
                ckpt_b = torch.load(ckpt_b_path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt_b["model_state_dict"])
        if not (args.resume and checkpoint is not None and "ema_state_dict" in checkpoint):
            ema = EMA(model, decay=cfg.ema_decay)
        else:
            ema.load_state_dict(checkpoint["ema_state_dict"])
            print("  ✓ Restored EMA weights from resumed checkpoint")

        set_phase_trainable(model, "C")
        train_ds = GeolocationDataset(
            train_df, cfg.get_path(cfg.train_img_dir), fine_centroids, fine_to_country, fine_to_coarse,
            max_offset_km=cfg.max_offset_km, transform=train_tf
        )
        loader_c = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory, worker_init_fn=seed_worker, generator=g
        )

        backbone_params = list(model.backbone.parameters()) + list(model.gem_pool.parameters())
        head_params = [p for n, p in model.named_parameters() if 'backbone' not in n and 'gem_pool' not in n]

        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': cfg.phase_c_backbone_lr},
            {'params': head_params, 'lr': cfg.phase_c_head_lr}
        ], weight_decay=cfg.weight_decay)

        def lr_lambda(ep: int) -> float:
            warmup_epochs = max(1, cfg.warmup_epochs)
            if ep < warmup_epochs:
                return 0.3 + 0.7 * float(ep + 1) / float(warmup_epochs + 1)
            progress = float(ep - warmup_epochs) / float(max(1, num_epochs_c - 1 - warmup_epochs))
            return cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        if args.resume and checkpoint is not None:
            if "optimizer_state_dict" in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    print("  ✓ Restored optimizer state from checkpoint")
                except Exception as e:
                    print(f"  Notice: Could not restore optimizer state: {e}")

            if "scheduler_state_dict" in checkpoint:
                try:
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                    print("  ✓ Restored scheduler state from checkpoint")
                except Exception as e:
                    print(f"  Notice: Could not restore scheduler state: {e}")

            if "scaler_state_dict" in checkpoint and scaler is not None:
                try:
                    scaler.load_state_dict(checkpoint["scaler_state_dict"])
                    print("  ✓ Restored scaler state from checkpoint")
                except Exception as e:
                    print(f"  Notice: Could not restore scaler state: {e}")

        metrics_log = []
        best_ckpt_path = os.path.join(target_exp_dir, cfg.checkpoint_best_name)
        last_ckpt_path = os.path.join(target_exp_dir, cfg.checkpoint_last_name)

        for ep in range(start_epoch, num_epochs_c):
            t_res = train_phase_c_epoch(
                model, loader_c, optimizer, scaler, ema,
                centroids_3d_tensor, centroids_latlng_tensor, fine_to_country_tensor,
                ep, cfg, device
            )
            scheduler.step()

            # Evaluate with EMA weights
            ema.apply_to(model)
            val_res = validate_epoch(
                model, val_loader, centroids_3d_tensor, centroids_latlng_tensor, fine_to_country_tensor, cfg, device
            )
            ema.restore(model)

            ep_summary = {
                "epoch": ep + 1,
                "train_loss": round(t_res["train_loss"], 4),
                **{k: round(v, 2) for k, v in val_res.items()}
            }
            metrics_log.append(ep_summary)

            print(
                f"[Phase C Ep {ep+1:02d}/{num_epochs_c:02d}] "
                f"Train Loss: {t_res['train_loss']:.4f} | "
                f"Val Median: {val_res['median_km']:.1f}km (Mean: {val_res['mean_km']:.1f}km) | "
                f"<200km: {val_res['pct_lt_200km']:.1f}% | Ctry Acc: {val_res['country_acc']:.1f}%"
            )

            # Save best checkpoint
            if val_res["median_km"] < best_val_median:
                best_val_median = val_res["median_km"]
                torch.save({
                    "epoch": ep,
                    "model_state_dict": ema.state_dict(),
                    "ema_state_dict": ema.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best_val_median": best_val_median,
                    "config": cfg.to_dict()
                }, best_ckpt_path)
                print(f"  --> Saved new best checkpoint to {best_ckpt_path} (Median: {best_val_median:.2f} km)")

            # Save last checkpoint
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_val_median": best_val_median,
                "config": cfg.to_dict()
            }, last_ckpt_path)

        # Save training metrics JSON
        metrics_path = os.path.join(target_exp_dir, "training_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_log, f, indent=2)

    # 4. Build Retrieval Database with Best Checkpoint
    best_ckpt_path = os.path.join(target_exp_dir, cfg.checkpoint_best_name)
    if os.path.exists(best_ckpt_path):
        print("\nLoading best model checkpoint for retrieval database generation...")
        best_model, _ = load_saved_model(best_ckpt_path, device=device)
        retrieval_db_path = os.path.join(target_exp_dir, cfg.retrieval_db_name)
        build_retrieval_database(
            best_model, train_df, cfg.get_path(cfg.train_img_dir),
            fine_centroids, fine_to_country, fine_to_coarse,
            retrieval_db_path, device=device, image_size=cfg.image_size, batch_size=cfg.batch_size
        )

        # 5. Generate and Save Out-of-Fold Validation Predictions
        print(f"\nGenerating validation predictions using best checkpoint...")
        best_model.eval()
        val_pred_lats, val_pred_lngs = [], []
        with torch.no_grad():
            for images, _, _, _, _, _ in tqdm(val_loader, desc="Validating", leave=False):
                images = images.to(device)
                cell_logits, _, country_logits, pred_offset, _, _ = best_model(images)
                p_lats, p_lngs = decode_coordinates_spherical(
                    cell_logits,
                    centroids_3d_tensor, centroids_latlng_tensor,
                    pred_offset,
                    country_logits=country_logits,
                    fine_to_country=fine_to_country_tensor,
                    top_k=cfg.cell_top_k,
                    temperature=cfg.decoder_temperature,
                    max_offset_km=cfg.max_offset_km,
                    country_weight=cfg.country_logit_weight,
                    local_neighborhood_km=cfg.neighborhood_radius_km,
                    country_top_k=cfg.decoder_country_top_k
                )
                val_pred_lats.extend(p_lats.cpu().numpy().tolist())
                val_pred_lngs.extend(p_lngs.cpu().numpy().tolist())

        val_out_df = val_df.copy()
        val_out_df['pred_lat'] = val_pred_lats
        val_out_df['pred_lng'] = val_pred_lngs
        val_out_df['error_km'] = [
            haversine_km(plat, plng, tlat, tlng)
            for plat, plng, tlat, tlng in zip(val_pred_lats, val_pred_lngs, val_df['lat'].values, val_df['lng'].values)
        ]
        val_preds_path = os.path.join(target_exp_dir, "val_predictions.csv")
        val_out_df.to_csv(val_preds_path, index=False)
        med_err = float(val_out_df['error_km'].median())
        mean_err = float(val_out_df['error_km'].mean())
        lt200 = float((val_out_df['error_km'] < 200).mean() * 100)
        lt750 = float((val_out_df['error_km'] < 750).mean() * 100)

        print("\n" + "=" * 70)
        print(f"FOLD VALIDATION SUMMARY ({'Fold ' + str(fold_idx) if is_cv else args.split}):")
        print(f"  Validation Samples: {len(val_out_df):,}")
        print(f"  Median Distance:    {med_err:.2f} km")
        print(f"  Mean Distance:      {mean_err:.2f} km")
        print(f"  < 200 km (city):    {lt200:.1f}%")
        print(f"  < 750 km (country): {lt750:.1f}%")
        print(f"✓ Saved validation predictions to: {val_preds_path}")
        print("=" * 70)

        # Copy checkpoint to root exp_dir if none exists yet
        root_best = os.path.join(cfg.exp_dir, cfg.checkpoint_best_name)
        if not os.path.exists(root_best) and target_exp_dir != cfg.exp_dir:
            import shutil
            shutil.copy2(best_ckpt_path, root_best)
            print(f"✓ Copied best model checkpoint to primary experiment dir: {root_best}")

    print("\n" + "=" * 70)
    print("TRAINING PIPELINE COMPLETE")
    print(f"Artifacts and logs saved in: {target_exp_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()
