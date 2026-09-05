import os
import json
import hashlib
from typing import Tuple, List, Dict, Optional, Any, Union
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.cluster import KMeans

COUNTRIES = [
    'Belarus', 'Finland', 'France', 'Germany', 'Iceland', 'Italy',
    'Norway', 'Poland', 'Spain', 'Sweden', 'Turkey', 'United_Kingdom'
]
COUNTRY_TO_IDX = {c: i for i, c in enumerate(COUNTRIES)}
IDX_TO_COUNTRY = {i: c for i, c in enumerate(COUNTRIES)}

# Mean and std for standard normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Earth radius in km (WGS84 mean)
EARTH_RADIUS_KM = 6371.0088
KM_PER_DEG_LAT = 111.195  # approximate km per degree latitude

# ------------------------------------------------------------------------------
# 1. Geographic Coordinate Mathematics & Numerical Stability
# ------------------------------------------------------------------------------

def coords_to_3d(lat_deg: Union[float, np.ndarray], lng_deg: Union[float, np.ndarray]) -> np.ndarray:
    """Converts lat/lng in degrees to 3D unit sphere Cartesian coordinates (x, y, z)."""
    lat_rad = np.radians(lat_deg)
    lng_rad = np.radians(lng_deg)
    x = np.cos(lat_rad) * np.cos(lng_rad)
    y = np.cos(lat_rad) * np.sin(lng_rad)
    z = np.sin(lat_rad)
    if isinstance(lat_deg, np.ndarray):
        return np.column_stack([x, y, z])
    return np.array([x, y, z], dtype=np.float32)

def cartesian_to_latlng(xyz: np.ndarray) -> np.ndarray:
    """Converts 3D Cartesian coordinates (x, y, z) back to lat/lng in decimal degrees."""
    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    xyz_norm = xyz / np.maximum(norm, 1e-8)
    if xyz_norm.ndim == 1:
        z = np.clip(xyz_norm[2], -1.0, 1.0)
        lat = np.degrees(np.arcsin(z))
        lng = np.degrees(np.arctan2(xyz_norm[1], xyz_norm[0]))
        return np.array([lat, lng], dtype=np.float32)
    else:
        z = np.clip(xyz_norm[:, 2], -1.0, 1.0)
        lat = np.degrees(np.arcsin(z))
        lng = np.degrees(np.arctan2(xyz_norm[:, 1], xyz_norm[:, 0]))
        return np.column_stack([lat, lng])

def coords_to_3d_torch(lat_deg: torch.Tensor, lng_deg: torch.Tensor) -> torch.Tensor:
    """Converts PyTorch lat/lng in degrees to 3D unit sphere Cartesian coordinates."""
    lat_rad = torch.deg2rad(lat_deg)
    lng_rad = torch.deg2rad(lng_deg)
    x = torch.cos(lat_rad) * torch.cos(lng_rad)
    y = torch.cos(lat_rad) * torch.sin(lng_rad)
    z = torch.sin(lat_rad)
    return torch.stack([x, y, z], dim=-1)

def cartesian_to_latlng_torch(xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Converts PyTorch 3D Cartesian coordinates to (lat, lng) in decimal degrees."""
    xyz_norm = F_normalize(xyz.float(), p=2, dim=-1)
    lat = torch.rad2deg(torch.asin(torch.clamp(xyz_norm[..., 2], -1.0 + 1e-6, 1.0 - 1e-6)))
    lng = torch.rad2deg(torch.atan2(xyz_norm[..., 1], xyz_norm[..., 0]))
    return lat, lng

def F_normalize(tensor: torch.Tensor, p: float = 2.0, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    norm = tensor.norm(p=p, dim=dim, keepdim=True).clamp(min=eps)
    return tensor / norm

def haversine_km(
    lat1: Union[float, np.ndarray],
    lng1: Union[float, np.ndarray],
    lat2: Union[float, np.ndarray],
    lng2: Union[float, np.ndarray],
    R: float = EARTH_RADIUS_KM
) -> Union[float, np.ndarray]:
    """Numerically stable Haversine distance in km between coordinates in decimal degrees."""
    lat1_r, lng1_r, lat2_r, lng2_r = map(np.radians, (lat1, lng1, lat2, lng2))
    dlat = lat2_r - lat1_r
    dlng = lng2_r - lng1_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlng / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    return 2.0 * R * np.arcsin(np.sqrt(a))

def haversine_km_torch(
    lat1: torch.Tensor,
    lng1: torch.Tensor,
    lat2: torch.Tensor,
    lng2: torch.Tensor,
    R: float = EARTH_RADIUS_KM
) -> torch.Tensor:
    """Differentiable and NaN-safe PyTorch Haversine distance in km."""
    lat1_r = torch.deg2rad(lat1.float())
    lng1_r = torch.deg2rad(lng1.float())
    lat2_r = torch.deg2rad(lat2.float())
    lng2_r = torch.deg2rad(lng2.float())
    dlat = lat2_r - lat1_r
    dlng = lng2_r - lng1_r
    a = torch.sin(dlat / 2.0) ** 2 + torch.cos(lat1_r) * torch.cos(lat2_r) * torch.sin(dlng / 2.0) ** 2
    # Clamping avoids infinite gradients when points coincide exactly (distance = 0)
    a_clamped = torch.clamp(a, 1e-12, 1.0 - 1e-7)
    return 2.0 * R * torch.asin(torch.sqrt(a_clamped))

# ------------------------------------------------------------------------------
# 2. Local Tangent-Plane East/North Offsets
# ------------------------------------------------------------------------------

def coords_to_offset_km(
    lat: Union[float, np.ndarray],
    lng: Union[float, np.ndarray],
    c_lat: Union[float, np.ndarray],
    c_lng: Union[float, np.ndarray]
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Converts target coordinates relative to cell centroid into North and East displacement in km.
    Includes proper longitude wraparound and high-latitude cosine stabilization.
    """
    delta_lat = lat - c_lat
    north_km = delta_lat * KM_PER_DEG_LAT

    # Longitude wraparound handling: difference mapped to [-180, 180]
    delta_lng = (lng - c_lng + 180.0) % 360.0 - 180.0
    
    # Use centroid latitude for local projection; clamp cosine for polar stability
    if isinstance(c_lat, np.ndarray):
        cos_lat = np.clip(np.cos(np.radians(c_lat)), 0.15, 1.0)
    else:
        cos_lat = max(float(np.cos(np.radians(c_lat))), 0.15)
        
    east_km = delta_lng * KM_PER_DEG_LAT * cos_lat
    return north_km, east_km

def offset_km_to_coords(
    c_lat: Union[float, np.ndarray],
    c_lng: Union[float, np.ndarray],
    north_km: Union[float, np.ndarray],
    east_km: Union[float, np.ndarray]
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Converts local tangent plane North and East displacement in km back to (lat, lng) degrees.
    Uses centroid latitude c_lat as tangent plane origin for exact linear invertibility.
    """
    lat = c_lat + north_km / KM_PER_DEG_LAT
    if isinstance(lat, np.ndarray):
        lat = np.clip(lat, -90.0, 90.0)
    else:
        lat = max(min(float(lat), 90.0), -90.0)

    if isinstance(c_lat, np.ndarray):
        cos_lat = np.clip(np.cos(np.radians(c_lat)), 0.15, 1.0)
    else:
        cos_lat = max(float(np.cos(np.radians(c_lat))), 0.15)

    lng = c_lng + east_km / (KM_PER_DEG_LAT * cos_lat)
    lng = (lng + 180.0) % 360.0 - 180.0
    return lat, lng

def offset_km_to_coords_torch(
    c_lat: torch.Tensor,
    c_lng: torch.Tensor,
    north_km: torch.Tensor,
    east_km: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch version of offset_km_to_coords with differentiable operations."""
    c_lat_f = c_lat.float()
    c_lng_f = c_lng.float()
    north_f = north_km.float()
    east_f = east_km.float()
    lat = c_lat_f + north_f / KM_PER_DEG_LAT
    lat = torch.clamp(lat, -90.0, 90.0)
    cos_lat = torch.clamp(torch.cos(torch.deg2rad(c_lat_f)), min=0.15)
    lng = c_lng_f + east_f / (KM_PER_DEG_LAT * cos_lat)
    lng = (lng + 180.0) % 360.0 - 180.0
    return lat, lng


def normalize_offset(
    north_km: Union[float, np.ndarray, torch.Tensor],
    east_km: Union[float, np.ndarray, torch.Tensor],
    max_offset_km: float = 50.0
) -> Tuple[Any, Any]:
    """Normalizes km offsets to [-1, 1] range via clamping."""
    if isinstance(north_km, torch.Tensor):
        norm_north = torch.clamp(north_km / max_offset_km, -1.0, 1.0)
        norm_east = torch.clamp(east_km / max_offset_km, -1.0, 1.0)
    else:
        norm_north = np.clip(north_km / max_offset_km, -1.0, 1.0)
        norm_east = np.clip(east_km / max_offset_km, -1.0, 1.0)
    return norm_north, norm_east

def unnormalize_offset(
    norm_north: Union[float, np.ndarray, torch.Tensor],
    norm_east: Union[float, np.ndarray, torch.Tensor],
    max_offset_km: float = 50.0
) -> Tuple[Any, Any]:
    """Unnormalizes [-1, 1] offsets back to km displacement."""
    return norm_north * max_offset_km, norm_east * max_offset_km

# ------------------------------------------------------------------------------
# 3. Geographic Hierarchy Construction & Verification
# ------------------------------------------------------------------------------

def compute_file_hash(filepath: str) -> str:
    """Computes SHA256 hash of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()

def build_geographic_hierarchy(
    train_df: pd.DataFrame,
    num_fine_cells: int = 384,
    num_coarse_regions: int = 48,
    seed: int = 42,
    output_dir: Optional[str] = None,
    config_hash: Optional[str] = None,
    manifest_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Builds a country-aware geographic hierarchy using strictly training samples:
    - Each fine cell belongs to exactly one country and one coarse region.
    - Default: 384 fine cells (32 per country) and 48 coarse regions (4 per country).
    - Centroids are computed in 3D spherical space via K-Means and converted back to lat/lng.
    """
    num_countries = len(COUNTRIES)
    assert num_fine_cells % num_countries == 0, f"num_fine_cells ({num_fine_cells}) must be divisible by {num_countries}"
    assert num_coarse_regions % num_countries == 0, f"num_coarse_regions ({num_coarse_regions}) must be divisible by {num_countries}"
    
    fine_per_country = num_fine_cells // num_countries
    coarse_per_country = num_coarse_regions // num_countries

    all_fine_centroids = []
    all_coarse_centroids = []
    fine_to_country = []
    fine_to_coarse = []

    coarse_offset = 0

    for country_idx, country_name in enumerate(COUNTRIES):
        c_df = train_df[train_df['country'] == country_name].reset_index(drop=True)
        if len(c_df) < fine_per_country:
            raise ValueError(f"Country {country_name} has only {len(c_df)} images, fewer than {fine_per_country} requested fine cells!")

        coords_deg = c_df[['lat', 'lng']].values
        coords_3d = coords_to_3d(coords_deg[:, 0], coords_deg[:, 1])

        # 1. Coarse clustering for this country
        kmeans_coarse = KMeans(n_clusters=coarse_per_country, random_state=seed + country_idx, n_init=10)
        kmeans_coarse.fit(coords_3d)
        coarse_centroids_c = cartesian_to_latlng(kmeans_coarse.cluster_centers_)
        all_coarse_centroids.append(coarse_centroids_c)

        # 2. Fine clustering for this country
        kmeans_fine = KMeans(n_clusters=fine_per_country, random_state=seed + country_idx * 100, n_init=10)
        kmeans_fine.fit(coords_3d)
        fine_centroids_c = cartesian_to_latlng(kmeans_fine.cluster_centers_)
        all_fine_centroids.append(fine_centroids_c)

        # Map each fine centroid to its nearest coarse centroid in 3D
        fine_3d = coords_to_3d(fine_centroids_c[:, 0], fine_centroids_c[:, 1])
        coarse_3d = coords_to_3d(coarse_centroids_c[:, 0], coarse_centroids_c[:, 1])
        # dist matrix: (fine_per_country, coarse_per_country)
        dists = np.linalg.norm(fine_3d[:, np.newaxis, :] - coarse_3d[np.newaxis, :, :], axis=-1)
        nearest_coarse_local = np.argmin(dists, axis=-1)
        nearest_coarse_global = nearest_coarse_local + coarse_offset

        fine_to_country.extend([country_idx] * fine_per_country)
        fine_to_coarse.extend(nearest_coarse_global.tolist())
        coarse_offset += coarse_per_country

    fine_centroids = np.vstack(all_fine_centroids).astype(np.float32)
    coarse_centroids = np.vstack(all_coarse_centroids).astype(np.float32)
    fine_to_country = np.array(fine_to_country, dtype=np.int64)
    fine_to_coarse = np.array(fine_to_coarse, dtype=np.int64)

    # Compute training manifest and label hashes
    manifest_hash = compute_file_hash(manifest_path) if manifest_path and os.path.exists(manifest_path) else "direct_df"
    labels_content = train_df[['filename', 'lat', 'lng']].to_csv(index=False).encode('utf-8')
    labels_hash = hashlib.sha256(labels_content).hexdigest()

    metadata = {
        "num_fine_cells": num_fine_cells,
        "num_coarse_regions": num_coarse_regions,
        "num_countries": num_countries,
        "country_list": COUNTRIES,
        "seed": seed,
        "config_hash": config_hash or "",
        "train_manifest_hash": manifest_hash,
        "train_labels_hash": labels_hash,
        "num_training_samples": len(train_df)
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, "fine_centroids.npy"), fine_centroids)
        np.save(os.path.join(output_dir, "coarse_centroids.npy"), coarse_centroids)
        np.save(os.path.join(output_dir, "fine_to_country.npy"), fine_to_country)
        np.save(os.path.join(output_dir, "fine_to_coarse.npy"), fine_to_coarse)
        with open(os.path.join(output_dir, "hierarchy_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    return {
        "fine_centroids": fine_centroids,
        "coarse_centroids": coarse_centroids,
        "fine_to_country": fine_to_country,
        "fine_to_coarse": fine_to_coarse,
        "metadata": metadata
    }

def load_geographic_hierarchy(
    artifacts_dir: str,
    expected_train_manifest: Optional[str] = None,
    expected_num_fine_cells: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Loads and validates saved geographic hierarchy artifacts from artifacts_dir.
    Refuses to load incompatible artifacts.
    """
    metadata_path = os.path.join(artifacts_dir, "hierarchy_metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Hierarchy metadata not found at {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if expected_num_fine_cells is not None and metadata.get("num_fine_cells") != expected_num_fine_cells:
        raise ValueError(
            f"Fine cell count mismatch: metadata has {metadata.get('num_fine_cells')} cells, "
            f"expected {expected_num_fine_cells} cells."
        )

    if expected_train_manifest and os.path.exists(expected_train_manifest):
        current_hash = compute_file_hash(expected_train_manifest)
        if metadata.get("train_manifest_hash") and metadata["train_manifest_hash"] != current_hash:
            raise ValueError(
                f"Centroid metadata mismatch! Metadata training manifest hash {metadata['train_manifest_hash']} "
                f"does not match current manifest hash {current_hash}."
            )

    fine_centroids = np.load(os.path.join(artifacts_dir, "fine_centroids.npy"))
    coarse_centroids = np.load(os.path.join(artifacts_dir, "coarse_centroids.npy"))
    fine_to_country = np.load(os.path.join(artifacts_dir, "fine_to_country.npy"))
    fine_to_coarse = np.load(os.path.join(artifacts_dir, "fine_to_coarse.npy"))

    if len(fine_centroids) != metadata["num_fine_cells"]:
        raise ValueError(f"Fine centroids count {len(fine_centroids)} != expected {metadata['num_fine_cells']}")
    if len(coarse_centroids) != metadata["num_coarse_regions"]:
        raise ValueError(f"Coarse centroids count {len(coarse_centroids)} != expected {metadata['num_coarse_regions']}")

    return fine_centroids, coarse_centroids, fine_to_country, fine_to_coarse, metadata

# ------------------------------------------------------------------------------
# 4. Validation Splits: Stratified Random & Spatially Grouped
# ------------------------------------------------------------------------------

def create_random_split(
    df: pd.DataFrame,
    val_ratio: float = 0.20,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Creates a deterministic 80/20 train/validation split stratified by country."""
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(splitter.split(df, df['country']))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    return train_df, val_df

def create_stratified_cv_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Creates deterministic 5-fold cross-validation splits stratified by country.
    
    Adheres strictly to the challenge benchmark rule:
    'Score with 5-fold cross-validation on train_labels.csv, stratified by country.
     Report the metrics pooled over all out-of-fold predictions.'
    
    Returns:
        List of (train_df, val_df) tuples for fold 0 through fold (n_splits - 1).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for train_idx, val_idx in skf.split(df, df['country']):
        train_fold = df.iloc[train_idx].reset_index(drop=True)
        val_fold = df.iloc[val_idx].reset_index(drop=True)
        folds.append((train_fold, val_fold))
    return folds

def create_spatial_split(
    df: pd.DataFrame,
    group_radius_km: float = 35.0,
    val_ratio: float = 0.20,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates a geographically grouped validation split:
    - Samples within group_radius_km are clustered into spatial groups.
    - An entire spatial group is assigned either to train or to validation (zero geographic group leak).
    - Balances country representation by partitioning groups per country.
    """
    np.random.seed(seed)
    all_train_indices = []
    all_val_indices = []

    for country in COUNTRIES:
        c_sub = df[df['country'] == country]
        c_indices = c_sub.index.values
        if len(c_indices) == 0:
            continue

        lats = c_sub['lat'].values
        lngs = c_sub['lng'].values
        coords_3d = coords_to_3d(lats, lngs)

        # Simple greedy spatial grouping within country
        unassigned = set(range(len(c_indices)))
        groups = []

        while unassigned:
            seed_idx = unassigned.pop()
            seed_vec = coords_3d[seed_idx]
            group = [seed_idx]
            
            # Find all unassigned within group_radius_km
            to_remove = []
            for other_idx in unassigned:
                d = haversine_km(lats[seed_idx], lngs[seed_idx], lats[other_idx], lngs[other_idx])
                if d <= group_radius_km:
                    group.append(other_idx)
                    to_remove.append(other_idx)
            for r in to_remove:
                unassigned.remove(r)
            groups.append(group)

        # Shuffle groups deterministically
        perm = np.random.permutation(len(groups))
        shuffled_groups = [groups[p] for p in perm]

        # Greedy allocation to achieve target val_ratio
        target_val_samples = int(len(c_indices) * val_ratio)
        current_val_samples = 0
        val_group_indices = []
        train_group_indices = []

        for grp in shuffled_groups:
            if current_val_samples + len(grp) <= target_val_samples or current_val_samples == 0:
                val_group_indices.extend(grp)
                current_val_samples += len(grp)
            else:
                train_group_indices.extend(grp)

        all_train_indices.extend(c_indices[train_group_indices])
        all_val_indices.extend(c_indices[val_group_indices])

    train_df = df.loc[all_train_indices].reset_index(drop=True)
    val_df = df.loc[all_val_indices].reset_index(drop=True)
    return train_df, val_df

def compute_split_diagnostics(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    img_dir: str
) -> Dict[str, Any]:
    """
    Reports validation split health and separation metrics:
    - Train and val sample sizes
    - Images per country
    - Duplicate coordinates / filenames
    - Missing files
    - Minimum geographic train-to-val distance
    - Percentages within 10, 25, 40, 50, 100 km
    """
    train_files = set(train_df['filename'])
    val_files = set(val_df['filename'])
    file_overlap = len(train_files.intersection(val_files))

    missing_train = sum(not os.path.exists(os.path.join(img_dir, f)) for f in train_df['filename'])
    missing_val = sum(not os.path.exists(os.path.join(img_dir, f)) for f in val_df['filename'])

    train_coords = train_df[['lat', 'lng']].values
    val_coords = val_df[['lat', 'lng']].values

    # Pairwise nearest distance from val to train
    train_3d = coords_to_3d(train_coords[:, 0], train_coords[:, 1])
    val_3d = coords_to_3d(val_coords[:, 0], val_coords[:, 1])

    # Batch compute minimum Haversine distance
    nearest_dists = []
    chunk_size = 500
    for i in range(0, len(val_coords), chunk_size):
        chunk_val_3d = val_3d[i:i+chunk_size]
        # Chord distance
        chord_dists = np.linalg.norm(chunk_val_3d[:, np.newaxis, :] - train_3d[np.newaxis, :, :], axis=-1)
        min_indices = np.argmin(chord_dists, axis=-1)
        
        # Exact haversine on min candidates
        c_lat1 = val_coords[i:i+chunk_size, 0]
        c_lng1 = val_coords[i:i+chunk_size, 1]
        c_lat2 = train_coords[min_indices, 0]
        c_lng2 = train_coords[min_indices, 1]
        dists = haversine_km(c_lat1, c_lng1, c_lat2, c_lng2)
        nearest_dists.extend(dists.tolist())

    nearest_dists = np.array(nearest_dists)
    min_dist = float(np.min(nearest_dists)) if len(nearest_dists) > 0 else 0.0

    diag = {
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "duplicate_filenames_overlap": file_overlap,
        "missing_train_images": missing_train,
        "missing_val_images": missing_val,
        "min_train_to_val_distance_km": round(min_dist, 2),
        "pct_val_within_10km": round(float(np.mean(nearest_dists <= 10.0) * 100.0), 2),
        "pct_val_within_25km": round(float(np.mean(nearest_dists <= 25.0) * 100.0), 2),
        "pct_val_within_40km": round(float(np.mean(nearest_dists <= 40.0) * 100.0), 2),
        "pct_val_within_50km": round(float(np.mean(nearest_dists <= 50.0) * 100.0), 2),
        "pct_val_within_100km": round(float(np.mean(nearest_dists <= 100.0) * 100.0), 2),
        "train_per_country": train_df['country'].value_counts().to_dict(),
        "val_per_country": val_df['country'].value_counts().to_dict(),
    }
    return diag

# ------------------------------------------------------------------------------
# 5. Geolocation Augmentations & Datasets
# ------------------------------------------------------------------------------

def get_train_transforms(image_size: int = 512, allow_hflip: bool = False) -> transforms.Compose:
    """
    Builds geolocation-safe augmentations.
    Preserves fine textual cues, road markings, architecture, and environmental details.
    Horizontal flipping is DISABLED by default to preserve asymmetric geographic cues.
    """
    t_list = [
        transforms.Resize((int(image_size * 1.06), int(image_size * 1.06))),
        transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
        transforms.RandomRotation(degrees=6),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
    ]
    if allow_hflip:
        t_list.append(transforms.RandomHorizontalFlip(p=0.5))
    t_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transforms.Compose(t_list)

def get_val_transforms(image_size: int = 512) -> transforms.Compose:
    """Standard evaluation transform."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

class GeolocationDataset(Dataset):
    """
    Primary dataset for geolocation supervision across fine cells, coarse regions,
    country classification, and local tangent-plane east/north offsets.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: str,
        fine_centroids: np.ndarray,
        fine_to_country: np.ndarray,
        fine_to_coarse: np.ndarray,
        max_offset_km: float = 50.0,
        transform: Optional[transforms.Compose] = None,
        tta_mode: str = "direct",
        image_size: int = 512
    ):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.fine_centroids = fine_centroids
        self.fine_to_country = fine_to_country
        self.fine_to_coarse = fine_to_coarse
        self.max_offset_km = max_offset_km
        self.transform = transform
        self.tta_mode = tta_mode
        self.image_size = image_size

        if self.tta_mode != "direct":
            self.norm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
            self.resize_direct = transforms.Resize((image_size, image_size))
            self.resize_larger = transforms.Resize((int(image_size * 1.06), int(image_size * 1.06)))
            self.five_crop = transforms.FiveCrop(image_size)

        # Precompute target fine cell and coarse region assignments
        lats = self.df['lat'].values
        lngs = self.df['lng'].values
        pts_3d = coords_to_3d(lats, lngs)
        fine_3d = coords_to_3d(self.fine_centroids[:, 0], self.fine_centroids[:, 1])

        cell_indices = []
        for i in range(len(self.df)):
            c_name = self.df.iloc[i]['country']
            c_idx = COUNTRY_TO_IDX.get(c_name, -1)
            valid_cells = np.where(self.fine_to_country == c_idx)[0] if c_idx >= 0 else np.arange(len(self.fine_centroids))
            if len(valid_cells) == 0:
                valid_cells = np.arange(len(self.fine_centroids))
            dists = np.linalg.norm(pts_3d[i] - fine_3d[valid_cells], axis=-1)
            cell_indices.append(valid_cells[np.argmin(dists)])

        self.cell_indices = np.array(cell_indices, dtype=np.int64)
        self.coarse_indices = self.fine_to_coarse[self.cell_indices]
        self.country_indices = np.array([COUNTRY_TO_IDX.get(c, -1) for c in self.df['country']], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img_name = row['filename']
        img_path = os.path.join(self.img_dir, img_name)

        if not os.path.exists(img_path):
            raise RuntimeError(f"Missing image file: {img_path}")

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Failed to read corrupted image {img_path}: {e}")

        if self.tta_mode != "direct":
            if self.tta_mode == "center":
                larger = self.resize_larger(image)
                crop = transforms.functional.center_crop(larger, (self.image_size, self.image_size))
                views = [self.norm(self.resize_direct(image)), self.norm(crop)]
            elif self.tta_mode == "5crop":
                larger = self.resize_larger(image)
                crops = self.five_crop(larger)
                views = [self.norm(c) for c in crops]
            elif self.tta_mode == "6view":
                larger = self.resize_larger(image)
                crops = self.five_crop(larger)
                views = [self.norm(c) for c in crops] + [self.norm(self.resize_direct(image))]
            elif self.tta_mode == "multiscale":
                v1 = self.norm(self.resize_direct(image))
                larger1 = transforms.Resize((int(self.image_size * 1.12), int(self.image_size * 1.12)))(image)
                c1 = transforms.functional.center_crop(larger1, (self.image_size, self.image_size))
                larger2 = transforms.Resize((int(self.image_size * 1.25), int(self.image_size * 1.25)))(image)
                c2 = transforms.functional.center_crop(larger2, (self.image_size, self.image_size))
                views = [v1, self.norm(c1), self.norm(c2)]
            else:
                views = [self.norm(self.resize_direct(image))]
            image = torch.stack(views, dim=0)
        elif self.transform:
            image = self.transform(image)

        lat = float(row['lat'])
        lng = float(row['lng'])

        cell_idx = int(self.cell_indices[idx])
        coarse_idx = int(self.coarse_indices[idx])
        country_idx = int(self.country_indices[idx])

        # Compute ground truth local offset relative to assigned fine cell centroid
        c_lat, c_lng = self.fine_centroids[cell_idx]
        north_km, east_km = coords_to_offset_km(lat, lng, c_lat, c_lng)
        norm_north, norm_east = normalize_offset(north_km, east_km, self.max_offset_km)

        return (
            image,
            torch.tensor([lat, lng], dtype=torch.float32),
            torch.tensor(cell_idx, dtype=torch.long),
            torch.tensor(coarse_idx, dtype=torch.long),
            torch.tensor(country_idx, dtype=torch.long),
            torch.tensor([norm_north, norm_east], dtype=torch.float32)
        )

class TwoViewGeolocationDataset(Dataset):
    """
    Dataset generating two distinct augmented views of each image for Phase A metric representation learning.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: str,
        fine_centroids: np.ndarray,
        fine_to_country: np.ndarray,
        transform: transforms.Compose
    ):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

        lats = self.df['lat'].values
        lngs = self.df['lng'].values
        pts_3d = coords_to_3d(lats, lngs)
        fine_3d = coords_to_3d(fine_centroids[:, 0], fine_centroids[:, 1])

        cell_indices = []
        for i in range(len(self.df)):
            c_name = self.df.iloc[i]['country']
            c_idx = COUNTRY_TO_IDX.get(c_name, -1)
            valid_cells = np.where(fine_to_country == c_idx)[0] if c_idx >= 0 else np.arange(len(fine_centroids))
            if len(valid_cells) == 0:
                valid_cells = np.arange(len(fine_centroids))
            dists = np.linalg.norm(pts_3d[i] - fine_3d[valid_cells], axis=-1)
            cell_indices.append(valid_cells[np.argmin(dists)])

        self.cell_indices = np.array(cell_indices, dtype=np.int64)
        self.country_indices = np.array([COUNTRY_TO_IDX.get(c, -1) for c in self.df['country']], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])

        if not os.path.exists(img_path):
            raise RuntimeError(f"Missing image file: {img_path}")

        try:
            pil_img = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Failed to read image {img_path}: {e}")

        view1 = self.transform(pil_img)
        view2 = self.transform(pil_img)

        lat = float(row['lat'])
        lng = float(row['lng'])

        return (
            view1,
            view2,
            torch.tensor([lat, lng], dtype=torch.float32),
            torch.tensor(self.country_indices[idx], dtype=torch.long),
            torch.tensor(self.cell_indices[idx], dtype=torch.long)
        )

class HoldoutDataset(Dataset):
    """Dataset for holdout evaluation and test-time augmentation (TTA)."""
    def __init__(
        self,
        img_dir: str,
        image_size: int = 512,
        mode: str = "direct"
    ):
        self.img_dir = img_dir
        self.image_size = image_size
        self.mode = mode
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Holdout directory not found: {img_dir}")

        valid_exts = ('.jpg', '.jpeg', '.png', '.webp')
        filenames = [f for f in os.listdir(img_dir) if f.lower().endswith(valid_exts)]
        filenames.sort()
        self.filenames = filenames

        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        self.resize_direct = transforms.Resize((image_size, image_size))
        self.resize_larger = transforms.Resize((int(image_size * 1.06), int(image_size * 1.06)))
        self.five_crop = transforms.FiveCrop(image_size)

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        fname = self.filenames[idx]
        path = os.path.join(self.img_dir, fname)

        try:
            image = Image.open(path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Failed to read holdout image {path}: {e}")

        if self.mode == "direct":
            views = [self.norm(self.resize_direct(image))]
        elif self.mode == "center":
            larger = self.resize_larger(image)
            crop = transforms.functional.center_crop(larger, (self.image_size, self.image_size))
            views = [self.norm(self.resize_direct(image)), self.norm(crop)]
        elif self.mode == "5crop":
            larger = self.resize_larger(image)
            crops = self.five_crop(larger)
            views = [self.norm(c) for c in crops]
        elif self.mode == "6view":
            larger = self.resize_larger(image)
            crops = self.five_crop(larger)
            views = [self.norm(c) for c in crops] + [self.norm(self.resize_direct(image))]
        elif self.mode == "multiscale":
            # Multi-scale crops all standardized to (image_size, image_size)
            v1 = self.norm(self.resize_direct(image))
            larger1 = transforms.Resize((int(self.image_size * 1.12), int(self.image_size * 1.12)))(image)
            c1 = transforms.functional.center_crop(larger1, (self.image_size, self.image_size))
            larger2 = transforms.Resize((int(self.image_size * 1.25), int(self.image_size * 1.25)))(image)
            c2 = transforms.functional.center_crop(larger2, (self.image_size, self.image_size))
            views = [v1, self.norm(c1), self.norm(c2)]
        else:
            views = [self.norm(self.resize_direct(image))]

        return torch.stack(views, dim=0), fname

# ------------------------------------------------------------------------------
# 6. Oracle Bounds & Dataset Feasibility Analysis
# ------------------------------------------------------------------------------

def run_oracle_analysis(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    fine_centroids: np.ndarray,
    coarse_centroids: np.ndarray,
    fine_to_country: np.ndarray,
    fine_to_coarse: np.ndarray,
    output_json_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Computes comprehensive oracle bounds and dataset feasibility metrics:
    - Nearest training coordinate distance for every validation sample
    - Oracle cell error (distance from sample to assigned cell centroid)
    - Population distribution across cells
    - Feasibility verdict for the <40 km stretch target
    """
    print("=" * 70)
    print("GEOLOCATION DATASET ORACLE & FEASIBILITY ANALYSIS")
    print("=" * 70)

    train_coords = train_df[['lat', 'lng']].values
    val_coords = val_df[['lat', 'lng']].values

    # 1. Nearest Training Coordinate Distance
    print("1. Computing Nearest-Training-Coordinate Distances...")
    train_3d = coords_to_3d(train_coords[:, 0], train_coords[:, 1])
    val_3d = coords_to_3d(val_coords[:, 0], val_coords[:, 1])

    nearest_dists = []
    chunk_size = 500
    for i in range(0, len(val_coords), chunk_size):
        chunk_val_3d = val_3d[i:i + chunk_size]
        chord_dists = np.linalg.norm(chunk_val_3d[:, np.newaxis, :] - train_3d[np.newaxis, :, :], axis=-1)
        min_indices = np.argmin(chord_dists, axis=-1)
        dists = haversine_km(
            val_coords[i:i + chunk_size, 0],
            val_coords[i:i + chunk_size, 1],
            train_coords[min_indices, 0],
            train_coords[min_indices, 1]
        )
        nearest_dists.extend(dists.tolist())

    nearest_dists = np.array(nearest_dists)
    median_nearest = float(np.median(nearest_dists))
    mean_nearest = float(np.mean(nearest_dists))

    rates = {
        "pct_within_10km": float(np.mean(nearest_dists <= 10.0) * 100.0),
        "pct_within_25km": float(np.mean(nearest_dists <= 25.0) * 100.0),
        "pct_within_40km": float(np.mean(nearest_dists <= 40.0) * 100.0),
        "pct_within_50km": float(np.mean(nearest_dists <= 50.0) * 100.0),
        "pct_within_100km": float(np.mean(nearest_dists <= 100.0) * 100.0),
        "pct_within_200km": float(np.mean(nearest_dists <= 200.0) * 100.0),
    }

    # 2. Per-Country Statistics
    per_country_stats = {}
    for country in COUNTRIES:
        mask = val_df['country'] == country
        if mask.sum() > 0:
            c_dists = nearest_dists[mask]
            per_country_stats[country] = {
                "val_samples": int(mask.sum()),
                "median_nearest_km": round(float(np.median(c_dists)), 2),
                "mean_nearest_km": round(float(np.mean(c_dists)), 2),
                "pct_within_40km": round(float(np.mean(c_dists <= 40.0) * 100.0), 2),
                "pct_within_100km": round(float(np.mean(c_dists <= 100.0) * 100.0), 2)
            }

    # 3. Fine-Cell & Coarse-Region Oracle Medians
    print("2. Computing Fine-Cell and Coarse-Region Oracle Medians...")
    fine_3d = coords_to_3d(fine_centroids[:, 0], fine_centroids[:, 1])
    val_cell_dists = []
    val_coarse_dists = []

    for i in range(len(val_df)):
        c_name = val_df.iloc[i]['country']
        c_idx = COUNTRY_TO_IDX.get(c_name, -1)
        valid_cells = np.where(fine_to_country == c_idx)[0] if c_idx >= 0 else np.arange(len(fine_centroids))
        if len(valid_cells) == 0:
            valid_cells = np.arange(len(fine_centroids))
        
        d = np.linalg.norm(val_3d[i] - fine_3d[valid_cells], axis=-1)
        best_cell = valid_cells[np.argmin(d)]
        best_coarse = fine_to_coarse[best_cell]

        dist_to_fine = haversine_km(val_coords[i, 0], val_coords[i, 1], fine_centroids[best_cell, 0], fine_centroids[best_cell, 1])
        dist_to_coarse = haversine_km(val_coords[i, 0], val_coords[i, 1], coarse_centroids[best_coarse, 0], coarse_centroids[best_coarse, 1])

        val_cell_dists.append(dist_to_fine)
        val_coarse_dists.append(dist_to_coarse)

    val_cell_dists = np.array(val_cell_dists)
    val_coarse_dists = np.array(val_coarse_dists)
    fine_cell_oracle_median = float(np.median(val_cell_dists))
    fine_cell_oracle_mean = float(np.mean(val_cell_dists))
    coarse_oracle_median = float(np.median(val_coarse_dists))

    # 4. Cell Population & Offset Distribution on Training Set
    print("3. Computing Training Cell Population & Offset Distributions...")
    train_cell_assignments = []
    train_offset_dists = []

    for i in range(len(train_df)):
        c_name = train_df.iloc[i]['country']
        c_idx = COUNTRY_TO_IDX.get(c_name, -1)
        valid_cells = np.where(fine_to_country == c_idx)[0] if c_idx >= 0 else np.arange(len(fine_centroids))
        if len(valid_cells) == 0:
            valid_cells = np.arange(len(fine_centroids))
        d = np.linalg.norm(train_3d[i] - fine_3d[valid_cells], axis=-1)
        best_cell = valid_cells[np.argmin(d)]
        train_cell_assignments.append(best_cell)
        
        offset_d = haversine_km(train_coords[i, 0], train_coords[i, 1], fine_centroids[best_cell, 0], fine_centroids[best_cell, 1])
        train_offset_dists.append(offset_d)

    train_cell_assignments = np.array(train_cell_assignments)
    train_offset_dists = np.array(train_offset_dists)

    cell_counts = np.bincount(train_cell_assignments, minlength=len(fine_centroids))
    empty_cells = int(np.sum(cell_counts == 0))
    undersampled_cells = int(np.sum((cell_counts > 0) & (cell_counts < 5)))

    offset_p50 = float(np.percentile(train_offset_dists, 50))
    offset_p90 = float(np.percentile(train_offset_dists, 90))
    offset_p95 = float(np.percentile(train_offset_dists, 95))
    offset_max = float(np.max(train_offset_dists))

    supported_by_data = (median_nearest <= 45.0) and (fine_cell_oracle_median <= 45.0)
    verdict = (
        "SUPPORTED: Data density and fine-cell discretization support an under-40 km stretch target."
        if supported_by_data else
        "CHALLENGING: Geographic sparsity in certain countries/regions pushes the lower bound above 40 km; "
        "model refinement and retrieval blending are critical to approach this target."
    )

    results = {
        "dataset_sizes": {"train": len(train_df), "val": len(val_df)},
        "nearest_training_coordinate": {
            "median_km": round(median_nearest, 2),
            "mean_km": round(mean_nearest, 2),
            **{k: round(v, 2) for k, v in rates.items()}
        },
        "oracles": {
            "fine_cell_oracle_median_km": round(fine_cell_oracle_median, 2),
            "fine_cell_oracle_mean_km": round(fine_cell_oracle_mean, 2),
            "coarse_region_oracle_median_km": round(coarse_oracle_median, 2)
        },
        "cell_population": {
            "total_fine_cells": len(fine_centroids),
            "empty_cells": empty_cells,
            "undersampled_cells_lt5": undersampled_cells,
            "min_samples_per_cell": int(np.min(cell_counts)),
            "max_samples_per_cell": int(np.max(cell_counts)),
            "mean_samples_per_cell": round(float(np.mean(cell_counts)), 1),
            "median_samples_per_cell": float(np.median(cell_counts))
        },
        "offset_distribution_km": {
            "p50": round(offset_p50, 2),
            "p90": round(offset_p90, 2),
            "p95": round(offset_p95, 2),
            "max": round(offset_max, 2)
        },
        "per_country_stats": per_country_stats,
        "stretch_target_under_40km_feasibility": verdict
    }

    print("\n" + "-" * 70)
    print("ORACLE & DATASET SUMMARY REPORT")
    print("-" * 70)
    print(f"Validation Samples:                 {len(val_df):,}")
    print(f"Training Samples:                   {len(train_df):,}")
    print(f"Nearest Train Coordinate Median:    {median_nearest:.2f} km (Mean: {mean_nearest:.2f} km)")
    print(f"  Within 10 km:                     {rates['pct_within_10km']:.1f}%")
    print(f"  Within 25 km:                     {rates['pct_within_25km']:.1f}%")
    print(f"  Within 40 km:                     {rates['pct_within_40km']:.1f}%")
    print(f"  Within 50 km:                     {rates['pct_within_50km']:.1f}%")
    print(f"  Within 100 km:                    {rates['pct_within_100km']:.1f}%")
    print(f"  Within 200 km:                    {rates['pct_within_200km']:.1f}%")
    print(f"Fine-Cell Oracle Median ({len(fine_centroids)}):      {fine_cell_oracle_median:.2f} km")
    print(f"Coarse-Region Oracle Median ({len(coarse_centroids)}):   {coarse_oracle_median:.2f} km")
    print(f"Offset Spread:                      90th%: {offset_p90:.2f} km | 95th%: {offset_p95:.2f} km | Max: {offset_max:.2f} km")
    print(f"Fine Cells Population:              Empty: {empty_cells} | Undersampled (<5): {undersampled_cells}")
    print(f"Stretch Target (<40km):             {verdict}")
    print("=" * 70)

    if output_json_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_json_path}")

    return results

if __name__ == "__main__":
    import argparse
    from config import get_default_config

    parser = argparse.ArgumentParser(description="Geolocation Dataset Utilities, Splits & Oracle Analysis")
    parser.add_argument("--generate-splits", action="store_true", help="Generate train and val split manifests")
    parser.add_argument("--oracle", action="store_true", help="Run oracle & dataset feasibility analysis")
    parser.add_argument("--split", type=str, default="spatial", choices=["spatial", "random", "cv"], help="Split type")
    parser.add_argument("--fold", type=int, default=0, help="Fold index (0..4) when using cv split")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path for oracle analysis")
    args = parser.parse_args()

    cfg = get_default_config()
    cfg.ensure_directories()
    raw_df = pd.read_csv(cfg.get_path(cfg.train_csv))

    if args.split == "cv":
        train_path = os.path.join(cfg.splits_dir, f"fold_{args.fold}_train.csv")
        val_path = os.path.join(cfg.splits_dir, f"fold_{args.fold}_val.csv")
    elif args.split == "spatial":
        train_path = cfg.get_path(cfg.spatial_train_manifest)
        val_path = cfg.get_path(cfg.spatial_val_manifest)
    else:
        train_path = cfg.get_path(cfg.random_train_manifest)
        val_path = cfg.get_path(cfg.random_val_manifest)

    if args.generate_splits or not (os.path.exists(train_path) and os.path.exists(val_path)):
        if args.split == "cv":
            print(f"Generating 5-fold country-stratified CV manifests (seed={cfg.seed})...")
            cv_folds = create_stratified_cv_splits(raw_df, n_splits=cfg.cv_num_folds, seed=cfg.seed)
            os.makedirs(cfg.splits_dir, exist_ok=True)
            for f_idx, (t_fold, v_fold) in enumerate(cv_folds):
                t_p = os.path.join(cfg.splits_dir, f"fold_{f_idx}_train.csv")
                v_p = os.path.join(cfg.splits_dir, f"fold_{f_idx}_val.csv")
                t_fold.to_csv(t_p, index=False)
                v_fold.to_csv(v_p, index=False)
                print(f"✓ Saved Fold {f_idx}: Train ({len(t_fold):,} rows), Val ({len(v_fold):,} rows)")
            train_df = pd.read_csv(train_path)
            val_df = pd.read_csv(val_path)
        else:
            print(f"Generating {args.split} split manifests (seed={cfg.seed})...")
            if args.split == "spatial":
                train_df, val_df = create_spatial_split(
                    raw_df, group_radius_km=cfg.spatial_group_radius_km, val_ratio=cfg.val_ratio, seed=cfg.seed
                )
            else:
                train_df, val_df = create_random_split(raw_df, val_ratio=cfg.val_ratio, seed=cfg.seed)

            os.makedirs(os.path.dirname(os.path.abspath(train_path)), exist_ok=True)
            train_df.to_csv(train_path, index=False)
            val_df.to_csv(val_path, index=False)
            print(f"✓ Saved {args.split} train manifest: {train_path} ({len(train_df):,} rows)")
            print(f"✓ Saved {args.split} val manifest:   {val_path} ({len(val_df):,} rows)")

            diag = compute_split_diagnostics(train_df, val_df, cfg.get_path(cfg.train_img_dir))
            diag_path = os.path.join(cfg.splits_dir, f"{args.split}_split_diagnostics.json")
            with open(diag_path, "w", encoding="utf-8") as f:
                json.dump(diag, f, indent=2)
            print(f"✓ Split diagnostics saved to: {diag_path}")
    else:
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)

    if args.oracle:
        hierarchy_dir = cfg.exp_dir
        hierarchy_meta = os.path.join(hierarchy_dir, "hierarchy_metadata.json")
        loaded = False
        if os.path.exists(hierarchy_meta):
            try:
                fine_c, coarse_c, f2c, f2r, _ = load_geographic_hierarchy(
                    hierarchy_dir, train_path, expected_num_fine_cells=cfg.num_fine_cells
                )
                loaded = True
            except ValueError as e:
                print(f"Existing hierarchy metadata does not match current configuration/manifest ({e}).\nRebuilding hierarchy for current split...")
        if not loaded:
            hierarchy_dict = build_geographic_hierarchy(
                train_df,
                num_fine_cells=cfg.num_fine_cells,
                num_coarse_regions=cfg.num_coarse_regions,
                seed=cfg.seed,
                output_dir=hierarchy_dir,
                config_hash=cfg.compute_config_hash(),
                manifest_path=train_path
            )
            fine_c = hierarchy_dict["fine_centroids"]
            coarse_c = hierarchy_dict["coarse_centroids"]
            f2c = hierarchy_dict["fine_to_country"]
            f2r = hierarchy_dict["fine_to_coarse"]

        out_path = args.output or os.path.join(cfg.exp_dir, "oracle_analysis.json")
        run_oracle_analysis(train_df, val_df, fine_c, coarse_c, f2c, f2r, output_json_path=out_path)

