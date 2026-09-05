import os
import json
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Dict, Any

@dataclass
class GeolocationConfig:
    # --- Experiment & Paths ---
    exp_name: str = "exp_regnety_baseline"
    base_dir: str = field(default_factory=lambda: os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    experiments_dir: str = "experiments"
    
    # Dataset locations (relative to base_dir)
    train_csv: str = "geo_dataset/train_labels.csv"
    train_img_dir: str = "geo_dataset/train"
    holdout_img_dir: str = "geo_dataset/holdout_public"
    
    # Split manifests (relative to base_dir or experiment dir)
    split_type: str = "random"  # 'random' (stratified country benchmark matching leaderboard), 'spatial', or 'cv'
    val_ratio: float = 0.20
    spatial_group_radius_km: float = 35.0
    random_train_manifest: str = "experiments/splits/random_train_manifest.csv"
    random_val_manifest: str = "experiments/splits/random_val_manifest.csv"
    spatial_train_manifest: str = "experiments/splits/spatial_train_manifest.csv"
    spatial_val_manifest: str = "experiments/splits/spatial_val_manifest.csv"
    cv_num_folds: int = 5
    fold: Optional[int] = None  # None for single-split, 0..4 for specific fold
    
    # Output file paths within experiment dir
    checkpoint_best_name: str = "best_model.pth"
    checkpoint_last_name: str = "last_model.pth"
    fine_centroids_name: str = "fine_centroids.npy"
    coarse_centroids_name: str = "coarse_centroids.npy"
    fine_to_country_name: str = "fine_to_country.npy"
    fine_to_coarse_name: str = "fine_to_coarse.npy"
    hierarchy_metadata_name: str = "hierarchy_metadata.json"
    retrieval_db_name: str = "train_retrieval_db.pt"
    prediction_output_name: str = "predictions.csv"
    
    # --- Reproducibility ---
    seed: int = 42
    deterministic: bool = True
    
    # --- Model Architecture ---
    model_arch: str = "regnety_004"
    pretrained: bool = False  # HARD RULE: Must always be False
    gem_p: float = 3.0
    gem_eps: float = 1e-6
    shared_proj_dim: int = 384
    dropout_rate: float = 0.2
    
    # Geographic hierarchy dimensions
    num_countries: int = 12
    num_coarse_regions: int = 48   # 4 coarse regions per country
    num_fine_cells: int = 576      # 48 fine cells per country (val oracle bound ~23.9 km)
    embedding_dim: int = 128       # Metric retrieval embedding dimension
    include_cartesian_head: bool = True  # 3D Cartesian head within param budget
    max_offset_km: float = 50.0    # Maximum local tangent-plane displacement (covers ~90-95% cell spread)
    
    # Parameter constraints
    max_allowed_params: int = 5_000_000
    
    # --- Data & Augmentation ---
    image_size: int = 512
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    horizontal_flip: bool = False  # DISABLED by default to protect asymmetric geographic cues
    
    # --- Optimization ---
    backbone_lr: float = 6e-4
    head_lr: float = 1.2e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    grad_accum_steps: int = 1
    warmup_epochs: int = 1
    min_lr_ratio: float = 0.05
    use_amp: bool = True
    ema_decay: float = 0.999
    
    # --- Training Curriculum (40 epochs total: 10 Ep Country warm-up + 30 Ep Joint Localization) ---
    # Phase A: Representation learning (RegNet-Y backbone + metric embedding)
    phase_a_epochs: int = 5
    phase_a_lr: float = 6e-4
    metric_temperature: float = 0.1
    metric_dist_scale: float = 50.0  # weight = exp(-dist / 50.0)
    
    # Phase B: Country and Coarse-Region learning
    phase_b_epochs: int = 10
    phase_b_lr: float = 1e-3
    phase_b_backbone_lr: float = 4e-4
    
    # Phase C: Joint Localization (30 epochs combined with Phase B = 40 total)
    phase_c_epochs: int = 30
    phase_c_backbone_lr: float = 6e-4
    phase_c_head_lr: float = 1.2e-3
    
    # Phase C Multi-Task Loss Weights
    loss_weight_country: float = 6.0
    loss_weight_coarse: float = 2.0
    loss_weight_fine: float = 2.0
    loss_weight_metric: float = 0.5
    loss_weight_offset: float = 1.0
    loss_weight_cartesian: float = 0.5
    loss_weight_haversine: float = 0.5
    haversine_warmup_epochs: int = 2
    
    # Decoding temperature schedule during training
    initial_train_temp: float = 0.5  # Soft initial temperature
    final_train_temp: float = 0.1
    
    # --- Inference & Decoding ---
    decoder_temperature: float = 0.10
    cell_top_k: int = 4
    country_logit_weight: float = 3.0
    neighborhood_radius_km: float = 150.0
    decoder_country_top_k: int = 2   # Allow candidate cells from top-2 countries in spatial neighborhood
    
    # Retrieval decoder settings
    retrieval_k: int = 20
    retrieval_country_top_k: int = 2
    retrieval_blend_alpha: float = 0.65  # alpha * cell_pred + (1 - alpha) * retrieval_pred
    use_geographic_medoid: bool = False  # Spherical mean vs geographic medoid
    
    # Test-Time Augmentation (TTA)
    tta_mode: str = "direct"  # 'direct', 'center', '5crop', '6view', 'multiscale'
    multiscale_sizes: List[int] = field(default_factory=lambda: [384, 448])

    @property
    def exp_dir(self) -> str:
        return os.path.join(self.base_dir, self.experiments_dir, self.exp_name)

    @property
    def splits_dir(self) -> str:
        return os.path.join(self.base_dir, self.experiments_dir, "splits")

    def get_path(self, relative_path: str) -> str:
        """Resolves a path relative to base_dir if not absolute."""
        if os.path.isabs(relative_path):
            return relative_path
        return os.path.join(self.base_dir, relative_path)

    def get_exp_path(self, filename: str) -> str:
        """Returns the full path inside the experiment output directory."""
        return os.path.join(self.exp_dir, filename)

    def get_fold_train_manifest(self, fold: int) -> str:
        """Returns the split manifest path for training fold."""
        return os.path.join(self.splits_dir, f"fold_{fold}_train.csv")

    def get_fold_val_manifest(self, fold: int) -> str:
        """Returns the split manifest path for validation fold."""
        return os.path.join(self.splits_dir, f"fold_{fold}_val.csv")

    def get_fold_dir(self, fold: int) -> str:
        """Returns the experiment sub-directory for a specific CV fold."""
        fdir = os.path.join(self.exp_dir, f"fold_{fold}")
        os.makedirs(fdir, exist_ok=True)
        return fdir

    def ensure_directories(self) -> None:
        """Ensures that experiment and split directories exist."""
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.splits_dir, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the configuration to a python dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeolocationConfig":
        """Constructs a GeolocationConfig instance from a dictionary."""
        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        current_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if "base_dir" in filtered_data:
            if not filtered_data["base_dir"] or not os.path.exists(filtered_data["base_dir"]):
                filtered_data["base_dir"] = current_repo_root
        else:
            filtered_data["base_dir"] = current_repo_root
        return cls(**filtered_data)

    def save_json(self, path: Optional[str] = None) -> str:
        """Saves configuration as formatted JSON."""
        if path is None:
            self.ensure_directories()
            path = os.path.join(self.exp_dir, "config.json")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load_json(cls, path: str) -> "GeolocationConfig":
        """Loads configuration from a JSON file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def compute_config_hash(self) -> str:
        """Computes deterministic SHA256 hash of key structural configuration values."""
        key_items = {
            "num_countries": self.num_countries,
            "num_coarse_regions": self.num_coarse_regions,
            "num_fine_cells": self.num_fine_cells,
            "embedding_dim": self.embedding_dim,
            "max_offset_km": self.max_offset_km,
            "seed": self.seed,
            "model_arch": self.model_arch,
        }
        encoded = json.dumps(key_items, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

def get_default_config() -> GeolocationConfig:
    """Returns the default central configuration object."""
    return GeolocationConfig()
