import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional
import timm

class GeMPooling(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer with learnable exponent p.
    Focuses on salient localized visual features (architectural details, road markings, landscape motifs)
    rather than uniform global average pooling.
    """
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super(GeMPooling, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, H, W)
        orig_dtype = x.dtype
        # Always compute GeM power in float32 to avoid float16 exponent overflow/underflow
        x_clamped = x.float().clamp(min=self.eps).pow(self.p)
        pooled = F.avg_pool2d(x_clamped, (x.size(-2), x.size(-1))).pow(1.0 / self.p)
        return pooled.squeeze(-1).squeeze(-1).to(orig_dtype)

class RegNetYGeolocationModel(nn.Module):
    """
    Primary Geolocation Model using RegNet-Y 400MF backbone with GeM pooling,
    compact shared feature projection, and multi-task geographic heads.
    Trained strictly from scratch with pretrained=False.
    Total parameters are verified dynamically to remain strictly <= 5,000,000.
    """
    def __init__(
        self,
        num_fine_cells: int = 576,
        num_coarse_regions: int = 48,
        num_countries: int = 12,
        embedding_dim: int = 128,
        max_offset_km: float = 50.0,
        shared_proj_dim: int = 384,
        dropout_rate: float = 0.2,
        include_cartesian_head: bool = True,
        gem_p: float = 3.0
    ):
        super(RegNetYGeolocationModel, self).__init__()
        self.num_fine_cells = num_fine_cells
        self.num_coarse_regions = num_coarse_regions
        self.num_countries = num_countries
        self.embedding_dim = embedding_dim
        self.max_offset_km = max_offset_km
        self.shared_proj_dim = shared_proj_dim
        self.include_cartesian_head = include_cartesian_head

        # HARD RULE: Must always be trained from scratch without external weights
        self.backbone = timm.create_model('regnety_004', pretrained=False, num_classes=0)
        in_features = self.backbone.num_features  # 440 for regnety_004

        self.gem_pool = GeMPooling(p=gem_p)
        
        # Compact shared feature projection to bottleneck dimensions and control parameter budget
        self.shared_proj = nn.Sequential(
            nn.Linear(in_features, shared_proj_dim),
            nn.BatchNorm1d(shared_proj_dim),
            nn.Hardswish(),
            nn.Dropout(p=dropout_rate)
        )

        # 1. 12-Country classification head
        self.country_head = nn.Sequential(
            nn.Linear(shared_proj_dim, num_countries)
        )

        # 2. Coarse-region classification head
        self.coarse_head = nn.Sequential(
            nn.Linear(shared_proj_dim, num_coarse_regions)
        )

        # 3. Fine-cell classification head
        self.cell_head = nn.Sequential(
            nn.Linear(shared_proj_dim, num_fine_cells)
        )

        # 4. 128-dimensional metric retrieval embedding head (L2-normalized)
        self.metric_head = nn.Sequential(
            nn.Linear(shared_proj_dim, embedding_dim)
        )

        # 5. Local tangent-plane east/north offset head (tanh in [-1, 1], scaled by max_offset_km)
        self.offset_head = nn.Sequential(
            nn.Linear(shared_proj_dim, 64),
            nn.Hardswish(),
            nn.Linear(64, 2),
            nn.Tanh()
        )

        # 6. Optional 3-dimensional normalized Cartesian unit sphere head
        if self.include_cartesian_head:
            self.cartesian_head = nn.Sequential(
                nn.Linear(shared_proj_dim, 64),
                nn.Hardswish(),
                nn.Linear(64, 3)
            )
        else:
            self.cartesian_head = None

        # Verify parameter count immediately upon initialization
        self.verify_parameter_budget()

    def count_parameters(self) -> Dict[str, int]:
        """Dynamically computes parameter breakdown across model components."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        gem_params = sum(p.numel() for p in self.gem_pool.parameters())
        shared_proj_params = sum(p.numel() for p in self.shared_proj.parameters())
        country_head_params = sum(p.numel() for p in self.country_head.parameters())
        coarse_head_params = sum(p.numel() for p in self.coarse_head.parameters())
        cell_head_params = sum(p.numel() for p in self.cell_head.parameters())
        metric_head_params = sum(p.numel() for p in self.metric_head.parameters())
        offset_head_params = sum(p.numel() for p in self.offset_head.parameters())
        cartesian_head_params = sum(p.numel() for p in self.cartesian_head.parameters()) if self.cartesian_head else 0

        return {
            "total": total_params,
            "trainable": trainable_params,
            "backbone": backbone_params,
            "gem_pool": gem_params,
            "shared_proj": shared_proj_params,
            "country_head": country_head_params,
            "coarse_head": coarse_head_params,
            "cell_head": cell_head_params,
            "metric_head": metric_head_params,
            "offset_head": offset_head_params,
            "cartesian_head": cartesian_head_params,
        }

    def verify_parameter_budget(self, max_allowed: int = 5_000_000) -> Dict[str, int]:
        """Calculates and asserts the strict parameter budget constraint."""
        counts = self.count_parameters()
        total = counts["total"]
        if total > max_allowed:
            raise ValueError(
                f"Model parameter limit exceeded! Model has {total:,} total parameters, "
                f"which exceeds the maximum allowed limit of {max_allowed:,}."
            )
        return counts

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts 128-dimensional L2-normalized metric embeddings for retrieval."""
        feat_map = self.backbone(x)
        feat = self.gem_pool(feat_map) if len(feat_map.shape) == 4 else feat_map
        proj = self.shared_proj(feat)
        metric_embed = self.metric_head(proj)
        return F.normalize(metric_embed, p=2, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass producing multi-task geographic outputs:
        - cell_logits: (B, num_fine_cells)
        - coarse_logits: (B, num_coarse_regions)
        - country_logits: (B, num_countries)
        - pred_offset: (B, 2) in normalized [-1, 1] range
        - pred_xyz: (B, 3) normalized unit sphere vector or None
        - metric_embed: (B, embedding_dim) L2-normalized representation
        """
        feat_map = self.backbone(x)
        feat = self.gem_pool(feat_map) if len(feat_map.shape) == 4 else feat_map
        proj = self.shared_proj(feat)

        cell_logits = self.cell_head(proj)
        coarse_logits = self.coarse_head(proj)
        country_logits = self.country_head(proj)
        pred_offset = self.offset_head(proj)  # [-1, 1]
        
        pred_xyz = None
        if self.cartesian_head is not None:
            pred_xyz = F.normalize(self.cartesian_head(proj), p=2, dim=-1)

        metric_embed = F.normalize(self.metric_head(proj), p=2, dim=-1)

        return cell_logits, coarse_logits, country_logits, pred_offset, pred_xyz, metric_embed

def set_phase_trainable(model: RegNetYGeolocationModel, phase: str) -> None:
    """
    Configures gradient requirements for the 3-phase curriculum:
    - Phase A: Train backbone + shared projection + metric embedding head only.
    - Phase B: Freeze backbone; train country and coarse-region heads.
    - Phase C: Full joint optimization across all modules.
    """
    phase = phase.upper()
    if phase == "A":
        # Representation learning
        for p in model.parameters():
            p.requires_grad = False
        for p in model.backbone.parameters():
            p.requires_grad = True
        for p in model.gem_pool.parameters():
            p.requires_grad = True
        for p in model.shared_proj.parameters():
            p.requires_grad = True
        for p in model.metric_head.parameters():
            p.requires_grad = True
    elif phase == "B":
        # Country & coarse-region training (backbone trainable for visual feature learning)
        for p in model.parameters():
            p.requires_grad = False
        for p in model.backbone.parameters():
            p.requires_grad = True
        for p in model.gem_pool.parameters():
            p.requires_grad = True
        for p in model.shared_proj.parameters():
            p.requires_grad = True
        for p in model.country_head.parameters():
            p.requires_grad = True
        for p in model.coarse_head.parameters():
            p.requires_grad = True
    elif phase == "C":
        # Joint localization: unfreeze everything
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unknown training phase '{phase}'. Expected 'A', 'B', or 'C'.")

def get_model(
    num_fine_cells: int = 576,
    num_coarse_regions: int = 48,
    num_countries: int = 12,
    embedding_dim: int = 128,
    max_offset_km: float = 50.0,
    shared_proj_dim: int = 384,
    include_cartesian_head: bool = True
) -> RegNetYGeolocationModel:
    """
    Instantiates the single RegNet-Y 400MF Geolocation model with pretrained=False,
    calculates and prints detailed parameter breakdowns, and enforces parameter constraints.
    """
    model = RegNetYGeolocationModel(
        num_fine_cells=num_fine_cells,
        num_coarse_regions=num_coarse_regions,
        num_countries=num_countries,
        embedding_dim=embedding_dim,
        max_offset_km=max_offset_km,
        shared_proj_dim=shared_proj_dim,
        include_cartesian_head=include_cartesian_head
    )
    counts = model.verify_parameter_budget()
    print("=" * 65)
    print("REGNET-Y 400MF GEOLOCATION MODEL INITIALIZED (FROM SCRATCH)")
    print("=" * 65)
    print(f"  Backbone (RegNet-Y 400MF):    {counts['backbone']:,} params")
    print(f"  GeM Pooling:                  {counts['gem_pool']:,} params")
    print(f"  Shared Feature Projection:    {counts['shared_proj']:,} params")
    print(f"  Country Classification Head:  {counts['country_head']:,} params")
    print(f"  Coarse-Region Head:           {counts['coarse_head']:,} params")
    print(f"  Fine-Cell Head:               {counts['cell_head']:,} params")
    print(f"  Metric Retrieval Head:        {counts['metric_head']:,} params")
    print(f"  East/North Offset Head:       {counts['offset_head']:,} params")
    if counts['cartesian_head'] > 0:
        print(f"  3D Cartesian Head:            {counts['cartesian_head']:,} params")
    print("-" * 65)
    print(f"  TOTAL PARAMETERS:             {counts['total']:,} / 5,000,000 max")
    print(f"  BUDGET REMAINING:             {5_000_000 - counts['total']:,} params")
    print("=" * 65)
    return model

def load_saved_model(
    checkpoint_path: str,
    device: torch.device = torch.device('cpu'),
    config_dict: Optional[Dict[str, Any]] = None
) -> Tuple[RegNetYGeolocationModel, Dict[str, Any]]:
    """
    Loads saved checkpoint weights into RegNetYGeolocationModel, verifying compatibility.
    Does not make any network requests. Automatically inspects tensor shapes for backward compatibility.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    loaded_config = checkpoint.get("config", config_dict or {})

    # Dynamically detect shapes from state_dict if available for maximum compatibility
    if "cell_head.0.weight" in state_dict:
        num_fine_cells = state_dict["cell_head.0.weight"].shape[0]
    else:
        num_fine_cells = loaded_config.get("num_fine_cells", 576)

    if "shared_proj.0.weight" in state_dict:
        shared_proj_dim = state_dict["shared_proj.0.weight"].shape[0]
    else:
        shared_proj_dim = loaded_config.get("shared_proj_dim", 384)

    if "coarse_head.0.weight" in state_dict:
        num_coarse_regions = state_dict["coarse_head.0.weight"].shape[0]
    else:
        num_coarse_regions = loaded_config.get("num_coarse_regions", 48)

    num_countries = loaded_config.get("num_countries", 12)
    embedding_dim = loaded_config.get("embedding_dim", 128)
    max_offset_km = loaded_config.get("max_offset_km", 50.0)
    include_cartesian = "cartesian_head.2.weight" in state_dict or "cartesian_head.0.weight" in state_dict

    model = RegNetYGeolocationModel(
        num_fine_cells=num_fine_cells,
        num_coarse_regions=num_coarse_regions,
        num_countries=num_countries,
        embedding_dim=embedding_dim,
        max_offset_km=max_offset_km,
        shared_proj_dim=shared_proj_dim,
        include_cartesian_head=include_cartesian
    )

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, loaded_config
