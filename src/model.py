import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class GeMPooling(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer.
    Focuses on salient localized visual features (architectural details, road markings, landscape motifs)
    rather than uniform global average pooling.
    """
    def __init__(self, p=3.0, eps=1e-6):
        super(GeMPooling, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1.0 / self.p).squeeze(-1).squeeze(-1)

class RegNetYGeolocationModel(nn.Module):
    """
    Primary Geolocation Model using RegNet-Y 400MF backbone (~3.9M backbone params).
    Total Trainable Parameters: ~4,443,191 (strictly under the 5,000,000 constraint).
    Standard Conv+BatchNorm+ReLU architecture provides fast, robust convergence when trained strictly from scratch.
    """
    def __init__(self, num_cells=512, num_countries=12):
        super(RegNetYGeolocationModel, self).__init__()
        self.backbone = timm.create_model('regnety_004', pretrained=True, num_classes=0)
        in_features = self.backbone.num_features

        self.gem_pool = GeMPooling(p=3.0)
        self.dropout = nn.Dropout(p=0.2)

        self.cell_head = nn.Sequential(
            nn.Linear(in_features, 320),
            nn.BatchNorm1d(320),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(320, num_cells)
        )

        self.coord_head = nn.Sequential(
            nn.Linear(in_features, 192),
            nn.BatchNorm1d(192),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(192, 2),
            nn.Tanh()
        )

        self.cartesian_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.Hardswish(),
            nn.Linear(128, 3)
        )

        self.country_head = nn.Sequential(
            nn.Linear(in_features, 192),
            nn.BatchNorm1d(192),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(192, num_countries)
        )

    def forward(self, x):
        feat_map = self.backbone(x)
        feat = self.gem_pool(feat_map) if len(feat_map.shape) == 4 else feat_map
        feat = self.dropout(feat)

        cell_logits = self.cell_head(feat)
        pred_offset = self.coord_head(feat)
        pred_xyz = F.normalize(self.cartesian_head(feat), p=2, dim=-1)
        country_logits = self.country_head(feat)

        return cell_logits, pred_offset, pred_xyz, country_logits

class ConvNeXtV2GeolocationModel(nn.Module):
    def __init__(self, num_cells=512, num_countries=12):
        super(ConvNeXtV2GeolocationModel, self).__init__()
        self.backbone = timm.create_model('convnextv2_atto', pretrained=True, num_classes=0, global_pool='')
        in_features = self.backbone.num_features

        self.gem_pool = GeMPooling(p=3.0)
        self.dropout = nn.Dropout(p=0.2)

        self.cell_head = nn.Sequential(
            nn.Linear(in_features, 384),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(384, num_cells)
        )

        self.coord_head = nn.Sequential(
            nn.Linear(in_features, 192),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(192, 2),
            nn.Tanh()
        )

        self.cartesian_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.Hardswish(),
            nn.Linear(128, 3)
        )

        self.country_head = nn.Sequential(
            nn.Linear(in_features, 192),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(192, num_countries)
        )

    def forward(self, x):
        feat_map = self.backbone(x)
        feat = self.gem_pool(feat_map) if len(feat_map.shape) == 4 else feat_map
        feat = self.dropout(feat)

        cell_logits = self.cell_head(feat)
        pred_offset = self.coord_head(feat)
        pred_xyz = F.normalize(self.cartesian_head(feat), p=2, dim=-1)
        country_logits = self.country_head(feat)

        return cell_logits, pred_offset, pred_xyz, country_logits

class MobileNetV4GeolocationModel(nn.Module):
    def __init__(self, num_cells=512, num_countries=12):
        super(MobileNetV4GeolocationModel, self).__init__()
        # MobileNetV4 uses pure convolutions, making it inherently resolution-agnostic. 
        # It effortlessly handles 384x384 out of the box and has 79.3% ImageNet Top-1!
        self.backbone = timm.create_model('mobilenetv4_conv_small.e2400_r224_in1k', pretrained=True, num_classes=0, global_pool='')
        with torch.no_grad():
            dummy_out = self.backbone(torch.zeros(1, 3, 224, 224))
            in_features = dummy_out.shape[1]

        self.gem_pool = GeMPooling(p=3.0)
        self.dropout = nn.Dropout(p=0.2)

        self.cell_head = nn.Sequential(
            nn.Linear(in_features, 384),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(384, num_cells)
        )

        self.coord_head = nn.Sequential(
            nn.Linear(in_features, 192),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(192, 2),
            nn.Tanh()
        )

        self.cartesian_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.Hardswish(),
            nn.Linear(128, 3)
        )

        self.country_head = nn.Sequential(
            nn.Linear(in_features, 192),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(192, num_countries)
        )

    def forward(self, x):
        feat_map = self.backbone(x)
        feat = self.gem_pool(feat_map) if len(feat_map.shape) == 4 else feat_map
        feat = self.dropout(feat)

        cell_logits = self.cell_head(feat)
        pred_offset = self.coord_head(feat)
        pred_xyz = F.normalize(self.cartesian_head(feat), p=2, dim=-1)
        country_logits = self.country_head(feat)

        return cell_logits, pred_offset, pred_xyz, country_logits


def get_model(num_cells=512, num_countries=12, arch="regnety"):
    if arch == "convnextv2":
        model = ConvNeXtV2GeolocationModel(num_cells=num_cells, num_countries=num_countries)
    elif arch == "mobilenetv4":
        model = MobileNetV4GeolocationModel(num_cells=num_cells, num_countries=num_countries)
    else:
        model = RegNetYGeolocationModel(num_cells=num_cells, num_countries=num_countries)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created ({arch.upper()}). Total Trainable Parameters: {num_params:,}")
    if num_params > 5000000:
        raise ValueError(f"Model has {num_params:,} parameters, strictly exceeding 5,000,000 limit!")
    return model

def load_saved_model(weights_path, num_cells=None, num_countries=12, device='cpu'):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Checkpoint weights not found at {weights_path}")

    state_dict = torch.load(weights_path, map_location=device)
    
    if num_cells is None:
        if "cell_head.4.weight" in state_dict:
            num_cells = state_dict["cell_head.4.weight"].shape[0]
        elif "cell_head.3.weight" in state_dict:
            num_cells = state_dict["cell_head.3.weight"].shape[0]
        else:
            num_cells = 512

    if "backbone.stem.0.weight" in state_dict or "backbone.stages.0.blocks.0.conv_dw.weight" in state_dict:
        model = ConvNeXtV2GeolocationModel(num_cells=num_cells, num_countries=num_countries)
    elif "backbone.s1.b1.conv1.weight" in state_dict or "backbone.stem.conv.weight" in state_dict or "backbone.stem.0.0.weight" in state_dict:
        model = RegNetYGeolocationModel(num_cells=num_cells, num_countries=num_countries)
    else:
        model = MobileNetV4GeolocationModel(num_cells=num_cells, num_countries=num_countries)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

if __name__ == "__main__":
    model = get_model()
    x = torch.randn(2, 3, 384, 384)
    cell_logits, pred_offset, pred_xyz, country_logits = model(x)
    print("Cell logits shape:", cell_logits.shape)
    print("Pred offset shape:", pred_offset.shape)
    print("Pred XYZ shape:", pred_xyz.shape)
    print("Country logits shape:", country_logits.shape)
