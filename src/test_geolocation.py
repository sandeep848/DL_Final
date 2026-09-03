import os
import sys
import numpy as np
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from dataset import coords_to_3d, cartesian_to_latlng
from model import get_model
from train import haversine_km_torch, soft_expectation_coords, coords_to_3d_torch, build_centroid_dist_matrix

def test_coordinate_conversion_roundtrip():
    lats = np.array([48.8566, 60.1699, 37.9838])
    lngs = np.array([2.3522, 24.9384, 23.7275])
    xyz = coords_to_3d(lats, lngs)
    latlng_rec = cartesian_to_latlng(xyz)
    np.testing.assert_allclose(lats, latlng_rec[:, 0], atol=1e-5)
    np.testing.assert_allclose(lngs, latlng_rec[:, 1], atol=1e-5)
    print("✓ test_coordinate_conversion_roundtrip passed!")

def test_haversine_nan_safety():
    lat1 = torch.tensor([48.8566, 48.8566], dtype=torch.float32, requires_grad=True)
    lng1 = torch.tensor([2.3522, 2.3522], dtype=torch.float32, requires_grad=True)
    lat2 = torch.tensor([48.8566, 50.0000], dtype=torch.float32)
    lng2 = torch.tensor([2.3522, 3.0000], dtype=torch.float32)
    
    dists = haversine_km_torch(lat1, lng1, lat2, lng2)
    loss = dists.sum()
    loss.backward()
    
    assert not torch.isnan(lat1.grad).any(), "NaN detected in lat1 gradient!"
    assert not torch.isnan(lng1.grad).any(), "NaN detected in lng1 gradient!"
    print("✓ test_haversine_nan_safety passed!")

def test_model_parameter_constraint_and_shapes():
    model = get_model(num_cells=1536, num_countries=12, arch="regnety")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert num_params <= 5000000, f"Model parameters {num_params} exceed 5,000,000 limit!"
    
    x = torch.randn(2, 3, 384, 384)
    cell_logits, pred_offset, pred_xyz, country_logits = model(x)
    assert cell_logits.shape == (2, 1536)
    assert pred_offset.shape == (2, 2)
    assert pred_xyz.shape == (2, 3)
    assert country_logits.shape == (2, 12)
    print(f"✓ test_model_parameter_constraint_and_shapes passed! ({num_params:,} params <= 5M limit)")

def test_spatially_constrained_soft_expectation():
    cell_logits = torch.randn(4, 1536)
    centroids = np.random.uniform(-50, 50, (1536, 2))
    centroids_tensor = torch.tensor(centroids, dtype=torch.float32)
    dist_matrix = build_centroid_dist_matrix(centroids, torch.device('cpu'))
    offsets = torch.zeros(4, 2)
    
    lat, lng = soft_expectation_coords(cell_logits, centroids_tensor, offsets, topk=12, temperature=0.18, centroid_dist_matrix=dist_matrix, radius_km=100.0)
    assert lat.shape == (4,)
    assert lng.shape == (4,)
    assert not torch.isnan(lat).any()
    assert not torch.isnan(lng).any()
    print("✓ test_spatially_constrained_soft_expectation passed!")

if __name__ == "__main__":
    print("Running RegNet-Y geolocation verification tests...")
    test_coordinate_conversion_roundtrip()
    test_haversine_nan_safety()
    test_model_parameter_constraint_and_shapes()
    test_spatially_constrained_soft_expectation()
    print("All unit tests passed successfully!")
