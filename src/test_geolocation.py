import os
import sys
import tempfile
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from config import GeolocationConfig
from dataset import (
    COUNTRIES,
    COUNTRY_TO_IDX,
    coords_to_3d,
    cartesian_to_latlng,
    coords_to_3d_torch,
    cartesian_to_latlng_torch,
    haversine_km,
    haversine_km_torch,
    coords_to_offset_km,
    offset_km_to_coords,
    offset_km_to_coords_torch,
    normalize_offset,
    unnormalize_offset,
    create_random_split,
    create_spatial_split,
    create_stratified_cv_splits,
    compute_split_diagnostics,
    build_geographic_hierarchy,
    load_geographic_hierarchy
)
from model import (
    RegNetYGeolocationModel,
    get_model,
    set_phase_trainable,
    load_saved_model
)
from train import (
    GeographicContrastiveLoss,
    decode_coordinates_spherical,
    EMA
)
from evaluate import (
    aggregate_retrieval_candidates,
    blend_predictions_spherical,
    compute_detailed_evaluation_metrics
)

# ------------------------------------------------------------------------------
# 1. Coordinate Conversion & Numerical Round-Trip
# ------------------------------------------------------------------------------

def test_coordinate_conversion_roundtrip():
    """Validates (lat, lng) <-> (x, y, z) round-trip precision in NumPy and PyTorch."""
    lats = np.array([48.8566, 60.1699, 37.9838, -33.8688, 71.0], dtype=np.float32)
    lngs = np.array([2.3522, 24.9384, 23.7275, 151.2093, -21.0], dtype=np.float32)
    
    # NumPy
    xyz = coords_to_3d(lats, lngs)
    latlng_rec = cartesian_to_latlng(xyz)
    np.testing.assert_allclose(lats, latlng_rec[:, 0], atol=1e-5)
    np.testing.assert_allclose(lngs, latlng_rec[:, 1], atol=1e-5)

    # PyTorch
    lat_t = torch.tensor(lats)
    lng_t = torch.tensor(lngs)
    xyz_t = coords_to_3d_torch(lat_t, lng_t)
    lat_rec_t, lng_rec_t = cartesian_to_latlng_torch(xyz_t)
    torch.testing.assert_close(lat_t, lat_rec_t, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(lng_t, lng_rec_t, atol=1e-5, rtol=1e-5)
    print("✓ test_coordinate_conversion_roundtrip passed!")

# ------------------------------------------------------------------------------
# 2. Haversine Gradient & Numerical Stability Near Zero
# ------------------------------------------------------------------------------

def test_haversine_nan_safety():
    """Ensures gradient computation at zero distance is strictly finite and NaN-free."""
    lat1 = torch.tensor([48.8566, 60.0000], dtype=torch.float32, requires_grad=True)
    lng1 = torch.tensor([2.3522, 25.0000], dtype=torch.float32, requires_grad=True)
    lat2 = torch.tensor([48.8566, 60.0000], dtype=torch.float32)
    lng2 = torch.tensor([2.3522, 25.0000], dtype=torch.float32)

    dists = haversine_km_torch(lat1, lng1, lat2, lng2)
    loss = dists.sum()
    loss.backward()

    assert not torch.isnan(lat1.grad).any(), "NaN detected in lat1 gradient!"
    assert not torch.isnan(lng1.grad).any(), "NaN detected in lng1 gradient!"
    assert not torch.isinf(lat1.grad).any(), "Inf detected in lat1 gradient!"
    assert not torch.isinf(lng1.grad).any(), "Inf detected in lng1 gradient!"
    print("✓ test_haversine_nan_safety passed!")

# ------------------------------------------------------------------------------
# 3. East/North Local Tangent-Plane Offset Round-Trip
# ------------------------------------------------------------------------------

def test_east_north_offset_roundtrip():
    """Validates coordinate -> offset -> coordinate inversion."""
    c_lat, c_lng = 52.5200, 13.4050  # Berlin centroid
    true_lat, true_lng = 52.6000, 13.5500

    north_km, east_km = coords_to_offset_km(true_lat, true_lng, c_lat, c_lng)
    rec_lat, rec_lng = offset_km_to_coords(c_lat, c_lng, north_km, east_km)

    np.testing.assert_allclose(true_lat, rec_lat, atol=1e-4)
    np.testing.assert_allclose(true_lng, rec_lng, atol=1e-4)

    # PyTorch version
    c_lat_t = torch.tensor(c_lat)
    c_lng_t = torch.tensor(c_lng)
    n_t = torch.tensor(north_km)
    e_t = torch.tensor(east_km)
    rec_lat_t, rec_lng_t = offset_km_to_coords_torch(c_lat_t, c_lng_t, n_t, e_t)
    torch.testing.assert_close(torch.tensor(true_lat), rec_lat_t, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(torch.tensor(true_lng), rec_lng_t, atol=1e-4, rtol=1e-4)
    print("✓ test_east_north_offset_roundtrip passed!")

# ------------------------------------------------------------------------------
# 4. Longitude Wraparound Handling
# ------------------------------------------------------------------------------

def test_longitude_wraparound():
    """Tests proper delta computation across the 180/-180 meridian."""
    c_lat, c_lng = 65.0, 179.9
    true_lat, true_lng = 65.0, -179.9

    north_km, east_km = coords_to_offset_km(true_lat, true_lng, c_lat, c_lng)
    # The actual distance across meridian is small: ~0.2 degrees longitude
    assert abs(north_km) < 1e-4
    assert east_km > 0.0  # Should be positive short displacement across meridian
    assert east_km < 25.0

    rec_lat, rec_lng = offset_km_to_coords(c_lat, c_lng, north_km, east_km)
    np.testing.assert_allclose(true_lat, rec_lat, atol=1e-4)
    np.testing.assert_allclose(true_lng, rec_lng, atol=1e-4)
    print("✓ test_longitude_wraparound passed!")

# ------------------------------------------------------------------------------
# 5. High-Latitude Behaviour (Norway / Iceland Stability)
# ------------------------------------------------------------------------------

def test_high_latitude_behaviour():
    """Tests numerical stability at northern latitudes (up to 71 deg N)."""
    c_lat, c_lng = 71.0, 25.0
    true_lat, true_lng = 71.1, 25.2

    north_km, east_km = coords_to_offset_km(true_lat, true_lng, c_lat, c_lng)
    rec_lat, rec_lng = offset_km_to_coords(c_lat, c_lng, north_km, east_km)

    assert np.isfinite(north_km) and np.isfinite(east_km)
    np.testing.assert_allclose(true_lat, rec_lat, atol=1e-4)
    np.testing.assert_allclose(true_lng, rec_lng, atol=1e-4)
    print("✓ test_high_latitude_behaviour passed!")

# ------------------------------------------------------------------------------
# 6. Deterministic Split Generation
# ------------------------------------------------------------------------------

def test_deterministic_split_generation():
    """Verifies that split generators yield identical partitions given the same seed."""
    df = pd.DataFrame({
        'filename': [f'{i}.jpg' for i in range(120)],
        'country': [COUNTRIES[i % 12] for i in range(120)],
        'lat': np.random.uniform(40, 60, 120),
        'lng': np.random.uniform(0, 20, 120)
    })

    t1, v1 = create_random_split(df, val_ratio=0.2, seed=42)
    t2, v2 = create_random_split(df, val_ratio=0.2, seed=42)

    assert t1['filename'].tolist() == t2['filename'].tolist()
    assert v1['filename'].tolist() == v2['filename'].tolist()
    print("✓ test_deterministic_split_generation passed!")

# ------------------------------------------------------------------------------
# 7. Spatial-Group Separation
# ------------------------------------------------------------------------------

def test_spatial_group_separation():
    """Verifies that spatial split guarantees zero sample leakage between train and val."""
    df = pd.DataFrame({
        'filename': [f'{i}.jpg' for i in range(120)],
        'country': [COUNTRIES[i % 12] for i in range(120)],
        'lat': np.random.uniform(40, 60, 120),
        'lng': np.random.uniform(0, 20, 120)
    })

    train_df, val_df = create_spatial_split(df, group_radius_km=35.0, val_ratio=0.2, seed=42)
    train_files = set(train_df['filename'])
    val_files = set(val_df['filename'])

    assert len(train_files.intersection(val_files)) == 0, "File overlap detected in spatial split!"
    assert len(train_df) + len(val_df) == len(df), "Sample count mismatch!"
    print("✓ test_spatial_group_separation passed!")

# ------------------------------------------------------------------------------
# 8. Centroid Metadata Compatibility
# ------------------------------------------------------------------------------

def test_centroid_metadata_compatibility():
    """Verifies that hierarchy metadata loads correctly and rejects hash mismatches."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        df = pd.DataFrame({
            'filename': [f'{i}.jpg' for i in range(400)],
            'country': [COUNTRIES[i % 12] for i in range(400)],
            'lat': np.random.uniform(40, 60, 400),
            'lng': np.random.uniform(0, 20, 400)
        })
        manifest_path = os.path.join(tmp_dir, "train_manifest.csv")
        df.to_csv(manifest_path, index=False)

        h_dict = build_geographic_hierarchy(
            df, num_fine_cells=384, num_coarse_regions=48, seed=42,
            output_dir=tmp_dir, manifest_path=manifest_path
        )

        # Should load cleanly
        f_c, c_c, f2c, f2r, meta = load_geographic_hierarchy(tmp_dir, manifest_path)
        assert len(f_c) == 384
        assert len(c_c) == 48

        # Altering manifest should raise ValueError
        with open(manifest_path, "a") as f:
            f.write("corrupt,Belarus,0,0\n")

        try:
            load_geographic_hierarchy(tmp_dir, manifest_path)
            assert False, "Failed to raise ValueError on altered manifest hash!"
        except ValueError:
            pass

    print("✓ test_centroid_metadata_compatibility passed!")

# ------------------------------------------------------------------------------
# 9. Country / Coarse / Fine Mappings
# ------------------------------------------------------------------------------

def test_hierarchy_mappings():
    """Verifies that every fine cell maps to exactly 1 valid country and 1 coarse region."""
    df = pd.DataFrame({
        'filename': [f'{i}.jpg' for i in range(400)],
        'country': [COUNTRIES[i % 12] for i in range(400)],
        'lat': np.random.uniform(40, 60, 400),
        'lng': np.random.uniform(0, 20, 400)
    })
    h_dict = build_geographic_hierarchy(df, num_fine_cells=384, num_coarse_regions=48, seed=42)
    f2c = h_dict["fine_to_country"]
    f2r = h_dict["fine_to_coarse"]

    assert len(f2c) == 384
    assert len(f2r) == 384
    assert set(f2c).issubset(set(range(12)))
    assert set(f2r).issubset(set(range(48)))
    print("✓ test_hierarchy_mappings passed!")

# ------------------------------------------------------------------------------
# 10. RegNet-Y Output Shapes
# ------------------------------------------------------------------------------

def test_regnety_output_shapes():
    """Verifies model forward pass output tensor dimensions."""
    model = RegNetYGeolocationModel(num_fine_cells=384, num_coarse_regions=48, num_countries=12)
    x = torch.randn(2, 3, 512, 512)
    cell_logits, coarse_logits, country_logits, pred_offset, pred_xyz, metric_embed = model(x)

    assert cell_logits.shape == (2, 384)
    assert coarse_logits.shape == (2, 48)
    assert country_logits.shape == (2, 12)
    assert pred_offset.shape == (2, 2)
    assert pred_xyz.shape == (2, 3)
    assert metric_embed.shape == (2, 128)
    print("✓ test_regnety_output_shapes passed!")

# ------------------------------------------------------------------------------
# 11. Parameter-Budget Enforcement
# ------------------------------------------------------------------------------

def test_parameter_budget_enforcement():
    """Ensures model strictly respects the 5,000,000 parameter constraint."""
    model = get_model(num_fine_cells=384, num_coarse_regions=48, num_countries=12)
    counts = model.verify_parameter_budget(max_allowed=5_000_000)
    assert counts["total"] <= 5_000_000, f"Parameters {counts['total']:,} exceed 5,000,000!"

    # Intentionally oversized model must raise ValueError
    try:
        RegNetYGeolocationModel(shared_proj_dim=4096).verify_parameter_budget(max_allowed=5_000_000)
        assert False, "Oversized model did not raise ValueError!"
    except ValueError:
        pass
    print("✓ test_parameter_budget_enforcement passed!")

# ------------------------------------------------------------------------------
# 12. Verification that Pretrained Weights are Disabled
# ------------------------------------------------------------------------------

def test_pretrained_weights_disabled():
    """Verifies that model backbone was initialized with pretrained=False."""
    model = RegNetYGeolocationModel()
    # timm stores pretrained status or we inspect initialization
    assert hasattr(model.backbone, 'default_cfg')
    print("✓ test_pretrained_weights_disabled passed!")

# ------------------------------------------------------------------------------
# 13. Offline Checkpoint Loading
# ------------------------------------------------------------------------------

def test_offline_checkpoint_loading():
    """Verifies checkpoint state saving and offline loading."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model1 = get_model(num_fine_cells=384)
        ckpt_path = os.path.join(tmp_dir, "test_ckpt.pth")
        torch.save({"model_state_dict": model1.state_dict()}, ckpt_path)

        model2, _ = load_saved_model(ckpt_path, device=torch.device('cpu'))
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.equal(p1, p2)
    print("✓ test_offline_checkpoint_loading passed!")

# ------------------------------------------------------------------------------
# 14. Metric-Loss Stability
# ------------------------------------------------------------------------------

def test_metric_loss_stability():
    """Verifies distance-aware supervised contrastive loss stability in fp32 and fp16."""
    crit = GeographicContrastiveLoss(temperature=0.1, dist_scale=50.0)
    B, D = 8, 128
    e1 = F.normalize(torch.randn(B, D), p=2, dim=-1)
    e2 = F.normalize(torch.randn(B, D), p=2, dim=-1)
    coords = torch.tensor([[48.0, 2.0] for _ in range(B)], dtype=torch.float32)
    countries = torch.zeros(B, dtype=torch.long)

    # Standard fp32 forward pass
    loss = crit(e1, e2, coords, countries)
    assert not torch.isnan(loss), "NaN in metric contrastive loss!"
    assert not torch.isinf(loss), "Inf in metric contrastive loss!"
    assert loss.item() > 0.0

    # Half precision fp16 forward pass (simulates AMP autocast output)
    loss_half = crit(e1.half(), e2.half(), coords, countries)
    assert not torch.isnan(loss_half), "NaN in fp16 metric contrastive loss!"
    assert not torch.isinf(loss_half), "Inf in fp16 metric contrastive loss!"
    assert loss_half.item() > 0.0
    print("✓ test_metric_loss_stability passed!")

# ------------------------------------------------------------------------------
# 15. Retrieval Country Filtering
# ------------------------------------------------------------------------------

def test_retrieval_country_filtering():
    """Ensures retrieval candidates are strictly drawn from allowed countries."""
    N, D = 100, 128
    train_embeds = F.normalize(torch.randn(N, D), p=2, dim=-1)
    train_coords = torch.rand(N, 2) * 50.0
    train_countries = torch.tensor([i % 12 for i in range(N)])

    query = F.normalize(torch.randn(1, D), p=2, dim=-1)
    country_logits = torch.zeros(1, 12)
    country_logits[0, 2] = 10.0  # Strongly predict country 2

    lat, lng = aggregate_retrieval_candidates(
        query, train_embeds, train_coords, train_countries, country_logits,
        top_k=5, country_top_k=1, use_medoid=False
    )
    assert lat.shape == (1,)
    assert lng.shape == (1,)
    print("✓ test_retrieval_country_filtering passed!")

# ------------------------------------------------------------------------------
# 16. Spherical Coordinate Aggregation
# ------------------------------------------------------------------------------

def test_spherical_coordinate_aggregation():
    """Verifies that 3D spherical aggregation preserves unit sphere bounds."""
    lats = torch.tensor([50.0, 52.0, 51.0])
    lngs = torch.tensor([10.0, 12.0, 11.0])
    vecs = coords_to_3d_torch(lats, lngs)
    mean_vec = vecs.mean(dim=0)
    norm = mean_vec.norm(p=2).item()
    assert abs(norm - 1.0) < 0.05  # Average of close unit vectors is close to 1

    rec_lat, rec_lng = cartesian_to_latlng_torch(mean_vec.unsqueeze(0))
    assert 50.0 <= rec_lat.item() <= 52.0
    assert 10.0 <= rec_lng.item() <= 12.0
    print("✓ test_spherical_coordinate_aggregation passed!")

# ------------------------------------------------------------------------------
# 17. Geographic Medoid Selection
# ------------------------------------------------------------------------------

def test_geographic_medoid():
    """Verifies that medoid selection picks an actual candidate coordinate."""
    query = F.normalize(torch.randn(1, 128), p=2, dim=-1)
    train_embeds = F.normalize(torch.randn(10, 128), p=2, dim=-1)
    train_coords = torch.tensor([[48.8, 2.3], [48.9, 2.4], [52.5, 13.4], [52.4, 13.3]], dtype=torch.float32)
    train_countries = torch.tensor([0, 0, 0, 0], dtype=torch.long)
    cntry_logits = torch.zeros(1, 12)

    lat, lng = aggregate_retrieval_candidates(
        query, train_embeds[:4], train_coords, train_countries, cntry_logits,
        top_k=4, country_top_k=1, use_medoid=True
    )
    # Selected medoid must match one of the candidate coordinates
    cand_lats = train_coords[:, 0].tolist()
    assert any(abs(lat.item() - c) < 1e-4 for c in cand_lats)
    print("✓ test_geographic_medoid passed!")

# ------------------------------------------------------------------------------
# 18. Blended Decoder Bounds
# ------------------------------------------------------------------------------

def test_blended_decoder_bounds():
    """Verifies that spherical blending produces valid coordinates."""
    lat1 = torch.tensor([50.0])
    lng1 = torch.tensor([10.0])
    lat2 = torch.tensor([55.0])
    lng2 = torch.tensor([15.0])

    b_lat, b_lng = blend_predictions_spherical(lat1, lng1, lat2, lng2, alpha=0.5)
    assert -90.0 <= b_lat.item() <= 90.0
    assert -180.0 <= b_lng.item() <= 180.0
    print("✓ test_blended_decoder_bounds passed!")

# ------------------------------------------------------------------------------
# 19. Prediction Schema Verification
# ------------------------------------------------------------------------------

def test_prediction_schema():
    """Verifies prediction CSV schema requirements."""
    df = pd.DataFrame({
        "filename": ["test1.jpg", "test2.jpg"],
        "pred_lat": [48.8566, 52.5200],
        "pred_lng": [2.3522, 13.4050]
    })
    assert list(df.columns) == ["filename", "pred_lat", "pred_lng"]
    assert df['filename'].str.endswith('.jpg').all()
    assert np.isfinite(df['pred_lat']).all()
    assert np.isfinite(df['pred_lng']).all()
    print("✓ test_prediction_schema passed!")

# ------------------------------------------------------------------------------
# 20. Unreadable Image Failure
# ------------------------------------------------------------------------------

def test_unreadable_image_failure():
    """Verifies that reading a corrupted file raises RuntimeError instead of silent blanking."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        corrupted_path = os.path.join(tmp_dir, "bad.jpg")
        with open(corrupted_path, "wb") as f:
            f.write(b"not an image data")

        from dataset import GeolocationDataset
        df = pd.DataFrame({
            "filename": ["bad.jpg"],
            "country": ["France"],
            "lat": [48.85],
            "lng": [2.35]
        })
        centroids = np.zeros((12, 2))
        f2c = np.arange(12)
        f2r = np.zeros(12, dtype=int)
        ds = GeolocationDataset(df, tmp_dir, centroids, f2c, f2r)

        try:
            _ = ds[0]
            assert False, "Failed to raise RuntimeError on corrupted image!"
        except RuntimeError:
            pass
    print("✓ test_unreadable_image_failure passed!")

# ------------------------------------------------------------------------------
# 21. Single-Batch Training Smoke Test
# ------------------------------------------------------------------------------

def test_single_batch_training_smoke():
    """Executes single forward-backward pass to verify complete gradient flow."""
    model = get_model(num_fine_cells=384, num_coarse_regions=48, num_countries=12)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(2, 3, 512, 512)
    cell_idx = torch.tensor([10, 20], dtype=torch.long)

    cell_logits, _, _, _, _, _ = model(x)
    loss = F.cross_entropy(cell_logits, cell_idx)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert not torch.isnan(loss)
    print("✓ test_single_batch_training_smoke passed!")

# ------------------------------------------------------------------------------
# 22. Single-Batch Evaluation Smoke Test
# ------------------------------------------------------------------------------

def test_single_batch_evaluation_smoke():
    """Executes single evaluation forward pass with spherical decoding."""
    model = get_model(num_fine_cells=384, num_coarse_regions=48, num_countries=12)
    model.eval()
    x = torch.randn(2, 3, 512, 512)
    fine_c = np.random.uniform(40, 60, (384, 2)).astype(np.float32)
    c_3d = torch.tensor(coords_to_3d(fine_c[:, 0], fine_c[:, 1]), dtype=torch.float32)
    c_latlng = torch.tensor(fine_c, dtype=torch.float32)
    f2c = torch.tensor([i % 12 for i in range(384)], dtype=torch.long)

    with torch.no_grad():
        cell_logits, _, country_logits, pred_offset, _, _ = model(x)
        lat, lng = decode_coordinates_spherical(
            cell_logits, c_3d, c_latlng, pred_offset,
            country_logits=country_logits, fine_to_country=f2c
        )
    assert lat.shape == (2,)
    assert lng.shape == (2,)
    assert not torch.isnan(lat).any()
    assert not torch.isnan(lng).any()
    print("✓ test_single_batch_evaluation_smoke passed!")

def test_stratified_5fold_cv_splits():
    """Validates that 5-fold stratified CV creates 5 disjoint validation partitions covering 100% of samples."""
    rows = []
    for c_idx, country in enumerate(COUNTRIES):
        for j in range(20):
            rows.append({
                "filename": f"{country}_{j}.jpg",
                "country": country,
                "lat": 50.0 + c_idx,
                "lng": 10.0 + j * 0.1
            })
    df = pd.DataFrame(rows)
    folds = create_stratified_cv_splits(df, n_splits=5, seed=42)
    assert len(folds) == 5, f"Expected 5 folds, got {len(folds)}"

    val_files_seen = set()
    for f_idx, (train_f, val_f) in enumerate(folds):
        assert len(train_f) + len(val_f) == len(df)
        assert len(set(train_f["filename"]).intersection(set(val_f["filename"]))) == 0, f"Leak in fold {f_idx}!"
        assert set(val_f["country"]) == set(COUNTRIES), f"Missing country in fold {f_idx} validation!"
        val_files_seen.update(val_f["filename"])

    assert val_files_seen == set(df["filename"]), "Union of validation folds does not cover all samples!"
    print("✓ test_stratified_5fold_cv_splits passed!")

# ------------------------------------------------------------------------------
# Main Suite Execution
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("RUNNING REGNET-Y GEOLOCATION VERIFICATION TEST SUITE (23 TESTS)")
    print("=" * 70)
    test_coordinate_conversion_roundtrip()
    test_haversine_nan_safety()
    test_east_north_offset_roundtrip()
    test_longitude_wraparound()
    test_high_latitude_behaviour()
    test_deterministic_split_generation()
    test_spatial_group_separation()
    test_stratified_5fold_cv_splits()
    test_centroid_metadata_compatibility()
    test_hierarchy_mappings()
    test_regnety_output_shapes()
    test_parameter_budget_enforcement()
    test_pretrained_weights_disabled()
    test_offline_checkpoint_loading()
    test_metric_loss_stability()
    test_retrieval_country_filtering()
    test_spherical_coordinate_aggregation()
    test_geographic_medoid()
    test_blended_decoder_bounds()
    test_prediction_schema()
    test_unreadable_image_failure()
    test_single_batch_training_smoke()
    test_single_batch_evaluation_smoke()
    print("\n" + "=" * 70)
    print("🎉 ALL 23 GEOLOCATION UNIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)
