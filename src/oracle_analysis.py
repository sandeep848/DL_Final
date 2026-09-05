"""
Consolidated into src/dataset.py.
Run: python src/dataset.py --oracle
"""
from dataset import run_oracle_analysis

if __name__ == "__main__":
    import os
    import argparse
    import pandas as pd
    from config import GeolocationConfig, get_default_config
    from dataset import load_geographic_hierarchy

    parser = argparse.ArgumentParser(description="Oracle Analysis (Forwarder to dataset.py)")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--split", type=str, default="spatial", choices=["spatial", "random"])
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    cfg = GeolocationConfig.load_json(args.config) if args.config else get_default_config()
    train_path = cfg.get_path(cfg.spatial_train_manifest if args.split == "spatial" else cfg.random_train_manifest)
    val_path = cfg.get_path(cfg.spatial_val_manifest if args.split == "spatial" else cfg.random_val_manifest)

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    loaded = False
    if os.path.exists(os.path.join(cfg.exp_dir, "hierarchy_metadata.json")):
        try:
            fine_c, coarse_c, f2c, f2r, _ = load_geographic_hierarchy(cfg.exp_dir, train_path)
            loaded = True
        except ValueError:
            pass
    if not loaded:
        from dataset import build_geographic_hierarchy
        h_dict = build_geographic_hierarchy(
            train_df, num_fine_cells=cfg.num_fine_cells, num_coarse_regions=cfg.num_coarse_regions,
            seed=cfg.seed, output_dir=cfg.exp_dir, config_hash=cfg.compute_config_hash(), manifest_path=train_path
        )
        fine_c = h_dict["fine_centroids"]
        coarse_c = h_dict["coarse_centroids"]
        f2c = h_dict["fine_to_country"]
        f2r = h_dict["fine_to_coarse"]
    out_path = args.output or os.path.join(cfg.exp_dir, "oracle_analysis.json")
    run_oracle_analysis(train_df, val_df, fine_c, coarse_c, f2c, f2r, output_json_path=out_path)
