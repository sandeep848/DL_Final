import os
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import load_saved_model
from dataset import get_cluster_centroids
from train import soft_expectation_coords, build_centroid_dist_matrix

class HoldoutDataset(Dataset):
    def __init__(self, img_dir, transform_norm=None):
        self.img_dir = img_dir
        self.transform_norm = transform_norm
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        self.img_names.sort()
        
        self.transform_resize = transforms.Resize((480, 480))
        self.transform_five_crop = transforms.FiveCrop(448)
        self.transform_direct = transforms.Resize((448, 448))
        
    def __len__(self):
        return len(self.img_names)
        
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (448, 448))
            
        img_resized = self.transform_resize(image)
        crops = self.transform_five_crop(img_resized)
        
        # 6-view TTA: 5 spatial crops + 1 full-frame view. NO horizontal flips to preserve language text & road markings!
        tensors_views = [self.transform_norm(crop) for crop in crops]
        tensors_views.append(self.transform_norm(self.transform_direct(image)))
            
        batch_views = torch.stack(tensors_views, dim=0)
        return batch_views, img_name

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(script_dir, ".."))
    train_csv = os.path.join(base_dir, "geo_dataset", "train_labels.csv")
    img_dir = os.path.join(base_dir, "geo_dataset", "holdout_public")
    weights_path = os.path.join(base_dir, "best_model.pth")
    centroids_path = os.path.join(base_dir, "hierarchical_cluster_centroids.npy")
    mapping_path = os.path.join(base_dir, "hierarchical_cluster_centroids_country_map.npy")
    output_csv = os.path.join(base_dir, "predictions.csv")
    
    if not os.path.exists(img_dir):
        print(f"WARNING: Holdout directory not found at {img_dir}.")
        return
        
    if not os.path.exists(weights_path):
        print(f"WARNING: Model weights not found at {weights_path}. Train the model first.")
        return
        
    if not os.path.exists(centroids_path):
        from dataset import get_hierarchical_cluster_centroids
        train_df = pd.read_csv(train_csv)
        cluster_centroids, cell_to_country = get_hierarchical_cluster_centroids(
            train_df, save_path=centroids_path, mapping_save_path=mapping_path, n_clusters=768
        )
    else:
        cluster_centroids = np.load(centroids_path)
        cell_to_country = np.load(mapping_path)
        
    centroids_tensor = torch.tensor(cluster_centroids, dtype=torch.float32).to(device)
    cell_to_country_tensor = torch.tensor(cell_to_country, dtype=torch.long).to(device)
    dist_matrix = build_centroid_dist_matrix(cluster_centroids, device)
    
    transform_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = HoldoutDataset(img_dir, transform_norm=transform_norm)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
    
    model = load_saved_model(weights_path, num_cells=len(cluster_centroids), num_countries=12, device=device)
    
    results = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Predicting (6-View TTA without flips)", leave=False)
        for batch_views, img_names in pbar:
            B, V, C, H, W = batch_views.shape
            batch_flattened = batch_views.view(B * V, C, H, W).to(device)
            
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                cell_logits_flat, offset_flat, _, country_logits_flat = model(batch_flattened)
            
            cell_logits = cell_logits_flat.view(B, V, -1)
            offset = offset_flat.view(B, V, -1)
            country_logits = country_logits_flat.view(B, V, -1)
            
            cell_logits_avg = cell_logits.mean(dim=1)
            offset_avg = offset.mean(dim=1)
            country_logits_avg = country_logits.mean(dim=1)
            
            lat_batch, lng_batch = soft_expectation_coords(
                cell_logits_avg, centroids_tensor, offset_avg,
                country_logits=country_logits_avg, cell_to_country=cell_to_country_tensor,
                topk=12, temperature=0.05, centroid_dist_matrix=dist_matrix, radius_km=100.0
            )
            
            final_lat = lat_batch.cpu().numpy()
            final_lng = lng_batch.cpu().numpy()
            
            for name, lat, lng in zip(img_names, final_lat, final_lng):
                results.append({
                    'filename': name,
                    'pred_lat': float(lat),
                    'pred_lng': float(lng)
                })
                
    df = pd.DataFrame(results)
    df = df[['filename', 'pred_lat', 'pred_lng']]
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")
    print(f"Generated {len(df)} predictions.")

if __name__ == "__main__":
    evaluate()
