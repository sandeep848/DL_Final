import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from dataset import get_dataloaders, unnormalize_coords, coords_to_3d
from model import get_model

def coords_to_3d_torch(lat_deg, lng_deg):
    """Converts PyTorch lat/lng in degrees to 3D unit sphere Cartesian coordinates (x, y, z)."""
    lat_rad = torch.deg2rad(lat_deg)
    lng_rad = torch.deg2rad(lng_deg)
    x = torch.cos(lat_rad) * torch.cos(lng_rad)
    y = torch.cos(lat_rad) * torch.sin(lng_rad)
    z = torch.sin(lat_rad)
    return torch.stack([x, y, z], dim=-1)

def haversine_km_torch(lat1, lng1, lat2, lng2, R=6371.0088):
    """PyTorch implementation of the haversine distance (lat/lng in decimal degrees)."""
    lat1, lng1, lat2, lng2 = map(torch.deg2rad, (lat1, lng1, lat2, lng2))
    d = (torch.sin((lat2 - lat1) / 2) ** 2
         + torch.cos(lat1) * torch.cos(lat2) * torch.sin((lng2 - lng1) / 2) ** 2)
    d_clamped = torch.clamp(d, 1e-7, 1.0 - 1e-7)
    return 2 * R * torch.arcsin(torch.sqrt(d_clamped))

def soft_expectation_coords(cell_logits, centroids_tensor, pred_offset, country_logits=None, cell_to_country=None, topk=12, temperature=0.18, centroid_dist_matrix=None, radius_km=100.0):
    """
    Computes Hierarchically-Conditioned Local Neighborhood Soft Expectation over Voronoi centroids + continuous offset.
    Country log-probabilities are added to cell logits to suppress cross-country hallucinations.
    """
    if country_logits is not None and cell_to_country is not None:
        country_log_probs = F.log_softmax(country_logits, dim=-1)
        country_boost = country_log_probs[:, cell_to_country]
        cell_logits = cell_logits + 2.0 * country_boost

    top1_idx = torch.argmax(cell_logits, dim=-1)
    B = cell_logits.size(0)

    if centroid_dist_matrix is not None:
        masked_logits = cell_logits.clone()
        for i in range(B):
            center_cell = top1_idx[i]
            dists = centroid_dist_matrix[center_cell]
            local_mask = dists <= radius_km
            masked_logits[i, ~local_mask] = -1000.0
        scaled_logits = masked_logits / temperature
    else:
        scaled_logits = cell_logits / temperature

    if topk is not None and topk < cell_logits.size(-1):
        topk_logits, topk_indices = torch.topk(scaled_logits, k=topk, dim=-1)
        topk_probs = F.softmax(topk_logits, dim=-1)
        topk_centroids = centroids_tensor[topk_indices]
        soft_lat = torch.sum(topk_probs * topk_centroids[:, :, 0], dim=-1)
        soft_lng = torch.sum(topk_probs * topk_centroids[:, :, 1], dim=-1)
    else:
        probs = F.softmax(scaled_logits, dim=-1)
        soft_lat = torch.matmul(probs, centroids_tensor[:, 0])
        soft_lng = torch.matmul(probs, centroids_tensor[:, 1])
    
    # Continuous Tanh local offset refinement
    lat_rad = torch.deg2rad(soft_lat)
    cos_lat = torch.clamp(torch.cos(lat_rad), min=0.2)
    lng_scale = 0.5 / cos_lat
    
    final_lat = soft_lat + pred_offset[:, 0] * 0.5
    final_lng = soft_lng + pred_offset[:, 1] * lng_scale
    return final_lat, final_lng

def build_centroid_dist_matrix(cluster_centroids, device):
    """Precomputes NxN pairwise Haversine distance matrix between all centroids on PyTorch GPU."""
    lat1 = torch.tensor(cluster_centroids[:, 0], dtype=torch.float32)
    lng1 = torch.tensor(cluster_centroids[:, 1], dtype=torch.float32)
    lat1_rad, lng1_rad = torch.deg2rad(lat1), torch.deg2rad(lng1)
    dlat = lat1_rad.unsqueeze(1) - lat1_rad.unsqueeze(0)
    dlng = lng1_rad.unsqueeze(1) - lng1_rad.unsqueeze(0)
    a = torch.sin(dlat/2)**2 + torch.cos(lat1_rad.unsqueeze(1)) * torch.cos(lat1_rad.unsqueeze(0)) * torch.sin(dlng/2)**2
    dist_matrix = (2 * 6371.0088 * torch.arcsin(torch.sqrt(torch.clamp(a, 1e-7, 1.0)))).to(device)
    return dist_matrix

def compute_sharp_spatial_targets(targets_cell, dist_matrix, p_true=0.85, neighbor_scale=15.0):
    """
    Computes sharp target distribution: 85% mass on the exact true Voronoi cell,
    with remaining 15% exponentially focused on immediate neighbors within 15 km.
    Eliminates the diffuse entropy floor and forces sharp, confident predictions.
    """
    dists = dist_matrix[targets_cell]
    neighbor_weights = torch.exp(-dists / neighbor_scale)
    neighbor_weights.scatter_(1, targets_cell.unsqueeze(1), 0.0)
    neighbor_sum = neighbor_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    neighbor_probs = neighbor_weights / neighbor_sum * (1.0 - p_true)
    
    targets = neighbor_probs.clone()
    targets.scatter_(1, targets_cell.unsqueeze(1), p_true)
    return targets

def train_epoch(model, loader, optimizer, criterion_country, centroids_tensor, dist_matrix, cell_to_country_tensor, scaler, device):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for images, targets_coords, targets_cell, targets_country in pbar:
        images = images.to(device)
        targets_coords = targets_coords.to(device)
        targets_cell = targets_cell.to(device)
        targets_country = targets_country.to(device)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            cell_logits, pred_offset, pred_xyz, country_logits = model(images)
            
            # Ground truth coordinates
            true_lat, true_lng = unnormalize_coords(targets_coords)
            true_xyz = coords_to_3d_torch(true_lat, true_lng)
            
            # 1. Sharp Spatial Classification Loss (85% on true cell, 15% on immediate neighbors <15km)
            sharp_targets = compute_sharp_spatial_targets(targets_cell, dist_matrix, p_true=0.85, neighbor_scale=15.0)
            loss_cell = torch.sum(-sharp_targets * F.log_softmax(cell_logits, dim=-1), dim=-1).mean()
            
            # 2. Country Classification Loss
            loss_country = criterion_country(country_logits, targets_country)
            
            # 3. Direct Coordinate Offset Supervision (L1 loss relative to target centroid)
            target_centroids = centroids_tensor[targets_cell]
            delta_lat = true_lat - target_centroids[:, 0]
            delta_lng = true_lng - target_centroids[:, 1]
            lat_rad = torch.deg2rad(true_lat)
            cos_lat = torch.clamp(torch.cos(lat_rad), min=0.2)
            lng_scale = 0.5 / cos_lat
            target_offset = torch.stack([
                torch.clamp(delta_lat / 0.5, -1.0, 1.0),
                torch.clamp(delta_lng / lng_scale, -1.0, 1.0)
            ], dim=-1)
            loss_offset = F.smooth_l1_loss(pred_offset, target_offset)
            
            # 4. Decoded coordinates for Focal Log-Haversine monitoring & guidance
            final_lat, final_lng = soft_expectation_coords(
                cell_logits, centroids_tensor, pred_offset,
                country_logits=country_logits, cell_to_country=cell_to_country_tensor,
                topk=12, temperature=0.05, centroid_dist_matrix=None, radius_km=100.0
            )
            dists_km = haversine_km_torch(final_lat, final_lng, true_lat, true_lng)
            focal_weight = 1.0 - torch.exp(-dists_km / 50.0)
            loss_log_hav = (focal_weight * torch.log(1.0 + dists_km / 5.0)).mean()
            loss_xyz = F.mse_loss(pred_xyz, true_xyz)
            
            # Balanced multi-task loss with high country supervision (eliminates foreign predictions)
            loss = 2.0 * loss_cell + 4.0 * loss_country + 2.0 * loss_offset + 1.0 * loss_log_hav + 0.2 * loss_xyz
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * images.size(0)
        pbar.set_postfix({'loss': f"{loss.item():.3f}", 'cell': f"{loss_cell.item():.2f}", 'cntry': f"{loss_country.item():.2f}", 'hav_km': f"{dists_km.median().item():.0f}"})
        
    return running_loss / len(loader.dataset)

def validate(model, loader, criterion_country, centroids_tensor, dist_matrix, cell_to_country_tensor, device):
    model.eval()
    running_loss = 0.0
    all_dists = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validating", leave=False)
        for images, targets_coords, targets_cell, targets_country in pbar:
            images = images.to(device)
            targets_coords = targets_coords.to(device)
            targets_cell = targets_cell.to(device)
            targets_country = targets_country.to(device)
            
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                cell_logits, pred_offset, pred_xyz, country_logits = model(images)
                
                final_lat, final_lng = soft_expectation_coords(
                    cell_logits, centroids_tensor, pred_offset,
                    country_logits=country_logits, cell_to_country=cell_to_country_tensor,
                    topk=12, temperature=0.05, centroid_dist_matrix=dist_matrix, radius_km=100.0
                )
                true_lat, true_lng = unnormalize_coords(targets_coords)
                true_xyz = coords_to_3d_torch(true_lat, true_lng)
                
                # 1. Sharp Spatial Classification Loss
                sharp_targets = compute_sharp_spatial_targets(targets_cell, dist_matrix, p_true=0.85, neighbor_scale=15.0)
                loss_cell = torch.sum(-sharp_targets * F.log_softmax(cell_logits, dim=-1), dim=-1).mean()
                
                loss_country = criterion_country(country_logits, targets_country)
                
                # Direct offset supervision
                target_centroids = centroids_tensor[targets_cell]
                delta_lat = true_lat - target_centroids[:, 0]
                delta_lng = true_lng - target_centroids[:, 1]
                lat_rad = torch.deg2rad(true_lat)
                cos_lat = torch.clamp(torch.cos(lat_rad), min=0.2)
                lng_scale = 0.5 / cos_lat
                target_offset = torch.stack([
                    torch.clamp(delta_lat / 0.5, -1.0, 1.0),
                    torch.clamp(delta_lng / lng_scale, -1.0, 1.0)
                ], dim=-1)
                loss_offset = F.smooth_l1_loss(pred_offset, target_offset)
                
                dists_km = haversine_km_torch(final_lat, final_lng, true_lat, true_lng)
                focal_weight = 1.0 - torch.exp(-dists_km / 50.0)
                loss_log_hav = (focal_weight * torch.log(1.0 + dists_km / 5.0)).mean()
                loss_xyz = F.mse_loss(pred_xyz, true_xyz)
                
                loss = 2.0 * loss_cell + 4.0 * loss_country + 2.0 * loss_offset + 1.0 * loss_log_hav + 0.2 * loss_xyz
                
            running_loss += loss.item() * images.size(0)
            all_dists.append(dists_km.cpu().numpy())
            
    val_loss = running_loss / len(loader.dataset)
    all_dists = np.concatenate(all_dists)
    median_dist = np.median(all_dists)
    mean_dist = np.mean(all_dists)
    
    return val_loss, median_dist, mean_dist

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(script_dir, ".."))
    csv_file = os.path.join(base_dir, "geo_dataset", "train_labels.csv")
    img_dir = os.path.join(base_dir, "geo_dataset", "train")
    model_save_path = os.path.join(base_dir, "best_model.pth")
    centroids_save_path = os.path.join(base_dir, "hierarchical_cluster_centroids.npy")
    
    # 768 clusters: 64 clusters per country (16 training images/cluster, 16.8 km oracle error)
    n_clusters = 768
    train_loader, val_loader, cluster_centroids, cell_to_country = get_dataloaders(
        csv_file, img_dir, batch_size=24, num_workers=4, n_clusters=n_clusters, centroids_save_path=centroids_save_path
    )
    centroids_tensor = torch.tensor(cluster_centroids, dtype=torch.float32).to(device)
    cell_to_country_tensor = torch.tensor(cell_to_country, dtype=torch.long).to(device)
    dist_matrix = build_centroid_dist_matrix(cluster_centroids, device)
    
    # Primary model: ConvNeXt-V2 Atto (~3.97M parameters total with 768 cells)
    model = get_model(num_cells=len(cluster_centroids), arch="convnextv2").to(device)
    
    criterion_country = nn.CrossEntropyLoss(label_smoothing=0.05)
    
    backbone_params = [p for n, p in model.named_parameters() if 'backbone' in n]
    head_params = [p for n, p in model.named_parameters() if 'backbone' not in n]
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 3e-4},
        {'params': head_params, 'lr': 1e-3}
    ], weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    
    num_epochs = 40
    warmup_epochs = 2
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        else:
            progress = float(epoch - warmup_epochs) / float(num_epochs - warmup_epochs)
            return 0.05 + 0.95 * 0.5 * (1.0 + np.cos(np.pi * progress))
            
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    best_median_dist = float('inf')
    
    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{num_epochs} [lr={current_lr:.6f}]", flush=True)
        train_loss = train_epoch(model, train_loader, optimizer, criterion_country, centroids_tensor, dist_matrix, cell_to_country_tensor, scaler, device)
        val_loss, median_dist, mean_dist = validate(model, val_loader, criterion_country, centroids_tensor, dist_matrix, cell_to_country_tensor, device)
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}", flush=True)
        print(f"Val Median Dist: {median_dist:.2f} km | Val Mean Dist: {mean_dist:.2f} km", flush=True)
        
        if median_dist < best_median_dist:
            best_median_dist = median_dist
            torch.save(model.state_dict(), model_save_path)
            print(f"--> Saved new best model checkpoint with Median Dist: {best_median_dist:.2f} km | Mean Dist: {mean_dist:.2f} km", flush=True)

if __name__ == "__main__":
    main()
