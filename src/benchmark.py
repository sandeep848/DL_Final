import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from dataset import get_dataloaders, unnormalize_coords, coords_to_3d
from model import get_model

def coords_to_3d_torch(lat_deg, lng_deg):
    lat_rad = torch.deg2rad(lat_deg)
    lng_rad = torch.deg2rad(lng_deg)
    x = torch.cos(lat_rad) * torch.cos(lng_rad)
    y = torch.cos(lat_rad) * torch.sin(lng_rad)
    z = torch.sin(lat_rad)
    return torch.stack([x, y, z], dim=-1)

def haversine_km_torch(lat1, lng1, lat2, lng2, R=6371.0088):
    lat1, lng1, lat2, lng2 = map(torch.deg2rad, (lat1, lng1, lat2, lng2))
    d = (torch.sin((lat2 - lat1) / 2) ** 2
         + torch.cos(lat1) * torch.cos(lat2) * torch.sin((lng2 - lng1) / 2) ** 2)
    d_clamped = torch.clamp(d, 1e-7, 1.0 - 1e-7)
    return 2 * R * torch.arcsin(torch.sqrt(d_clamped))

def soft_expectation_coords(cell_logits, centroids_tensor, pred_offset, country_logits=None, cell_to_country=None, topk=100, temperature=0.1, centroid_dist_matrix=None, radius_km=100.0):
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
    
    lat_rad = torch.deg2rad(soft_lat)
    cos_lat = torch.clamp(torch.cos(lat_rad), min=0.2)
    lng_scale = 0.5 / cos_lat
    
    final_lat = soft_lat + pred_offset[:, 0] * 0.5
    final_lng = soft_lng + pred_offset[:, 1] * lng_scale
    return final_lat, final_lng

def build_centroid_dist_matrix(cluster_centroids, device):
    lat1 = torch.tensor(cluster_centroids[:, 0], dtype=torch.float32)
    lng1 = torch.tensor(cluster_centroids[:, 1], dtype=torch.float32)
    lat1_rad, lng1_rad = torch.deg2rad(lat1), torch.deg2rad(lng1)
    dlat = lat1_rad.unsqueeze(1) - lat1_rad.unsqueeze(0)
    dlng = lng1_rad.unsqueeze(1) - lng1_rad.unsqueeze(0)
    a = torch.sin(dlat/2)**2 + torch.cos(lat1_rad.unsqueeze(1)) * torch.cos(lat1_rad.unsqueeze(0)) * torch.sin(dlng/2)**2
    dist_matrix = (2 * 6371.0088 * torch.arcsin(torch.sqrt(torch.clamp(a, 1e-7, 1.0)))).to(device)
    return dist_matrix

def train_epoch(model, loader, optimizer, criterion_country, centroids_tensor, dist_matrix, cell_to_country_tensor, scaler, device):
    model.train()
    running_loss = 0.0
    
    for images, targets_coords, targets_cell, targets_country in loader:
        images = images.to(device)
        targets_coords = targets_coords.to(device)
        targets_cell = targets_cell.to(device)
        targets_country = targets_country.to(device)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            cell_logits, pred_offset, pred_xyz, country_logits = model(images)
            
            final_lat, final_lng = soft_expectation_coords(
                cell_logits, centroids_tensor, pred_offset, 
                country_logits=country_logits, cell_to_country=cell_to_country_tensor,
                topk=100, temperature=0.1, centroid_dist_matrix=None
            )
            true_lat, true_lng = unnormalize_coords(targets_coords)
            true_xyz = coords_to_3d_torch(true_lat, true_lng)
            
            cell_dists = dist_matrix[targets_cell]
            soft_targets = 1.0 / (1.0 + (cell_dists / 30.0)**2)
            soft_targets = soft_targets / soft_targets.sum(dim=-1, keepdim=True)
            loss_cell = torch.sum(-soft_targets * F.log_softmax(cell_logits, dim=-1), dim=-1).mean()
            
            loss_country = criterion_country(country_logits, targets_country)
            
            dists_km = haversine_km_torch(final_lat, final_lng, true_lat, true_lng)
            focal_weight = 1.0 - torch.exp(-dists_km / 50.0)
            loss_log_hav = (focal_weight * torch.log(1.0 + dists_km / 5.0)).mean()
            loss_xyz = F.mse_loss(pred_xyz, true_xyz)
            
            loss = 5.0 * loss_cell + 1.0 * loss_country + 1.0 * loss_log_hav + 0.2 * loss_xyz
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * images.size(0)
        
    return running_loss / len(loader.dataset)

def validate(model, loader, criterion_country, centroids_tensor, dist_matrix, cell_to_country_tensor, device):
    model.eval()
    running_loss = 0.0
    all_dists = []
    country_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, targets_coords, targets_cell, targets_country in loader:
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
                
                cell_dists = dist_matrix[targets_cell]
                soft_targets = 1.0 / (1.0 + (cell_dists / 30.0)**2)
                soft_targets = soft_targets / soft_targets.sum(dim=-1, keepdim=True)
                loss_cell = torch.sum(-soft_targets * F.log_softmax(cell_logits, dim=-1), dim=-1).mean()
                
                loss_country = criterion_country(country_logits, targets_country)
                dists_km = haversine_km_torch(final_lat, final_lng, true_lat, true_lng)
                focal_weight = 1.0 - torch.exp(-dists_km / 50.0)
                loss_log_hav = (focal_weight * torch.log(1.0 + dists_km / 5.0)).mean()
                loss_xyz = F.mse_loss(pred_xyz, true_xyz)
                
                loss = 5.0 * loss_cell + 1.0 * loss_country + 1.0 * loss_log_hav + 0.2 * loss_xyz
                
            running_loss += loss.item() * images.size(0)
            all_dists.append(dists_km.cpu().numpy())
            
            preds_country = torch.argmax(country_logits, dim=-1)
            country_correct += (preds_country == targets_country).sum().item()
            total_samples += images.size(0)
            
    val_loss = running_loss / len(loader.dataset)
    all_dists = np.concatenate(all_dists)
    median_dist = np.median(all_dists)
    mean_dist = np.mean(all_dists)
    acc_200 = np.mean(all_dists < 200.0) * 100.0
    acc_750 = np.mean(all_dists < 750.0) * 100.0
    country_acc = (country_correct / total_samples) * 100.0
    
    return val_loss, median_dist, mean_dist, acc_200, acc_750, country_acc

def benchmark_model(arch_name, train_loader, val_loader, cluster_centroids, cell_to_country, num_epochs=15, device='cuda'):
    print(f"\n========================================================")
    print(f"BENCHMARKING CANDIDATE: {arch_name.upper()} ({num_epochs} Epochs)")
    print(f"========================================================")
    
    centroids_tensor = torch.tensor(cluster_centroids, dtype=torch.float32).to(device)
    cell_to_country_tensor = torch.tensor(cell_to_country, dtype=torch.long).to(device)
    dist_matrix = build_centroid_dist_matrix(cluster_centroids, device)
    
    model = get_model(num_cells=len(cluster_centroids), arch=arch_name).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    criterion_country = nn.CrossEntropyLoss(label_smoothing=0.05)
    
    backbone_params = [p for n, p in model.named_parameters() if 'backbone' in n]
    head_params = [p for n, p in model.named_parameters() if 'backbone' not in n]
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 1e-4},
        {'params': head_params, 'lr': 1e-3}
    ], weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    
    warmup_epochs = 2
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        else:
            progress = float(epoch - warmup_epochs) / float(num_epochs - warmup_epochs)
            return 0.5 * (1.0 + np.cos(np.pi * progress))
            
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    best_median = float('inf')
    best_metrics = {}
    
    start_time = time.time()
    for epoch in range(num_epochs):
        ep_start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion_country, centroids_tensor, dist_matrix, cell_to_country_tensor, scaler, device)
        val_loss, median_dist, mean_dist, acc_200, acc_750, country_acc = validate(model, val_loader, criterion_country, centroids_tensor, dist_matrix, cell_to_country_tensor, device)
        scheduler.step()
        ep_duration = time.time() - ep_start
        
        print(f"[{arch_name.upper()}] Ep {epoch+1:02d}/{num_epochs:02d} ({ep_duration:.1f}s) | Val Median: {median_dist:.1f} km | Mean: {mean_dist:.1f} km | <200km: {acc_200:.1f}% | Ctry Acc: {country_acc:.1f}%", flush=True)
        
        if median_dist < best_median:
            best_median = median_dist
            best_metrics = {
                'arch': arch_name,
                'params': param_count,
                'best_median': median_dist,
                'mean_dist': mean_dist,
                'acc_200': acc_200,
                'acc_750': acc_750,
                'country_acc': country_acc,
                'best_epoch': epoch + 1
            }
            ckpt_path = f"benchmark_{arch_name}.pth"
            torch.save(model.state_dict(), ckpt_path)
            
    total_time = time.time() - start_time
    best_metrics['total_time_min'] = total_time / 60.0
    return best_metrics

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(script_dir, ".."))
    csv_file = os.path.join(base_dir, "geo_dataset", "train_labels.csv")
    img_dir = os.path.join(base_dir, "geo_dataset", "train")
    centroids_save_path = os.path.join(base_dir, "hierarchical_cluster_centroids.npy")
    
    n_clusters = 2048
    print(f"Preparing 448x448 dataloaders with {n_clusters} hierarchical clusters...")
    train_loader, val_loader, cluster_centroids, cell_to_country = get_dataloaders(
        csv_file, img_dir, batch_size=24, num_workers=4, n_clusters=n_clusters, centroids_save_path=centroids_save_path
    )
    
    candidates = ['mobilenetv4', 'convnextv2', 'regnety']
    benchmark_results = []
    
    for arch in candidates:
        res = benchmark_model(arch, train_loader, val_loader, cluster_centroids, cell_to_country, num_epochs=15, device=device)
        benchmark_results.append(res)
        
    print("\n" + "="*80)
    print("🏆 MULTI-MODEL BENCHMARK RESULTS (15 Epochs, Identical Split, 448x448 High-Res)")
    print("="*80)
    print(f"{'Model Architecture':<18} | {'Params':<10} | {'Median Dist':<12} | {'Mean Dist':<11} | {'<200km':<8} | {'Country Acc':<11}")
    print("-" * 80)
    
    benchmark_results.sort(key=lambda x: x['best_median'])
    for r in benchmark_results:
        print(f"{r['arch'].upper():<18} | {r['params']:<10,d} | {r['best_median']:<9.2f} km | {r['mean_dist']:<8.2f} km | {r['acc_200']:<7.1f}% | {r['country_acc']:<10.1f}%")
    print("="*80)
    
    winner = benchmark_results[0]
    print(f"\n🎉 WINNING CHAMPION: {winner['arch'].upper()} with Median Distance: {winner['best_median']:.2f} km!")
    print(f"Copying {winner['arch']} checkpoint to best_model.pth...")
    
    src_ckpt = f"benchmark_{winner['arch']}.pth"
    dst_ckpt = os.path.join(base_dir, "best_model.pth")
    if os.path.exists(src_ckpt):
        import shutil
        shutil.copy(src_ckpt, dst_ckpt)
        print(f"Successfully saved champion weights to {dst_ckpt}!")

if __name__ == "__main__":
    main()
