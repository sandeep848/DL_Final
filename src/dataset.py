import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

COUNTRIES = ['Belarus', 'Finland', 'France', 'Germany', 'Iceland', 'Italy', 
             'Norway', 'Poland', 'Spain', 'Sweden', 'Turkey', 'United_Kingdom']
COUNTRY_TO_IDX = {country: idx for idx, country in enumerate(COUNTRIES)}

def coords_to_3d(lat_deg, lng_deg):
    """Converts lat/lng in degrees to 3D unit sphere Cartesian coordinates (x, y, z)."""
    lat_rad = np.radians(lat_deg)
    lng_rad = np.radians(lng_deg)
    x = np.cos(lat_rad) * np.cos(lng_rad)
    y = np.cos(lat_rad) * np.sin(lng_rad)
    z = np.sin(lat_rad)
    return np.column_stack([x, y, z]) if isinstance(lat_deg, np.ndarray) else np.array([x, y, z])

def cartesian_to_latlng(xyz):
    """Converts 3D Cartesian coordinates (x, y, z) back to lat/lng in decimal degrees."""
    norm = np.linalg.norm(xyz, axis=1, keepdims=True)
    xyz_norm = xyz / np.maximum(norm, 1e-8)
    lat = np.degrees(np.arcsin(np.clip(xyz_norm[:, 2], -1.0, 1.0)))
    lng = np.degrees(np.arctan2(xyz_norm[:, 1], xyz_norm[:, 0]))
    return np.column_stack([lat, lng])

def get_hierarchical_cluster_centroids(csv_file_or_df, save_path=None, mapping_save_path=None, n_clusters=2048, random_state=42):
    """
    Computes hierarchical spatial cluster centroids partitioned strictly by country.
    Guarantees that every spatial Voronoi cell belongs to exactly one country (100% pure).
    """
    if save_path and os.path.exists(save_path) and (mapping_save_path is None or os.path.exists(mapping_save_path)):
        centroids = np.load(save_path)
        if len(centroids) == n_clusters:
            mapping = np.load(mapping_save_path) if mapping_save_path and os.path.exists(mapping_save_path) else None
            return centroids, mapping
        print(f"Centroids count mismatch (found {len(centroids)}, requested {n_clusters}). Recomputing...")

    if isinstance(csv_file_or_df, pd.DataFrame):
        df = csv_file_or_df
    else:
        df = pd.read_csv(csv_file_or_df)

    num_countries = len(COUNTRIES)
    base_k = n_clusters // num_countries
    rem = n_clusters % num_countries

    all_centroids = []
    cell_to_country = []

    for idx, country in enumerate(COUNTRIES):
        k_c = base_k + (1 if idx < rem else 0)
        c_df = df[df['country'] == country]
        coords_deg = c_df[['lat', 'lng']].values
        coords_3d = coords_to_3d(coords_deg[:, 0], coords_deg[:, 1])

        kmeans = KMeans(n_clusters=k_c, random_state=random_state, n_init=10).fit(coords_3d)
        c_centroids = cartesian_to_latlng(kmeans.cluster_centers_)
        all_centroids.append(c_centroids)
        cell_to_country.extend([idx] * k_c)

    centroids_deg = np.vstack(all_centroids)
    cell_to_country = np.array(cell_to_country, dtype=np.int64)

    if save_path:
        np.save(save_path, centroids_deg)
        print(f"Saved {n_clusters} hierarchical centroids to {save_path}")
    if mapping_save_path:
        np.save(mapping_save_path, cell_to_country)
        print(f"Saved cell-to-country mapping to {mapping_save_path}")

    return centroids_deg, cell_to_country

def get_cluster_centroids(csv_file_or_df, save_path=None, n_clusters=2048, random_state=42):
    mapping_path = save_path.replace(".npy", "_country_map.npy") if save_path else None
    centroids, _ = get_hierarchical_cluster_centroids(csv_file_or_df, save_path, mapping_path, n_clusters, random_state)
    return centroids

class GeolocationDataset(Dataset):
    def __init__(self, df, img_dir, cluster_centroids, cell_to_country=None, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.cluster_centroids_3d = coords_to_3d(cluster_centroids[:, 0], cluster_centroids[:, 1])
        
        lats = self.df['lat'].values
        lngs = self.df['lng'].values
        pts_3d = coords_to_3d(lats, lngs)
        
        if cell_to_country is not None and 'country' in self.df.columns:
            # Hierarchical mapping: assign each point to nearest cluster WITHIN its ground truth country!
            cell_indices = []
            country_indices = np.array([COUNTRY_TO_IDX.get(c, -1) for c in self.df['country']])
            for i in range(len(self.df)):
                c_idx = country_indices[i]
                valid_cells = np.where(cell_to_country == c_idx)[0]
                if len(valid_cells) > 0:
                    dists = np.linalg.norm(pts_3d[i] - self.cluster_centroids_3d[valid_cells], axis=-1)
                    cell_indices.append(valid_cells[np.argmin(dists)])
                else:
                    dists = np.linalg.norm(pts_3d[i] - self.cluster_centroids_3d, axis=-1)
                    cell_indices.append(np.argmin(dists))
            self.cell_indices = np.array(cell_indices, dtype=np.int64)
        else:
            # Pairwise distance (N, K)
            dists = np.linalg.norm(pts_3d[:, np.newaxis, :] - self.cluster_centroids_3d[np.newaxis, :, :], axis=-1)
            self.cell_indices = np.argmin(dists, axis=-1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['filename']
        img_path = os.path.join(self.img_dir, img_name)

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (448, 448))

        if self.transform:
            image = self.transform(image)

        lat = row['lat']
        lng = row['lng']

        lat_norm = lat / 90.0
        lng_norm = lng / 180.0
        target_coords = (lat_norm, lng_norm)

        cell_idx = int(self.cell_indices[idx])

        country_idx = -1
        if 'country' in row:
            country_idx = COUNTRY_TO_IDX.get(row['country'], -1)

        return (image, 
                torch.tensor(target_coords, dtype=torch.float32), 
                torch.tensor(cell_idx, dtype=torch.long), 
                torch.tensor(country_idx, dtype=torch.long))

def get_dataloaders(csv_file, img_dir, batch_size=24, num_workers=4, val_split=0.2, n_clusters=2048, centroids_save_path=None):
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"{csv_file} not found.")
        
    df = pd.read_csv(csv_file)
    
    if 'country' in df.columns:
        train_df, val_df = train_test_split(df, test_size=val_split, stratify=df['country'], random_state=42)
    else:
        train_df, val_df = train_test_split(df, test_size=val_split, random_state=42)
        
    mapping_save_path = centroids_save_path.replace(".npy", "_country_map.npy") if centroids_save_path else None
    cluster_centroids, cell_to_country = get_hierarchical_cluster_centroids(
        train_df, save_path=centroids_save_path, mapping_save_path=mapping_save_path, n_clusters=n_clusters
    )
        
    # High resolution 448x448 image transforms: No horizontal flip to preserve language text and road signs!
    train_transform = transforms.Compose([
        transforms.Resize((480, 480)),
        transforms.RandomResizedCrop(448, scale=(0.85, 1.0)),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_dataset = GeolocationDataset(train_df, img_dir, cluster_centroids, cell_to_country=cell_to_country, transform=train_transform)
    val_dataset = GeolocationDataset(val_df, img_dir, cluster_centroids, cell_to_country=cell_to_country, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, cluster_centroids, cell_to_country

def unnormalize_coords(coords):
    """Unnormalize coords back to original range (lat, lng)"""
    lat = coords[:, 0] * 90.0
    lng = coords[:, 1] * 180.0
    return lat, lng

