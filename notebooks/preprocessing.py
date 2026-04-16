import cv2

def calculate_greenery_score(image_path):
    img = cv2.imread(image_path)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define range for "Green" (Trees/Grass)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    mask = cv2.inRange(img_hsv, lower_green, upper_green)
    green_pixels = np.sum(mask > 0)
    total_pixels = img.shape[0] * img.shape[1]

    return green_pixels / total_pixels

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from PIL import Image
import os  

# --- CONFIGURATION ---
# Use relative paths that assume the user has a 'data' folder
CSV_PATH = 'data/train.csv' 
IMAGE_DIR = 'data/images/'

# 1. Load and Clean Data
df = pd.read_csv(CSV_PATH)

# --- STEP 1: HARD SYNC (The Image Check) ---
existing_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')]
existing_ids = set([int(f.split('.')[0]) for f in existing_files])

# Filter DF so it ONLY contains rows where images actually exist
df = df[df['id'].isin(existing_ids)].reset_index(drop=True)
df = df[df['bedrooms'] < 10] # Remove the famous 33-bedroom outlier in this dataset

print(f"Dataset synced: {len(df)} properties with matching images found.")

# 2. Feature Engineering
# Target: Log-transform price to handle the wide range of house values
df['log_price'] = np.log1p(df['price'])

# 1. Create 'House Age' (Assuming data is from 2015)
df['house_age'] = 2015 - df['yr_built']

# 2. 'Renovated' status (Binary)
df['is_renovated'] = df['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)

# 3. 'Neighborhood Comparison' (Crucial Insight!)
# This captures if the house is an "outlier" (much bigger/smaller than neighbors)
df['living_vs_neighbor'] = df['sqft_living'] / df['sqft_living15']
df['lot_vs_neighbor'] = df['sqft_lot'] / df['sqft_lot15']

# 4. 'Has Basement' (Binary)
df['has_basement'] = df['sqft_basement'].apply(lambda x: 1 if x > 0 else 0)

# 5. Handle Highly Skewed Features
# 'sqft_lot' often has massive outliers (huge farms vs tiny city lots)
# Log-transforming it helps the Neural Network converge faster.
df['sqft_lot_log'] = np.log1p(df['sqft_lot'])
df['sqft_lot15_log'] = np.log1p(df['sqft_lot15'])

# Update your feature list
features = [
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_above', 'sqft_basement',
    'sqft_lot_log', 'floors', 'waterfront', 'view', 'condition', 'grade',
    'house_age', 'is_renovated', 'living_vs_neighbor', 'lot_vs_neighbor',
    'sqft_living15', 'sqft_lot15_log', 'lat', 'long'
]

from sklearn.cluster import KMeans

# 1. Create Neighborhood Clusters based on Location
# 10-15 clusters is usually good for this dataset size
coords = df[['lat', 'long']]
kmeans = KMeans(n_clusters=15, random_state=42, n_init=10)
df['neighborhood_cluster'] = kmeans.fit_transform(coords).argmin(axis=1)

# 2. Add this to your feature list
features.append('neighborhood_cluster')

# Add this to your dataframe
df['greenery_score'] = df['id'].apply(lambda x: calculate_greenery_score(f"{IMAGE_DIR}/{int(x)}.jpg"))
features.append('greenery_score')

X = df[features]
y = df['log_price']

# Split Data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# 3. Define Image Transformations
# ResNet expects 224x224 images normalized with specific means/stds

# Training: Heavy lifting for robustness
train_transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)), # Force focus on edges/context
    transforms.RandomHorizontalFlip(), # Flips the satellite view
    transforms.RandomRotation(15), 
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1), # Handles different sun lighting
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Validation: Simple and fast
val_transform = transforms.Compose([
   transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# 4. Custom Multimodal Dataset
class PropertyDataset(Dataset):
    def __init__(self, df, tabular_data, image_dir, transform=None):
        self.df = df
        self.tabular_data = torch.tensor(tabular_data, dtype=torch.float32)
        self.image_dir = image_dir
        self.transform = transform
        self.labels = torch.tensor(df['log_price'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load Tabular
        tab = self.tabular_data[idx]

        # Load Image
        prop_id = self.df.iloc[idx]['id']
        img_path = os.path.join(self.image_dir, f"{prop_id}.jpg")
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, tab, label

# Create DataLoaders
train_ds = PropertyDataset(df.iloc[X_train.index], X_train_scaled, IMAGE_DIR, transform=train_transform)
val_ds = PropertyDataset(df.iloc[X_val.index], X_val_scaled, IMAGE_DIR, transform=val_transform)

# Create DataLoaders with multiple workers and pinned memory
train_loader = DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True,
    num_workers=2,        # Set to 2 or 4 in Colab, 0 for cpu
    pin_memory=True,     # Speeds up CPU to GPU transfer
    prefetch_factor=2    # Pre-loads batches
)

val_loader = DataLoader(
    val_ds,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)
