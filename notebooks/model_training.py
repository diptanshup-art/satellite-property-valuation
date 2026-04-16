import torch
import torch.nn as nn
from torchvision import models

class GatedFusionModel(nn.Module):
    def __init__(self, num_tab_features):
        super(GatedFusionModel, self).__init__()
        # CNN Branch (ResNet18)
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.cnn_features = nn.Sequential(*list(resnet.children())[:-1])

        # Tabular Branch
        self.tab_mlp = nn.Sequential(
            nn.Linear(num_tab_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # THE UNIQUE PART: Gated Fusion Unit
        # We learn a 'weight' for both branches
        self.gate = nn.Sequential(
            nn.Linear(512 + 128, 1),
            nn.Sigmoid()
        )

        self.regressor = nn.Sequential(
            nn.Linear(512 + 128, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, img, tab):
        img_feats = self.cnn_features(img).view(img.size(0), -1) # 512
        tab_feats = self.tab_mlp(tab) # 128

        combined = torch.cat((img_feats, tab_feats), dim=1)

        # Calculate the Gating value (how much to trust each branch)
        g = self.gate(combined)

        # Apply the gate to emphasize specific features
        fused = combined * g

        return self.regressor(fused)

import torch.optim as optim
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt   

# 1. Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Initialize Model
# 'num_features' should match the number of columns in your scaled tabular data
model = GatedFusionModel(num_tab_features=X_train_scaled.shape[1]).to(device)

# 3. Loss and Optimizer
import torch.nn.functional as F

def weighted_huber_loss(inputs, targets, delta=1.0):
    # 1. Calculate standard Huber Loss
    base_loss = F.huber_loss(inputs, targets, reduction='none', delta=delta)

    # 2. Create weights: Higher log-prices get more importance
    # We use exp(targets) to move back toward linear scale for weighting
    weights = torch.exp(targets) / torch.exp(targets).mean()

    # 3. Apply weights and return mean
    return (weights * base_loss).mean()

optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

# 4. Training Function
def train_model(model, train_loader, val_loader, epochs=20):
    best_r2 = -float('inf') # Save based on R2 now

    history = {'train_loss': [], 'val_loss': [], 'val_r2': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for images, tabs, labels in train_loader:
            images, tabs, labels = images.to(device), tabs.to(device), labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images, tabs)        
            loss = weighted_huber_loss(outputs, labels, delta=1.0)
            loss.backward()

            # It "clips" the gradients to a max value of 1.0 to stop validation spikes
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, tabs, labels in val_loader:
                images, tabs, labels = images.to(device), tabs.to(device), labels.to(device).unsqueeze(1)
                outputs = model(images, tabs)
                val_loss += weighted_huber_loss(outputs, labels, delta=1.0).item()

                # Store for R2 calculation
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        # Calculate R2 (on log prices)
        r2 = r2_score(all_labels, all_preds)

        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        history['val_r2'].append(r2)

        # Clip predictions to prevent infinity during RMSE calc in early epochs
        safe_preds = np.clip(all_preds, 10, 16)
        rmse = np.sqrt(np.mean((np.expm1(safe_preds) - np.expm1(all_labels))**2))

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | R2: {r2:.3f} | RMSE: ${rmse:,.0f}")

        # 5. Save on Best R2
        if r2 > best_r2:
            best_r2 = r2
            torch.save(model.state_dict(), 'best_multimodal_model.pth')
            print(f"--> NEW BEST R2: {best_r2:.4f} (Model Saved)")

        scheduler.step(avg_val)

    def plot_history(history):
        plt.plot(history['train_loss'], label='Train MSE')
        plt.plot(history['val_loss'], label='Val MSE')
        plt.title('Training and Validation Loss')
        plt.ylabel('Loss (MSE)')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show() 

    plot_history(history)    
