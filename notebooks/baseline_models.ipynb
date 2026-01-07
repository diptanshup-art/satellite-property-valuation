import torch
import torch.nn as nn  
import torch.optim as optim
from sklearn.metrics import r2_score
import numpy as np

class TabularBaselineModel(nn.Module):
    def __init__(self, num_tab_features):
        super(TabularBaselineModel, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_tab_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, tab):
        return self.mlp(tab)

# 1. Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- MODEL 1: TABULAR ONLY ---
print("\n--- Training Tabular Only Baseline ---")
tab_model = TabularBaselineModel(num_tab_features=X_train_scaled.shape[1]).to(device)
optimizer = torch.optim.AdamW(tab_model.parameters(), lr=1e-4)
criterion = nn.HuberLoss(delta=1.0)
    
# Simple training loop for baseline
best_tab_r2 = -float('inf')
for epoch in range(15): # Faster convergence for MLP
        tab_model.train()
        for _, tabs, labels in train_loader:
            tabs, labels = tabs.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(tab_model(tabs), labels)
            loss.backward()
            optimizer.step()
            
        # Eval
        tab_model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for _, tabs, labels in val_loader:
                preds = tab_model(tabs.to(device))
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        r2 = r2_score(all_labels, all_preds)
        best_tab_r2 = max(best_tab_r2, r2)

print(f"Tabular Only best R2: {best_tab_r2}")

import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error

def train_xgboost_baseline(X_train, X_val, y_train, y_val):
    print("--- Training XGBoost Baseline ---")
    
    # Initialize the Regressor
    # These parameters are a good starting point for the King County dataset
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42
    )

    # Train
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Predict
    preds = xgb_model.predict(X_val)
    
    # Calculate Metrics
    r2 = r2_score(y_val, preds)
    # Convert log-prices back to real dollars for RMSE/MAE
    rmse = np.sqrt(np.mean((np.expm1(preds) - np.expm1(y_val))**2))
    
    print(f"XGBoost Baseline R2: {r2:.4f}")
    print(f"XGBoost Baseline RMSE: ${rmse:,.0f}")
    
    return r2, rmse

# Run it
xgb_r2, xgb_rmse = train_xgboost_baseline(X_train_scaled, X_val_scaled, y_train, y_val)
