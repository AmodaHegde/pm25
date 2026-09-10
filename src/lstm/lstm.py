import os
import yaml
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow
import mlflow.pytorch

# 1. Configuration & Utilities
with open("params.yaml", "r") as f:
    config = yaml.safe_load(f)

def flatten_dict(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class PM25LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super(PM25LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def create_sliding_windows(data, target_idx, window_size=24):
    sequences, targets = [], []
    for i in range(len(data) - window_size):
        seq = data[i : i + window_size, :]
        label = data[i + window_size, target_idx]
        sequences.append(seq)
        targets.append(label)
    return np.array(sequences), np.array(targets)

# 2. Data Preparation
os.makedirs(os.path.dirname(config["model"]["save_path"]), exist_ok=True)
df = pd.read_csv(config["data"]["dataset_path"])

feature_cols = [
    "pm2_5", "temperature_2m", "relative_humidity_2m", "wind_u", "wind_v",
    "boundary_layer_height", "precipitation", "nitrogen_dioxide", "hour_sin", "hour_cos"
]
data = df[feature_cols].values
target_col_idx = feature_cols.index(config["data"]["target_column"])
LOOKBACK = config["data"]["lookback_window"]

# Split into dev (train + val for CV) and test
test_ratio = 1.0 - (config["data"]["train_split"] + config["data"]["val_split"])
dev_size = int(len(data) * (1.0 - test_ratio))

dev_raw = data[:dev_size]
test_raw = data[dev_size - LOOKBACK:]

# Fit scaler only on development data
scaler = MinMaxScaler()
dev_scaled = scaler.fit_transform(dev_raw)
test_scaled = scaler.transform(test_raw)
joblib.dump(scaler, config["model"]["scaler_path"])

X_dev, y_dev = create_sliding_windows(dev_scaled, target_col_idx, LOOKBACK)
X_test, y_test = create_sliding_windows(test_scaled, target_col_idx, LOOKBACK)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tscv = TimeSeriesSplit(n_splits=5)

# 3. MLflow Parent Run
mlflow.set_experiment(config["mlflow"]["experiment_name"])

with mlflow.start_run(run_name="TimeSeries_CV_Training") as parent_run:
    mlflow.log_params(flatten_dict(config))
    mlflow.log_artifact("params.yaml")

    cv_val_losses, cv_val_rmses, cv_val_maes = [], [], []

    # 4. K-Fold Cross Validation Loop
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_dev)):
        with mlflow.start_run(run_name=f"Fold_{fold+1}", nested=True):
            train_loader = DataLoader(
                TimeSeriesDataset(X_dev[train_idx], y_dev[train_idx]),
                batch_size=config["train"]["batch_size"],
                shuffle=True
            )
            val_loader = DataLoader(
                TimeSeriesDataset(X_dev[val_idx], y_dev[val_idx]),
                batch_size=config["train"]["batch_size"],
                shuffle=False
            )

            model = PM25LSTM(
                input_dim=len(feature_cols),
                hidden_dim=config["model"]["hidden_dim"],
                num_layers=config["model"]["num_layers"],
                dropout=config["model"]["dropout"]
            ).to(device)

            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["learning_rate"])

            # Train Fold
            for epoch in range(config["train"]["epochs"]):
                model.train()
                for bx, by in train_loader:
                    bx, by = bx.to(device), by.to(device)
                    optimizer.zero_grad()
                    out = model(bx)
                    loss = criterion(out, by)
                    loss.backward()
                    optimizer.step()

            # Evaluate Fold
            model.eval()
            val_preds, val_targets = [], []
            val_loss = 0.0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    preds = model(bx)
                    val_loss += criterion(preds, by).item()
                    val_preds.extend(preds.cpu().numpy())
                    val_targets.extend(by.cpu().numpy())

            val_loss /= len(val_loader)
            val_preds = np.array(val_preds).squeeze()
            val_targets = np.array(val_targets).squeeze()

            scale_range = scaler.data_max_[target_col_idx] - scaler.data_min_[target_col_idx]
            val_preds_unscaled = val_preds * scale_range + scaler.data_min_[target_col_idx]
            val_targets_unscaled = val_targets * scale_range + scaler.data_min_[target_col_idx]

            fold_rmse = np.sqrt(mean_squared_error(val_targets_unscaled, val_preds_unscaled))
            fold_mae = mean_absolute_error(val_targets_unscaled, val_preds_unscaled)

            mlflow.log_metrics({
                "val_loss": val_loss,
                "val_rmse": fold_rmse,
                "val_mae": fold_mae
            })

            cv_val_losses.append(val_loss)
            cv_val_rmses.append(fold_rmse)
            cv_val_maes.append(fold_mae)

    # 5. Log CV Averages to Parent Run
    mlflow.log_metrics({
        "mean_cv_loss": np.mean(cv_val_losses),
        "mean_cv_rmse": np.mean(cv_val_rmses),
        "mean_cv_mae": np.mean(cv_val_maes)
    })

    # 6. Final Model Retrained on full Dev set
    final_train_loader = DataLoader(
        TimeSeriesDataset(X_dev, y_dev),
        batch_size=config["train"]["batch_size"],
        shuffle=True
    )
    final_model = PM25LSTM(
        input_dim=len(feature_cols),
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"]
    ).to(device)

    final_criterion = nn.MSELoss()
    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=config["train"]["learning_rate"])

    for epoch in range(config["train"]["epochs"]):
        final_model.train()
        for bx, by in final_train_loader:
            bx, by = bx.to(device), by.to(device)
            final_optimizer.zero_grad()
            out = final_model(bx)
            loss = final_criterion(out, by)
            loss.backward()
            final_optimizer.step()

    torch.save(final_model.state_dict(), config["model"]["save_path"])

    # 7. Final Evaluation on Unseen Test Data
    test_loader = DataLoader(
        TimeSeriesDataset(X_test, y_test),
        batch_size=config["train"]["batch_size"],
        shuffle=False
    )

    final_model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            preds = final_model(bx)
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(by.numpy())

    test_preds = np.array(test_preds).squeeze()
    test_targets = np.array(test_targets).squeeze()

    test_preds_unscaled = test_preds * scale_range + scaler.data_min_[target_col_idx]
    test_targets_unscaled = test_targets * scale_range + scaler.data_min_[target_col_idx]

    test_rmse = np.sqrt(mean_squared_error(test_targets_unscaled, test_preds_unscaled))
    test_mae = mean_absolute_error(test_targets_unscaled, test_preds_unscaled)

    mlflow.log_metrics({
        "test_rmse": test_rmse,
        "test_mae": test_mae
    })

    sample_input = torch.tensor(X_test[:1], dtype=torch.float32).to(device)
    mlflow.pytorch.log_model(
        pytorch_model=final_model,
        artifact_path="model",
        input_example=sample_input.cpu().numpy()
    )

print(f"CV Validation Results -> Mean RMSE: {np.mean(cv_val_rmses):.2f}, Mean MAE: {np.mean(cv_val_maes):.2f}")
print(f"Final Test Evaluation -> RMSE: {test_rmse:.2f} µg/m³, MAE: {test_mae:.2f} µg/m³")