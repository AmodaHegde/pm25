import yaml
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow

with open("params.yaml", "r") as f:
    config = yaml.safe_load(f)

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

df = pd.read_csv(config["data"]["dataset_path"])
feature_cols = [
    "pm2_5", "temperature_2m", "relative_humidity_2m", "wind_u", "wind_v",
    "boundary_layer_height", "precipitation", "nitrogen_dioxide", "hour_sin", "hour_cos"
]
data = df[feature_cols].values
target_col_idx = feature_cols.index(config["data"]["target_column"])

LOOKBACK = config["data"]["lookback_window"]
train_size = int(len(data) * config["data"]["train_split"])
val_size = int(len(data) * config["data"]["val_split"])

val_raw = data[train_size - LOOKBACK : train_size + val_size]

scaler = joblib.load(config["model"]["scaler_path"])
val_scaled = scaler.transform(val_raw)

X_val, y_val = create_sliding_windows(val_scaled, target_col_idx, LOOKBACK)
val_dataset = TimeSeriesDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=config["train"]["batch_size"], shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PM25LSTM(
    input_dim=len(feature_cols),
    hidden_dim=config["model"]["hidden_dim"],
    num_layers=config["model"]["num_layers"],
    dropout=config["model"]["dropout"]
).to(device)

model.load_state_dict(torch.load(config["model"]["save_path"], map_location=device))
model.eval()

criterion = nn.MSELoss()
val_loss = 0.0
val_preds = []

with torch.no_grad():
    for batch_x, batch_y in val_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        preds = model(batch_x)
        loss = criterion(preds, batch_y)
        val_loss += loss.item()
        val_preds.extend(preds.cpu().numpy())

val_loss /= len(val_loader)
val_preds = np.array(val_preds)

scale_factor = scaler.data_max_[target_col_idx] - scaler.data_min_[target_col_idx]
y_val_unscaled = y_val * scale_factor + scaler.data_min_[target_col_idx]
preds_unscaled = val_preds.squeeze() * scale_factor + scaler.data_min_[target_col_idx]

rmse = np.sqrt(mean_squared_error(y_val_unscaled, preds_unscaled))
mae = mean_absolute_error(y_val_unscaled, preds_unscaled)

mlflow.set_experiment(config["mlflow"]["experiment_name"])
with mlflow.start_run():
    mlflow.log_metrics({
        "val_loss": val_loss,
        "val_rmse": rmse,
        "val_mae": mae
    })

print(f"Validation Evaluation:")
print(f"  Loss: {val_loss:.6f}")
print(f"  RMSE: {rmse:.2f} µg/m³")
print(f"  MAE:  {mae:.2f} µg/m³")