import yaml
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow
import mlflow.pytorch

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

test_raw = data[(train_size + val_size) - LOOKBACK :]

scaler = joblib.load(config["model"]["scaler_path"])
test_scaled = scaler.transform(test_raw)

X_test, y_test = create_sliding_windows(test_scaled, target_col_idx, LOOKBACK)
test_dataset = TimeSeriesDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=config["train"]["batch_size"], shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PM25LSTM(
    input_dim=len(feature_cols),
    hidden_dim=config["model"]["hidden_dim"],
    num_layers=config["model"]["num_layers"],
    dropout=config["model"]["dropout"]
).to(device)

model.load_state_dict(torch.load(config["model"]["save_path"], map_location=device))
model.eval()

test_preds = []
with torch.no_grad():
    for batch_x, _ in test_loader:
        batch_x = batch_x.to(device)
        preds = model(batch_x)
        test_preds.extend(preds.cpu().numpy())

test_preds = np.array(test_preds)

scale_factor = scaler.data_max_[target_col_idx] - scaler.data_min_[target_col_idx]
y_test_unscaled = y_test * scale_factor + scaler.data_min_[target_col_idx]
preds_unscaled = test_preds.squeeze() * scale_factor + scaler.data_min_[target_col_idx]

rmse = np.sqrt(mean_squared_error(y_test_unscaled, preds_unscaled))
mae = mean_absolute_error(y_test_unscaled, preds_unscaled)

mlflow.set_experiment(config["mlflow"]["experiment_name"])
with mlflow.start_run():
    mlflow.log_metrics({
        "test_rmse": rmse,
        "test_mae": mae
    })

    sample_input = torch.tensor(X_test[:1], dtype=torch.float32).to(device)
    mlflow.pytorch.log_model(
        pytorch_model=model,
        name="model",
        input_example=sample_input.cpu().numpy()
    )

print(f"Test Evaluation:")
print(f"  RMSE: {rmse:.2f} µg/m³")
print(f"  MAE:  {mae:.2f} µg/m³")