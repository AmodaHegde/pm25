import os
import yaml
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
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

def flatten_dict(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

os.makedirs(os.path.dirname(config["model"]["save_path"]), exist_ok=True)

df = pd.read_csv(config["data"]["dataset_path"])
feature_cols = [
    "pm2_5", "temperature_2m", "relative_humidity_2m", "wind_u", "wind_v",
    "boundary_layer_height", "precipitation", "nitrogen_dioxide", "hour_sin", "hour_cos"
]
data = df[feature_cols].values
target_col_idx = feature_cols.index(config["data"]["target_column"])

train_size = int(len(data) * config["data"]["train_split"])
train_raw = data[:train_size]

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_raw)

joblib.dump(scaler, config["model"]["scaler_path"])

LOOKBACK = config["data"]["lookback_window"]
X_train, y_train = create_sliding_windows(train_scaled, target_col_idx, LOOKBACK)

train_dataset = TimeSeriesDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=config["train"]["batch_size"], shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PM25LSTM(
    input_dim=len(feature_cols),
    hidden_dim=config["model"]["hidden_dim"],
    num_layers=config["model"]["num_layers"],
    dropout=config["model"]["dropout"]
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["learning_rate"])

mlflow.set_experiment(config["mlflow"]["experiment_name"])

with mlflow.start_run():
    mlflow.log_params(flatten_dict(config))
    mlflow.log_artifact("params.yaml")

    for epoch in range(config["train"]["epochs"]):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        mlflow.log_metric("train_loss", avg_loss, step=epoch)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{config['train']['epochs']}] - Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), config["model"]["save_path"])
    print(f"Model saved to {config['model']['save_path']}")