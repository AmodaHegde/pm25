import math
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow
import mlflow.pytorch

with open("params.yaml", "r") as f:
    config = yaml.safe_load(f)

# ============================================================
# Dataset (identical to lstm.py)
# ============================================================

class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

# ============================================================
# Positional encoding (Transformers have no built-in notion of
# sequence order the way an LSTM does, so this has to be added
# explicitly before the encoder layers)
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]

# ============================================================
# Model
# ============================================================

class PM25Transformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.2):
        super(PM25Transformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        out = self.transformer_encoder(x)
        out = self.fc(out[:, -1, :])
        return out

# ============================================================
# Helpers (identical to lstm.py)
# ============================================================

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

# ============================================================
# Data pipeline (identical to lstm.py)
# ============================================================

df = pd.read_csv(config["data"]["dataset_path"])

feature_cols = [
    "pm2_5",
    "temperature_2m",
    "relative_humidity_2m",
    "wind_u",
    "wind_v",
    "boundary_layer_height",
    "precipitation",
    "nitrogen_dioxide",
    "hour_sin",
    "hour_cos",
]

data = df[feature_cols].values
target_col_idx = feature_cols.index(config["data"]["target_column"])

train_size = int(len(data) * config["data"]["train_split"])
train_raw = data[:train_size]
test_raw = data[train_size:]

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_raw)
test_scaled = scaler.transform(test_raw)

LOOKBACK = config["data"]["lookback_window"]
X_train, y_train = create_sliding_windows(train_scaled, target_col_idx, LOOKBACK)
X_test, y_test = create_sliding_windows(test_scaled, target_col_idx, LOOKBACK)

train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=config["train"]["batch_size"], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config["train"]["batch_size"], shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# NOTE: params.yaml wasn't shared, so nhead isn't an existing key in your
# config's "model" section. Falls back to 4 if not present - add
# model.nhead to params.yaml to control it explicitly. hidden_dim must be
# divisible by nhead (default hidden_dim=64 / nhead=4 works out of the box).
model = PM25Transformer(
    input_dim=len(feature_cols),
    d_model=config["model"]["hidden_dim"],
    nhead=config["model"].get("nhead", 4),
    num_layers=config["model"]["num_layers"],
    dropout=config["model"]["dropout"],
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["learning_rate"])

mlflow.set_experiment(config["mlflow"]["experiment_name"])

with mlflow.start_run(run_name="transformer"):
    mlflow.log_params(flatten_dict(config))
    mlflow.log_param("model_type", "transformer")
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

    model.eval()
    test_preds = []
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device)
            preds = model(batch_x)
            test_preds.extend(preds.cpu().numpy())

    test_preds = np.array(test_preds)

    y_test_unscaled = y_test * (scaler.data_max_[target_col_idx] - scaler.data_min_[target_col_idx]) + scaler.data_min_[target_col_idx]
    preds_unscaled = test_preds.squeeze() * (scaler.data_max_[target_col_idx] - scaler.data_min_[target_col_idx]) + scaler.data_min_[target_col_idx]

    rmse = np.sqrt(mean_squared_error(y_test_unscaled, preds_unscaled))
    mae = mean_absolute_error(y_test_unscaled, preds_unscaled)

    mlflow.log_metrics({
        "test_rmse": rmse,
        "test_mae": mae,
    })

    sample_input = torch.tensor(X_test[:1], dtype=torch.float32).to(device)
    mlflow.pytorch.log_model(
        pytorch_model=model,
        name="model",
        input_example=sample_input.cpu().numpy(),
        serialization_format="pickle",
    )

    print(f"Test Evaluation:")
    print(f"  RMSE: {rmse:.2f} ug/m3")
    print(f"  MAE:  {mae:.2f} ug/m3")