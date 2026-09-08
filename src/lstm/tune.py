import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import mlflow
import optuna

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
raw_features = df[feature_cols].values
target_col_idx = feature_cols.index(config["data"]["target_column"])
lookback = config["data"]["lookback_window"]

dev_size = int(len(raw_features) * (config["data"]["train_split"] + config["data"]["val_split"]))
dev_raw = raw_features[:dev_size]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def objective(trial):
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.0, 0.4, step=0.1) if num_layers > 1 else 0.0
    learning_rate = trial.suggest_float("learning_rate", 0.1, 0.5, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    epochs = 15

    tscv = TimeSeriesSplit(n_splits=3)
    fold_rmses = []

    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
        mlflow.log_params(trial.params)

        for fold, (train_idx, val_idx) in enumerate(tscv.split(dev_raw)):
            fold_scaler = MinMaxScaler()
            train_fold_raw = dev_raw[train_idx]
            train_fold_scaled = fold_scaler.fit_transform(train_fold_raw)

            val_fold_raw = dev_raw[max(0, val_idx[0] - lookback) : val_idx[-1] + 1]
            val_fold_scaled = fold_scaler.transform(val_fold_raw)

            X_tr, y_tr = create_sliding_windows(train_fold_scaled, target_col_idx, lookback)
            X_vl, y_vl = create_sliding_windows(val_fold_scaled, target_col_idx, lookback)

            train_loader = DataLoader(TimeSeriesDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(TimeSeriesDataset(X_vl, y_vl), batch_size=batch_size, shuffle=False)

            model = PM25LSTM(
                input_dim=len(feature_cols),
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout
            ).to(device)

            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            for epoch in range(epochs):
                model.train()
                for bx, by in train_loader:
                    bx, by = bx.to(device), by.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(bx), by)
                    loss.backward()
                    optimizer.step()

                model.eval()
                val_epoch_loss = 0.0
                with torch.no_grad():
                    for bx, by in val_loader:
                        bx, by = bx.to(device), by.to(device)
                        preds_batch = model(bx)
                        val_epoch_loss += criterion(preds_batch, by).item()
                val_epoch_loss /= len(val_loader)

                global_step = fold * epochs + epoch
                trial.report(val_epoch_loss, step=global_step)
                if trial.should_prune():
                    mlflow.set_tag("pruned", "True")
                    raise optuna.TrialPruned()

            preds = []
            with torch.no_grad():
                for bx, _ in val_loader:
                    bx = bx.to(device)
                    preds.extend(model(bx).cpu().numpy())

            preds = np.array(preds).squeeze()
            scale_factor = fold_scaler.data_max_[target_col_idx] - fold_scaler.data_min_[target_col_idx]
            y_vl_unscaled = y_vl * scale_factor + fold_scaler.data_min_[target_col_idx]
            preds_unscaled = preds * scale_factor + fold_scaler.data_min_[target_col_idx]

            fold_rmse = np.sqrt(mean_squared_error(y_vl_unscaled, preds_unscaled))
            fold_rmses.append(fold_rmse)

        mean_rmse = float(np.mean(fold_rmses))
        mlflow.log_metric("mean_cv_rmse", mean_rmse)

        return mean_rmse

mlflow.set_experiment(config["mlflow"]["experiment_name"])

pruner = optuna.pruners.MedianPruner(
    n_startup_trials=5,
    n_warmup_steps=5,
    interval_steps=1
)

with mlflow.start_run(run_name="optuna_study_parent"):
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=15)

    mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
    mlflow.log_metric("best_cv_rmse", study.best_value)

    config["model"]["hidden_dim"] = study.best_params.get("hidden_dim", config["model"]["hidden_dim"])
    config["model"]["num_layers"] = study.best_params.get("num_layers", config["model"]["num_layers"])
    config["model"]["dropout"] = study.best_params.get("dropout", config["model"]["dropout"])
    config["train"]["learning_rate"] = study.best_params.get("learning_rate", config["train"]["learning_rate"])
    config["train"]["batch_size"] = study.best_params.get("batch_size", config["train"]["batch_size"])

    with open("params.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)