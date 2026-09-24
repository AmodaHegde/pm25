import argparse
import copy
import random

import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import optuna

FEATURE_COLS = [
    "pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide",
    "sulphur_dioxide", "ozone", "dust",
    "temperature_2m", "relative_humidity_2m", "apparent_temperature",
    "precipitation", "rain", "surface_pressure",
    "boundary_layer_height", "wind_u", "wind_v", "hour_sin", "hour_cos",
    "doy_sin", "doy_cos", "dow_sin", "dow_cos", "pm_ratio",
    "ventilation_index", "temp_humidity_interaction", "pm2_5_lag_1h",
    "pm2_5_lag_2h", "pm2_5_lag_3h", "pm2_5_lag_6h", "pm2_5_lag_24h",
    "pm2_5_roll_mean_6h", "pm2_5_roll_std_6h", "pm2_5_roll_mean_24h",
]

class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index], self.targets[index]

class PM25LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, inputs):
        outputs, _ = self.lstm(inputs)
        return self.fc(outputs[:, -1, :])

def create_sliding_windows(data, target_idx, window_size):
    sequences = []
    targets = []
    for index in range(len(data) - window_size):
        sequences.append(data[index : index + window_size])
        targets.append(data[index + window_size, target_idx])
    return np.asarray(sequences), np.asarray(targets)

def load_config():
    with open("params.yaml", "r") as config_file:
        raw_config = yaml.safe_load(config_file)
    return raw_config, raw_config.get("lstm", raw_config)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def objective(trial, dev_raw, target_idx, lookback, config, device):
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.0, 0.4, step=0.1)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    epochs = config["tuning_epochs"]
    fold_rmses = []

    for fold, (train_idx, val_idx) in enumerate(TimeSeriesSplit(n_splits=3).split(dev_raw)):
        train_raw = dev_raw[train_idx]
        val_raw = dev_raw[max(0, val_idx[0] - lookback) : val_idx[-1] + 1]
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_raw)
        val_scaled = scaler.transform(val_raw)
        X_train, y_train = create_sliding_windows(train_scaled, target_idx, lookback)
        X_val, y_val = create_sliding_windows(val_scaled, target_idx, lookback)

        train_loader = DataLoader(
            TimeSeriesDataset(X_train, y_train), batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TimeSeriesDataset(X_val, y_val), batch_size=batch_size, shuffle=False
        )
        model = PM25LSTM(
            input_dim=len(FEATURE_COLS),
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        ).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            model.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()

            model.eval()
            validation_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    validation_loss += criterion(model(batch_x), batch_y).item()
            validation_loss /= len(val_loader)
            trial.report(validation_loss, step=fold * epochs + epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        predictions = []
        model.eval()
        with torch.no_grad():
            for batch_x, _ in val_loader:
                predictions.extend(model(batch_x.to(device)).cpu().numpy().ravel())
        scale_range = scaler.data_max_[target_idx] - scaler.data_min_[target_idx]
        predictions = np.asarray(predictions) * scale_range + scaler.data_min_[target_idx]
        targets = y_val * scale_range + scaler.data_min_[target_idx]
        fold_rmse = np.sqrt(mean_squared_error(targets, predictions))
        fold_rmses.append(fold_rmse)
        print(
            f"Trial {trial.number}, fold {fold + 1}/3 complete: "
            f"RMSE={fold_rmse:.4f}",
            flush=True,
        )

    return float(np.mean(fold_rmses))

def update_lstm_config(raw_config, best_params):
    updated_config = copy.deepcopy(raw_config)
    lstm_config = updated_config["lstm"] if "lstm" in updated_config else updated_config
    lstm_config["model"].update(
        {
            "hidden_dim": best_params["hidden_dim"],
            "num_layers": best_params["num_layers"],
            "dropout": best_params["dropout"],
        }
    )
    lstm_config["train"].update(
        {
            "learning_rate": best_params["learning_rate"],
            "batch_size": best_params["batch_size"],
        }
    )
    return updated_config

def main():
    parser = argparse.ArgumentParser(description="Tune the LSTM with Optuna.")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--study-name", default="pm25_lstm_tuning")
    parser.add_argument("--storage", default="sqlite:///optuna_study.db")
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Training epochs per fold for each trial (default: 10).",
    )
    parser.add_argument("--no-update-config", action="store_true")
    args = parser.parse_args()

    raw_config, config = load_config()
    config["tuning_epochs"] = args.epochs
    seed_everything(config.get("seed", 42))
    df = pd.read_csv(config["data"]["dataset_path"])
    missing_features = sorted(set(FEATURE_COLS) - set(df.columns))
    if missing_features:
        raise ValueError(f"Dataset is missing LSTM features: {missing_features}")

    data = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    target_idx = FEATURE_COLS.index(config["data"]["target_column"])
    dev_size = int(
        len(data) * (config["data"]["train_split"] + config["data"]["val_split"])
    )
    dev_raw = data[:dev_size]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )
    study.optimize(
        lambda trial: objective(
            trial,
            dev_raw,
            target_idx,
            config["data"]["lookback_window"],
            config,
            device,
        ),
        n_trials=args.n_trials,
    )

    print(f"Best validation RMSE: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")

    if not args.no_update_config:
        updated_config = update_lstm_config(raw_config, study.best_params)
        with open("params.yaml", "w") as config_file:
            yaml.safe_dump(updated_config, config_file, sort_keys=False)
        print("Updated the lstm section in params.yaml.")

if __name__ == "__main__":
    main()
