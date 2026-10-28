"""PINN with 5-fold expanding-window time-series cross-validation.

Preprocessing is fitted independently inside each fold to avoid leakage:
- StandardScaler for continuous input features and the PM2.5 target
- OrdinalEncoder for the categorical station feature
- Median imputation for missing numeric predictor values

The first three model inputs are always x (km), y (km), and time (hours),
so the physics derivatives are taken with respect to those three inputs only.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient

from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------
CSV_FILE = os.environ.get(
    "PINN_CSV_FILE", "data/silver/delhi_air_weather.csv"
)
OUTPUT_METRICS = os.environ.get(
    "PINN_OUTPUT_METRICS", "results/pinn_timeseries_cv_metrics.csv"
)
OUTPUT_PREDICTIONS = os.environ.get(
    "PINN_OUTPUT_PREDICTIONS", "results/pinn_timeseries_cv_predictions.csv"
)
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "")
MLFLOW_EXPERIMENT_NAME = os.environ.get(
    "MLFLOW_EXPERIMENT_NAME", "Baseline_PINN_TimeSeries_CV"
)

LAT_COL = "lat"
LON_COL = "lon"
TIME_COL = "timestamp"
PM_COL = "pm25"
STATION_COL = "station"
N_SPLITS = 5
EPOCHS = 3000
LEARNING_RATE = 0.001
PHYSICS_WEIGHT = 1.0
DATA_WEIGHT = 1.0
HIDDEN_UNITS = 64

# Illustrative physical parameters. Validate/calibrate these for your study.
# x/y are in km and time is in hours.
u = 0.0       # East-west wind velocity (km/hour)
v = 0.0       # North-south wind velocity (km/hour)
D = 1.0       # Effective diffusion coefficient (km^2/hour)
source = 0.0  # Source/sink term (concentration/hour)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# In Colab, set MLFLOW_TRACKING_URI to the HTTPS ngrok URL for the server.
# Keep ngrok and MLflow credentials in Colab secrets/environment variables.
if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
mlflow_client = MlflowClient()
experiment = mlflow_client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
parent_run = mlflow_client.create_run(
    experiment.experiment_id,
    tags={
        "mlflow.runName": "PINN_TimeSeries_CV",
        "model_type": "baseline_pinn",
        "cv_strategy": "expanding_window_time_series",
    },
)
parent_run_id = parent_run.info.run_id
mlflow_client.log_params(parent_run_id, {
    "n_splits": N_SPLITS,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "physics_weight": PHYSICS_WEIGHT,
    "data_weight": DATA_WEIGHT,
    "hidden_units": HIDDEN_UNITS,
    "seed": SEED,
    "device": str(device),
    "target_column": PM_COL,
})

# ---------------------------------------------------------------------
# 2. Load and clean data
# ---------------------------------------------------------------------
df = pd.read_csv(CSV_FILE)
required = [LAT_COL, LON_COL, TIME_COL, PM_COL, STATION_COL]
missing_required = [c for c in required if c not in df.columns]
if missing_required:
    raise ValueError(f"Missing required columns: {missing_required}")

df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
df = df.dropna(subset=required).copy()
df = df.sort_values(TIME_COL).reset_index(drop=True)
if df[TIME_COL].nunique() <= N_SPLITS:
    raise ValueError(
        f"Need more than {N_SPLITS} unique timestamps for {N_SPLITS}-fold CV."
    )

# Use numeric meteorological/other predictor columns when present.
# Exclude identifiers, coordinates, target, and timestamp from this list.
excluded = {
    "id", LAT_COL, LON_COL, TIME_COL, PM_COL, STATION_COL,
    "x", "y", "time_hours", "x_scaled", "y_scaled", "time_scaled"
}
numeric_predictors = [
    c for c in df.columns
    if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
]
print("Additional numeric predictors:", numeric_predictors)

# Split unique timestamps, not rows. Every station record at a timestamp
# is assigned to the same fold.
unique_times = np.sort(df[TIME_COL].dropna().unique())
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

fold_metrics = []
fold_predictions = []

# ---------------------------------------------------------------------
# 3. Cross-validation loop
# ---------------------------------------------------------------------
for fold, (train_time_idx, val_time_idx) in enumerate(
    tscv.split(unique_times), start=1
):
    train_times = unique_times[train_time_idx]
    val_times = unique_times[val_time_idx]

    train_df = df[df[TIME_COL].isin(train_times)].copy()
    val_df = df[df[TIME_COL].isin(val_times)].copy()

    if train_df.empty or val_df.empty:
        print(f"Skipping fold {fold}: empty train or validation set")
        continue

    print(f"\n{'=' * 18} FOLD {fold}/{N_SPLITS} {'=' * 18}")
    print(f"Train: {train_df[TIME_COL].min()} -> {train_df[TIME_COL].max()} "
          f"({len(train_df)} rows)")
    print(f"Valid: {val_df[TIME_COL].min()} -> {val_df[TIME_COL].max()} "
          f"({len(val_df)} rows)")

    # ----- Coordinate features in physical units (km) -----
    # Fit the local projection anchor using training rows only.
    lat0 = train_df[LAT_COL].mean()
    lon0 = train_df[LON_COL].mean()

    for part in (train_df, val_df):
        part["x_km"] = (
            (part[LON_COL] - lon0)
            * 111.32 * np.cos(np.radians(lat0))
        )
        part["y_km"] = (part[LAT_COL] - lat0) * 111.32

    # Convert timestamps to hours relative to the first training timestamp.
    time_origin = train_df[TIME_COL].min()
    for part in (train_df, val_df):
        part["time_hours"] = (
            (part[TIME_COL] - time_origin).dt.total_seconds() / 3600.0
        )

    # First three continuous inputs must remain x, y, time for the PDE.
    coordinate_time_cols = ["x_km", "y_km", "time_hours"]
    continuous_cols = coordinate_time_cols + numeric_predictors

    # ----- Numeric missing-value imputation: fit on training only -----
    imputer = SimpleImputer(strategy="median")
    X_train_raw = imputer.fit_transform(train_df[continuous_cols])
    X_val_raw = imputer.transform(val_df[continuous_cols])

    # ----- Standard scaling: fit on training only -----
    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train_raw)
    X_val_scaled = x_scaler.transform(X_val_raw)

    # ----- Categorical station encoding: fit on training only -----
    # Unseen validation stations are assigned -1. This permits inference but
    # does not provide learned station-specific information for unseen stations.
    station_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        dtype=np.float64,
    )
    station_train = station_encoder.fit_transform(
        train_df[[STATION_COL]].astype(str)
    )
    station_val = station_encoder.transform(
        val_df[[STATION_COL]].astype(str)
    )

    X_train_all = np.column_stack([X_train_scaled, station_train])
    X_val_all = np.column_stack([X_val_scaled, station_val])

    # Scale target using training targets only.
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(train_df[[PM_COL]]).astype(np.float32)
    y_val_scaled = y_scaler.transform(val_df[[PM_COL]]).astype(np.float32)

    X_train = torch.tensor(X_train_all, dtype=torch.float32, device=device)
    X_val = torch.tensor(X_val_all, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train_scaled, dtype=torch.float32, device=device)
    y_val = torch.tensor(y_val_scaled, dtype=torch.float32, device=device)

    # Model input = all scaled continuous features + encoded station.
    input_dim = X_train.shape[1]
    model = nn.Sequential(
        nn.Linear(input_dim, HIDDEN_UNITS),
        nn.Tanh(),
        nn.Linear(HIDDEN_UNITS, HIDDEN_UNITS),
        nn.Tanh(),
        nn.Linear(HIDDEN_UNITS, HIDDEN_UNITS),
        nn.Tanh(),
        nn.Linear(HIDDEN_UNITS, 1),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Scaler scales are standard deviations in the raw physical units.
    # x/y/time scaled inputs = (raw - mean) / scale.
    sx, sy, st = [float(s) for s in x_scaler.scale_[:3]]
    pm_scale = float(y_scaler.scale_[0])

    # ----- Train fold model -----
    for epoch in range(EPOCHS):
        model.train()
        X_phys = X_train.detach().clone().requires_grad_(True)
        pred_train = model(X_phys)
        data_loss = torch.mean((pred_train - y_train) ** 2)

        # Gradients with respect to the scaled x, y, and time inputs.
        grads = torch.autograd.grad(
            outputs=pred_train,
            inputs=X_phys,
            grad_outputs=torch.ones_like(pred_train),
            create_graph=True,
        )[0]
        C_x_scaled = grads[:, 0:1]
        C_y_scaled = grads[:, 1:2]
        C_t_scaled = grads[:, 2:3]

        C_xx_scaled = torch.autograd.grad(
            C_x_scaled, X_phys,
            grad_outputs=torch.ones_like(C_x_scaled),
            create_graph=True,
        )[0][:, 0:1]
        C_yy_scaled = torch.autograd.grad(
            C_y_scaled, X_phys,
            grad_outputs=torch.ones_like(C_y_scaled),
            create_graph=True,
        )[0][:, 1:2]

        # Convert derivatives of standardized target to physical PM2.5 units.
        # Target: C = y_scaler.mean_[0] + pm_scale * C_standardized.
        C_x = pm_scale * C_x_scaled / sx
        C_y = pm_scale * C_y_scaled / sy
        C_t = pm_scale * C_t_scaled / st
        C_xx = pm_scale * C_xx_scaled / (sx ** 2)
        C_yy = pm_scale * C_yy_scaled / (sy ** 2)

        # Advection-diffusion residual in physical units.
        residual = C_t + u * C_x + v * C_y - D * (C_xx + C_yy) - source
        physics_loss = torch.mean(residual ** 2)

        loss = DATA_WEIGHT * data_loss + PHYSICS_WEIGHT * physics_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 500 == 0 or epoch == 0:
            print(
                f"Fold {fold} | Epoch {epoch + 1}/{EPOCHS} | "
                f"Data loss: {data_loss.item():.6f} | "
                f"Physics loss: {physics_loss.item():.6f}"
            )

    # ----- Evaluate training and future timestamps in original PM2.5 units -----
    model.eval()
    with torch.no_grad():
        pred_train_scaled = model(X_train).cpu().numpy()
        pred_val_scaled = model(X_val).cpu().numpy()

    pred_train = y_scaler.inverse_transform(pred_train_scaled).ravel()
    actual_train = train_df[PM_COL].to_numpy()
    pred_val = y_scaler.inverse_transform(pred_val_scaled).ravel()
    actual_val = val_df[PM_COL].to_numpy()
    train_rmse = float(np.sqrt(mean_squared_error(actual_train, pred_train)))
    train_mae = float(mean_absolute_error(actual_train, pred_train))
    train_r2 = float(r2_score(actual_train, pred_train))
    test_rmse = float(np.sqrt(mean_squared_error(actual_val, pred_val)))
    test_mae = float(mean_absolute_error(actual_val, pred_val))
    test_r2 = float(r2_score(actual_val, pred_val))

    train_station_set = set(train_df[STATION_COL].astype(str).unique())
    val_station_set = set(val_df[STATION_COL].astype(str).unique())
    unseen_stations = sorted(val_station_set - train_station_set)

    fold_metrics.append({
        "fold": fold,
        "train_start": train_df[TIME_COL].min(),
        "train_end": train_df[TIME_COL].max(),
        "validation_start": val_df[TIME_COL].min(),
        "validation_end": val_df[TIME_COL].max(),
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "train_stations": len(train_station_set),
        "validation_stations": len(val_station_set),
        "unseen_validation_stations": len(unseen_stations),
        "train_rmse": train_rmse,
        "train_mae": train_mae,
        "train_r2": train_r2,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "test_r2": test_r2,
    })

    fold_result = val_df[[TIME_COL, STATION_COL, PM_COL]].copy()
    fold_result["fold"] = fold
    fold_result["predicted_pm25"] = pred_val
    fold_predictions.append(fold_result)

    fold_run = mlflow_client.create_run(
        experiment.experiment_id,
        tags={
            "mlflow.runName": f"Fold_{fold}",
            "mlflow.parentRunId": parent_run_id,
            "fold": str(fold),
        },
    )
    fold_run_id = fold_run.info.run_id
    mlflow_client.log_params(fold_run_id, {
        "train_rows": len(train_df),
        "test_rows": len(val_df),
        "train_start": str(train_df[TIME_COL].min()),
        "train_end": str(train_df[TIME_COL].max()),
        "test_start": str(val_df[TIME_COL].min()),
        "test_end": str(val_df[TIME_COL].max()),
        "train_stations": len(train_station_set),
        "test_stations": len(val_station_set),
        "unseen_test_stations": len(unseen_stations),
    })
    mlflow_client.log_metrics(fold_run_id, {
        "train_rmse": train_rmse,
        "train_mae": train_mae,
        "train_r2": train_r2,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "test_r2": test_r2,
    })
    mlflow_client.set_terminated(fold_run_id, status="FINISHED")

    print(
        f"Training metrics | RMSE: {train_rmse:.3f} µg/m³ | "
        f"MAE: {train_mae:.3f} µg/m³ | R²: {train_r2:.4f}"
    )
    print(
        f"Testing metrics  | RMSE: {test_rmse:.3f} µg/m³ | "
        f"MAE: {test_mae:.3f} µg/m³ | R²: {test_r2:.4f}"
    )
    if unseen_stations:
        print("WARNING: stations not seen in training:", unseen_stations)

# ---------------------------------------------------------------------
# 4. Save cross-validation metrics and out-of-fold predictions
# ---------------------------------------------------------------------
if not fold_metrics:
    raise RuntimeError("No folds completed. Check the dataset and split settings.")

metrics_df = pd.DataFrame(fold_metrics)
os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_METRICS)), exist_ok=True)
os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_PREDICTIONS)), exist_ok=True)
metrics_df.to_csv(OUTPUT_METRICS, index=False)

predictions_df = pd.concat(fold_predictions, ignore_index=True)
predictions_df.to_csv(OUTPUT_PREDICTIONS, index=False)

for metric_name in (
    "train_rmse", "train_mae", "train_r2",
    "test_rmse", "test_mae", "test_r2",
):
    mlflow_client.log_metric(parent_run_id, metric_name, float(metrics_df[metric_name].mean()))
mlflow_client.log_artifact(parent_run_id, OUTPUT_METRICS, artifact_path="results")
mlflow_client.log_artifact(parent_run_id, OUTPUT_PREDICTIONS, artifact_path="results")
mlflow_client.set_terminated(parent_run_id, status="FINISHED")

print("\nCross-validation results")
metric_columns = [
    "fold", "train_rmse", "train_mae", "train_r2",
    "test_rmse", "test_mae", "test_r2", "unseen_validation_stations",
]
print(metrics_df[metric_columns].to_string(index=False))
for split in ("train", "test"):
    print(
        f"\nMean {split} RMSE: {metrics_df[f'{split}_rmse'].mean():.3f} µg/m³ | "
        f"MAE: {metrics_df[f'{split}_mae'].mean():.3f} µg/m³ | "
        f"R²: {metrics_df[f'{split}_r2'].mean():.4f}"
    )
print(f"Saved fold metrics to: {OUTPUT_METRICS}")
print(f"Saved out-of-fold predictions to: {OUTPUT_PREDICTIONS}")

# Plot observed versus out-of-fold predicted PM2.5.
actual = predictions_df[PM_COL].to_numpy()
predicted = predictions_df["predicted_pm25"].to_numpy()
limits = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]

plt.figure(figsize=(10, 5))
plt.scatter(actual, predicted, alpha=0.3)
plt.plot(limits, limits, "r--", label="Perfect prediction")
plt.xlabel("Observed PM2.5 (µg/m³)")
plt.ylabel("Predicted PM2.5 (µg/m³)")
plt.title("PINN: 5-Fold Time-Series CV — Observed vs Predicted PM2.5")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
