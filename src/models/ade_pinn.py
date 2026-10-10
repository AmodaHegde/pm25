"""Basic spatiotemporal PINN for PM2.5 using the advection-diffusion equation.

The neural network learns C(x, y, t) from three inputs: local x/y coordinates
(km) and elapsed time (hours). It is trained with:
  1) supervised loss on observed PM2.5 values, and
  2) physics loss from C_t + u*C_x + v*C_y - D*(C_xx + C_yy) - S = 0.

Important:
- The data must contain multiple distinct latitude AND longitude values.
- Constant u, v, D, and S are simplifying assumptions, not calibrated physics.
- Coordinates are expected as columns in each row. A single Delhi coordinate is
  not enough to identify the spatial derivatives in a 2-D advection-diffusion PDE.

Example:
  python advection_diffusion_pinn.py --csv delhi_data.csv \
      --time-col time --lat-col latitude --lon-col longitude \
      --target-col pm2_5 --epochs 1000 --output-dir pinn_output

If your wind_u/wind_v components are reliable, in m/s, and point eastward/northward,
you can use their training-set means as constant advection velocities:
  --wind-u-col wind_u --wind-v-col wind_v --wind-unit m/s
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn


class PINN(nn.Module):
    """Small MLP: (x, y, t) -> normalized PM2.5 concentration."""

    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 1),
        )

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        return self.network(xyt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Input CSV file")
    parser.add_argument("--time-col", default="time")
    parser.add_argument("--lat-col", default="latitude")
    parser.add_argument("--lon-col", default="longitude")
    parser.add_argument("--target-col", default="pm2_5")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--collocation-size", type=int, default=1024)
    parser.add_argument("--physics-weight", type=float, default=0.01)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--u", type=float, default=0.0,
                        help="Constant eastward velocity in km/hour; used if wind columns are not supplied")
    parser.add_argument("--v", type=float, default=0.0,
                        help="Constant northward velocity in km/hour; used if wind columns are not supplied")
    parser.add_argument("--diffusivity", type=float, default=0.1,
                        help="Constant diffusivity in km^2/hour (illustrative default)")
    parser.add_argument("--source", type=float, default=0.0,
                        help="Constant net source/sink S in concentration-units/hour; default 0")
    parser.add_argument("--wind-u-col", default=None,
                        help="Optional eastward wind-component column; its training mean is used")
    parser.add_argument("--wind-v-col", default=None,
                        help="Optional northward wind-component column; its training mean is used")
    parser.add_argument("--wind-unit", choices=["m/s", "km/h"], default=None,
                        help="Required when wind component columns are supplied")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="pinn_output")
    return parser.parse_args()


def local_coordinates_km(
    latitude: np.ndarray,
    longitude: np.ndarray,
    lat_ref: float,
    lon_ref: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate lat/lon as local east/north distances in kilometres."""
    x_km = 111.32 * np.cos(np.deg2rad(lat_ref)) * (longitude - lon_ref)
    y_km = 110.574 * (latitude - lat_ref)
    return x_km, y_km


def mean_wind_components(
    train_df: pd.DataFrame, args: argparse.Namespace
) -> tuple[float, float]:
    """Return constant u/v in km/hour, optionally from mean wind columns."""
    use_u = args.wind_u_col is not None
    use_v = args.wind_v_col is not None
    if use_u != use_v:
        raise ValueError("Pass both --wind-u-col and --wind-v-col, or neither.")

    if not use_u:
        return float(args.u), float(args.v)

    missing = [c for c in (args.wind_u_col, args.wind_v_col) if c not in train_df.columns]
    if missing:
        raise ValueError(f"Wind column(s) not found: {missing}")
    if args.wind_unit is None:
        raise ValueError("Set --wind-unit to m/s or km/h when using wind component columns.")

    u = pd.to_numeric(train_df[args.wind_u_col], errors="coerce").mean()
    v = pd.to_numeric(train_df[args.wind_v_col], errors="coerce").mean()
    if not np.isfinite(u) or not np.isfinite(v):
        raise ValueError("Wind component columns do not contain valid numeric values.")
    multiplier = 3.6 if args.wind_unit == "m/s" else 1.0
    return float(u * multiplier), float(v * multiplier)


def make_pde_residual(
    model: nn.Module,
    xyt_norm: torch.Tensor,
    scales: dict[str, float],
    u: float,
    v: float,
    diffusivity: float,
    source: float,
) -> torch.Tensor:
    """Calculate the dimensional PDE residual at normalized (x, y, t) points."""
    xyt = xyt_norm.detach().clone().requires_grad_(True)
    c_norm = model(xyt)

    # First derivatives of normalized concentration w.r.t. normalized x, y, t.
    grad_norm = torch.autograd.grad(
        outputs=c_norm,
        inputs=xyt,
        grad_outputs=torch.ones_like(c_norm),
        create_graph=True,
    )[0]

    c_std = scales["c_std"]
    lx, ly, duration = scales["lx"], scales["ly"], scales["duration"]

    # Convert derivatives to physical units: concentration, km, and hours.
    c_x = c_std * grad_norm[:, 0:1] / lx
    c_y = c_std * grad_norm[:, 1:2] / ly
    c_t = c_std * grad_norm[:, 2:3] / duration

    # Second derivatives for the diffusion term.
    grad_x = torch.autograd.grad(
        outputs=grad_norm[:, 0:1],
        inputs=xyt,
        grad_outputs=torch.ones_like(grad_norm[:, 0:1]),
        create_graph=True,
    )[0]
    grad_y = torch.autograd.grad(
        outputs=grad_norm[:, 1:2],
        inputs=xyt,
        grad_outputs=torch.ones_like(grad_norm[:, 1:2]),
        create_graph=True,
    )[0]
    c_xx = c_std * grad_x[:, 0:1] / (lx**2)
    c_yy = c_std * grad_y[:, 1:2] / (ly**2)

    # PDE: dC/dt + u*dC/dx + v*dC/dy - D*(d2C/dx2 + d2C/dy2) - S = 0
    return c_t + u * c_x + v * c_y - diffusivity * (c_xx + c_yy) - source


def batch_predict(model: nn.Module, X: torch.Tensor, batch_size: int = 8192) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            chunks.append(model(X[start : start + batch_size]).cpu().numpy().reshape(-1))
    return np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.collocation_size < 1:
        raise ValueError("epochs, batch-size, and collocation-size must be positive.")
    if not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("validation-fraction must be between 0 and 0.5.")
    if args.diffusivity < 0:
        raise ValueError("diffusivity must be non-negative.")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    required = [args.time_col, args.lat_col, args.lon_col, args.target_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s): {missing}. The 2-D PINN needs timestamp, "
            "latitude, longitude, and measured PM2.5 columns. Use the --*-col options "
            "to match your CSV. Add latitude/longitude to every row if they are not present."
        )

    df[args.time_col] = pd.to_datetime(df[args.time_col], errors="coerce", utc=True)
    for col in (args.lat_col, args.lon_col, args.target_col):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required).copy()
    df = df.sort_values(args.time_col).reset_index(drop=True)
    if len(df) < 20:
        raise ValueError("Need at least 20 valid observations for a train/validation split.")

    unique_times = np.sort(df[args.time_col].dropna().unique())
    if len(unique_times) < 3:
        raise ValueError("Need at least 3 distinct timestamps for a chronological split.")
    split_index = int((1.0 - args.validation_fraction) * len(unique_times))
    split_index = min(max(split_index, 1), len(unique_times) - 1)
    cutoff_time = pd.Timestamp(unique_times[split_index])
    train_df = df[df[args.time_col] < cutoff_time].copy()
    val_df = df[df[args.time_col] >= cutoff_time].copy()
    if len(train_df) < 10 or len(val_df) < 1:
        raise ValueError("Train/validation split is too small; adjust validation-fraction or provide more data.")

    # Coordinates from the training period only.
    lat_ref = float(train_df[args.lat_col].mean())
    lon_ref = float(train_df[args.lon_col].mean())
    x_train, y_train = local_coordinates_km(
        train_df[args.lat_col].to_numpy(dtype=float),
        train_df[args.lon_col].to_numpy(dtype=float), lat_ref, lon_ref
    )
    lx = float(np.max(x_train) - np.min(x_train))
    ly = float(np.max(y_train) - np.min(y_train))
    if lx < 1e-3 or ly < 1e-3:
        raise ValueError(
            "This is a 2-D spatial PINN, but the training data has almost no spread in "
            "latitude or longitude. A single Delhi coordinate is not enough to learn "
            "the spatial derivatives in the advection-diffusion equation. Collect data "
            "at multiple distinct coordinates (with meaningful spatial coverage), or use "
            "a temporal model instead."
        )

    # Convert all rows to local x/y (km), and time to elapsed hours.
    x_all, y_all = local_coordinates_km(
        df[args.lat_col].to_numpy(dtype=float),
        df[args.lon_col].to_numpy(dtype=float), lat_ref, lon_ref
    )
    times_all = pd.to_datetime(df[args.time_col], utc=True)
    t0 = train_df[args.time_col].min()
    t_all = (times_all - t0).dt.total_seconds().to_numpy(dtype=float) / 3600.0

    # Training coordinate ranges are used for normalization and chain-rule scaling.
    x_min, y_min = float(np.min(x_train)), float(np.min(y_train))
    t_train = (train_df[args.time_col] - t0).dt.total_seconds().to_numpy(dtype=float) / 3600.0
    t_min, t_max = float(np.min(t_train)), float(np.max(t_train))
    duration = t_max - t_min
    if duration <= 0:
        raise ValueError("Training timestamps must span more than one instant.")

    x_norm_all = (x_all - x_min) / lx
    y_norm_all = (y_all - y_min) / ly
    t_norm_all = (t_all - t_min) / duration
    X_all_np = np.column_stack([x_norm_all, y_norm_all, t_norm_all]).astype(np.float32)

    train_mask = (df[args.time_col] < cutoff_time).to_numpy()
    val_mask = ~train_mask
    y_all_raw = df[args.target_col].to_numpy(dtype=np.float32)
    y_mean = float(np.mean(y_all_raw[train_mask]))
    y_std = float(np.std(y_all_raw[train_mask]))
    if not np.isfinite(y_std) or y_std < 1e-8:
        raise ValueError("Training PM2.5 values have near-zero variance; normalization is not possible.")
    Y_all_np = ((y_all_raw - y_mean) / y_std).astype(np.float32).reshape(-1, 1)

    u, v = mean_wind_components(train_df, args)
    scales = {"lx": lx, "ly": ly, "duration": duration, "c_mean": y_mean, "c_std": y_std}
    residual_scale = y_std * max(
        1.0 / duration,
        abs(u) / lx,
        abs(v) / ly,
        args.diffusivity / (lx**2),
        args.diffusivity / (ly**2),
        1e-8,
    )
    residual_scale = max(float(residual_scale), 1e-8)

    X_all = torch.tensor(X_all_np, dtype=torch.float32, device=device)
    Y_all = torch.tensor(Y_all_np, dtype=torch.float32, device=device)
    train_indices = np.flatnonzero(train_mask)
    val_indices = np.flatnonzero(val_mask)
    X_train, Y_train = X_all[train_indices], Y_all[train_indices]
    X_val, Y_val = X_all[val_indices], Y_all[val_indices]

    model = PINN(width=args.width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history: list[dict[str, float]] = []

    print(f"Device: {device}")
    print(f"Rows: {len(df):,} | train: {len(train_df):,} | validation: {len(val_df):,}")
    print(f"Spatial span: {lx:.2f} km east-west x {ly:.2f} km north-south")
    print(f"Training time span: {duration:.1f} hours")
    print(f"PDE coefficients: u={u:.4g}, v={v:.4g} km/h; D={args.diffusivity:.4g} km^2/h; S={args.source:.4g}")
    if args.wind_u_col is None:
        print("Wind note: using the constant --u/--v values. The wind_u/wind_v columns, if present, are not used automatically.")
    else:
        print("Wind note: using the training-set mean wind components as constant coefficients.")
    print("Note: these constant coefficients and S=0 are baseline assumptions, not a calibrated atmospheric model.")

    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_n = min(args.batch_size, len(X_train))
        row_idx = torch.randint(0, len(X_train), (batch_n,), device=device)
        x_data, y_data = X_train[row_idx], Y_train[row_idx]

        # Forward pass on observed data -> supervised/data loss.
        predicted = model(x_data)
        data_loss = torch.mean((predicted - y_data) ** 2)

        # Collocation points -> require the PDE to hold even where there are no measurements.
        collocation = torch.rand(args.collocation_size, 3, device=device)
        residual = make_pde_residual(
            model, collocation, scales, u, v, args.diffusivity, args.source
        )
        physics_loss = torch.mean((residual / residual_scale) ** 2)
        total_loss = data_loss + args.physics_weight * physics_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            val_pred_norm = batch_predict(model, X_val)
            val_pred = val_pred_norm * y_std + y_mean
            val_true = y_all_raw[val_indices]
            val_rmse = float(np.sqrt(np.mean((val_pred - val_true) ** 2)))
            val_mae = float(np.mean(np.abs(val_pred - val_true)))
            record = {
                "epoch": float(epoch),
                "total_loss": float(total_loss.detach().cpu()),
                "data_loss": float(data_loss.detach().cpu()),
                "physics_loss": float(physics_loss.detach().cpu()),
                "validation_rmse": val_rmse,
                "validation_mae": val_mae,
            }
            history.append(record)
            print(
                f"Epoch {epoch:5d}/{args.epochs} | total={record['total_loss']:.5f} "
                f"data={record['data_loss']:.5f} physics={record['physics_loss']:.5f} "
                f"val_RMSE={val_rmse:.3f} val_MAE={val_mae:.3f}"
            )

    # Make predictions for every valid row; save the original columns plus predictions.
    model.eval()
    pred_norm_all = batch_predict(model, X_all)
    pred_all = pred_norm_all * y_std + y_mean
    output_df = df.copy()
    output_df["pm2_5_pinn_predicted"] = pred_all
    output_csv = out_dir / "pm25_pinn_predictions.csv"
    output_df.to_csv(output_csv, index=False)

    history_df = pd.DataFrame(history)
    history_csv = out_dir / "training_history.csv"
    history_df.to_csv(history_csv, index=False)

    # Plot training losses (these are sampled training-batch losses).
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history_df["epoch"], history_df["data_loss"], label="Data loss")
    ax.plot(history_df["epoch"], history_df["physics_loss"], label="Physics loss")
    ax.plot(history_df["epoch"], history_df["total_loss"], label="Total loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Basic advection-diffusion PINN training losses")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    loss_plot = out_dir / "training_losses.png"
    fig.savefig(loss_plot, dpi=150)
    plt.close(fig)

    checkpoint: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "architecture": {"input_features": ["x_km_norm", "y_km_norm", "t_hours_norm"], "width": args.width},
        "scaling": {
            "lat_ref": lat_ref, "lon_ref": lon_ref,
            "x_min": x_min, "y_min": y_min, "lx": lx, "ly": ly,
            "t_min_hours": t_min, "duration_hours": duration,
            "pm25_mean": y_mean, "pm25_std": y_std,
        },
        "pde_parameters": {"u_km_per_hour": u, "v_km_per_hour": v,
                           "diffusivity_km2_per_hour": args.diffusivity,
                           "source_concentration_per_hour": args.source,
                           "physics_weight": args.physics_weight},
    }
    checkpoint_path = out_dir / "pinn_model.pt"
    torch.save(checkpoint, checkpoint_path)

    final_val = history_df.iloc[-1]
    print("\nFinished.")
    print(f"Final validation RMSE: {final_val['validation_rmse']:.3f}")
    print(f"Final validation MAE:  {final_val['validation_mae']:.3f}")
    print(f"Predictions: {output_csv}")
    print(f"Training history: {history_csv}")
    print(f"Loss plot: {loss_plot}")
    print(f"Model checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
