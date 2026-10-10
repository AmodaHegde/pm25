"""Basic Physics-Informed Neural Network (PINN) starter for PM2.5.

Model: C_t + u*C_x + v*C_y - D*(C_xx + C_yy) = 0

Coordinates:
  x, y: kilometres in a local projection
  t: hours since the first observation
  C: PM2.5 concentration (for example, micrograms/m^3)

This is a deliberately simplified prototype. Constant wind components and
constant diffusivity are illustrative defaults, not calibrated parameters.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


class PINN(nn.Module):
    """Small fully-connected network mapping (x, y, t) to normalized PM2.5."""

    def __init__(self, width: int = 64, depth: int = 4) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(3, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.Tanh()])
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

        # A smooth, reproducible starting point for this small example.
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        return self.network(xyt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Input CSV file")
    parser.add_argument("--time-col", default="timestamp", help="Timestamp column")
    parser.add_argument("--lat-col", default="latitude", help="Latitude column, in degrees")
    parser.add_argument("--lon-col", default="longitude", help="Longitude column, in degrees")
    parser.add_argument("--target-col", default="PM2.5", help="PM2.5 target column")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--collocation-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--physics-weight", type=float, default=0.01)
    parser.add_argument("--u", type=float, default=1.0,
                        help="Constant eastward wind/advection speed in km/hour (illustrative)")
    parser.add_argument("--v", type=float, default=0.0,
                        help="Constant northward wind/advection speed in km/hour (illustrative)")
    parser.add_argument("--diffusivity", type=float, default=0.1,
                        help="Diffusivity in km^2/hour (illustrative; must be non-negative)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="pinn_output")
    return parser.parse_args()


def load_and_prepare_data(args: argparse.Namespace):
    path = Path(args.csv)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)
    required = [args.time_col, args.lat_col, args.lon_col, args.target_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}. Available columns: {list(df.columns)}. "
            "If your data only has station names, join a station-to-latitude/longitude table first."
        )

    df = df.copy()
    df[args.time_col] = pd.to_datetime(df[args.time_col], errors="coerce")
    for column in (args.lat_col, args.lon_col, args.target_col):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=required).copy()
    df = df[np.isfinite(df[args.target_col])].copy()
    df = df.sort_values(args.time_col).reset_index(drop=True)

    if len(df) < 20:
        raise ValueError(f"Only {len(df)} valid rows found; provide more observations for training.")
    if (df[args.target_col] < 0).any():
        print("Warning: negative PM2.5 values found. Check sensor errors or missing-value codes.")

    # Local equirectangular projection. Suitable as a simple approximation for one city/region.
    lat0 = float(df[args.lat_col].mean())
    lon0 = float(df[args.lon_col].mean())
    df["_x_km"] = 111.32 * np.cos(np.deg2rad(lat0)) * (df[args.lon_col] - lon0)
    df["_y_km"] = 110.574 * (df[args.lat_col] - lat0)
    df["_t_hours"] = (
        (df[args.time_col] - df[args.time_col].min()).dt.total_seconds() / 3600.0
    )

    x_min, x_max = float(df["_x_km"].min()), float(df["_x_km"].max())
    y_min, y_max = float(df["_y_km"].min()), float(df["_y_km"].max())
    t_min, t_max = float(df["_t_hours"].min()), float(df["_t_hours"].max())
    # Avoid division by zero if all rows share one coordinate or timestamp.
    Lx, Ly, T = max(x_max - x_min, 1e-6), max(y_max - y_min, 1e-6), max(t_max - t_min, 1e-6)

    # The network receives coordinates in [0, 1]. Derivatives are converted back
    # to physical units inside the PDE residual via Lx, Ly and T.
    xyt = np.column_stack([
        (df["_x_km"].to_numpy() - x_min) / Lx,
        (df["_y_km"].to_numpy() - y_min) / Ly,
        (df["_t_hours"].to_numpy() - t_min) / T,
    ]).astype(np.float32)

    pm25_mean = float(df[args.target_col].mean())
    pm25_std = float(df[args.target_col].std())
    if not np.isfinite(pm25_std) or pm25_std < 1e-8:
        pm25_std = 1.0
    target_norm = ((df[args.target_col].to_numpy() - pm25_mean) / pm25_std).astype(np.float32)

    scale = {
        "x_min": x_min, "y_min": y_min, "t_min": t_min,
        "Lx": Lx, "Ly": Ly, "T": T,
        "pm25_mean": pm25_mean, "pm25_std": pm25_std,
    }
    return df, xyt, target_norm, scale


def advection_diffusion_residual(
    model: nn.Module,
    xyt_norm: torch.Tensor,
    scale: dict[str, float],
    u: float,
    v: float,
    diffusivity: float,
) -> torch.Tensor:
    """Return the PDE residual in physical concentration/time units."""
    xyt = xyt_norm.detach().clone().requires_grad_(True)
    c_norm = model(xyt)

    # First derivatives with respect to normalized x, y, t.
    grad = torch.autograd.grad(
        outputs=c_norm,
        inputs=xyt,
        grad_outputs=torch.ones_like(c_norm),
        create_graph=True,
    )[0]

    # Second spatial derivatives.
    grad_x = torch.autograd.grad(grad[:, 0].sum(), xyt, create_graph=True)[0]
    grad_y = torch.autograd.grad(grad[:, 1].sum(), xyt, create_graph=True)[0]
    c_xx_norm = grad_x[:, 0:1]
    c_yy_norm = grad_y[:, 1:2]

    c_std = scale["pm25_std"]
    Lx, Ly, T = scale["Lx"], scale["Ly"], scale["T"]

    c_t = c_std * grad[:, 2:3] / T
    c_x = c_std * grad[:, 0:1] / Lx
    c_y = c_std * grad[:, 1:2] / Ly
    c_xx = c_std * c_xx_norm / (Lx ** 2)
    c_yy = c_std * c_yy_norm / (Ly ** 2)

    # C_t + u*C_x + v*C_y - D*(C_xx + C_yy) = 0
    return c_t + u * c_x + v * c_y - diffusivity * (c_xx + c_yy)


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.collocation_size < 1:
        raise ValueError("epochs, batch-size and collocation-size must all be positive.")
    if args.diffusivity < 0:
        raise ValueError("diffusivity must be non-negative.")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df, xyt_np, target_np, scale = load_and_prepare_data(args)
    X = torch.tensor(xyt_np, dtype=torch.float32, device=device)
    Y = torch.tensor(target_np[:, None], dtype=torch.float32, device=device)

    model = PINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    n_rows = len(df)
    batch_size = min(args.batch_size, n_rows)
    # A characteristic hourly concentration-change scale for dimensionless PDE loss.
    # Tune this and --physics-weight for your dataset and physical parameter choices.
    residual_scale = max(scale["pm25_std"] / 24.0, 1e-6)

    model.train()
    for epoch in range(1, args.epochs + 1):
        # Randomly sample observations to keep each update inexpensive on large CSVs.
        idx = torch.randint(0, n_rows, (batch_size,), device=device)
        x_data, y_data = X[idx], Y[idx]
        prediction = model(x_data)
        data_loss = torch.mean((prediction - y_data) ** 2)

        # Collocation points are locations/times where we enforce the PDE even when
        # there is no direct PM2.5 measurement. They are sampled in the coordinate box.
        collocation = torch.rand(args.collocation_size, 3, device=device)
        residual = advection_diffusion_residual(
            model, collocation, scale, args.u, args.v, args.diffusivity
        )
        physics_loss = torch.mean((residual / residual_scale) ** 2)
        total_loss = data_loss + args.physics_weight * physics_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 100 == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:5d}/{args.epochs} | "
                f"total={total_loss.item():.6f} | "
                f"data={data_loss.item():.6f} | physics={physics_loss.item():.6f}"
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "pinn_pm25.pt"
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "scale": scale,
        "u_km_per_hour": args.u,
        "v_km_per_hour": args.v,
        "diffusivity_km2_per_hour": args.diffusivity,
        "input_columns": [args.lon_col, args.lat_col, args.time_col],
        "target_column": args.target_col,
    }
    torch.save(checkpoint, model_path)

    # Save fitted predictions in chunks to avoid allocating a large intermediate tensor.
    model.eval()
    preds_norm = []
    with torch.no_grad():
        for start in range(0, len(X), 8192):
            preds_norm.append(model(X[start:start + 8192]).cpu().numpy().reshape(-1))
    preds = np.concatenate(preds_norm) * scale["pm25_std"] + scale["pm25_mean"]
    result = df.copy()
    result["PINN_PM25_Predicted"] = preds
    result_path = output_dir / "pm25_predictions.csv"
    result.to_csv(result_path, index=False)

    rmse = float(np.sqrt(np.mean((preds - result[args.target_col].to_numpy()) ** 2)))
    mae = float(np.mean(np.abs(preds - result[args.target_col].to_numpy())))
    print(f"\nRows used: {len(df):,}")
    print(f"Training-fit RMSE: {rmse:.4f}")
    print(f"Training-fit MAE:  {mae:.4f}")
    print(f"Saved model:       {model_path}")
    print(f"Saved predictions: {result_path}")
    print("Note: these are training-fit metrics, not held-out validation metrics.")
    print("Note: use measured wind components and calibrated diffusivity before drawing scientific conclusions.")


if __name__ == "__main__":
    main()
