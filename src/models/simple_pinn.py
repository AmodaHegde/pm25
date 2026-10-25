
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 1. Configuration
CSV_FILE = "D:/miniproject/data/silver/delhi_air_weather.csv"

LAT_COL = "lat"
LON_COL = "lon"
TIME_COL = "timestamp"
PM_COL = "pm25"

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Load and clean data
df = pd.read_csv(CSV_FILE)

df[TIME_COL] = pd.to_datetime(df[TIME_COL])
df = df.dropna(
    subset=[LAT_COL, LON_COL, TIME_COL, PM_COL]
).copy()

# Convert latitude/longitude into approximate local coordinates
# in kilometres, suitable for a compact study region.
lat0 = df[LAT_COL].mean()
lon0 = df[LON_COL].mean()

df["x"] = (
    (df[LON_COL] - lon0)
    * 111.32 * np.cos(np.radians(lat0))
)
df["y"] = (df[LAT_COL] - lat0) * 111.32

# Normalize x, y, and time to [0, 1]
# Keep the scale values for derivative conversion.
x_min, x_max = df["x"].min(), df["x"].max()
y_min, y_max = df["y"].min(), df["y"].max()

time_min = df[TIME_COL].min()
time_max = df[TIME_COL].max()

df["x_norm"] = (df["x"] - x_min) / (x_max - x_min)
df["y_norm"] = (df["y"] - y_min) / (y_max - y_min)

duration_hours = (
    (time_max - time_min).total_seconds() / 3600
)
df["t_norm"] = (
    (df[TIME_COL] - time_min).dt.total_seconds()
    / (time_max - time_min).total_seconds()
)

# PM2.5 scaling
pm_mean = df[PM_COL].mean()
pm_std = df[PM_COL].std()

df["c_norm"] = (df[PM_COL] - pm_mean) / pm_std

# Convert to tensors
X = torch.tensor(
    df[["x_norm", "y_norm", "t_norm"]].values,
    dtype=torch.float32,
    device=device
)

C_obs = torch.tensor(
    df["c_norm"].values.reshape(-1, 1),
    dtype=torch.float32,
    device=device
)

# 3. Define the neural network
model = nn.Sequential(
    nn.Linear(3, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 1)
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 4. Advection-diffusion parameters
# These are illustrative starting values.
# x/y are in km and time is in hours.
u = 0.0       # East-west wind velocity (km/hour)
v = 0.0       # North-south wind velocity (km/hour)
D = 1.0       # Effective diffusion coefficient (km^2/hour)
source = 0.0  # Source/sink term in concentration/hour

# 5. Training
epochs = 3000
physics_weight = 1.0
data_weight = 1.0

for epoch in range(epochs):

    # Coordinates must require gradients for the PDE loss
    X_train = X.detach().clone().requires_grad_(True)

    # Predicted normalized concentration
    C_pred = model(X_train)

    # Data loss: match observed PM2.5
    data_loss = torch.mean((C_pred - C_obs) ** 2)

    # First derivatives in normalized coordinates
    grads = torch.autograd.grad(
        C_pred,
        X_train,
        grad_outputs=torch.ones_like(C_pred),
        create_graph=True
    )[0]

    C_xn = grads[:, 0:1]
    C_yn = grads[:, 1:2]
    C_tn = grads[:, 2:3]

    # Second spatial derivatives
    C_xxn = torch.autograd.grad(
        C_xn, X_train,
        grad_outputs=torch.ones_like(C_xn),
        create_graph=True
    )[0][:, 0:1]

    C_yyn = torch.autograd.grad(
        C_yn, X_train,
        grad_outputs=torch.ones_like(C_yn),
        create_graph=True
    )[0][:, 1:2]

    # Convert derivatives to physical coordinates.
    # C = pm_mean + pm_std * C_norm
    C_t = pm_std * C_tn / duration_hours
    C_x = pm_std * C_xn / (x_max - x_min)
    C_y = pm_std * C_yn / (y_max - y_min)

    C_xx = (
        pm_std * C_xxn / (x_max - x_min) ** 2
    )
    C_yy = (
        pm_std * C_yyn / (y_max - y_min) ** 2
    )

    # PDE residual:
    # dC/dt + u*dC/dx + v*dC/dy
    # = D*(d2C/dx2 + d2C/dy2) + source
    residual = (
        C_t + u * C_x + v * C_y
        - D * (C_xx + C_yy) - source
    )

    physics_loss = torch.mean(residual ** 2)

    loss = (
        data_weight * data_loss
        + physics_weight * physics_loss
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Data: {data_loss.item():.6f} | "
            f"Physics: {physics_loss.item():.6f}"
        )

# 6. Predict PM2.5 at the observed coordinates and times
model.eval()

with torch.no_grad():
    C_pred_norm = model(X).cpu().numpy().flatten()

df["predicted_pm25"] = C_pred_norm * pm_std + pm_mean

# 7. Evaluate fit to observations
actual = df[PM_COL].values
predicted = df["predicted_pm25"].values

rmse = np.sqrt(np.mean((actual - predicted) ** 2))
mae = np.mean(np.abs(actual - predicted))

print(f"\nTraining RMSE: {rmse:.3f} µg/m³")
print(f"Training MAE:  {mae:.3f} µg/m³")

# 8. Plot actual versus predicted
plt.figure(figsize=(10, 5))
plt.scatter(actual, predicted, alpha=0.3)
limits = [
    min(actual.min(), predicted.min()),
    max(actual.max(), predicted.max())
]
plt.plot(limits, limits, "r--", label="Perfect prediction")
plt.xlabel("Observed PM2.5 (µg/m³)")
plt.ylabel("Predicted PM2.5 (µg/m³)")
plt.title("PINN: Observed vs Predicted PM2.5")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 9. Save predictions
#df.to_csv("pinn_pm25_predictions.csv", index=False)
#print("Saved predictions to pinn_pm25_predictions.csv")

res_df = pd.DataFrame({
    "train_rmse:": rmse,
    "train_mae":mae})

res_df.to_csv("pinn_pm25_metrics.csv", index=False)
print("Saved metrics to pinn_pm25_metrics.csv")