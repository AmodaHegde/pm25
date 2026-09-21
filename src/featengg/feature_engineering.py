import numpy as np
import pandas as pd

# 1. Load Bronze Dataset
df = pd.read_csv("D:/miniproject/data/bronze/pm25.csv")


# 2. Feature Engineering Functions
def add_temporal_features(df):
    timestamps = pd.to_datetime(df["time"])

    # Add wind vector components
    wind_rad = np.radians(df["wind_direction_10m"])
    df["wind_u"] = -df["wind_speed_10m"] * np.sin(wind_rad)
    df["wind_v"] = -df["wind_speed_10m"] * np.cos(wind_rad)

    # Cyclical hour encoding
    hours = timestamps.dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)

    # Day of Year (1 - 365)
    doy = timestamps.dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # Day of Week (0 - 6)
    dow = timestamps.dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    return df


def add_domain_features(df):
    # Coarse to fine particulate ratio (avoid division by zero)
    df["pm_ratio"] = df["pm10"] / (df["pm2_5"] + 1e-6)

    # Reconstruct 10m wind speed from u and v components
    wind_speed = np.sqrt(df["wind_u"] ** 2 + df["wind_v"] ** 2)

    # Ventilation Index = Wind Speed * Boundary Layer Height
    df["ventilation_index"] = wind_speed * df["boundary_layer_height"]

    # Temperature-Humidity Interaction
    df["temp_humidity_interaction"] = (
        df["temperature_2m"] * df["relative_humidity_2m"]
    )

    return df


def create_lags_and_rolling(df, target_cols=["pm2_5"], lags=[1, 2, 3, 6, 24]):
    for col in target_cols:
        # Generate autoregressive lag features
        for lag in lags:
            df[f"{col}_lag_{lag}h"] = df[col].shift(lag)

        # Rolling 6-hour and 24-hour moving averages & standard deviations
        df[f"{col}_roll_mean_6h"] = df[col].shift(1).rolling(window=6).mean()
        df[f"{col}_roll_std_6h"] = df[col].shift(1).rolling(window=6).std()
        df[f"{col}_roll_mean_24h"] = df[col].shift(1).rolling(window=24).mean()

    return df


# 3. Apply Transformations
df = add_temporal_features(df)
df = add_domain_features(df)
df = create_lags_and_rolling(df, target_cols=["pm2_5"])

# 4. Define Complete Silver Column List (Includes engineered features)
final_columns = [
    "time",
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "us_aqi",
    "european_aqi",
    "dust",
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "precipitation",
    "rain",
    "surface_pressure",
    "weather_code",
    "boundary_layer_height",
    "wind_u",
    "wind_v",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    "dow_sin",
    "dow_cos",
    "pm_ratio",
    "ventilation_index",
    "temp_humidity_interaction",
    "pm2_5_lag_1h",
    "pm2_5_lag_2h",
    "pm2_5_lag_3h",
    "pm2_5_lag_6h",
    "pm2_5_lag_24h",
    "pm2_5_roll_mean_6h",
    "pm2_5_roll_std_6h",
    "pm2_5_roll_mean_24h",
]

# Drop ammonia if present in raw bronze data (100% null values)
if "ammonia" in df.columns:
    df = df.drop(columns=["ammonia"])

# Subset to Silver schema
df_final = df[final_columns].copy()

# 5. Handle missing values (Interpolate + Bfill/Ffill for lag NaNs)
numeric_cols = df_final.select_dtypes(include=[np.number]).columns
df_final[numeric_cols] = (
    df_final[numeric_cols].interpolate(method="linear").bfill().ffill()
)

# 6. Save Processed Silver Dataset
df_final.to_csv("D:/miniproject/data/silver/pm25.csv", index=False)

print(f"Silver dataset successfully created with shape: {df_final.shape}")