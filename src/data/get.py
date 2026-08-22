import requests
import numpy as np
import pandas as pd

LATITUDE = 28.6139
LONGITUDE = 77.2090
START_DATE = "2025-10-15"
END_DATE = "2026-02-15"

aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
aq_params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "hourly": ["pm2_5", "nitrogen_dioxide"],
    "timezone": "Asia/Kolkata"
}

aq_response = requests.get(aq_url, params=aq_params)
aq_data = aq_response.json()
df_aq = pd.DataFrame(aq_data["hourly"])

weather_url = "https://archive-api.open-meteo.com/v1/archive"
weather_params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "hourly": [
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "wind_direction_10m",
        "precipitation",
        "boundary_layer_height"
    ],
    "timezone": "Asia/Kolkata"
}

weather_response = requests.get(weather_url, params=weather_params)
weather_data = weather_response.json()
df_weather = pd.DataFrame(weather_data["hourly"])

df = pd.merge(df_aq, df_weather, on="time", how="inner")

wind_rad = np.radians(df["wind_direction_10m"])
df["wind_u"] = -df["wind_speed_10m"] * np.sin(wind_rad)
df["wind_v"] = -df["wind_speed_10m"] * np.cos(wind_rad)

timestamps = pd.to_datetime(df["time"])
hours = timestamps.dt.hour
df["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
df["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)

final_columns = [
    "time",
    "pm2_5",
    "temperature_2m",
    "relative_humidity_2m",
    "wind_u",
    "wind_v",
    "boundary_layer_height",
    "precipitation",
    "nitrogen_dioxide",
    "hour_sin",
    "hour_cos"
]

df_final = df[final_columns].copy()
numeric_cols = df_final.select_dtypes(include=[np.number]).columns
df_final[numeric_cols] = df_final[numeric_cols].interpolate(method="linear").bfill().ffill()
df_final.to_csv("delhi_pm25_lstm_dataset.csv", index=False)

print(f"Dataset successfully created with shape: {df_final.shape}")