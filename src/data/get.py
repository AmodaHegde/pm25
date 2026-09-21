import requests
import numpy as np
import pandas as pd

LATITUDE = 28.6139
LONGITUDE = 77.2090
START_DATE = "2025-10-01"
END_DATE = "2026-02-28"

aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
aq_params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "hourly": [
        "pm2_5",
        "pm10",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
        "us_aqi",
        "european_aqi",
        "dust",
        "ammonia",
    ],
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
        "apparent_temperature",
        "precipitation",
        "rain",
        "surface_pressure",
        "wind_speed_10m",
        "wind_direction_10m",
        "weather_code",
        "boundary_layer_height"
    ],
    "timezone": "Asia/Kolkata"
}

weather_response = requests.get(weather_url, params=weather_params)
weather_data = weather_response.json()
df_weather = pd.DataFrame(weather_data["hourly"])

df = pd.merge(df_aq, df_weather, on="time", how="inner")

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
    "ammonia",
    "temperature_2m",
        "relative_humidity_2m",
        "apparent_temperature",
        "precipitation",
        "rain",
        "surface_pressure",
        "wind_speed_10m",
        "wind_direction_10m",
        "weather_code",
        "boundary_layer_height"
]

df_final = df[final_columns].copy()
df_final.to_csv("D:/miniproject/data/bronze/pm25.csv", index=False)

print(f"Bronze dataset successfully created with shape: {df_final.shape}")