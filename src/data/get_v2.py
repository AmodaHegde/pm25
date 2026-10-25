
import os
import time
import requests
import pandas as pd
import numpy as np

# ---------------- CONFIGURATION ----------------

STATIONS_FILE = "D:/miniproject/utils/stations.csv"
OUTPUT_DIR = "era5_weather"

START_DATE = "2025-01-01"
END_DATE = "2025-12-31"

API_URL = "https://archive-api.open-meteo.com/v1/archive"

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "rain",
    "surface_pressure",
    "boundary_layer_height"
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- LOAD STATIONS ----------------

stations = pd.read_csv(STATIONS_FILE)

required = {"station_name", "latitude", "longitude"}
if not required.issubset(stations.columns):
    raise ValueError(
        f"stations.csv must contain: {required}"
    )

stations["latitude"] = pd.to_numeric(
    stations["latitude"], errors="raise"
)
stations["longitude"] = pd.to_numeric(
    stations["longitude"], errors="raise"
)

# ---------------- DOWNLOAD DATA ----------------

all_weather = []

for _, station in stations.iterrows():

    name = station["station_name"]
    lat = station["latitude"]
    lon = station["longitude"]

    print(f"Downloading: {name}")

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(HOURLY_VARIABLES),
        "models": "era5",
        "timezone": "Asia/Kolkata",
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm"
    }

    # Retry temporary network/API failures
    for attempt in range(3):
        try:
            response = requests.get(
                API_URL, params=params, timeout=120
            )
            response.raise_for_status()
            data = response.json()

            if "hourly" not in data:
                raise ValueError(
                    f"Unexpected API response: {data}"
                )

            break

        except (requests.RequestException, ValueError):
            if attempt == 2:
                raise
            time.sleep(3 * (attempt + 1))

    weather = pd.DataFrame(data["hourly"])
    weather = weather.rename(columns={"time": "timestamp"})

    # Convert meteorological wind direction to components.
    # Direction is where wind comes FROM, clockwise from north.
    direction_rad = np.radians(
        weather["wind_direction_10m"]
    )
    speed = weather["wind_speed_10m"]

    weather["wind_u"] = -speed * np.sin(direction_rad)
    weather["wind_v"] = -speed * np.cos(direction_rad)

    # Wind components are undefined if direction is missing.
    missing_direction = weather["wind_direction_10m"].isna()
    weather.loc[missing_direction, ["wind_u", "wind_v"]] = np.nan

    weather.insert(1, "station", name)
    weather["latitude"] = lat
    weather["longitude"] = lon

    # Save an individual station file
    safe_name = (
        str(name).strip().lower()
        .replace(" ", "_")
        .replace("/", "_")
    )

    station_file = os.path.join(
        OUTPUT_DIR, f"{safe_name}_weather_2025.csv"
    )
    weather.to_csv(station_file, index=False)

    all_weather.append(weather)

    print(
        f"Saved {station_file} "
        f"({len(weather)} hourly rows)"
    )

    # Avoid sending requests too rapidly
    time.sleep(1)

# ---------------- COMBINE ALL STATIONS ----------------

combined = pd.concat(all_weather, ignore_index=True)

combined_file = os.path.join(
    OUTPUT_DIR, "all_stations_weather_2025.csv"
)
combined.to_csv(combined_file, index=False)

print("\nDownload complete.")
print(f"Stations downloaded: {len(all_weather)}")
print(f"Combined rows: {len(combined)}")
print(f"Combined file: {combined_file}")

print("\nMissing values by column:")
print(combined.isna().sum())
