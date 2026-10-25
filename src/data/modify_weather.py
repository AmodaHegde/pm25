import pandas as pd
from pathlib import Path

weather = pd.read_csv("D:/miniproject/era5_weather/all_stations_weather_2025.csv")
OUTPUT_FILE = Path(
    r"D:/miniproject/data/bronze/all_stations_weather_2025.csv"
)

weather.drop(
    columns=["latitude", "longitude"],
    inplace=True
)

weather.to_csv(OUTPUT_FILE, index=False)