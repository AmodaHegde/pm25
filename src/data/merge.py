import pandas as pd
from pathlib import Path

air = pd.read_csv("D:/miniproject/data/bronze/delhi_combined.csv")
weather = pd.read_csv("D:/miniproject/era5_weather/all_stations_weather_2025.csv")
OUTPUT_FILE = Path(
    r"D:/miniproject/data/bronze/delhi_air_climate_2025.csv"
)

air["timestamp"] = pd.to_datetime(air["timestamp"])
weather["timestamp"] = pd.to_datetime(weather["timestamp"])

invalid_air = air["timestamp"].isna().sum()
invalid_weather = weather["timestamp"].isna().sum()

print("Invalid air timestamps:", invalid_air)
print("Invalid weather timestamps:", invalid_weather)

if invalid_air > 0 or invalid_weather > 0:
    raise ValueError(
        "Invalid timestamps found. Check the source CSV files."
    )

air["timestamp"] = air["timestamp"].dt.floor("h")
weather["timestamp"] = weather["timestamp"].dt.floor("h")

air["station_name"] = air["station_name"].astype(str).str.strip()
weather["station_name"] = (
    weather["station_name"].astype(str).str.strip()
)

air_duplicates = air.duplicated(
    subset=["station_name", "timestamp"]
).sum()

weather_duplicates = weather.duplicated(
    subset=["station_name", "timestamp"]
).sum()

print("Duplicate air station-hours:", air_duplicates)
print("Duplicate weather station-hours:", weather_duplicates)

if air_duplicates > 0:
    raise ValueError(
        "Air data has duplicate station-timestamp pairs."
    )

if weather_duplicates > 0:
    raise ValueError(
        "Weather data has duplicate station-timestamp pairs."
    )

overlapping_columns = (
    set(air.columns) & set(weather.columns)
) - {
    "station_name",
    "timestamp",
    "latitude",
    "longitude"
}

weather = weather.drop(
    columns=list(overlapping_columns),
    errors="ignore"
)

merged = air.merge(
    weather,
    left_on=["station_name", "timestamp"],
    right_on=["station_name", "timestamp"],
    how="left",
    validate="one_to_one",
    indicator=True
)

matched = (merged["_merge"] == "both").sum()
unmatched = (merged["_merge"] == "left_only").sum()

print("\nMerge results")
print("------------------------------")
print("Air rows:", len(air))
print("Weather rows:", len(weather))
print("Merged rows:", len(merged))
print("Matched rows:", matched)
print("Unmatched air rows:", unmatched)

# Remove helper columns
merged.drop(
    columns=["_merge"],
    inplace=True,
    errors="ignore"
)

# Confirm that no air records were lost or duplicated
if len(merged) != len(air):
    raise ValueError(
        "Merged row count differs from air data row count."
    )

# --------------------------------------------------
# 10. Check missing values in climate columns
# --------------------------------------------------

climate_columns = [
    col for col in weather.columns
    if col not in {"station_name", "timestamp"}
]

# print("\nMissing climate values")
# print("------------------------------")

# for col in climate_columns:
#     print(f"{col}: {merged[col].isna().sum()}")

# --------------------------------------------------
# 11. Save merged dataset
# --------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

merged.to_csv(OUTPUT_FILE, index=False)

print("\nMerge completed successfully.")
print("Final shape:", merged.shape)
print("Saved to:", OUTPUT_FILE)