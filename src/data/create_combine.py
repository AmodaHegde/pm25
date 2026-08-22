import glob
import os
import re
import pandas as pd

metadata_file = "D:/miniproject/data/stations_metadata.csv"
station_metadata = pd.read_csv(metadata_file)
all_files = glob.glob("D:/miniproject/data/*.csv")
dataframes = []

for file_path in all_files:
  filename = os.path.basename(file_path).replace(".csv", "")
  if filename == os.path.basename(metadata_file).replace(".csv", ""):
    continue
  match = re.match(r"^(.*?)[_-]?(\d{4})$", filename)
  if not match:
    continue
  station_name = match.group(1).strip("_")
  year = int(match.group(2))
  temp_df = pd.read_csv(file_path)
  temp_df["station"] = station_name
  temp_df["Year"] = year
  dataframes.append(temp_df)

combined_df = pd.concat(dataframes, ignore_index=True)
combined_df["Timestamp"] = pd.to_datetime(combined_df["Timestamp"])
combined_df = combined_df.merge(station_metadata, on="station", how="left")
combined_df = combined_df.sort_values(
    by=["station", "Timestamp"]
).reset_index(drop=True)
combined_df.to_parquet("processed_air_quality_5yr.parquet", index=False)