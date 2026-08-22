import glob
import os
import re
import pandas as pd

raw_data_dir = "D:/miniproject/data"
output_file = "D:/miniproject/data/combined_data/combined_air_quality.csv"
metadata_file_name = "D:/miniproject/data/metadata/stations_metadata.csv"
metadata_path = os.path.join(raw_data_dir, metadata_file_name)
metadata_df = pd.read_csv(metadata_path)
csv_files = glob.glob(os.path.join(raw_data_dir, "*.csv"))
dataframes = []

for file_path in csv_files:
  filename = os.path.basename(file_path)
  if filename == metadata_file_name:
    continue
  name_without_ext = filename.replace(".csv", "")
  match = re.match(r"^(.*?)[_-]?(\d{4})$", name_without_ext)
  if not match:
    continue
  station_name = match.group(1).strip("_")
  year = int(match.group(2))
  df = pd.read_csv(file_path)
  df["station"] = station_name
  df["Year"] = year
  dataframes.append(df)

combined_df = pd.concat(dataframes, ignore_index=True)
combined_df["Timestamp"] = pd.to_datetime(combined_df["Timestamp"])
combined_df = combined_df.merge(metadata_df, on="station", how="left")
combined_df = combined_df.sort_values(
    by=["station", "Timestamp"]
).reset_index(drop=True)
combined_df.to_csv(output_file, index=False)