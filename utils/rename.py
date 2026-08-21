import re
import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

FOLDER = Path(r"D:\miniproject\data")


# ============================================================
# EXTRACT STATION NAME
# ============================================================

def extract_station_name(filename):
    """
    Handles filenames such as:

    raw_data_hourly_bapuji_nagar,_bengaluru_-_kspcb_1H (2).csv
        -> bapuji_nagar

    raw_data_hourly_btm_layout,_bengaluru_-_cpcb_1H.csv
        -> btm_layout

    silk_board.csv
        -> silk_board
    """

    name = Path(filename).stem

    # --------------------------------------------------------
    # Case 1:
    # raw_data_hourly_station,_bengaluru_-_kspcb_1H
    # --------------------------------------------------------

    if name.lower().startswith("raw_data_hourly_"):

        # Remove prefix
        name = re.sub(
            r"^raw_data_hourly_",
            "",
            name,
            flags=re.IGNORECASE
        )

        # Remove Windows duplicate suffix
        # (1), (2), (3), etc.
        name = re.sub(
            r"\s*\(\d+\)$",
            "",
            name
        )

        # Extract station before ",_bengaluru"
        match = re.match(
            r"(.+?),_bengaluru_-_.*$",
            name,
            flags=re.IGNORECASE
        )

        if match:
            station = match.group(1)
        else:
            station = name

    # --------------------------------------------------------
    # Case 2:
    # Already clean filename
    #
    # Example:
    # silk_board.csv
    # --------------------------------------------------------

    else:
        station = name

    # --------------------------------------------------------
    # Clean station name
    # --------------------------------------------------------

    station = station.strip()

    # Replace spaces with underscores
    station = station.replace(" ", "_")

    # Remove commas
    station = station.replace(",", "")

    return station


# ============================================================
# EXTRACT YEAR FROM CSV
# ============================================================

def extract_year(csv_file):
    """
    Supports BOTH:

    01-01-2020 00:00
    01-01-2020 23:00

    AND

    2025-01-01 00:00:00
    2025-12-31 23:00:00
    """

    # Read only first column
    df = pd.read_csv(
        csv_file,
        usecols=[0],
        dtype=str,
        encoding="utf-8-sig",
        low_memory=False
    )

    if df.empty:
        raise ValueError("CSV file is empty")

    timestamp_column = df.columns[0]

    timestamps = (
        df[timestamp_column]
        .astype(str)
        .str.strip()
    )

    # Remove empty / NaN-like values
    timestamps = timestamps[
        ~timestamps.str.lower().isin(
            ["nan", "nat", "none", ""]
        )
    ]

    if timestamps.empty:
        raise ValueError("Timestamp column is empty")

    # ========================================================
    # FORMAT 1:
    #
    # DD-MM-YYYY HH:MM
    #
    # Example:
    # 01-01-2020 00:00
    # ========================================================

    years_dd_mm_yyyy = timestamps.str.extract(
        r"^\d{2}-\d{2}-(\d{4})"
    )[0]

    # ========================================================
    # FORMAT 2:
    #
    # YYYY-MM-DD HH:MM:SS
    #
    # Example:
    # 2025-01-01 00:00:00
    # ========================================================

    years_yyyy_mm_dd = timestamps.str.extract(
        r"^(\d{4})-\d{2}-\d{2}"
    )[0]

    # ========================================================
    # Combine both formats
    # ========================================================

    years = years_dd_mm_yyyy.fillna(
        years_yyyy_mm_dd
    )

    years = years.dropna()

    if years.empty:
        raise ValueError(
            "No supported timestamps found. "
            f"First few values were: "
            f"{timestamps.head().tolist()}"
        )

    # Find unique years
    unique_years = sorted(
        years.astype(int).unique()
    )

    # ========================================================
    # Check whether file contains multiple years
    # ========================================================

    if len(unique_years) > 1:
        raise ValueError(
            f"File contains multiple years: {unique_years}"
        )

    return int(unique_years[0])


# ============================================================
# GENERATE UNIQUE OUTPUT NAME
# ============================================================

def get_unique_filename(folder, station, year):

    base_name = f"{station}_{year}"

    new_file = folder / f"{base_name}.csv"

    counter = 2

    while new_file.exists():

        new_file = folder / (
            f"{base_name}_{counter}.csv"
        )

        counter += 1

    return new_file


# ============================================================
# MAIN
# ============================================================

def rename_files():

    csv_files = list(FOLDER.glob("*.csv"))

    if not csv_files:

        print("No CSV files found.")

        return

    print("=" * 70)
    print("BENGALURU AIR QUALITY CSV RENAMER")
    print("=" * 70)

    print(f"Folder: {FOLDER}")
    print(f"Files found: {len(csv_files)}")
    print()

    successful = 0
    failed = 0

    for csv_file in csv_files:

        try:

            print("Processing:")
            print(f"  {csv_file.name}")

            # ------------------------------------------------
            # Extract station
            # ------------------------------------------------

            station = extract_station_name(
                csv_file.name
            )

            # ------------------------------------------------
            # Extract year from Timestamp
            # ------------------------------------------------

            year = extract_year(csv_file)

            # ------------------------------------------------
            # Generate new filename
            # ------------------------------------------------

            new_file = get_unique_filename(
                FOLDER,
                station,
                year
            )

            # ------------------------------------------------
            # Rename
            # ------------------------------------------------

            csv_file.rename(new_file)

            print(f"  Station : {station}")
            print(f"  Year    : {year}")
            print(f"  NEW     : {new_file.name}")
            print()

            successful += 1

        except Exception as e:

            print(f"  ERROR: {e}")
            print()

            failed += 1

    # ========================================================
    # SUMMARY
    # ========================================================

    print("=" * 70)
    print("COMPLETED")
    print("=" * 70)

    print(f"Successfully renamed : {successful}")
    print(f"Failed               : {failed}")

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    rename_files()