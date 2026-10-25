
from pathlib import Path
import pandas as pd

# ---------- Configuration ----------
INPUT_DIR = Path("D:/Downloads/delhi_cpcb")
OUTPUT_DIR = Path("missing_data_analysis")

TIME_COL = "Timestamp"
PM25_COL = "PM2.5 (µg/m³)"

EXPECTED_INTERVAL = "1h"
LONG_GAP_HOURS = 24


def find_column(df, requested_name):
    """Find a column without being sensitive to letter case."""
    matches = {
        str(col).strip().lower(): col
        for col in df.columns
    }
    key = requested_name.strip().lower()

    if key not in matches:
        raise ValueError(
            f"Column '{requested_name}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    return matches[key]


def analyze_station(file_path):
    df = pd.read_csv(file_path)

    time_col = find_column(df, TIME_COL)
    pm25_col = find_column(df, PM25_COL)

    df[time_col] = pd.to_datetime(
        df[time_col], errors="coerce"
    )
    df[pm25_col] = pd.to_numeric(
        df[pm25_col], errors="coerce"
    )

    # Remove rows with invalid timestamps and sort chronologically.
    df = df.dropna(subset=[time_col]).sort_values(time_col)

    if df.empty:
        raise ValueError("No valid timestamps found.")

    # Duplicate timestamps can make hourly reindexing ambiguous.
    duplicate_count = int(df[time_col].duplicated().sum())

    # If duplicates exist, retain a single row per timestamp.
    # Mean is used for duplicate PM2.5 measurements.
    df = (
        df.groupby(time_col, as_index=True)[pm25_col]
        .mean()
        .sort_index()
    )

    # Build the complete hourly timeline, including absent rows.
    full_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=EXPECTED_INTERVAL
    )

    original_times = df.index
    observed = df.reindex(full_index)

    # Missing target readings include both null values in existing
    # rows and timestamps that were entirely absent.
    missing = observed[pm25_col].isna()

    # Identify the beginning and end of each consecutive missing run.
    run_id = missing.ne(missing.shift(fill_value=False)).cumsum()
    missing_rows = observed.loc[missing].copy()
    missing_rows["run_id"] = run_id[missing].values

    gap_records = []

    for _, run in missing_rows.groupby("run_id"):
        start = run.index[0]
        end = run.index[-1]

        # With hourly data, a run of N timestamps represents N hours
        # of missing hourly readings.
        duration_hours = len(run)

        absent_timestamps = sum(
            timestamp not in original_times
            for timestamp in run.index
        )

        null_existing_rows = (
            duration_hours - absent_timestamps
        )

        gap_records.append({
            "Station": file_path.stem,
            "Gap_Start": start,
            "Gap_End": end,
            "Duration_Hours": duration_hours,
            "Duration_Days": round(duration_hours / 24, 2),
            "Absent_Timestamps": absent_timestamps,
            "Null_PM25_Existing_Rows": null_existing_rows,
            "Long_Gap_24h_or_More": (
                duration_hours >= LONG_GAP_HOURS
            )
        })

    # Station-level summary
    expected_rows = len(full_index)
    missing_count = int(missing.sum())
    gaps = pd.DataFrame(gap_records)

    summary = {
        "Station": file_path.stem,
        "Original_Unique_Timestamps": len(df),
        "Expected_Hourly_Timestamps": expected_rows,
        "Absent_Timestamps": sum(
            timestamp not in original_times
            for timestamp in full_index
        ),
        "Missing_PM25_Hours": missing_count,
        "Missing_PM25_Percentage": round(
            missing_count / expected_rows * 100, 2
        ),
        "Number_of_Gap_Runs": len(gaps),
        "Longest_Gap_Hours": (
            int(gaps["Duration_Hours"].max())
            if not gaps.empty else 0
        ),
        "Gaps_24h_or_More": (
            int(gaps["Long_Gap_24h_or_More"].sum())
            if not gaps.empty else 0
        ),
        "Duplicate_Timestamps": duplicate_count
    }

    return summary, gap_records


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(INPUT_DIR.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {INPUT_DIR.resolve()}")
        return

    summaries = []
    all_gaps = []

    for file_path in csv_files:
        try:
            summary, gaps = analyze_station(file_path)
            summaries.append(summary)
            all_gaps.extend(gaps)

            print(
                f"{file_path.name}: "
                f"{summary['Missing_PM25_Percentage']}% missing, "
                f"longest gap = "
                f"{summary['Longest_Gap_Hours']} hours"
            )

        except Exception as error:
            print(f"Error in {file_path.name}: {error}")

    if summaries:
        summary_df = pd.DataFrame(summaries)
        summary_df.to_csv(
            OUTPUT_DIR / "station_gap_summary.csv",
            index=False
        )

        gap_columns = [
            "Station", "Gap_Start", "Gap_End",
            "Duration_Hours", "Duration_Days",
            "Absent_Timestamps", "Null_PM25_Existing_Rows",
            "Long_Gap_24h_or_More"
        ]

        gaps_df = pd.DataFrame(
            all_gaps, columns=gap_columns
        )
        gaps_df.to_csv(
            OUTPUT_DIR / "consecutive_missing_gaps.csv",
            index=False
        )

        print("\nAnalysis complete.")
        print(f"Reports saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
