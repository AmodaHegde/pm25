
from pathlib import Path
import pandas as pd

# Folder containing one CSV file per station
INPUT_DIR = Path("D:\Downloads\delhi_cpcb")
OUTPUT_FILE = Path("null_percentage_report.csv")


def analyze_null_values(file_path):
    """Calculate null counts and percentages for each column."""
    df = pd.read_csv(file_path)

    total_rows = len(df)

    report = pd.DataFrame({
        "Station": file_path.stem,
        "Column": df.columns,
        "Total_Rows": total_rows,
        "Null_Count": df.isnull().sum().values,
        "Null_Percentage": (
            df.isnull().mean().values * 100
        ).round(2),
    })

    return report


def main():
    csv_files = sorted(INPUT_DIR.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in: {INPUT_DIR.resolve()}")
        return

    reports = []

    for file_path in csv_files:
        try:
            report = analyze_null_values(file_path)
            reports.append(report)
            print(f"Processed: {file_path.name}")
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

    if reports:
        final_report = pd.concat(reports, ignore_index=True)
        final_report.to_csv(OUTPUT_FILE, index=False)

        print(f"\nProcessed {len(reports)} station files.")
        print(f"Report saved to: {OUTPUT_FILE.resolve()}")

        # Display columns with missing values
        print("\nColumns with missing values:")
        print(
            final_report[final_report["Null_Count"] > 0]
            .sort_values(
                ["Station", "Null_Percentage"],
                ascending=[True, False]
            )
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
