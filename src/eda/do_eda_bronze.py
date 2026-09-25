import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_plot(fig, filename):
    output_path = RESULTS_DIR / filename
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


df = pd.read_csv("D:/miniproject/data/bronze/pm25-1.csv")
df["time"] = pd.to_datetime(df["time"])

feature_cols = [
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
target_cols = ["pm2_5"]

# STAGE 1: Data Completeness & Quality Inspection

def inspect_data_quality(dataframe):
    # Check shape, data types, missing values, and duplicate timestamps
    summary = pd.DataFrame(
        {
            "dtype": dataframe.dtypes,
            "null_count": dataframe.isnull().sum(),
            "null_pct (%)": (dataframe.isnull().sum() / len(dataframe)) * 100,
        }
    )

    summary_path = RESULTS_DIR / "bronze_data_quality_summary.csv"
    summary.to_csv(summary_path)

    print("--- DATA QUALITY SUMMARY ---")
    print(summary)
    print(f"Saved data quality summary to: {summary_path}")


# STAGE 2: Univariate Statistical Analysis

def analyze_distributions(dataframe, target_cols):
    # Summary statistics
    stats = dataframe[target_cols].describe().T[["mean", "std", "min", "50%", "max"]]
    stats["skewness"] = dataframe[target_cols].skew()
    stats_path = RESULTS_DIR / "bronze_univariate_statistics.csv"
    stats.to_csv(stats_path)

    print("\n--- UNIVARIATE STATISTICS ---")
    print(stats)
    print(f"Saved univariate stats to: {stats_path}")

    # Boxplots to detect extreme pollution spikes / outliers
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.boxplot(data=dataframe[target_cols], ax=ax)
    ax.set_title("Distribution & Outlier Detection for Atmospheric Variables")
    ax.tick_params(axis="x", rotation=45)
    save_plot(fig, "bronze_distribution_boxplot.png")
    plt.show()


# STAGE 3: Temporal & Seasonal Dynamics Analysis

def analyze_time_patterns(dataframe):
    dataframe["hour"] = dataframe["time"].dt.hour
    dataframe["month"] = dataframe["time"].dt.month_name()
    dataframe["day_of_week"] = dataframe["time"].dt.day_name()

    # Diurnal (hourly) cycle of key pollutants (e.g., PM2.5, NO2)
    diurnal_profile = dataframe.groupby("hour")[
        ["pm2_5", "nitrogen_dioxide", "temperature_2m"]
    ].mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        diurnal_profile.index,
        diurnal_profile["pm2_5"],
        label="PM2.5",
        marker="o",
    )
    ax.plot(
        diurnal_profile.index,
        diurnal_profile["nitrogen_dioxide"],
        label="NO2",
        marker="s",
    )
    ax.set_xlabel("Hour of Day (IST)")
    ax.set_ylabel("Concentration")
    ax.set_title("Average Diurnal (24-Hour) Pollution Pattern in Delhi")
    ax.legend()
    ax.grid(True)
    save_plot(fig, "bronze_diurnal_pollution_pattern.png")
    plt.show()


# STAGE 4: Cross-Variable & Interaction Analysis

def analyze_correlations(dataframe, feature_cols):
    # Calculate Pearson/Spearman correlation matrix
    corr_matrix = dataframe[feature_cols].corr(method="spearman")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Spearman Rank Correlation: Weather Parameters vs. Air Quality")
    save_plot(fig, "bronze_correlation_heatmap.png")
    plt.show()


# 1. Execute Data Quality Check
inspect_data_quality(df)

# 2. Render Boxplots & Distribution Summary
analyze_distributions(df, target_cols=target_cols)

# 3. Render Hourly Diurnal Line Plot
analyze_time_patterns(df)

# 4. Render Correlation Heatmap
analyze_correlations(df, feature_cols=feature_cols)

# Based on the above findings, we shall be dropping "Ammonia" as it is completely null.

