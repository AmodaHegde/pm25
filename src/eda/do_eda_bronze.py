import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime

df = pd.read_csv("D:/miniproject/data/bronze/pm25.csv")
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

    print("--- DATA QUALITY SUMMARY ---")
    print(summary)


# STAGE 2: Univariate Statistical Analysis

def analyze_distributions(dataframe, target_cols):
    # Summary statistics
    stats = dataframe[target_cols].describe().T[["mean", "std", "min", "50%", "max"]]
    stats["skewness"] = dataframe[target_cols].skew()
    print("\n--- UNIVARIATE STATISTICS ---")
    print(stats)

    # Boxplots to detect extreme pollution spikes / outliers
    plt.figure(figsize=(12, 4))
    sns.boxplot(data=dataframe[target_cols])
    plt.title("Distribution & Outlier Detection for Atmospheric Variables")
    plt.xticks(rotation=45)
    plt.tight_layout()
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

    plt.figure(figsize=(10, 4))
    plt.plot(
        diurnal_profile.index,
        diurnal_profile["pm2_5"],
        label="PM2.5",
        marker="o",
    )
    plt.plot(
        diurnal_profile.index,
        diurnal_profile["nitrogen_dioxide"],
        label="NO2",
        marker="s",
    )
    plt.xlabel("Hour of Day (IST)")
    plt.ylabel("Concentration")
    plt.title("Average Diurnal (24-Hour) Pollution Pattern in Delhi")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# STAGE 4: Cross-Variable & Interaction Analysis

def analyze_correlations(dataframe, feature_cols):
    # Calculate Pearson/Spearman correlation matrix
    corr_matrix = dataframe[feature_cols].corr(method="spearman")

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    plt.title(
        "Spearman Rank Correlation: Weather Parameters vs. Air Quality"
    )
    plt.tight_layout()
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

