import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_plot(fig, filename):
    fig.savefig(RESULTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)

df = pd.read_csv("D:/miniproject/data/silver/pm25.csv")
df["time"] = pd.to_datetime(df["time"])

target_cols = ["pm2_5"]
feature_cols = [
    column
    for column in df.select_dtypes(include=[np.number]).columns
    if column not in target_cols
]


def numeric_columns(dataframe, columns):
    return [column for column in columns if column in dataframe and pd.api.types.is_numeric_dtype(dataframe[column])]

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


# STAGE 2B: Feature-Wide Outlier Analysis

def analyze_feature_outliers(dataframe, feature_cols):
    numeric_features = numeric_columns(dataframe, feature_cols)
    quartiles = dataframe[numeric_features].quantile([0.25, 0.75]).T
    quartiles.columns = ["q1", "q3"]
    quartiles["iqr"] = quartiles["q3"] - quartiles["q1"]
    quartiles["lower_bound"] = quartiles["q1"] - 1.5 * quartiles["iqr"]
    quartiles["upper_bound"] = quartiles["q3"] + 1.5 * quartiles["iqr"]

    outlier_mask = dataframe[numeric_features].lt(quartiles["lower_bound"]) | dataframe[numeric_features].gt(quartiles["upper_bound"])
    summary = pd.DataFrame(
        {
            "outlier_count": outlier_mask.sum(),
            "outlier_pct (%)": outlier_mask.mean() * 100,
        }
    ).join(quartiles)
    summary = summary.sort_values("outlier_count", ascending=False)

    print("\n--- FEATURE OUTLIER SUMMARY (1.5 x IQR) ---")
    print(summary[["outlier_count", "outlier_pct (%)", "lower_bound", "upper_bound"]])
    summary.to_csv(RESULTS_DIR / "silver_feature_outlier_summary.csv")

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.boxplot(data=dataframe[numeric_features], ax=ax, showfliers=True)
    ax.set_title("Feature Distributions and IQR Outliers")
    ax.tick_params(axis="x", rotation=75)
    ax.set_ylabel("Value")
    save_plot(fig, "silver_feature_outlier_boxplots.png")
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


def analyze_seasonal_patterns(dataframe):
    seasonal_data = dataframe.assign(month=dataframe["time"].dt.month)
    month_names = pd.date_range("2024-01-01", periods=12, freq="MS").month_name()
    monthly_profile = seasonal_data.groupby("month")["pm2_5"].agg(["mean", "median", "std", "min", "max", "count"])
    monthly_profile = monthly_profile.reindex(range(1, 13))
    monthly_profile.index = month_names

    print("\n--- SEASONAL PM2.5 SUMMARY ---")
    print(monthly_profile)
    monthly_profile.to_csv(RESULTS_DIR / "silver_seasonal_pm25_summary.csv")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(monthly_profile.index, monthly_profile["mean"], marker="o", label="Mean PM2.5")
    ax.plot(monthly_profile.index, monthly_profile["median"], marker="s", label="Median PM2.5")
    ax.set_xlabel("Month")
    ax.set_ylabel("PM2.5")
    ax.set_title("Seasonal Variation in Average PM2.5")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    ax.grid(True)
    save_plot(fig, "silver_seasonal_pm25_pattern.png")
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=seasonal_data, x=seasonal_data["time"].dt.month_name(), y="pm2_5", order=month_names, ax=ax)
    ax.set_title("Monthly PM2.5 Distributions")
    ax.set_xlabel("Month")
    ax.set_ylabel("PM2.5")
    ax.tick_params(axis="x", rotation=45)
    save_plot(fig, "silver_seasonal_pm25_boxplots.png")
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


# STAGE 5: Bivariate Statistical Analysis

def analyze_bivariate_relationships(dataframe, feature_cols):
    numeric_features = numeric_columns(dataframe, feature_cols)
    rows = []
    for feature in numeric_features:
        paired = dataframe[["pm2_5", feature]].dropna()
        x_values = paired[feature].to_numpy()
        y_values = paired["pm2_5"].to_numpy()
        pearson = np.corrcoef(x_values, y_values)[0, 1]
        spearman = paired["pm2_5"].corr(paired[feature], method="spearman")
        slope, intercept = np.polyfit(x_values, y_values, 1)
        predictions = slope * x_values + intercept
        ss_res = np.sum((y_values - predictions) ** 2)
        ss_tot = np.sum((y_values - y_values.mean()) ** 2)
        rows.append(
            {
                "feature": feature,
                "pearson_correlation": pearson,
                "spearman_correlation": spearman,
                "covariance": paired["pm2_5"].cov(paired[feature]),
                "linear_slope": slope,
                "linear_r_squared": 1 - (ss_res / ss_tot if ss_tot else np.nan),
                "sample_size": len(paired),
            }
        )

    summary = pd.DataFrame(rows).set_index("feature").sort_values(
        "spearman_correlation", key=lambda values: values.abs(), ascending=False
    )
    print("\n--- BIVARIATE STATISTICS: PM2.5 VS FEATURES ---")
    print(summary)
    summary.to_csv(RESULTS_DIR / "silver_bivariate_pm25_statistics.csv")

    plot_features = summary.head(min(6, len(summary))).index
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    for axis, feature in zip(axes, plot_features):
        sns.regplot(data=dataframe, x=feature, y="pm2_5", scatter_kws={"alpha": 0.25, "s": 12}, line_kws={"color": "black"}, ax=axis)
        axis.set_title(f"PM2.5 vs {feature}")
    for axis in axes[len(plot_features):]:
        axis.remove()
    fig.tight_layout()
    save_plot(fig, "silver_bivariate_pm25_relationships.png")
    plt.show()


# 1. Execute Data Quality Check
inspect_data_quality(df)

# 2. Render Boxplots & Distribution Summary
analyze_distributions(df, target_cols=target_cols)

# 2B. Investigate Outliers Across All Numeric Features
analyze_feature_outliers(df, feature_cols=feature_cols)

# 3. Render Hourly Diurnal Line Plot
analyze_time_patterns(df)

# 3B. Render Monthly Seasonal PM2.5 Plots
analyze_seasonal_patterns(df)

# 4. Render Correlation Heatmap
analyze_correlations(df, feature_cols=feature_cols)

# 5. Run Bivariate PM2.5-versus-Feature Analysis
analyze_bivariate_relationships(df, feature_cols=feature_cols)

# Based on the above findings, we shall be dropping "Ammonia" as it is completely null.

