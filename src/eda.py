"""
src/eda.py
Exploratory Data Analysis module with production-quality, reusable visualization functions.
"""

import os
from typing import Optional, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Set styling defaults
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 150

CLASS_COLORS = {
    "Light_Load": "#2ecc71",
    "Medium_Load": "#f39c12",
    "Maximum_Load": "#e74c3c",
}


def _ensure_dir(path: Optional[str]) -> None:
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)


def plot_target_distribution(
    df: pd.DataFrame,
    target_col: str = "Load_Type",
    save_path: Optional[str] = "outputs/figures/eda_target_distribution.png"
) -> plt.Figure:
    """Plots the distribution and class balance of Load_Type."""
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df[target_col].value_counts()
    percentages = (df[target_col].value_counts(normalize=True) * 100).round(2)

    palette = [CLASS_COLORS.get(c, "#3498db") for c in counts.index]
    bars = sns.barplot(x=counts.index, y=counts.values, palette=palette, ax=ax)

    for i, p in enumerate(bars.patches):
        height = p.get_height()
        ax.annotate(
            f"{int(height):,} ({percentages.iloc[i]}%)",
            (p.get_x() + p.get_width() / 2.0, height / 2.0),
            ha="center",
            va="center",
            color="white",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_title("Power System Load Type Distribution (Full Dataset)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Load Category", fontsize=12)
    ax.set_ylabel("Record Count", fontsize=12)
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[EDA] Saved target distribution plot to {save_path}")

    return fig


def plot_missing_values(
    df: pd.DataFrame,
    save_path: Optional[str] = "outputs/figures/eda_missing_values.png"
) -> plt.Figure:
    """Plots the percentage of missing values per column."""
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]

    fig, ax = plt.subplots(figsize=(10, 5))
    if len(missing_pct) == 0:
        ax.text(0.5, 0.5, "No Missing Values Found", ha="center", va="center", fontsize=14)
        ax.set_title("Missing Values Overview", fontsize=14, fontweight="bold")
    else:
        bars = sns.barplot(x=missing_pct.values, y=missing_pct.index, color="#e67e22", ax=ax)
        for i, val in enumerate(missing_pct.values):
            ax.text(val + 0.1, i, f"{val:.2f}% ({df[missing_pct.index[i]].isnull().sum():,})", va="center", fontsize=10)

        ax.set_title("Missing Values Percentage by Feature", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Missing Percentage (%)", fontsize=12)
        ax.set_ylabel("Column Name", fontsize=12)
        ax.set_xlim(0, max(missing_pct.values) * 1.35)

    plt.tight_layout()
    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[EDA] Saved missing values plot to {save_path}")

    return fig


def plot_numerical_distributions(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    save_path: Optional[str] = "outputs/figures/eda_numerical_distributions.png"
) -> plt.Figure:
    """Plots histograms with KDE overlays for key numerical features."""
    if feature_cols is None:
        feature_cols = [
            "Usage_kWh",
            "Lagging_Current_Reactive.Power_kVarh",
            "Leading_Current_Reactive_Power_kVarh",
            "CO2(tCO2)",
            "Lagging_Current_Power_Factor",
            "Leading_Current_Power_Factor",
        ]
    cols_to_plot = [c for c in feature_cols if c in df.columns]

    n_cols = 3
    n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(cols_to_plot):
        data_clean = df[col].dropna()
        sns.histplot(data_clean, kde=True, ax=axes[i], color="#3498db", bins=30)
        axes[i].set_title(f"Distribution of {col}", fontsize=11, fontweight="bold")
        axes[i].set_xlabel(col, fontsize=10)
        axes[i].set_ylabel("Frequency", fontsize=10)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[EDA] Saved numerical distributions plot to {save_path}")

    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    save_path: Optional[str] = "outputs/figures/eda_correlation_heatmap.png"
) -> plt.Figure:
    """Plots Pearson correlation heatmap between numerical features."""
    num_df = df.select_dtypes(include=[np.number])
    if feature_cols:
        num_df = num_df[[c for c in feature_cols if c in num_df.columns]]

    corr = num_df.corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"},
        annot=True,
        fmt=".2f",
        annot_kws={"size": 9},
        ax=ax,
    )
    ax.set_title("Correlation Matrix of Power System Features", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[EDA] Saved correlation heatmap to {save_path}")

    return fig


def plot_usage_over_time(
    df: pd.DataFrame,
    date_col: str = "Date_Time",
    target_col: str = "Load_Type",
    save_path: Optional[str] = "outputs/figures/eda_usage_over_time.png"
) -> plt.Figure:
    """Plots daily aggregated Usage_kWh over time with moving averages."""
    data = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
        data[date_col] = pd.to_datetime(data[date_col])

    daily_usage = data.set_index(date_col)["Usage_kWh"].resample("D").mean()
    rolling_7d = daily_usage.rolling(window=7, min_periods=1).mean()
    rolling_30d = daily_usage.rolling(window=30, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(daily_usage.index, daily_usage.values, alpha=0.35, color="#34495e", label="Daily Mean Usage (kWh)")
    ax.plot(rolling_7d.index, rolling_7d.values, color="#e67e22", linewidth=2, label="7-Day Moving Average")
    ax.plot(rolling_30d.index, rolling_30d.values, color="#e74c3c", linewidth=2.5, label="30-Day Trend")

    # Mark the final test month boundary (December 2018)
    split_date = pd.Timestamp("2018-12-01")
    if split_date >= daily_usage.index.min() and split_date <= daily_usage.index.max():
        ax.axvline(split_date, color="#2980b9", linestyle="--", linewidth=2, label="Test Set Split (2018-12-01)")

    ax.set_title("Power Consumption (Usage_kWh) Over Time with Rolling Averages", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Mean Usage (kWh per 15-min)", fontsize=12)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[EDA] Saved usage over time plot to {save_path}")

    return fig


def plot_load_type_over_time(
    df: pd.DataFrame,
    date_col: str = "Date_Time",
    target_col: str = "Load_Type",
    save_path: Optional[str] = "outputs/figures/eda_load_type_over_time.png"
) -> plt.Figure:
    """Plots weekly distribution of Load_Type states over time."""
    data = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
        data[date_col] = pd.to_datetime(data[date_col])

    data["Week"] = data[date_col].dt.to_period("W").dt.to_timestamp()
    weekly_dist = pd.crosstab(data["Week"], data[target_col], normalize="index") * 100

    # Ensure consistent order
    for col in ["Light_Load", "Medium_Load", "Maximum_Load"]:
        if col not in weekly_dist.columns:
            weekly_dist[col] = 0.0
    weekly_dist = weekly_dist[["Light_Load", "Medium_Load", "Maximum_Load"]]

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = [CLASS_COLORS[c] for c in weekly_dist.columns]
    weekly_dist.plot(kind="area", stacked=True, color=colors, alpha=0.8, ax=ax)

    ax.set_title("Proportion of Load Types Over Time (Weekly Aggregation)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Proportion of Records (%)", fontsize=12)
    ax.set_ylim(0, 100)
    ax.legend(title="Load Type", loc="lower right", frameon=True)
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[EDA] Saved load type over time plot to {save_path}")

    return fig


def plot_usage_by_load_type(
    df: pd.DataFrame,
    save_path: Optional[str] = "outputs/figures/eda_usage_by_load_type.png"
) -> plt.Figure:
    """Plots boxplot and violin plot of Usage_kWh across Load_Type classes."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    palette = [CLASS_COLORS.get(c, "#3498db") for c in ["Light_Load", "Medium_Load", "Maximum_Load"]]

    sns.boxplot(
        data=df,
        x="Load_Type",
        y="Usage_kWh",
        order=["Light_Load", "Medium_Load", "Maximum_Load"],
        palette=palette,
        ax=axes[0],
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"},
    )
    axes[0].set_title("Usage (kWh) Boxplot by Load Type", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Load Type", fontsize=11)
    axes[0].set_ylabel("Usage (kWh)", fontsize=11)

    sns.violinplot(
        data=df,
        x="Load_Type",
        y="Usage_kWh",
        order=["Light_Load", "Medium_Load", "Maximum_Load"],
        palette=palette,
        ax=axes[1],
        inner="quartile",
    )
    axes[1].set_title("Usage (kWh) Density Distribution by Load Type", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Load Type", fontsize=11)
    axes[1].set_ylabel("Usage (kWh)", fontsize=11)

    plt.tight_layout()
    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[EDA] Saved usage by load type plot to {save_path}")

    return fig


def plot_load_type_by_month(
    df: pd.DataFrame,
    date_col: str = "Date_Time",
    target_col: str = "Load_Type",
    save_path: Optional[str] = "outputs/figures/eda_load_type_by_month.png"
) -> plt.Figure:
    """Plots Load_Type breakdown across calendar months."""
    data = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
        data[date_col] = pd.to_datetime(data[date_col])

    data["Month_Name"] = data[date_col].dt.strftime("%b")
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    ct = pd.crosstab(data["Month_Name"], data[target_col]).reindex(month_order)
    for col in ["Light_Load", "Medium_Load", "Maximum_Load"]:
        if col not in ct.columns:
            ct[col] = 0
    ct = ct[["Light_Load", "Medium_Load", "Maximum_Load"]]

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = [CLASS_COLORS[c] for c in ct.columns]
    ct.plot(kind="bar", stacked=False, color=colors, ax=ax, width=0.8)

    ax.set_title("Load Type Frequency by Calendar Month (2018)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Record Count", fontsize=12)
    ax.legend(title="Load Type", frameon=True)
    plt.xticks(rotation=0)
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[EDA] Saved load type by month plot to {save_path}")

    return fig


def plot_load_type_by_hour(
    df: pd.DataFrame,
    date_col: str = "Date_Time",
    target_col: str = "Load_Type",
    save_path: Optional[str] = "outputs/figures/eda_load_type_by_hour.png"
) -> plt.Figure:
    """Plots Load_Type frequency across 24 hours of the day."""
    data = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
        data[date_col] = pd.to_datetime(data[date_col])

    data["Hour"] = data[date_col].dt.hour
    ct = pd.crosstab(data["Hour"], data[target_col])
    for col in ["Light_Load", "Medium_Load", "Maximum_Load"]:
        if col not in ct.columns:
            ct[col] = 0
    ct = ct[["Light_Load", "Medium_Load", "Maximum_Load"]]

    fig, ax = plt.subplots(figsize=(13, 5))
    colors = [CLASS_COLORS[c] for c in ct.columns]
    ct.plot(kind="bar", stacked=True, color=colors, ax=ax, width=0.85)

    ax.set_title("Hourly Distribution of Load Types Across 24-Hour Cycle", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Hour of Day (00:00 to 23:00)", fontsize=12)
    ax.set_ylabel("Total Number of Observations", fontsize=12)
    ax.legend(title="Load Type", frameon=True)
    plt.xticks(rotation=0)
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[EDA] Saved load type by hour plot to {save_path}")

    return fig


def plot_power_features_relationships(
    df: pd.DataFrame,
    save_path: Optional[str] = "outputs/figures/eda_power_features_relationships.png"
) -> plt.Figure:
    """Plots key power relationships (Lagging Reactive Power, Power Factor, CO2) vs Load Type."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    palette = [CLASS_COLORS.get(c, "#3498db") for c in ["Light_Load", "Medium_Load", "Maximum_Load"]]

    # 1. Lagging Reactive Power vs Load Type
    sns.boxplot(
        data=df,
        x="Load_Type",
        y="Lagging_Current_Reactive.Power_kVarh",
        order=["Light_Load", "Medium_Load", "Maximum_Load"],
        palette=palette,
        ax=axes[0],
        showfliers=False,
    )
    axes[0].set_title("Lagging Reactive Power (kVarh)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Load Type", fontsize=10)
    axes[0].set_ylabel("Lagging Reactive Power (kVarh)", fontsize=10)

    # 2. Lagging Power Factor vs Load Type
    sns.boxplot(
        data=df,
        x="Load_Type",
        y="Lagging_Current_Power_Factor",
        order=["Light_Load", "Medium_Load", "Maximum_Load"],
        palette=palette,
        ax=axes[1],
        showfliers=False,
    )
    axes[1].set_title("Lagging Power Factor (%)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Load Type", fontsize=10)
    axes[1].set_ylabel("Power Factor", fontsize=10)

    # 3. CO2 vs Load Type
    sns.boxplot(
        data=df,
        x="Load_Type",
        y="CO2(tCO2)",
        order=["Light_Load", "Medium_Load", "Maximum_Load"],
        palette=palette,
        ax=axes[2],
        showfliers=False,
    )
    axes[2].set_title("CO2 Emissions (tCO2)", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Load Type", fontsize=10)
    axes[2].set_ylabel("CO2 (tCO2)", fontsize=10)

    plt.tight_layout()
    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[EDA] Saved power features relationships plot to {save_path}")

    return fig


def generate_all_eda_figures(df: pd.DataFrame, output_dir: str = "outputs/figures") -> None:
    """Generates and saves all core EDA figures in batch."""
    os.makedirs(output_dir, exist_ok=True)
    plot_target_distribution(df, save_path=os.path.join(output_dir, "eda_target_distribution.png"))
    plot_missing_values(df, save_path=os.path.join(output_dir, "eda_missing_values.png"))
    plot_numerical_distributions(df, save_path=os.path.join(output_dir, "eda_numerical_distributions.png"))
    plot_correlation_heatmap(df, save_path=os.path.join(output_dir, "eda_correlation_heatmap.png"))
    plot_usage_over_time(df, save_path=os.path.join(output_dir, "eda_usage_over_time.png"))
    plot_load_type_over_time(df, save_path=os.path.join(output_dir, "eda_load_type_over_time.png"))
    plot_usage_by_load_type(df, save_path=os.path.join(output_dir, "eda_usage_by_load_type.png"))
    plot_load_type_by_month(df, save_path=os.path.join(output_dir, "eda_load_type_by_month.png"))
    plot_load_type_by_hour(df, save_path=os.path.join(output_dir, "eda_load_type_by_hour.png"))
    plot_power_features_relationships(df, save_path=os.path.join(output_dir, "eda_power_features_relationships.png"))
    plt.close("all")
    print(f"[EDA] All EDA figures successfully generated and saved to {output_dir}")
