import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for script runs

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import config

OUTPUT_DIR = config.project_root / "data" / "processed"


def _save(fig: plt.Figure, filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved -> {path}")

def print_descriptive_stats(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("DESCRIPTIVE STATISTICS -- NUMERIC COLUMNS")
    print("=" * 60)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    desc = df[numeric_cols].describe().T
    desc["median"] = df[numeric_cols].median()
    print(desc[["count", "mean", "median", "std", "min", "25%", "75%", "max"]].to_string())

    print("\n" + "=" * 60)
    print("FREQUENCY COUNTS -- CATEGORICAL COLUMNS")
    print("=" * 60)
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        print(f"\n  [{col}]")
        print(df[col].value_counts().to_string())


def plot_hist_selling_price(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Selling Price Distribution", fontsize=14, fontweight="bold")

    axes[0].hist(df["selling_price"], bins=50, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("Selling Price (INR)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Raw scale")

    axes[1].hist(df["selling_price"], bins=50, color="steelblue", edgecolor="white", log=True)
    axes[1].set_xlabel("Selling Price (INR)")
    axes[1].set_ylabel("Count (log)")
    axes[1].set_title("Log-scale Y axis")

    _save(fig, "hist_selling_price.png")


def plot_hist_km_driven(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Km Driven Distribution", fontsize=14, fontweight="bold")

    axes[0].hist(df["km_driven"], bins=50, color="darkorange", edgecolor="white")
    axes[0].set_xlabel("Km Driven")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Raw scale")

    axes[1].hist(df["km_driven"], bins=50, color="darkorange", edgecolor="white", log=True)
    axes[1].set_xlabel("Km Driven")
    axes[1].set_ylabel("Count (log)")
    axes[1].set_title("Log-scale Y axis")

    _save(fig, "hist_km_driven.png")


def plot_boxplots(df: pd.DataFrame) -> None:
    cols = ["selling_price", "km_driven", "year"]
    fig, axes = plt.subplots(1, len(cols), figsize=(14, 5))
    fig.suptitle("Box Plots -- Outlier Detection", fontsize=14, fontweight="bold")

    for ax, col in zip(axes, cols):
        ax.boxplot(df[col].dropna(), vert=True, patch_artist=True,
                   boxprops=dict(facecolor="lightcyan", color="navy"),
                   medianprops=dict(color="red", linewidth=2),
                   flierprops=dict(marker="o", markersize=3, alpha=0.4, color="gray"))
        ax.set_title(col)
        ax.set_ylabel(col)

    _save(fig, "boxplot_outliers.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Correlation Matrix -- Numeric Features", fontsize=13, fontweight="bold")
    _save(fig, "correlation_heatmap.png")


def plot_pairplot(df: pd.DataFrame) -> None:
    pair_cols = ["selling_price", "km_driven", "year", "owner"]
    pair_df = df[[c for c in pair_cols if c in df.columns]].dropna()

    grid = sns.pairplot(
        pair_df,
        diag_kind="kde",
        plot_kws={"alpha": 0.4, "s": 15},
        height=2.5,
    )
    grid.figure.suptitle(
        "Pairplot -- Key Features vs Selling Price",
        y=1.02, fontsize=13, fontweight="bold",
    )
    _save(grid.figure, "pairplot.png")


def run_eda(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    print_descriptive_stats(df)

    print("\nGenerating plots...")
    plot_hist_selling_price(df)
    plot_hist_km_driven(df)
    plot_boxplots(df)
    plot_correlation_heatmap(df)
    plot_pairplot(df)

    print("\nEDA complete. All plots saved to data/processed/")


if __name__ == "__main__":
    from data_loader import load_data, encode_owner
    data = load_data()
    data = encode_owner(data)
    run_eda(data)
