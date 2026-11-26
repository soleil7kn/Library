
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Dataset configuration
# ============================================================

DATASETS = {
    "Weather": {
        "path": "../dataset/weather/weather.csv",
        "interval_minutes": 10,
    },
    "ETTh1": {
        "path": "../dataset/ETT-small/ETTh1.csv",
        "interval_minutes": 60,
    },
    "ETTh2": {
        "path": "../dataset/ETT-small/ETTh2.csv",
        "interval_minutes": 60,
    },
    "ETTm1": {
        "path": "../dataset/ETT-small/ETTm1.csv",
        "interval_minutes": 15,
    },
    "ETTm2": {
        "path": "../dataset/ETT-small/ETTm2.csv",
        "interval_minutes": 15,
    },
}


# ============================================================
# Configuration
# ============================================================

WINDOW_DAYS = 7
STRIDE_DAYS = 1

OUTPUT_DIR = Path("./correlation_stability_results")

# CoV calculation: avoid division by near-zero mean correlation
COV_EPSILON = 0.05


# ============================================================
# Load dataset
# ============================================================

def load_dataset(path):
    df = pd.read_csv(path)

    # Remove date column from correlation calculation
    date_col = next(
        (c for c in df.columns if c.lower() == "date"),
        None
    )

    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col])
        dates = df[date_col]
        df = df.drop(columns=[date_col])
    else:
        dates = pd.Series(np.arange(len(df)))

    # Use numeric columns only
    df = df.select_dtypes(include=[np.number])

    # Remove constant columns
    df = df.loc[:, df.nunique() > 1]

    # Handle missing values
    df = df.replace([np.inf, -np.inf], np.nan)

    return df, dates


# ============================================================
# Calculate rolling correlation matrices
# ============================================================

def calculate_rolling_correlations(
    df,
    interval_minutes,
    window_days,
    stride_days
):
    window_size = int(
        window_days * 24 * 60 / interval_minutes
    )

    stride = int(
        stride_days * 24 * 60 / interval_minutes
    )

    if len(df) < window_size:
        raise ValueError(
            f"Not enough data: {len(df)} rows, "
            f"but {window_size} rows are required."
        )

    matrices = []
    window_indices = []

    for start in range(
        0,
        len(df) - window_size + 1,
        stride
    ):
        end = start + window_size

        window = df.iloc[start:end]

        corr = window.corr(
            method="pearson",
            min_periods=max(30, int(window_size * 0.5))
        )

        matrices.append(corr.to_numpy())
        window_indices.append(start)

    return (
        np.stack(matrices),
        df.columns.tolist(),
        window_indices,
        window_size
    )


# ============================================================
# Calculate stability metrics
# ============================================================


def calculate_metrics(matrices, columns):
    n_windows, n_vars, _ = matrices.shape

    upper_i, upper_j = np.triu_indices(
        n_vars, k=1
    )

    # Extract correlation coefficients for each pair
    pair_corr = matrices[:, upper_i, upper_j]

    # Calculate valid observation ratio for each pair
    valid_ratio = np.mean(
        np.isfinite(pair_corr), axis=0
    )

    # Keep pairs with at least 95% valid windows
    valid_pairs = valid_ratio >= 0.95

    pair_corr = pair_corr[:, valid_pairs]
    upper_i = upper_i[valid_pairs]
    upper_j = upper_j[valid_pairs]
    valid_ratio = valid_ratio[valid_pairs]

    if pair_corr.shape[1] == 0:
        raise ValueError(
            "No variable pairs with sufficient valid windows."
        )

    # Pair-level statistics using available windows
    mean_corr = np.nanmean(pair_corr, axis=0)

    std_corr = np.nanstd(
        pair_corr, axis=0, ddof=0
    )

    mean_abs_corr = np.nanmean(
        np.abs(pair_corr), axis=0
    )

    # Coefficient of variation
    cov = np.full_like(mean_corr, np.nan)

    valid_cov = np.abs(mean_corr) >= COV_EPSILON

    cov[valid_cov] = (
        std_corr[valid_cov] /
        np.abs(mean_corr[valid_cov])
    )

    # Consecutive-window correlation changes
    pair_diff = np.abs(np.diff(pair_corr, axis=0))

    mean_pair_change = np.nanmean(
        pair_diff, axis=0
    )

    # Pair-level results
    pair_results = pd.DataFrame({
        "variable_1": [
            columns[i] for i in upper_i
        ],
        "variable_2": [
            columns[j] for j in upper_j
        ],
        "valid_window_ratio": valid_ratio,
        "mean_correlation": mean_corr,
        "mean_absolute_correlation": mean_abs_corr,
        "std_correlation": std_corr,
        "mean_temporal_change": mean_pair_change,
        "cov": cov,
    })

    # Dataset-level results
    summary = {
        "n_variables": n_vars,
        "n_windows": n_windows,
        "n_pairs": len(pair_corr[0]),
        "mean_absolute_correlation":
            np.mean(mean_abs_corr),
        "mean_correlation_std":
            np.mean(std_corr),
        "mean_cov":
            np.nanmean(cov),
        "mean_temporal_change":
            np.nanmean(mean_pair_change),
        "median_temporal_change":
            np.nanmedian(mean_pair_change),
    }

    return summary, pair_results, mean_pair_change



# ============================================================
# Plot correlation stability
# ============================================================

def plot_temporal_change(
    dataset_name,
    temporal_change,
    output_dir
):
    if len(temporal_change) == 0:
        return

    plt.figure(figsize=(10, 4))

    plt.plot(
        np.arange(len(temporal_change)),
        temporal_change
    )

    plt.xlabel("Window transition")
    plt.ylabel("Mean absolute correlation change")
    plt.title(
        f"{dataset_name}: Temporal correlation change"
    )

    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_dir /
        f"{dataset_name}_temporal_change.png",
        dpi=300
    )

    plt.close()


def plot_summary(summary_df, output_dir):
    metrics = [
        ("mean_absolute_correlation",
         "Mean absolute correlation"),
        ("mean_correlation_std",
         "Correlation standard deviation"),
        ("mean_temporal_change",
         "Temporal correlation change"),
    ]

    for column, ylabel in metrics:
        plt.figure(figsize=(9, 5))

        plt.bar(
            summary_df["dataset"],
            summary_df[column]
        )

        plt.ylabel(ylabel)
        plt.title(ylabel + " across datasets")
        plt.xticks(rotation=30)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            output_dir / f"comparison_{column}.png",
            dpi=300
        )

        plt.close()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(DATASETS.keys()),
        help="Dataset names to analyze"
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    summaries = []

    for name in args.datasets:
        if name not in DATASETS:
            print(f"[Skip] Unknown dataset: {name}")
            continue

        config = DATASETS[name]
        path = config["path"]

        print(f"\nAnalyzing {name}...")
        print(f"Path: {path}")

        if not Path(path).exists():
            print(f"[Skip] File not found: {path}")
            continue

        df, dates = load_dataset(path)

        matrices, columns, indices, window_size = (
            calculate_rolling_correlations(
                df,
                config["interval_minutes"],
                WINDOW_DAYS,
                STRIDE_DAYS
            )
        )

        summary, pair_results, temporal_change = (
            calculate_metrics(
                matrices,
                columns
            )
        )

        summary["dataset"] = name
        summaries.append(summary)

        # Save pair-level statistics
        pair_results.to_csv(
            OUTPUT_DIR /
            f"{name}_pair_statistics.csv",
            index=False
        )

        # Save temporal correlation changes
        pd.DataFrame({
            "window_transition": np.arange(
                len(temporal_change)
            ),
            "mean_absolute_change": temporal_change
        }).to_csv(
            OUTPUT_DIR /
            f"{name}_temporal_change.csv",
            index=False
        )

        # Save rolling correlation matrices
        np.save(
            OUTPUT_DIR /
            f"{name}_correlation_matrices.npy",
            matrices
        )

        plot_temporal_change(
            name,
            temporal_change,
            OUTPUT_DIR
        )

        print("\nResults:")
        for key, value in summary.items():
            print(f"  {key}: {value}")

    if not summaries:
        print("No datasets were analyzed.")
        return

    summary_df = pd.DataFrame(summaries)

    summary_df = summary_df[
        [
            "dataset",
            "n_variables",
            "n_windows",
            "n_pairs",
            "mean_absolute_correlation",
            "mean_correlation_std",
            "mean_cov",
            "mean_temporal_change",
            "median_temporal_change",
        ]
    ]

    summary_df.to_csv(
        OUTPUT_DIR / "summary.csv",
        index=False
    )

    plot_summary(
        summary_df,
        OUTPUT_DIR
    )

    print("\n================================")
    print("Analysis completed!")
    print(f"Results: {OUTPUT_DIR.resolve()}")
    print("================================")

    print("\nSummary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
