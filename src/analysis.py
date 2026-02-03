from __future__ import annotations

"""Offline analysis for time-based patterns in negative tweets."""

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

from src import config
from src.features_time import DAY_ORDER, SEASON_ORDER, TIME_INTERVAL_ORDER, add_time_features
from src.io import read_csv


def _negative_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Return only negative tweets using either legacy or normalized labels."""
    if "Durum" in df.columns:
        return df.loc[df["Durum"] == -1]

    if "label" in df.columns:
        labels = df["label"]
        if labels.dtype.kind in {"i", "f"}:
            return df.loc[labels == -1]
        return df.loc[labels.astype(str).str.lower().isin({"negative", "negatif", "-1"})]

    raise ValueError("No sentiment label column found (expected 'Durum' or 'label').")


def _print_frequency_table(df: pd.DataFrame, col: str) -> None:
    """Print counts and ratios for a single categorical column."""
    counts = df[col].value_counts(dropna=False)
    ratio = 100 * counts / len(df) if len(df) else 0
    print(pd.DataFrame({col: counts, "Ratio": ratio}))
    print("-" * 45)


def _save_bar_plot(df: pd.DataFrame, col: str, output_path: Path, order: Iterable[str]) -> None:
    """Save a simple matplotlib bar chart (no seaborn dependency)."""
    counts = df[col].value_counts().reindex(order, fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    ax.set_title(f"Negative tweets by {col}")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def run_analysis() -> None:
    """Compute negative sentiment distributions and write plots to reports/."""
    data_path = config.DATA_DIR / "tweets_labeled.csv"
    df = read_csv(data_path)

    df = add_time_features(df)
    negative_df = _negative_filter(df)

    if negative_df.empty:
        print("No negative tweets found for analysis.")
        return

    # Keep the original reporting order from the prototype.
    cols = ["time_interval", "days", "seasons"]
    for col in cols:
        _print_frequency_table(negative_df, col)

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _save_bar_plot(
        negative_df,
        "time_interval",
        config.REPORTS_DIR / "neg_by_time_interval.png",
        TIME_INTERVAL_ORDER,
    )
    _save_bar_plot(
        negative_df,
        "days",
        config.REPORTS_DIR / "neg_by_day.png",
        DAY_ORDER,
    )
    _save_bar_plot(
        negative_df,
        "seasons",
        config.REPORTS_DIR / "neg_by_season.png",
        SEASON_ORDER,
    )


if __name__ == "__main__":
    run_analysis()
