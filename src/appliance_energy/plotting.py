from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_model_mae(scores: pd.DataFrame, path: Path) -> None:
    ordered = scores.sort_values("MAE", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(ordered["model"], ordered["MAE"])
    ax.set_xlabel("MAE (Wh)")
    ax.set_title("Model comparison on the held-out period")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_forecast_comparison(forecasts: pd.DataFrame, path: Path,
                             max_origins: int = 4) -> None:
    subset = forecasts[forecasts["origin_id"] <= max_origins]
    actual = subset.drop_duplicates("timestamp").sort_values("timestamp")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(actual["timestamp"], actual["actual"], linewidth=2.2, label="Actual")
    for model, group in subset.groupby("model"):
        ax.plot(group["timestamp"], group["prediction"], label=model, alpha=0.8)
    ax.set_ylabel("Appliance energy (Wh)")
    ax.set_title(f"Forecast comparison: first {max_origins} origins")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
