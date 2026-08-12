from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DAILY_PERIOD, WEEKLY_PERIOD, HORIZON


def mean_forecast(history, horizon=HORIZON):
    return np.full(horizon, history.mean())


def naive_forecast(history, horizon=HORIZON):
    return np.full(horizon, history.iloc[-1])


def seasonal_naive_forecast(history, horizon=HORIZON, seasonality=DAILY_PERIOD):
    if len(history) < seasonality:
        raise ValueError("History shorter than seasonal period")
    recent = list(history.values[-seasonality:])
    return np.array([recent[step % seasonality] for step in range(horizon)], dtype=float)


def drift_forecast(history, horizon=HORIZON):
    if len(history) < 2:
        return naive_forecast(history, horizon)
    slope = (history.iloc[-1] - history.iloc[0]) / (len(history) - 1)
    return np.array([history.iloc[-1] + slope * step for step in range(1, horizon + 1)])


BENCHMARKS = {
    "mean": lambda h, n: mean_forecast(h, n),
    "naive": lambda h, n: naive_forecast(h, n),
    "seasonal_naive_daily": lambda h, n: seasonal_naive_forecast(h, n, DAILY_PERIOD),
    "seasonal_naive_weekly": lambda h, n: seasonal_naive_forecast(h, n, WEEKLY_PERIOD),
    "drift": lambda h, n: drift_forecast(h, n),
}


def rolling_benchmark_forecasts(series: pd.Series, origins: list[dict], horizon: int = HORIZON) -> pd.DataFrame:
    records = []
    for name, fn in BENCHMARKS.items():
        for origin in origins:
            cutoff = origin["cutoff_pos"]
            history = series.iloc[:cutoff]
            actual = series.iloc[cutoff:cutoff + horizon]
            pred = np.asarray(fn(history, horizon), dtype=float)
            records.append(pd.DataFrame({
                "origin_id": origin["origin_id"],
                "timestamp": actual.index,
                "step": np.arange(1, len(actual) + 1),
                "actual": actual.to_numpy(),
                "prediction": pred[:len(actual)],
                "model": name,
            }))
    return pd.concat(records, ignore_index=True)
