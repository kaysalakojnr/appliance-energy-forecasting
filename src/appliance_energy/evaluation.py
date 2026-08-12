from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import DAILY_PERIOD, HORIZON, TEST_STEPS


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))


def rmse(y_true, y_pred) -> float:
    err = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(err ** 2)))


def bias(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)))


def smape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    terms = np.divide(np.abs(y_true - y_pred), denom,
                      out=np.zeros_like(denom, dtype=float), where=denom != 0)
    return float(np.mean(terms) * 100)


def mase_scale(y_train, seasonality: int = DAILY_PERIOD) -> float:
    values = np.asarray(y_train, dtype=float)
    if len(values) <= seasonality:
        raise ValueError("Training data must exceed the MASE seasonality.")
    return float(np.mean(np.abs(values[seasonality:] - values[:-seasonality])))


def evaluate(name, y_true, y_pred, scale) -> dict:
    value_mae = mae(y_true, y_pred)
    return {
        "model": name,
        "MAE": value_mae,
        "RMSE": rmse(y_true, y_pred),
        "MASE": value_mae / scale if scale else np.nan,
        "sMAPE": smape(y_true, y_pred),
        "Bias": bias(y_true, y_pred),
    }


def picp(y_true, lower, upper) -> float:
    y = np.asarray(y_true, dtype=float)
    return float(np.mean((y >= np.asarray(lower)) & (y <= np.asarray(upper))))


def mpiw(lower, upper) -> float:
    return float(np.mean(np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)))


def make_origins(index, test_steps: int = TEST_STEPS, horizon: int = HORIZON) -> list[dict]:
    """Create non-overlapping rolling origins that tile the holdout exactly."""
    if test_steps % horizon:
        raise ValueError("test_steps must be divisible by horizon")
    test_start = len(index) - test_steps
    if test_start <= 0:
        raise ValueError("Not enough data for the requested holdout.")
    return [
        {
            "origin_id": block + 1,
            "cutoff_pos": test_start + block * horizon,
            "train_end": index[test_start + block * horizon - 1],
            "first_forecast": index[test_start + block * horizon],
            "last_forecast": index[test_start + (block + 1) * horizon - 1],
            "n_train": test_start + block * horizon,
        }
        for block in range(test_steps // horizon)
    ]


def score_forecast_frame(frame: pd.DataFrame, scale: float) -> pd.DataFrame:
    rows = []
    for model, group in frame.groupby("model"):
        record = evaluate(model, group["actual"], group["prediction"], scale)
        if {"lower", "upper"}.issubset(group.columns) and group[["lower", "upper"]].notna().all().all():
            record["PICP"] = picp(group["actual"], group["lower"], group["upper"])
            record["MPIW"] = mpiw(group["lower"], group["upper"])
        rows.append(record)
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)


def diebold_mariano(error_a, error_b, horizon: int = 1) -> dict:
    """Harvey-Leybourne-Newbold corrected DM test using absolute-error loss."""
    e1 = np.asarray(error_a, dtype=float)
    e2 = np.asarray(error_b, dtype=float)
    if len(e1) != len(e2):
        raise ValueError("Error series must have equal length.")
    d = np.abs(e1) - np.abs(e2)
    n = len(d)
    mean_d = d.mean()
    gamma0 = np.mean((d - mean_d) ** 2)
    variance = gamma0
    for lag in range(1, min(horizon, n)):
        gamma = np.mean((d[lag:] - mean_d) * (d[:-lag] - mean_d))
        variance += 2 * gamma
    if variance <= 0:
        return {"statistic": np.nan, "p_value": np.nan}
    dm = mean_d / np.sqrt(variance / n)
    correction = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    dm *= correction
    p = 2 * stats.t.sf(abs(dm), df=n - 1)
    return {"statistic": float(dm), "p_value": float(p)}
