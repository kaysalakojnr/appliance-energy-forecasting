from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TARGET, HORIZON, INDOOR_TEMP, INDOOR_RH, WEATHER_COLS

TARGET_LAGS = [24, 25, 26, 48, 72, 168, 336]
ROLLING_WINDOWS = [24, 72, 168]


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Deterministic calendar features known for any future timestamp."""
    idx = frame.index
    out = pd.DataFrame(index=idx)
    out["hour"] = idx.hour
    out["dayofweek"] = idx.dayofweek
    out["day_of_month"] = idx.day
    out["week_of_year"] = idx.isocalendar().week.astype(int).to_numpy()
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    for k in (1, 2, 3):
        out[f"hour_sin{k}"] = np.sin(2 * np.pi * k * idx.hour / 24)
        out[f"hour_cos{k}"] = np.cos(2 * np.pi * k * idx.hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    out["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    out["is_evening_peak"] = idx.hour.isin([17, 18, 19]).astype(int)
    out["is_overnight"] = idx.hour.isin([0, 1, 2, 3, 4, 5]).astype(int)
    out["is_morning_ramp"] = idx.hour.isin([6, 7, 8, 9]).astype(int)
    return out


def add_target_lags(frame: pd.DataFrame, target: str = TARGET, horizon: int = HORIZON) -> pd.DataFrame:
    """Lag/rolling target features using no target value newer than the horizon."""
    out = pd.DataFrame(index=frame.index)
    values = frame[target]
    for lag in TARGET_LAGS:
        out[f"lag_{lag}"] = values.shift(lag)
    safe = values.shift(horizon)
    for window in ROLLING_WINDOWS:
        out[f"roll_mean_{window}"] = safe.rolling(window).mean()
        out[f"roll_std_{window}"] = safe.rolling(window).std()
        out[f"roll_min_{window}"] = safe.rolling(window).min()
        out[f"roll_max_{window}"] = safe.rolling(window).max()
    out["same_hour_mean_7d"] = safe.groupby(frame.index.hour).transform(
        lambda s: s.rolling(7, min_periods=1).mean())
    out["same_hour_std_7d"] = safe.groupby(frame.index.hour).transform(
        lambda s: s.rolling(7, min_periods=1).std())
    out["diff_24_48"] = out["lag_24"] - out["lag_48"]
    out["diff_24_168"] = out["lag_24"] - out["lag_168"]
    out["ratio_24_roll168"] = out["lag_24"] / (out["roll_mean_168"] + 1e-6)
    return out


def add_sensor_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Indoor aggregates and physically interpretable derived quantities."""
    out = pd.DataFrame(index=frame.index)
    temps = frame[INDOOR_TEMP]
    rh = frame[INDOOR_RH]
    out["indoor_temp_mean"] = temps.mean(axis=1)
    out["indoor_temp_std"] = temps.std(axis=1)
    out["indoor_temp_range"] = temps.max(axis=1) - temps.min(axis=1)
    out["indoor_rh_mean"] = rh.mean(axis=1)
    out["indoor_rh_std"] = rh.std(axis=1)
    out["indoor_rh_range"] = rh.max(axis=1) - rh.min(axis=1)
    out["temp_gradient"] = out["indoor_temp_mean"] - frame["T_out"]
    out["rh_gradient"] = out["indoor_rh_mean"] - frame["RH_out"]
    out["dewpoint_depression"] = frame["T_out"] - frame["Tdewpoint"]
    out["kitchen_living_temp"] = (frame["T1"] + frame["T2"]) / 2
    out["kitchen_rh_delta"] = frame["RH_1"] - out["indoor_rh_mean"]
    return out


def build_feature_table(frame: pd.DataFrame, regime: str = "A", target: str = TARGET,
                        horizon: int = HORIZON) -> pd.DataFrame:
    """Build Regime A (forecast-realistic) or Regime B (conditional) features."""
    if regime not in {"A", "B"}:
        raise ValueError("regime must be 'A' or 'B'")
    channels = add_sensor_features(frame).join(
        frame[INDOOR_TEMP + INDOOR_RH + WEATHER_COLS + ["lights"]]
    )
    lagged = channels.shift(horizon)
    lagged.columns = [f"{c}_lag{horizon}" for c in channels.columns]
    blocks = [add_time_features(frame), add_target_lags(frame, target, horizon), lagged]
    if regime == "B":
        future = channels.copy()
        future.columns = [f"{c}_future" for c in channels.columns]
        blocks.append(future)
    table = pd.concat(blocks, axis=1)
    table[target] = frame[target]
    return table.dropna()


def feature_groups(table: pd.DataFrame) -> dict[str, list[str]]:
    """Return the groups used in the Part 6 feature-ablation experiment."""
    columns = [c for c in table.columns if c != TARGET]
    calendar = [c for c in columns if c in set(add_time_features(pd.DataFrame(index=table.index)).columns)]
    target_lags = [c for c in columns if c.startswith("lag_") or c.startswith("diff_")]
    target_rolling = [c for c in columns if c.startswith("roll_") or c.startswith("same_hour_") or c.startswith("ratio_")]
    weather_tokens = tuple(WEATHER_COLS)
    weather = [c for c in columns if c.endswith(f"_lag{HORIZON}") and any(w in c for w in weather_tokens)]
    lights = [c for c in columns if c.startswith("lights_")]
    used = set(calendar + target_lags + target_rolling + weather + lights)
    indoor = [c for c in columns if c not in used]
    return {
        "calendar": calendar,
        "target_rolling": target_rolling,
        "target_lags": target_lags,
        "indoor_sensors": indoor,
        "lights": lights,
        "weather": weather,
    }
