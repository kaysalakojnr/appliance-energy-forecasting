from __future__ import annotations

import itertools
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from ..config import (DAILY_PERIOD, HORIZON, SARIMA_ORDER, SARIMA_SEASONAL_ORDER,
                      SARIMAX_WEATHER_COLS)


def build_calendar(index) -> pd.DataFrame:
    frame = pd.DataFrame(index=index)
    hour = index.hour.to_numpy()
    dow = index.dayofweek.to_numpy()
    for k in (1, 2, 3):
        frame[f"hour_sin{k}"] = np.sin(2 * np.pi * k * hour / 24)
        frame[f"hour_cos{k}"] = np.cos(2 * np.pi * k * hour / 24)
    frame["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    frame["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    frame["is_weekend"] = (dow >= 5).astype(float)
    return frame


def fit_sarimax(y, order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL_ORDER, exog=None):
    return SARIMAX(y, exog=exog, order=order, seasonal_order=seasonal_order,
                   enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)


def rolling_sarimax_forecast(series: pd.Series, origins: list[dict], name: str,
                             exog: pd.DataFrame | None = None,
                             order=SARIMA_ORDER,
                             seasonal_order=SARIMA_SEASONAL_ORDER,
                             horizon: int = HORIZON,
                             alpha: float = 0.05) -> pd.DataFrame:
    """Fit once at the first origin and state-update without parameter refits."""
    first_cutoff = origins[0]["cutoff_pos"]
    current = fit_sarimax(series.iloc[:first_cutoff], order, seasonal_order,
                          None if exog is None else exog.iloc[:first_cutoff])
    records = []
    for origin in origins:
        cutoff = origin["cutoff_pos"]
        if cutoff > first_cutoff:
            new_endog = series.iloc[cutoff - horizon:cutoff]
            new_exog = None if exog is None else exog.iloc[cutoff - horizon:cutoff]
            current = current.append(new_endog, exog=new_exog, refit=False)
        actual = series.iloc[cutoff:cutoff + horizon]
        future_exog = None if exog is None else exog.iloc[cutoff:cutoff + horizon]
        forecast = current.get_forecast(steps=len(actual), exog=future_exog)
        interval = forecast.conf_int(alpha=alpha)
        records.append(pd.DataFrame({
            "origin_id": origin["origin_id"], "timestamp": actual.index,
            "step": np.arange(1, len(actual) + 1), "actual": actual.to_numpy(),
            "prediction": forecast.predicted_mean.to_numpy(),
            "lower": interval.iloc[:, 0].to_numpy(),
            "upper": interval.iloc[:, 1].to_numpy(), "model": name,
        }))
    return pd.concat(records, ignore_index=True)


def make_exogenous_frames(hourly: pd.DataFrame) -> dict[str, pd.DataFrame | None]:
    calendar = build_calendar(hourly.index)
    weather = pd.concat([calendar, hourly[SARIMAX_WEATHER_COLS]], axis=1)
    return {"sarima": None, "sarimax_calendar": calendar, "sarimax_weather": weather}


def grid_search_nonseasonal(y: pd.Series) -> pd.DataFrame:
    """Required 147-model p,d,q search by AIC."""
    records = []
    for p, d, q in itertools.product(range(7), range(3), range(7)):
        try:
            fitted = ARIMA(y, order=(p, d, q), enforce_stationarity=False,
                           enforce_invertibility=False).fit()
            records.append({"p": p, "d": d, "q": q, "aic": fitted.aic,
                            "bic": fitted.bic, "converged": True})
        except Exception:
            records.append({"p": p, "d": d, "q": q, "aic": np.nan,
                            "bic": np.nan, "converged": False})
    return pd.DataFrame(records).sort_values("aic", na_position="last").reset_index(drop=True)


def grid_search_seasonal(y: pd.Series, base_orders: list[tuple[int, int, int]], period: int = DAILY_PERIOD) -> pd.DataFrame:
    """Search (P,D,Q) in {0,1}^3 on the strongest base orders."""
    records = []
    for order in base_orders:
        for P, D, Q in itertools.product((0, 1), repeat=3):
            seasonal = (P, D, Q, period)
            try:
                fitted = SARIMAX(y, order=order, seasonal_order=seasonal,
                                 enforce_stationarity=False,
                                 enforce_invertibility=False).fit(disp=False)
                records.append({"order": str(order), "seasonal_order": str(seasonal),
                                "p": order[0], "d": order[1], "q": order[2],
                                "P": P, "D": D, "Q": Q,
                                "aic": fitted.aic, "bic": fitted.bic,
                                "converged": True})
            except Exception:
                records.append({"order": str(order), "seasonal_order": str(seasonal),
                                "p": order[0], "d": order[1], "q": order[2],
                                "P": P, "D": D, "Q": Q,
                                "aic": np.nan, "bic": np.nan,
                                "converged": False})
    return pd.DataFrame(records).sort_values("aic", na_position="last").reset_index(drop=True)
