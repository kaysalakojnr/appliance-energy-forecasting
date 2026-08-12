from __future__ import annotations

import inspect
import numpy as np
import pandas as pd

from ..config import CHRONOS_MODEL, CHRONOS_CONTEXT, CHRONOS_QUANTILES, HORIZON


def load_chronos(model_name: str = CHRONOS_MODEL):
    """Load Chronos-Bolt lazily so core tests do not require the dependency."""
    try:
        import torch
        from chronos import BaseChronosPipeline
    except ImportError as exc:
        raise RuntimeError(
            "Chronos is unavailable. Install `chronos-forecasting` and `torch`."
        ) from exc
    if torch.cuda.is_available():
        device, dtype = "cuda", torch.bfloat16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device, dtype = "mps", torch.float32
    else:
        device, dtype = "cpu", torch.float32
    pipeline = BaseChronosPipeline.from_pretrained(
        model_name, device_map=device, torch_dtype=dtype)
    return pipeline, torch


def _predict_quantiles(pipeline, tensor, horizon, quantiles):
    try:
        return pipeline.predict_quantiles(tensor, horizon, quantiles)
    except TypeError:
        first = list(inspect.signature(pipeline.predict_quantiles).parameters)[0]
        return pipeline.predict_quantiles(**{
            first: tensor, "prediction_length": horizon,
            "quantile_levels": quantiles,
        })


def _to_array(quantile_forecast, horizon, n_quantiles):
    values = quantile_forecast[0]
    values = values.detach().cpu().numpy() if hasattr(values, "detach") else np.asarray(values)
    if values.shape == (horizon, n_quantiles):
        return values
    if values.shape == (n_quantiles, horizon):
        return values.T
    raise ValueError(f"Unexpected Chronos shape: {values.shape}")


def rolling_chronos_forecast(series: pd.Series, origins: list[dict],
                             context_length: int = CHRONOS_CONTEXT,
                             horizon: int = HORIZON,
                             quantiles: list[float] = CHRONOS_QUANTILES,
                             model_name: str = CHRONOS_MODEL) -> pd.DataFrame:
    pipeline, torch = load_chronos(model_name)
    records = []
    for origin in origins:
        cutoff = origin["cutoff_pos"]
        context = series.iloc[max(0, cutoff - context_length):cutoff]
        actual = series.iloc[cutoff:cutoff + horizon]
        tensor = torch.tensor(context.to_numpy(dtype="float32"))
        q_forecast, _ = _predict_quantiles(pipeline, tensor, horizon, quantiles)
        values = _to_array(q_forecast, horizon, len(quantiles))
        records.append(pd.DataFrame({
            "origin_id": origin["origin_id"], "timestamp": actual.index,
            "step": np.arange(1, horizon + 1), "actual": actual.to_numpy(),
            "prediction": values[:, quantiles.index(0.5)],
            "lower": values[:, quantiles.index(0.1)],
            "upper": values[:, quantiles.index(0.9)],
            "q25": values[:, quantiles.index(0.25)],
            "q75": values[:, quantiles.index(0.75)],
            "model": "chronos_zeroshot",
        }))
    return pd.concat(records, ignore_index=True)
