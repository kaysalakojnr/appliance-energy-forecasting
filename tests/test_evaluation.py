import numpy as np
from appliance_energy.evaluation import mae, rmse, mase_scale, evaluate, make_origins

def test_perfect_forecast_has_zero_error():
    y = np.arange(1, 50, dtype=float)
    scale = mase_scale(y, seasonality=1)
    scores = evaluate("perfect", y, y, scale)
    assert scores["MAE"] == 0
    assert scores["RMSE"] == 0
    assert scores["MASE"] == 0

def test_origins_tile_holdout():
    import pandas as pd
    idx = pd.date_range("2016-01-01", periods=500, freq="h")
    origins = make_origins(idx, test_steps=336, horizon=24)
    assert len(origins) == 14
    assert origins[0]["cutoff_pos"] == 164
    assert origins[-1]["last_forecast"] == idx[-1]
