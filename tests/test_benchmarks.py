import numpy as np
import pandas as pd
from appliance_energy.models.benchmarks import seasonal_naive_forecast, naive_forecast

def test_naive_length_and_value():
    s = pd.Series(np.arange(200, dtype=float))
    pred = naive_forecast(s, 24)
    assert len(pred) == 24
    assert np.all(pred == s.iloc[-1])

def test_weekly_seasonal_naive_uses_lag_168():
    s = pd.Series(np.arange(300, dtype=float))
    pred = seasonal_naive_forecast(s, horizon=24, seasonality=168)
    np.testing.assert_array_equal(pred, s.iloc[-168:-144].to_numpy())
